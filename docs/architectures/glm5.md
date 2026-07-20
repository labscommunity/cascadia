# GLM-5.2 (`glm5`)

GLM-5.2 (Z.ai / Zhipu, ~744B MoE, ~40B active/token, MIT weights) exported to
int4 and run distributed across **N Intel AI-PCs** (target 4× 32 GB) via the
`cascadia-engine-sparse-moe` engine — one pipeline stage per node. The rank count
is a runtime parameter (`total`), never hardcoded; layers split evenly across
ranks. Attention is DeepSeek-V3-style **MLA + DSA**, a different family from the
V4 shell in this crate, so `src/glm/` is mostly a rewrite, not an adapt.

**Status: implemented** on `pawan/glm-5-2` (single branch, incremental commits).
Numerics are golden-tested 1:1 against a Python CPU reference (`tools/glm5_ref`);
the FP8→int4 exporter is validated by a synthetic round-trip; the pipeline is
parity-tested across N ranks over the real loopback transport. Not yet run on the
744B checkpoint or real hardware — that is the export + deploy phase.

## Architecture

| Param | Value |
|---|---|
| layers | 78 (`first_k_dense_replace=3` real dense MLP + 75 MoE); MTP `num_nextn_predict_layers=1` at layer 78 |
| hidden / vocab | 6144 / 154,880 |
| dense FFN | `intermediate_size=12288` (layers 0–2) |
| experts | 256 routed + 1 shared, **top-8**, `moe_intermediate=2048`; `n_group=1, topk_group=1` (NO grouped/node-limited routing) |
| routing | `scoring_func="sigmoid"` + `noaux_tc` bias, norm-topk, `routed_scaling_factor=2.5` |
| attention | **classic V3 MLA**: `q_lora_rank=2048`, `kv_lora_rank=512`, 64 heads (no GQA), `qk_nope=192` + `qk_rope=64` (qk_head_dim 256), `v_head_dim=256`; KV latent = 512 + 64 rope = **576 f/tok** |
| sparse attn | **DSA / IndexShare**: lightning indexer scores **raw** positions → top-`index_topk` (2048) causal keys; `index_n_heads=32`, `index_head_dim=128`. **IndexShare**: only `"full"` layers (`config.indexer_types`, 22 of 79 — layers 0,1,2 then every 4th) own an indexer; `"shared"` layers reuse the previous full layer's top-k (carry-forward). Indexer query proj = `wq_b`; interleaved rope; weights FP8 |
| residual | plain `[b, s, 6144]` — no Hyper-Connections |
| rope | `rope_theta=8e6`, interleaved, `rope_type="default"` (**no YaRN**), `max_position=1,048,576` (1M ctx) |
| quant | int4 experts (per-row scales, group-32), FP8 block-128 source; MTP head int8 (int4 collapses accept) |

Numeric contract (Rust shell == CPU ref): bf16 rounding after each linear /
norm / rope; f32 accumulation in the attention absorb core, router logits, and
softmax. MLA `q_a`/`kv_a` layernorms use eps 1e-6 (HF RMSNorm default), the
other norms `rms_norm_eps` 1e-5. Absorbed-latent MLA decode (576-float latent
cache) is the only memory-feasible form at 1M ctx.

## Size / feasibility

```
int4 expert        ~= 18.9 MB   (3 * 6144 * 2048 * 0.5 B)
routed total       ~= 356 GB    (75 MoE layers * 256) + ~1.4 GB shared
int4 export        ~= 386 GB    (from a ~755 GB FP8 checkpoint)
per node (N=4)     ~= 89 GB experts  vs 32 GB RAM
```

Throughput is gated by **residency** (how much of the expert set is RAM-resident),
not engine math. At N=4 ~35% is resident, so the node streams the rest from NVMe
→ ~0.4–0.6 tok/s single-stream: async/batch + structured-output territory (the
coding-agent workload), not interactive. Residency improves with N (each node's
slice shrinks): N=8 → ~45 GB, N=16 → ~22 GB (fits RAM). Cap `max_seq` from the
deployment, never from `max_position` (1M would preallocate TB-scale KV).

## What's implemented

- **Primitives** (`src/glm/`), each golden-tested vs `tools/glm5_ref`: sigmoid+
  noaux_tc gate, interleaved RoPE, MLA attention (absorbed decode) **with DSA
  indexer selection**, DSA lightning indexer, SwiGLU FFN, MoE block, MTP head,
  full model.
- **Exporter** (`tools/export_glm5.py`): FP8 e4m3 block-128 dequant → int4
  repack + bf16 shells + DSA indexer weights; hard-fails on config surprises;
  resumable (`.done` markers + disk pre-flight). MTP weights not exported yet.
- **Loader / stage / pipeline**: per-rank layer slice, `arch=="glm5"` sniff,
  N-general even split, shared dsv4 TCP transport. `{1,2,4}`-rank chains match
  single-process bit-for-bit (incl. middle-relay ranks).
- **Throughput**: batch-union MoE (dedup expert loads) wired through single-
  process prefill, N=1 serve, and the N>1 pipeline driver (`ForwardBatchPrefill`
  frame). Grammar-constrained decoding with forced-run batching (K forced tokens
  ≈ 1 forward; no draft head). Learned-pin residency (`.coli_usage` histogram +
  `mlock` of the hottest experts, per-node budget from the layer slice).

## Reuse map (vs the `dsv4` shell)

| Reuse | Rewrite (different family) | Drop |
|---|---|---|
| `dsv4::math` (bf16/linear/dot/rmsnorm), `rope::apply_rope_row`, `expert_mmap` int4 dispatch, the pipeline wire (already N-rank), `sampling`, `StagedRunner` | `attn`/`indexer`/`model`/`loader`/`stage`: V3 MLA + raw-position DSA + sigmoid gate + real dense first-3 | Hyper-Connections, block-compressed KV, hash routing |

## Key decisions

1. **Sibling `src/glm/`, not a generalized core** — dsv4 is shipped; share only
   pure leaves.
2. **Absorbed MLA decode** — the only memory-feasible form at long ctx + DSA.
3. **Family dispatch via `arch=="glm5"`** on the raw manifest.
4. **Python CPU ref is the single bit-exact ground truth**; the exporter
   hard-fails on config surprises rather than defaulting silently.
5. **N-node, not 4-node** — every budget / pin set / shard derives from the
   runtime `total` and this rank's slice; test matrix spans {1,2,4}.

## Optimization playbook (ranked; residency > kernel)

1. **RAM-residency budget** (`cap_for_ram`) + OOM refusal — done.
2. **Learned pin** (`.coli_usage` + AUTOPIN, `mlock`) — done.
3. **Batch-union MoE** (dedup experts, block-of-64) — done (prefill path).
4. **Grammar-forced drafting** — done (mechanism; real JSON/GBNF grammar needs
   the deployment tokenizer).
5. **AVX-VNNI int4×int8 kernel** (Core Ultra: 256-bit VNNI, no AVX-512) —
   deferred; must be benchmarked on Intel hardware.
6. **Router-lookahead + coupling prefetch** to hide interconnect — deferred.
7. **MTP int8 spec-decode** — deferred, conditional on a batched-verify microbench
   showing net gain; needs KV rewind + accept/reject.
8. **KV persistence** for zero re-prefill across restarts — deferred.

## Remaining before / at the hardware phase

- Real 744B export (needs the FP8 checkpoint + ~386 GB disk) — this also confirms
  the DSA/MTP tensor names against the real config (the exporter loud-fails on a
  mismatch).
- Deploy across the AI-PCs; measure real tok/s and residency hit-rate.
- The deferred optimizations above, prioritized by what the measurements show.

## Open risks

- **Residency / hit-rate** (highest) — mitigate with more nodes + the residency
  playbook + grammar drafting.
- **MLA attention shape** — the #1 correctness hazard; locked against the CPU ref.
- **Top-k tie ordering** vs `torch.topk` in the sigmoid gate — covered by goldens.
- **Cross-node sharding is net-new** — hiding interconnect latency behind
  lookahead/prefetch, parameterized by `total`, is unproven at scale.
