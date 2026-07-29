# Kimi-K3 (`kimi_k3`)

Moonshot **Kimi-K3** (`moonshotai/Kimi-K3`, ~2.8T MoE, 1M ctx) analysed against the
`cascadia-engine-sparse-moe` engine.

**Status: implemented, not yet run on real weights.** The shell
(`crates/cascadia-engine-sparse-moe/src/k3/`), the exporter
(`tools/export_kimi_k3.py`) and the CPU reference (`tools/kimi_k3_ref/`) are
complete and golden-tested; `{1,2,3,4,6}`-rank pipelines are bit-identical to a
single process. What remains is the real 1.56 TB export and bring-up.

**That is blocked on hardware.** K3 does **not fit the 4× 32 GB AI-PC fleet**
that dsv4 and glm5 target — the always-resident bf16 shell alone is ~112 GB
against 128 GB of total fleet RAM — and the Xeon bench host has neither the RAM
nor the disk. K3 is a single-big-host target (the K2.6 / MiniMax-M2 deployment
model) or it is parked. See [Feasibility](#feasibility).

## Architecture

From the real `config.json`. The checkpoint is multimodal
(`KimiK3ForConditionalGeneration`, 27-layer ViT); we drop the vision tower and
serve `text_config` only (`model_type: kimi_linear`).

| Param | Value |
|---|---|
| layers | 93 (`first_k_dense_replace=1` dense + 92 MoE, `moe_layer_freq=1`) |
| hidden / vocab | 7168 / 163,840; dense FFN `intermediate_size=33792`; `rms_norm_eps=1e-5` |
| experts | **896 routed + 2 shared, top-16**, `moe_intermediate_size=3072` |
| **LatentMoE** | routed experts run in a **3584-dim latent**, not hidden (`routed_expert_hidden_size=3584`, `latent_moe_use_norm=true`) — per-layer 7168↔3584 projections + RMSNorm |
| routing | `sigmoid` + `noaux_tc` bias, norm-topk, `routed_scaling_factor=1.0`, `num_expert_group=1, topk_group=1` (no group masking) |
| activation | **SiTU** (`hidden_act="situ"`, β=4.0, linear β=25.0) — model-global, NOT SwiGLU |
| attention | **hybrid 3:1** — 69 KDA layers : 24 full-attention layers |
| ├ KDA | Kimi Delta Attention (gated delta-rule linear attn): 96 heads × 128, `short_conv_kernel_size=4`, `gate_lower_bound=-5.0`, `use_full_rank_gate=true`. Fixed-size recurrent state, **no KV** |
| └ full attn | MLA: `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope=128` + `qk_rope=64`, `v_head_dim=128`, 96 heads. **`mla_use_nope=true`** (no RoPE at all) + **`mla_use_output_gate=true`** |
| residual | **AttnRes** — attention residual carried across 12-layer blocks (`attn_res_block_size=12`) |
| rope | none on full-attn layers (NoPE); `max_position_embeddings=1,048,576` |
| quant | **`mxfp4-pack-quantized`** (compressed-tensors): FP4 **e2m1** values + **E8M0 u8** group-32 scales, symmetric. Only routed experts are quantized — `self_attn`, `shared_experts`, dense MLP and `lm_head` are bf16 |

## Feasibility

### Routed experts (streamed)

```
per expert    = 3 * 3584 * 3072            =  33.0M params
routed total  = 92 layers * 896 * 33.0M    =  2.72T params   (matches the "2.8T" claim)
on disk @fp4  = 2.72T * 0.5 B * 1.0625     =  ~1.45 TB       (E8M0 u8 scales = +6.25%)
active/token  = 16 * 92 * 33.0M            =  48.6B params   = ~24 GB streamed/token at 0% hit
```

### Always-resident bf16 shell — the blocker

Everything below is in the quantization `ignore` list, so it is bf16 and must be
RAM-resident before a single routed expert is pinned:

| Component | Derivation | bf16 |
|---|---|---:|
| KDA attention | 69 × (4×7168×12288 qkvo + 7168×12288 full-rank gate) = 30.4B | **60.7 GB** |
| shared experts | 92 × 2 × 3 × 7168 × 3072 = 12.2B | **24.3 GB** |
| gated MLA | 24 × ~232M = 5.6B | 11.1 GB |
| LatentMoE projections | 92 × 2 × 7168 × 3584 = 4.7B | 9.4 GB |
| embed + lm_head | 2 × 163,840 × 7168 = 2.35B | 4.7 GB |
| dense layer 0 | 3 × 7168 × 33,792 = 727M | 1.5 GB |
| | | **~112 GB** |

Confirmed from the modeling source: `shared_experts` is built with no
`hidden_size` override, so it runs at 7168 rather than in the 3584 latent —
the more expensive of the two possibilities.

### Verdict

| Target | Result |
|---|---|
| **4 × 32 GB AI-PC fleet** | **infeasible** — 112 GB of resident shell vs 128 GB total fleet RAM; ~29 GB/node of shell against a 32 GB budget leaves nothing for experts |
| 8 × 32 GB | ~15 GB/node shell, ~15 GB/node for experts against a 181 GB/node slice → **~8% residency**, ~22 GB/token → **~0.2 tok/s**, worse than GLM-5.2 at N=4 |
| N for true residency | ~1.45 TB experts + shell over 32 GB nodes → **N ≈ 55** |
| **Xeon bench host** (~172 GB RAM, ~1.6 TB free) | **export: does not fit; run: ~0.1 tok/s** — see below |
| single host, ≥768 GB RAM + ~2 TB striped NVMe ≥10 GB/s | **viable** — 45–60% residency, ~10–13 GB/token → ~0.5–1 tok/s |

For scale: GLM-5.2 is ~386 GB int4 and already only reaches 0.4–0.6 tok/s at N=4
with ~35% residency. K3 is ~3.8× larger with 2× the active experts per token.

#### On the Xeon bench host

The same host produced the GLM-5.2 export (source FP8 755 GB + int4 output
386 GB = 1.14 TB, comfortably inside 1.6 TB). K3 does not have that headroom:

```
source checkpoint  = 1.45 TB routed (native mxfp4) + ~112 GB bf16  = ~1.57 TB
export output      = ~same (fp4 repack, no regrind)                = ~1.57 TB
both concurrently                                                  = ~3.1 TB   vs 1.6 TB free
```

The source **alone** is ~98% of free space. Even streaming with
`--free-source-shards`-style deletion there is no room for the output, and any
re-run means re-downloading ~1.57 TB. **Export needs roughly +2 TB of storage.**

Running is separately blocked: 172 GB RAM − ~112 GB resident shell leaves ~60 GB
for experts against 1.45 TB → **~4% residency** → ~23 GB streamed/token → at a
few GB/s **≈ 8–12 s/token ≈ 0.1 tok/s**. More storage fixes the export; only more
RAM fixes the run.

**K3 is a single-big-host model (the K2.6 / M2 deployment class), not a fleet
model — and no host currently in reach clears the bar.** The gap is ~2× on the
bench host's disk and ~4.5× on its RAM.

## Port plan (if the platform gate clears)

Follows the established sibling-shell pattern (`src/dsv4/`, `src/glm/`) — a Rust
shell validated bit-for-bit against a Python CPU reference, not OpenVINO-traced
graphs.

### Reuse map

| Reuse as-is | Adapt | Net-new |
|---|---|---|
| `dsv4::math` (bf16/linear/dot/rmsnorm), `staged::StagedRunner`, `sampling`, the dsv4 TCP pipeline wire, `glm::residency` (pin/mlock), `glm::gate::moe_gate` — sigmoid + `noaux_tc` + norm-topk is an **exact** match for K3's router | `glm::attn` **absorbed-decode structure only** (`qabs = W_UKᵀ·q`; `score = qabs·Lc`; `ctx = W_UV·clat`) — rewritten, not flagged: NoPE deletes the whole `Rc`/`k_pe` path, and the output gate has no hook | KDA layer (short conv + gated delta recurrence + full-rank gate), SiTU, LatentMoE block + projections, AttnRes block carry, **fp4 e2m1 expert kernel** |

### Corrections to the obvious-but-wrong approach

1. **`dsv4::expert_mmap` cannot decode K3 experts.** It is a *linear symmetric*
   grid — `Σ (nibble−8) · bf16_scale · x` (`expert_mmap.rs:438`). FP4 e2m1 is
   *nonlinear* (`{0, .5, 1, 1.5, 2, 3, 4, 6}±`) with u8 E8M0 scales. Add a new
   fp4 bin format + a 16-entry-LUT SIMD kernel (a mechanical variant of
   `dequant_row_dot_avx2/512`). Do **not** have the exporter regrind mxfp4 →
   linear int4: re-quantizing an already-4-bit grid whose levels (0.5/1.5/3)
   aren't on the linear grid is an unevaluated quality risk, and it costs
   +12.5% disk for bf16 scales instead of +6.25%.
2. **Batched prefill is mandatory from day one, not an optimization.**
   Per-token prefill streams the full active set per position: a 4k prompt at
   ~8% residency is ~4000 × 22 GB ≈ 80 TB of reads. Batch-union saturates
   essentially all 896 experts per layer in one pass ≈ 1.36 TB — a ~60×
   difference. KDA's sequential recurrence does **not** conflict with this:
   `glm/stage.rs:601` already runs per-position attention inside the batched
   layer loop, then unions the MoE across rows. K3 does the same, walking the
   recurrent state row by row (~1.6M MACs/position — noise next to the MoE).
   Chunked-scan (parallel delta rule) KDA stays deferred.
3. **AttnRes: widen the wire — boundary-snapping does not work.**
   *(Corrected after reading `modeling_kimi_linear.py`; an earlier revision of
   this doc said the opposite.)* AttnRes is **not** a carried anchor. Each layer
   holds a growing **stack** of per-block residuals and mixes over all of them
   with a learned softmax (`_apply_attn_res`, applied **twice per layer** — once
   before attention, once before the MLP). Because the mixture attends over
   *every* prior block, snapping rank starts to block boundaries saves nothing:
   the stack still has to cross the wire. The inter-stage activation is
   therefore `prefix_sum [H]` **+** `block_residual [n_blocks, H]` — up to
   9 × 7168 f32 ≈ 258 KB/token. This is the dsv4 Hyper-Connections situation
   (`dsv4/stage.rs:75`, `hidden = hc_mult * hidden_size`), not the glm5
   `index_aligned_split` one. Use an even layer split and widen the wire.
4. **Keep the sibling module.** 69 of 93 layers are linear attention — there is
   no MLA-shaped core to generalize dsv4/glm5 into. Cross-importing leaves is
   already the in-tree precedent (`glm/attn.rs` uses `dsv4::math`; `glm/stage.rs`
   uses `dsv4::stage::even_layer_split`).

### Phases

1. **Platform gate** — confirm a ≥768 GB host exists, or stop.
2. **`tools/kimi_k3_ref/`** — CPU reference + fixtures ported from the HF
   modeling source. Pins the AttnRes anchor semantics, the SiTU formula, the
   shared-expert width, and the MLA cache form *before* any Rust is written.
3. **`tools/export_kimi_k3.py`** — unwrap the multimodal wrapper / drop the ViT;
   routed experts to the new fp4 bin format (no regrind); bf16 shells incl. the
   LatentMoE projections; hard-fail on config surprises; resumable.
4. **`src/k3/` primitives**, each golden-tested against the ref: fp4 LUT expert
   kernel, SiTU, LatentMoE block, KDA layer, gated NoPE-MLA, AttnRes carry.
5. **Runner** — `forward_layers_batch` + batch-union MoE from the start;
   `block_aligned_split`; `reset()` must clear conv + recurrent state;
   `arch == "kimi_k3"` sniff in `engine.rs`.
6. **Parity** — tiny/med synthetic end-to-end; {1,2,4}-rank pipeline bit-match.
7. **Real export + single-host bring-up**, then residency tuning per the glm5
   playbook.

## Resolved math

Extracted from the real upstream sources, vendored under
`tools/kimi_k3_ref/upstream/` (`modeling_kimi_linear.py` from the HF repo;
`kda_naive.py` + `kda_gate.py` from `fla-org/flash-linear-attention`, MIT). These
are the load-bearing details `config.json` alone could not answer.

### SiTU (`SituAndMul`) — model-global activation

```
gate, up = split(x, 2)                      # computed in f32, cast back
situ_a   = β · tanh(gate/β) · sigmoid(gate)          β = 4.0
up'      = linear_β · tanh(up/linear_β)              linear_β = 25.0
out      = situ_a · up'
```

MLP is `w2( SiTU(cat[w1(x), w3(x)]) )` — w1=gate, w3=up, w2=down. Three separate
matrices in the checkpoint (concatenated only at runtime), so the existing
gate/up/down section layout in `expert_mmap` still applies.

### AttnRes — learned mixture over a growing block stack

```python
_apply_attn_res(prefix_sum, block_residual, proj, norm):
    v       = cat([block_residual (T,nb,H), prefix_sum (T,1,H)], dim=1)
    k       = v * rsqrt(mean(v², -1) + eps)          # RMS-normalise, weight not yet applied
    score_w = norm.weight * proj.weight              # proj: Linear(H, 1, bias=False)
    probs   = softmax((k * score_w).sum(-1), -1)
    return probs @ v
```

Per layer (`_forward_attn_residual`), with two independent (proj, norm) pairs:

```
prefix_sum = hidden_in
if block_residual non-empty:
    hidden = _apply_attn_res(prefix_sum, block_residual, self_attention_res_proj, self_attention_res_norm)
if layer_idx % 12 == 0:
    block_residual = cat([block_residual, prefix_sum])   # grow the stack
    prefix_sum     = None
hidden     = self_attn(input_layernorm(hidden))
prefix_sum = (prefix_sum + hidden) if prefix_sum is not None else hidden
hidden     = _apply_attn_res(prefix_sum, block_residual, mlp_res_proj, mlp_res_norm)
hidden     = moe_or_mlp(post_attention_layernorm(hidden))
prefix_sum = prefix_sum + hidden
return prefix_sum, block_residual
```

Appends occur at layers 0, 12, …, 84 → **8 stack entries** over 93 layers.

### LatentMoE

```
topk_idx, topk_w = gate(x)                  # gate reads HIDDEN (7168), not the latent
x_lat = routed_expert_down_proj(x)          # 7168 -> 3584
y     = Σ_k w_k · expert_k(x_lat)           # experts in 3584, moe_inter 3072
y     = routed_expert_norm(y)               # RMSNorm(3584), applied to the COMBINED output
y     = routed_expert_up_proj(y)            # 3584 -> 7168
out   = y + shared_experts(x)               # shared on HIDDEN 7168, inter = 3072 × 2
```

Shared experts take **no `hidden_size` override** → they run at 7168, confirming
the 24.3 GB line in the shell table (an earlier revision flagged 3584 as
possible; it is not).

### Layer indexing — `linear_attn_config` is 1-indexed

`kda_layers` and `full_attn_layers` list layers **1-indexed**. Subtracting 1
yields an exact partition of `0..92` (69 KDA + 24 MLA), which the checkpoint's
tensor index confirms directly: layer 0 carries `self_attn.A_log` (KDA) and
layer 3 carries `self_attn.kv_b_proj` (MLA).

Read as 0-indexed the lists look wrong in two ways — layer 0 appears in neither
and `full_attn_layers` ends at 93, out of range — which is exactly the shape of
an off-by-one. The exporter shifts on load, so `manifest.json` is 0-indexed.

### KDA (Kimi Delta Attention)

```
q, k = q_proj(x), k_proj(x)                 # 7168 -> 96×128
v    = v_proj(x)                            # 7168 -> 96×128
q, k, v = silu(shortconv_k4(·))             # per-tensor causal depthwise conv + own conv state
g_raw = f_b_proj(f_a_proj(x))               # 7168 -> 128 -> 12288  (low-rank)
g     = -5.0 · sigmoid( exp(A_log)[h] · (g_raw + dt_bias) )      # lower-bound gate variant
β     = sigmoid(b_proj(x))                  # 7168 -> 96, per head
q, k  = l2norm(q), l2norm(k);  q *= 128^-0.5

# per head, state S is [K=128, V=128]:
S = S · exp(g)[:, None]
S = S + (β·k) ⊗ (v − Sᵀk)
o = Sᵀ q

o = FusedRMSNormGated(o, g_proj(x), act=sigmoid)   # full-rank gate, per head_dim
o = o_proj(o)                                       # 12288 -> 7168
```

`gate_lower_bound = -5.0` selects the `lower_bound · sigmoid(exp(A_log)·g)`
branch, **not** the `-exp(A_log) · softplus(g)` default
(`kda_gate.py`, `USE_LOWER_BOUND`).

### Gated NoPE MLA (24 layers)

`rotary_emb = None` and `assert use_nope` — **no rotation anywhere**. The
`qk_rope_head_dim=64` slice still exists dimensionally but passes through
unrotated, and `k_rot` is MQA-shared (`[B,1,T,64]`, broadcast over all 96 heads).
`scaling = q_head_dim^-0.5 = 192^-0.5`. Output gate:

```
g = sigmoid(g_proj(x))                      # 7168 -> 12288, full rank
attn_out = attn_out * g                     # after head-concat, BEFORE o_proj
attn_out = o_proj(attn_out)
```

The HF reference caches expanded k/v; we use glm5's absorbed-latent decode
instead (**576 floats/token** = 512 latent + 64 shared rot), which is
mathematically equivalent and the only memory-feasible form at long context.

## Findings from checkpoint verification

Both were found by validating against the real checkpoint's metadata (the
tensor index and the safetensors headers — ~1.6 MB fetched, no weights).

### 1. `A_log` is zero-padded in the checkpoint — RESOLVED

The released weights ship `A_log` as `[128]` on every KDA layer, while
`modeling_kimi_linear.py` declares `torch.empty(num_heads)` = **96** and fla's
gate does `A_log.view(H, 1)` with `H = g.shape[-2] = 96`. vLLM's Kimi-Linear
implementation also stores it per head. `view(96, 1)` cannot take 128 elements,
so on the face of it the published modeling file cannot run the published
weights.

Reading the actual 512 bytes settles it — the tensor is **96 real values
zero-padded to `head_dim`**:

```
idx 0..95  : nonzero 96/96, exp(A_log) in [0.471, 11.776]   (init: log(uniform(1,16)))
idx 96..127: nonzero 0/32,  all exactly 0.0
```

So the decay is per HEAD, as every implementation says, and the shell is
correct as written. The loader drops the padding.

Dropping it is not cosmetic: `exp(0) = 1` is *no decay*, so consuming the tail
would leave 32 heads' recurrent state never decaying — output that looks
plausible and degrades as context grows. `kda.rs` has a test pinning that
rationale so the truncation is not "simplified away" later.

### 2. No `tokenizer.json` — blocks serving, not loading

K3 ships `tiktoken.model` plus a custom `TikTokenTokenizer`, and **no chat
template**. Every engine in this crate loads `tokenizer.json` via the HF
`tokenizers` crate, and there is no tiktoken support in the Rust tree, so the
API rank refuses to start. Pre-tokenized input and benchmarking are unaffected.

The exporter now copies the tokenizer artifacts into the export but
deliberately does **not** synthesise a `tokenizer.json`: the tiktoken `pat_str`
relies on Java/ICU character-class intersection (`&&` against `\p{Han}`), which
neither the HF tokenizers nor Rust regex engines accept. A naive translation
mis-splits text and presents as a model quality problem rather than a tokenizer
one. Converting it correctly, and validating token-for-token against the Python
tokenizer, is a prerequisite for serving.

## Open risks

- **Platform (highest)** — see the verdict. Everything else is moot until it clears.
- **KDA numerics** — the gated delta rule is the #1 correctness hazard and has no
  precedent in this crate; lock it against the CPU ref before anything else.
- **fp4 e2m1 kernel** — new quant grid, new bin format; needs its own round-trip test.
- **Linear-attention state cannot rewind.** The current append-only generate loop
  is fine, but any future spec-decode / MTP accept-reject needs *state
  checkpointing*, not KV truncation.
- **Prefix cache** — a K3 snapshot must carry the KDA recurrent state
  (96×128×128 f32 × 69 layers ≈ 430 MB) plus conv windows, not just KV. Defer.
- `max_seq` sizing actually *improves* vs glm5: only the 24 full-attn layers
  scale with context (512-float latent/token).
