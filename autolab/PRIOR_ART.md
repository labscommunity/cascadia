# Prior-art synthesis — perf learnings going into this loop

Everything verified by past PRs on this repo. Each entry: claim, source,
status. The loop should NOT re-test these unless it has reason to
believe the claim no longer holds (hardware change, model change, OV
version bump).

## Verified positive (productionized)

| ID | Claim | Source | Status |
|----|-------|--------|--------|
| P1 | **Async target wire overlap with draft compute** = +6-19% on Llama 3.1 8B INT4 v5 shards over TB4. Hides ~one draft.feed worth of compute per spec round. | PR #5 (`perf/dist-spec-async-overlap`) | Live on main. Sparse-MoE has no draft yet — applies once we add one. |
| P2 | **FastDraft 150M beats Llama 1B draft by +35-38%** for spec decode on alpha+charlie. Smaller draft → lower per-round overhead. | PR #1 / #4 → #5 | Llama-class result; need to retest on K2.6 if/when we add a K2.6 draft. |
| P3 | **K=4 wins distributed, K=5 wins single-node.** Round overhead grows with K. | PR #1 / #4 | Llama; sparse-MoE has no spec decode yet. |
| P4 | **Long output (256-1024) amortizes spec rounds** — 38.49 tok/s peak at 1024-out vs 28.29 at 256-out single-node. | PR #1 / #4 | Llama. |
| P5 | **Layer-0 KV cache (no more O(N²) layer-0 attn)** = architectural win; net-neutral at 4-token decode but eliminates a long-context cliff. | PR #10 (`feat/sparse-moe-followups`) | Live. |
| P6 | **Pre-allocated KV with geometric grow** in shells. O(N) cumulative alloc/copy instead of O(N²). Net-neutral at short prompts; long-context win. | PR #10 | Live. |
| P7 | **Rust shell forward via cdylib > OV shell IRs.** Same int4 kernel; OV shell IR was buggy. ~200 LOC net reduction in runner. | PR #9 (commit 8759b93) | Live. Layer 0 + head stay on OV. |
| P8 | **Pipeline-parallel inference across N ranks works** — F32 hidden tensor [1,1,7168] = ~28 KB per token over TCP. 100 round trips averaged 22.4 ms over Tailscale DERP. | PR #9 | Live. |

## Verified negative (do not re-run)

| ID | Claim | Source | Why dead |
|----|-------|--------|---------|
| N1 | **Tensor-parallelism over TB4 is fundamentally not viable.** -90% throughput vs single-node. Needs NVLink-class fabric. | PR #4 d4 | TP requires 32+ all-reduces/token; TB4 RTT × 32 > compute time. PP (1 transfer/token) is the only viable distributed shape over TB4. |
| N2 | **Gather-before-Decompress IR rewrite for experts is 1.84× slower.** Cosine 1.000000 (math perfect) but OV's CPU plugin optimizes the dense decompress+gather pattern better. | rainier autolab | Don't re-try this graph rewrite path on OV CPU. Pure Rust int4 GEMM beats it. |
| N3 | **Tree-spec (width-2 root, parallel drafts) loses on this hardware.** Wire scaling on charlie's stage_1 grows ~3.5× for ~1.83× more tokens. Only viable if (a) EAGLE draft, (b) batched serving, or (c) wire scaling fixed. | autolab/distributed-perf | Don't re-run; preserved on the autolab branch. |
| N4 | **v6 4D additive mask shards + ForwardV6 frame** correct but slightly slower than v5 due to f16 mask wire overhead on chain-spec. | autolab/distributed-perf | v6 only wins if both draft and target are v6 + INT4; NNCF tied-embedding hang blocks INT4 v6 on Llama 3.2 1B/3B. Sparse-MoE doesn't use spec yet, so moot for now. |
| N5 | **PA per-stage on multi-stage gives ~0% per-token speedup for single-user sequential.** PA wins are batching-driven. | PR #4 d7 | Don't pursue PA until we have batched serving. |
| N6 | **Layer split direction matters: slower node should NOT get more layers.** 14/18 charlie-heavier was -25% vs 16/16. | PR #4 d5 | When designing 2-box K2.6 splits, give matias-03 (or whichever is slower) ≤ half the layers. |

## Carryover open questions (worth re-attacking on K2.6)

- **OV 2026.1 CPU snippets bug on shell IRs** (different seq sizes within one run). Workaround was token-by-token prefill. The Rust shell sidesteps it but layer 0 still uses OV — re-check on K2.6 with current OV.
- **Right-direction layer rebalance** for the 2-box K2.6 pipeline. Memory says matias-02 + 03 are identical Lunar Lake, so it's a wash, but worth measuring per-rank step time to see if one is consistently faster.
- **paged-attention re-export** on K2.6 shells. Was deferred for Llama; may matter for K2.6 long-context.

## Hardware constraints already measured

- Miner: DDR4-2133 5-channel, **58 GB/s empirical read peak**. Caps single-stage K2.6 at ~3.5 tok/s even if disk were infinite.
- Matias (Lunar Lake 258V): 32 GB RAM; pagefile must be auto-managed (256 MB default → OOM during shell compile). 30 OV shells = ~37 GB committed.
- Tailscale DERP relay "sea" between matias boxes: 22.4 ms avg, 19.5 ms p50, 102 ms p99 (first-round). Cold connection adds ~10 ms.
- Direct UDP between matias boxes blocked by Intel firewall — DERP is forced.
- K2.6 model: 553 GB original, ~290 GB per box after 2-stage split. 60 MoE layers, 384 experts/layer, top-8 dispatch. Hidden 7168 bf16. 64 heads, qk_head_dim 192, v_head_dim 128.

## Where the time goes (current 2-box steady-state ~17-20 s/tok)

From [[k26-state]] + code reading. Order-of-magnitude only; the loop should re-measure.

| Stage | Time/token (rough) | Notes |
|---|---|---|
| Shell attention × 30 layers/rank | ~2-3 s | Rust int4 GEMM on Lunar Lake CPU; bf16 path |
| Expert dispatch × 30 × top-8 | ~10-12 s | Disk-page-bound on cold experts; AVX-512 int4 kernel |
| RMSNorm + router × 30 | <0.1 s | Negligible |
| Layer 0 / head | <0.2 s | Once per token; OV IR |
| Cross-rank Forward send/recv | ~0.05 s | 28 KB / 22 ms DERP |
| Sampler | <0.001 s | Greedy argmax |

**The big rocks are expert-page latency and shell compute.** Anything
that reduces top-K, prefetches experts, compresses experts, or
overlaps attention with expert page-in is likely to move the needle.
