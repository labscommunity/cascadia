# Router-aware MoE expert streaming — implemented in this PR (0002 patch)

MoE models no longer stream every expert per token. `--elastic` now reads only
the experts the router selects, detected automatically for **any** MoE the
runtime can execute — no model list, no new flags.

## How it works

- **Automatic detection**: a pre-scan marks every streamed weight consumed by
  `GGML_OP_MUL_MAT_ID` with an expert-stack axis (`ne[2] > 1`). Qwen3.6-35B-A3B
  yields exactly its 120 routed tensors (3 per layer x 40); a dense or hybrid
  model triggers nothing.
- **Selective uploads**: routed tensors leave the whole-tensor slot pool
  (49.8 MiB instead of ~505 MiB) and read per-expert slices into per-tensor
  arenas at the tensor's own `nb[2]` stride — so every existing kernel
  (fused MoE GEMV, raw + reordered, all K-quants) addresses slices unchanged.
- **Decode**: the existing single-token fused GEMV runs over the arena with a
  private slot-id table; the router's original ids and every downstream
  consumer (probs, biases, scales, LoRA) are untouched.
- **Prefill**: the counting-sort grouped loop loads only experts with routed rows.
- **Hot-expert cache**: opt-in via `CASCADIA_EXPERT_CACHE_MB` (default 0, keeps
  `--elastic-vram` semantics exact); per-tensor LRU arenas + a scratch window.

## Measured on the B70 (Qwen3.6-35B-A3B UD-Q4_K_XL, ctx 4096, KV q8_0, -fa on)

| --elastic-vram | before (every expert) | after (router-aware) | speedup | peak VRAM before -> after |
|---|---:|---:|---:|---|
| 0 (fully streamed) | 0.25 t/s | **2.61 t/s** (n=4, ±0.01) | **10.4x** | 2.59 -> **1.89 GiB** |
| 4 | 0.30 t/s | **3.08 t/s** | 10.1x | 6.06 -> **5.32 GiB** |
| 10 | 0.47 t/s | **4.91 t/s** | 10.5x | 12.42 -> **11.92 GiB** |
| 16 | 1.12 t/s | **11.35 t/s** | 10.2x | 18.31 -> **17.63 GiB** |
| auto (fits) | 77.5 t/s | 78.1 t/s | stock path | 21.50 GiB |

- Prefill: 0.62 -> 3.36 t/s fully streamed (5.4x); 2.10 -> 6.20 at 16 GiB.
- Weight read traffic: 19.80 -> ~2.06 GiB per token (9.6x fewer bytes; 32x on
  the routed-expert share). Device counters confirm ~586 MiB of expert uploads
  per token vs the 0.573 GiB selected-expert payload arithmetic.
- Load time fully streamed: 20.4 -> 13.5 s.
- **Correctness**: every streamed/auto arm produces byte-identical greedy
  tokens to its reference (MoE vs stock; dense 1.5B / hybrid 27B vs unfused
  stock, as before); the all-fit path is untouched. Final tally **40/40**
  token-identical runs — the campaign itself caught one real bug (a slot
  could be evicted within the same call when per-tensor cache slots < top-k,
  breaking parity at 512 MiB), fixed by pinning freshly placed slices and
  re-verified.
- **Hot-expert cache (opt-in)**: 512 MiB = 2.88 t/s (**+10%**, n=2); 2 GiB
  = 3.10 t/s in 2 of 3 runs (+19%) with one anomalous slow outlier and
  59-67% hit rates — needs more repetitions before a recommendation, hence
  default off.

## Artifacts

- `patches/llama.cpp/0002-sycl-router-aware-moe.patch` (+334 lines, ggml-sycl
  only), applied after 0001 by both build scripts; fresh-chain build verified.
- `cascadia doctor` reports the capability; `CASCADIA_EXPERT_CACHE_MB` passes
  through to the child.
- Figures refreshed from measured data only: fig1/fig3/fig4 MoE bars, fig8
  now draws the before/after curves (dashed = every expert).
