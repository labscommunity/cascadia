# Moonshot taxonomy

Seed list of 60 moonshot candidates for the K2.6 pipeline. Each gets a
one-liner, expected delta direction + magnitude class, risk axis,
required code-change scope, and the campaign ID it ultimately runs as
(filled in when designed).

**Classes:**
- `XS` <5% expected delta — defensible micro-opt
- `S` 5-20% — solid eng-level win
- `M` 20-100% — design change
- `L` 2-5× — moonshot magnitude
- `XL` >5× — paradigm shift

The loop is biased toward `M+` per the user directive ("optimize for
moonshots"). `XS` items only run if a parent moonshot makes them
trivial to fold in.

| ID | Bucket | Candidate | Expected | Risk | Scope |
|----|--------|-----------|---------:|------|-------|
| A1 | quant | Int2 expert weights, mixed-precision with Int4 fallback for top-K hot | L | quality | int4-gemm + rainier exporter |
| A2 | quant | Per-token expert pruning (drop experts with routing weight < threshold) | S-M | quality | shell_int4 router only |
| A3 | quant | Top-K reduction K=8 → K=4 / K=6 with quality measurement | S-M | quality | shell_int4 router only |
| A4 | quant | Fp4 (MXFP4) with shared-exponent blocks for experts | M | quality+code | int4-gemm new kernel |
| A5 | quant | BitNet-style ternary (-1/0/+1) experts | L | quality | int4-gemm + retrain |
| A6 | quant | Int8 / Int6 shell projection weights (currently bf16) | S | quality | shell + rainier |
| A7 | quant | Activation-aware quant (AWQ) for the most outlier-prone expert layers | M | quality+code | rainier + int4-gemm |
| A8 | quant | KV cache fp16/bf16/int8 instead of f32 (halves KV BW) | S-M | quality | shell_int4 cache |
| A9 | quant | KV cache fp8 (E4M3) with per-token scaling | M | quality | shell_int4 cache + kernel |
| A10 | quant | Distill K2.6 experts down to half-size + Int2 — accuracy-preserving compression | L | quality+time | rainier + retrain |
| B1 | kv-attn | KV layout `[cap, NUM_HEADS, D]` (cap-major) vs current `[NUM_HEADS, cap, D]` (head-major) — measure GEMV BW | S | code | shell_int4 |
| B2 | kv-attn | Flash-Attention v3 fused kernel for shell attention (CPU AVX-512 port) | M | code | new kernel in int4-gemm |
| B3 | kv-attn | KV cache eviction (drop least-attended slots beyond window N) | S-M | quality | shell_int4 |
| B4 | kv-attn | Sliding-window attention for prefill (don't compute attention to far-past) | S | quality | shell_int4 |
| B5 | kv-attn | Multi-query attention emulation (share K/V across head groups) | M | quality+code | shell + rainier |
| B6 | kv-attn | Paged KV with 64-token blocks (for batched serving later) | S | code | shell_int4 |
| B7 | kv-attn | Skip-low-weight-head attention (drop heads with low attn entropy) | S-M | quality | shell_int4 |
| B8 | kv-attn | Pre-RoPE'd K cache (cache K post-rotation) | S | code | shell_int4 |
| B9 | kv-attn | YARN-RoPE precomputed cos/sin table per position (currently per-token) | XS-S | code | shell |
| B10 | kv-attn | Attention head pruning (drop K2.6 heads with low importance) | M | quality | rainier + shell |
| C1 | dispatch | Pre-fetch top-K experts (overlap mmap+touch with attention) | M | code | runner + new prefetcher |
| C2 | dispatch | Pin top-K% hot experts in RAM (mlock or shared-memory pool) | S-M | mem | runner |
| C3 | dispatch | Co-locate all 60 layers' active experts in one mmap to share page cache | S | code | runner + safetensors |
| C4 | dispatch | Sort tokens by routed expert set, batch through one GEMM call | M | code | shell_int4 |
| C5 | dispatch | SIMD expert dispatch sum (currently scalar over top-K) | XS | code | shell_int4 |
| C6 | dispatch | Fused MoE GEMM: single kernel for gate × W_up × silu × W_down | S-M | code | new kernel |
| C7 | dispatch | Expert weight prewarm during model load (currently touched lazily) | S-M | time/mem | runner |
| C8 | dispatch | Async expert page-in: start I/O for next layer while computing this layer | M | code | runner restructure |
| C9 | dispatch | Per-layer JIT specialization of expert kernel (const-folded scales) | XS-S | code | int4-gemm |
| C10 | dispatch | AVX-512 VNNI int8 path for accumulators (currently fp32) | S | code | kernel_avx512 |
| C11 | dispatch | AMX path for matias' future Granite Rapids (not Lunar Lake) | M | hw | new kernel |
| D1 | wire | F32 → BF16 hidden tensor wire (50% wire BW) | S | quality | dist.rs frame fmt |
| D2 | wire | F16 hidden wire | S | quality | dist.rs |
| D3 | wire | Per-token compression (zfp / SZ float compression) | S-M | code | dist.rs |
| D4 | wire | Async pipeline: rank N starts T+1 while rank N+1 still on T | M | code | engine + dist.rs |
| D5 | wire | Speculative downstream send (send hidden before sampling completes) | S | code | engine |
| D6 | wire | Persistent connection pool (already on; verify no per-token reconnect) | XS | code | transport |
| D7 | wire | Batch frame: send 4 tokens worth during prefill | S | code | dist.rs |
| D8 | wire | UDP-with-NACK transport vs TCP (sub-ms RTT removal) | S-M | code | new transport |
| D9 | wire | tailscale → direct WireGuard (skip DERP relay) | S | infra | network |
| D10 | wire | UCX / rdma-rs backend (no IB on Lunar Lake but measure) | XS | code | new transport |
| E1 | topo | 3-box pipeline: 20/20/20 split across 3 matias boxes | M | infra | runner + stage box |
| E2 | topo | Heterogeneous split: 25/35 with rank 0 hosting API + layer 0 | S | code | shard config |
| E3 | topo | PP×TP hybrid: PP=2, TP=2 attention split within rank on top-K hot layers | L | code | new engine |
| E4 | topo | Cyclic micro-batched pipeline (fill pipeline before draining) | M | code | engine restructure |
| E5 | topo | NUMA-aware shell placement on miner (2-socket Xeon scenario) | S | hw | runner |
| E6 | topo | iGPU offload of layer 0 only (use Lunar Lake Xe iGPU via OV GPU plugin) | S-M | code | runner |
| E7 | topo | NPU offload of router / gate (Lunar Lake has NPU) | S | code | runner + new engine path |
| E8 | topo | Speculative decode within pipeline: small draft on rank 0, K=2 verify | M | code | new engine + draft model |
| E9 | topo | 4-box matias pipeline: 15/15/15/15 split | L | infra | needs 2 more boxes staged |
| E10 | topo | Pinning rank 0 cores to NUMA0, rank-N+1 cores to NUMA1 — measure on miner | XS-S | hw | runner |
| F1 | sched | Fuse RMSNorm into expert input (skip materializing normalized output) | S | code | shell_int4 |
| F2 | sched | SwiGLU fusion in expert kernel (currently silu(W_up x) * (W_gate x)) | S | code | kernel_avx512 |
| F3 | sched | Token pipelining within rank: T+1 attention while T in expert | M | code | engine restructure |
| F4 | sched | Multi-thread per shell (currently single-thread per token) — rayon over heads | S-M | code | shell_int4 |
| F5 | sched | Multi-thread expert dispatch over top-K (currently sequential) | S | code | runner |
| F6 | sched | Continuous-batching during multi-prompt eval (overlap N users) | M | code | engine + API |
| G1 | algo | EAGLE-style speculation with shared trunk | L | code | new engine + draft head |
| G2 | algo | Medusa-style multi-head LM head | M | code | head + runner |
| G3 | algo | Lookahead decoding (Jacobi) | M | code | runner |
| G4 | algo | Skip-layer decoding (drop shallow layers for low-uncertainty tokens) | S-M | quality | runner |
| G5 | algo | Early exit (stop after layer K if confidence high) | M | quality | runner + head |
| G6 | algo | Speculative MoE: predict routed experts, prefetch, fall back if wrong | M | code | router + runner |

**Total: 60 candidates** across 7 buckets. Loop will:
- Order by `(expected magnitude × inverse code cost)` after literature search refines estimates
- Spawn additional candidates as it learns (target 100+ tracked by end)
- Mark each `running` → `complete` with outcome in the rightmost column once executed

## Anti-list (do NOT pursue without strong new evidence)

From [[PRIOR_ART]] verified-negatives, ruled out at the start:

- ❌ Tensor parallelism over Lunar-Lake-class fabric (TP needs NVLink-class)
- ❌ Gather-before-Decompress IR rewrite (1.84× slower on OV CPU)
- ❌ Tree-spec on existing draft architecture (loses on wire)
- ❌ v6 4D additive mask shards (slightly slower than v5 chain-spec)
- ❌ Paged-attention without batched serving (no per-token win for single-user)
