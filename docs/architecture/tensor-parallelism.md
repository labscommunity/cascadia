# Tensor parallelism

Pipeline parallelism splits the model **vertically**: each node owns a contiguous range of layers and activations flow point-to-point. Tensor parallelism splits **horizontally**: every node holds the same layer range but a 1/N slice of every weight matrix, and partial results are summed across the group after each attention block and each MLP block.

PP and TP compose. A typical exo-style cluster of four nodes might run as `total=2, tp_size=2`: two pipeline stages of 16 layers each, with each stage held by two TP ranks cooperatively.

## Status

This commit ships the foundation:

- `tahoma.parallel.TPGroup` — ring-based all-reduce-sum over TCP (validated against numpy ground truth for tp_size in {2, 3, 4}, fp16 + fp32).
- `ShardSpec.tp_size` / `ShardSpec.tp_rank` carried through CLI as `--tp-size` / `--tp-rank`.
- Engine ABCs unchanged. Engines that don't opt in to TP must reject `tp_size > 1` with a clear error.

What's **not** here yet (per-engine work, requires re-exported shards):

1. **Column / row-parallel shard exports.** Today's v5 shards bake the full weight matrices into each stage; running them on multiple TP ranks would just compute the same thing twice. The export script needs to slice:
   - `q_proj`, `k_proj`, `v_proj`: column-parallel (split along output dim).
   - Attention `o_proj`: row-parallel (split along input dim).
   - MLP `gate_proj`, `up_proj`: column-parallel.
   - MLP `down_proj`: row-parallel.
2. **Engine integration.** After attention's `o_proj` and MLP's `down_proj` the engine must call `TPGroup.all_reduce_sum_inplace(activations)` before passing the result to the residual stream / next layer. Engines that opt in declare it on their builder.
3. **KV cache sharding.** With column-parallel attention each TP rank only computes `num_heads / tp_size` heads; the per-rank stateful KV cache shrinks proportionally.

## Wire protocol

`TPGroup` opens one outbound connection to `(tp_rank + 1) mod tp_size` and accepts one inbound from `(tp_rank - 1) mod tp_size`. Collectives use the standard ring algorithm: reduce-scatter (`tp_size - 1` rounds) followed by all-gather (`tp_size - 1` rounds). Per round each rank sends `bytes / tp_size` and receives the same — bandwidth-optimal for a given tensor size.

Frame format per chunk: `[4B byte_count BE][raw bytes]`. No tensor metadata — the dtype + shape come from the caller's contract that every rank pass an identically-shaped tensor.

## Launch

For a 2 × 2 grid (PP=2, TP=2), each of the four nodes runs a worker with both pipeline coordinates and TP coordinates set:

```bash
# rank 0, tp_rank 0  (stage 0, TP rank 0)
tahoma worker --rank 0 --total 2 --tp-size 2 --tp-rank 0 \
              --engine ov-runtime-tp --device GPU --model ... \
              --next 10.0.0.2:9100 --api :8000

# rank 0, tp_rank 1
tahoma worker --rank 0 --total 2 --tp-size 2 --tp-rank 1 \
              --engine ov-runtime-tp --device GPU --model ... \
              --next 10.0.0.2:9101

# rank 1, tp_rank 0/1: similar
```

The TP group's listen ports are sequential from `--listen` (one per peer). Discovery doesn't yet handle TP-group rendezvous; it's currently the user's responsibility to pass the right ports.

## When TP is worth it

- Single-node TP across two iGPUs / dGPUs: rare on Intel AI PCs (most ship with one iGPU), but useful on workstations with an Arc dGPU + iGPU.
- Multi-node TP over a high-bandwidth fabric (Thunderbolt 4/5, 10 GbE+): cuts per-token latency for very large layers when network bandwidth ≫ layer compute.
- TP over Wi-Fi / LAN: usually a loss. The all-reduce traffic is `2 × (1 - 1/tp_size) × hidden_size × bytes_per_token`, which dominates when bandwidth is < ~1 Gbps.

The placement engine in `tahoma/master/` does not yet model TP cost. It's a follow-up: a TP edge is `2x` the per-link bandwidth use of a PP edge, and the placement search should account for that.
