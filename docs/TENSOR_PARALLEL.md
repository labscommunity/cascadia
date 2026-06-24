# Tensor parallelism (`tp-runtime`)

Megatron-style 2-way tensor parallelism for cascadia: each rank holds a 1/N
slice of every layer's weight matrices and computes its partial of attention and
MLP; the ranks all-reduce the partials over the existing transport. Unlike
pipeline parallelism (which splits the model by contiguous *layers* and gives no
single-stream latency benefit), tensor parallelism splits *within* each layer so
both machines work on the *same* token in parallel — aggregating DRAM bandwidth.

## Pieces

- **Exporter** `tools/tp_export.py` — produces, per TP rank, segmented OpenVINO IR
  (`rank_{r}/{embed,attn_i,mlp_i,head}/openvino_model.xml`) with column/row-parallel
  sliced weights and per-rank KV heads. q/k/v_proj + gate/up_proj are
  column-parallel (slice output rows); o_proj + down_proj are row-parallel (slice
  input cols → partials are summed by the all-reduce). Llama-family, INT4 via nncf.
- **Engine** `crates/cascadia-engine-openvino/src/tp_runtime.rs` (`--engine tp-runtime`)
  — loads the per-rank segments and runs, per token, per layer:
  `pa = attn_i(hidden); hidden += all_reduce(pa); pm = mlp_i(hidden); hidden += all_reduce(pm)`.
  Residual adds in f32. all-reduce = concurrent `send(own f16 partial)`/`recv(peer)`,
  summed in f32 (own rounded to f16 first so both ranks stay bit-identical).
  Topology is a 2-node bidirectional peer link (`--rank`=tp_rank, `--total`=tp_size,
  `--next`=peer, `--listen`=own); rank 0 drives + broadcasts each step's input_ids,
  rank 1 relays.
- **Validation** `tools/tp_reference.py` (PyTorch: tp=1 fp32 == tp=2 fp32, exact, and
  == HF greedy) and `tools/tp_ov_validate.py` (OV tp=1 == tp=2, bit-identical tokens).

## Correctness

Validated at every level. tp=1 fp32 == tp=2 fp32 == HF greedy (exact). The OV
export tp=1 == tp=2 (bit-identical). End-to-end on two Arc Xe2 NUCs the `tp-runtime`
engine produces output **identical to the monolithic and pipeline runs**:
`"The capital of France is"` → `"The capital of France is Paris."`

## Performance (Llama-3.2-1B INT4, 2× Core Ultra X7 358H Arc Xe2, I226-LM cable)

Same session, same client, 128-token greedy decode, median:

| Config | decode tok/s | TTFT | per-token |
|---|---|---|---|
| Monolithic (1 node) | **79.0** | 32 ms | 12.7 ms |
| Pipeline parallel (2-stage, 2 nodes) | **50.3** | 59 ms | 19.9 ms |
| **Tensor parallel (2-way, 2 nodes)** | **22.6** | 130 ms | 44.3 ms |

**TP is correct but ~2.2× slower than pipeline on this stack, and the bottleneck
is GPU dispatch, not the network.** Megatron requires 2 all-reduces per layer, which
forces the forward pass to be broken into `2L+2 = 34` separate OpenVINO infers per
token (one per all-reduce segment). On OpenVINO's per-infer dispatch model that is
~38 ms/token of segment dispatch+compute; the 32 all-reduces are only ~5.8 ms (~13%).

So the theoretical TP latency win (bandwidth aggregation → ~halve compute) is real
but **masked by per-segment dispatch overhead** — the same dispatch-bound regime
documented for batch-1 decode on Xe2. Two levers would unmask it, in order of impact:

1. **Collapse the segments** (CUDA-Graphs-equivalent SYCL Graph capture/replay, or a
   persistent megakernel) so the 34 infers become ~1 dispatch — this attacks the ~87%.
2. **Faster all-reduce** (e.g. the speedeth poll-mode datapath) — attacks the ~13%.

In other words: tensor parallelism is the right structure for lowering single-stream
latency on a memory-forced multi-node model, but on the current OpenVINO runtime it
is dispatch-bound; the dispatch-amortization work is the prerequisite for it to pay off.

## Run

```
# export (once)
python tools/tp_export.py --model <hf-id-or-dir> --output-dir <out> --tp-size 2 --quantization int4
# copy <out> to both nodes; then:
# node B (rank 1):
cascadia worker --engine tp-runtime --rank 1 --total 2 --device GPU --model <out> \
                --listen 0.0.0.0:9401 --next <nodeA-ip>:9400
# node A (rank 0):
cascadia worker --engine tp-runtime --rank 0 --total 2 --device GPU --model <out> \
                --listen 0.0.0.0:9400 --next <nodeB-ip>:9401 --api 127.0.0.1:8000
```

## Limitations / next

- Llama-family only; greedy (argmax) only; TP=2 validated (the slicing generalizes to
  TP=N where N divides num_kv_heads, but the peer link + all-reduce are 2-node today —
  N-node needs a ring/tree all-reduce).
- Activations cross f16 on the wire (the all-reduce), so very deep models accumulate a
  little more rounding than a single-graph fp16 forward; INT4 quant dominates anyway.
- The dispatch bottleneck (above) is the headline next step.
