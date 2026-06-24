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

### Profiling the bottleneck (`gpu_ms`/`allreduce_ms` instrumentation + per-segment probes)

The engine logs `gpu_ms` (sum of segment infers) and `allreduce_ms` per task. A 2-rank
loopback run (both ranks on one node, so the all-reduce is free) totals the *same*
~44 ms/token as the 2-node run, split **gpu ~87% / all-reduce ~9%** — the work is in
the segment infers, not the wire. Per-segment GPU probe (alpha Arc Xe2, int4):

| segment | per-infer wall | GPU-busy | host/dispatch overhead |
|---|---|---|---|
| mlp_i (stateless) | 278 µs | 226 µs | ~52 µs |
| attn_i (stateful KV + dynamic shape + beam_idx Gather) | 817 µs | **150 µs** | **~667 µs** |

The attention segments are ~5× their own compute — almost all OV per-infer overhead
(stateful-KV management + dynamic-shape re-inference). **Root cause:** pipeline parallelism
runs 8 layers in *one* stateful graph (overhead paid 2× per token); TP runs *one layer per
attn segment* (overhead paid 16×). So TP pays the stateful per-infer overhead ~8× more, and
the 34 infers can't be merged (a network all-reduce sits between every pair).

**Quantified target:** raw segment compute is only ~16·150 µs (attn) + 16·226 µs (mlp) +
embed/head ≈ **~7 ms/token**, vs the ~39 ms with per-infer overhead. Collapsing the 34 infers
into a single persistent GPU dispatch (the megakernel: one launch for the whole forward,
signaling the host at all-reduce points) would put TP at ~7 ms + all-reduce → **~2–3× faster
than pipeline**. Lesser tweaks (leaner all-reduce, static-shape attn, dropping the beam_idx
Gather) trim the secondary costs but cannot overcome the 34-infer dispatch wall — they do not
beat pipeline on their own. The persistent megakernel is the load-bearing next step
(this is the `cascadia-megakernel` effort), with speedeth's fast wire attacking the residual
all-reduce.

### Decode-only profile, measured 2-node (alpha rank0 + charlie rank1, int4)

The instrumentation now resets after prefill and splits the *steady-state decode*
forward into four buckets (`SE_TP_TUNE`, `decode profile` log line). Measured over 4–8
reps of 64–96-token greedy decode, per token:

| Bucket | µs/token | share | what it is |
|---|---|---|---|
| **GPU segment infers** | **~37,000** | **81%** | 34 OV `InferRequest`s × ~1.05 ms each |
| All-reduce | ~7,200 | 16% | 32 TCP round-trips × ~225 µs (192.168.0.x LAN) |
| Host setup | ~1,080 | 2.4% | `set_input` + f32↔bytes conversions |
| Other | ~97 | 0.2% | residual adds, argmax, control broadcast |

This is the hard confirmation of the loopback estimate: **the wall is GPU dispatch (81%)**,
and the host path is negligible (2.4%) — so set_input/tensor-reuse micro-optimization buys
nothing and was not pursued. The only OV-level bucket with real headroom is the all-reduce
(16%), which is exactly speedeth's small-message-RTT regime.

### What OV-level tuning recovers (`SE_TP_TUNE` A/B)

Pinning the GPU plugin to the single-stream low-latency schedule
(`PERFORMANCE_HINT=LATENCY` + `NUM_STREAMS=1`, default-on; `SE_TP_TUNE=0` reverts) is the
cheapest possible lever — pure plugin config, no re-export, no kernel work. Measured A/B,
same nodes/client/model:

| Config | decode tok/s (median) | GPU bucket |
|---|---|---|
| baseline (`SE_TP_TUNE=0`) | 22.2 | ~37.0 ms/tok |
| tuned (`SE_TP_TUNE=1`) | **23.4** | ~34.0 ms/tok |

**~+5–8%, and that is the ceiling for OV-config tuning.** The LATENCY hint shaves only ~8%
off the GPU bucket because the per-infer cost is the irreducible OV-plugin/Level-Zero
*per-invocation* floor — a half-width attn segment costs about as much as a full monolithic
layer. No config knob touches that; only **fewer dispatches** (impossible in standard
2-all-reduce-per-layer Megatron — the MLP's column-parallel gate/up needs the full post-attn
hidden, so neither all-reduce can be skipped or merged) or a **lighter dispatch path**
(the megakernel) can. Confirmed empirically: the cheap lever is worth single digits, the
structural fix is the megakernel.

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
