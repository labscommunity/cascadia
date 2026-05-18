# 037 — F5 sliding-window attention bench (autolab iter 037 / F5 retry)

Single-stage miner bench of `--attention-window` (F5 / iter 029).

## Method

- Host: miner (192.168.86.51), CPU-only, sparse-MoE engine, top_k=8 (manifest default).
- Worker built from `perf/f5-sparse-attention-029` @ `769c8b0` with
  `cargo build --release -p tahoma --features openvino` (linked
  against `/home/tatef/openvino_2026.1.0`).
- 1 prompt per W setting: `"The capital of France is"` (7 tokens),
  greedy decode (temperature=0), substring eval matches `"paris"` in
  the lowercased output.
- Worker is killed and restarted between each W since
  `--attention-window` is fixed at startup.
- Bench script: `k26_bench_f5.sh` (per-prompt timing, jsonl output).

## Setup notes

- Old worker (PID 2702179, K=8 default) was killed before the run; no
  other tahoma worker is on miner during the bench. Matias 2-box is
  off-limits (A8+C1 combined bench is using it; we collide nothing).
- LD_LIBRARY_PATH = `/home/tatef/openvino_2026.1.0/...` (via
  `setupvars.sh`).
- Memory pressure on miner is severe (~750 MB free of 130 GB, 5 GB
  swap), driven by K2.6's ~105 GB working set per worker; this is the
  same condition every bench in this branch has run under and is the
  reason A3-K6 / C1 / A8 results landed where they did.

## Files

- `bench_w<W>_mt<MT>.jsonl` — raw bench output for each setting
- `worker_w<W>_mt<MT>.log` — full worker log
- `k26_bench_f5.sh` — bench driver (deployed to miner at
  `/tmp/k26_bench_f5.sh`)
- `run_bench.sh` — kill-old / start-new / wait-ready / run-bench
  one-shot wrapper
- `compare.sh` — produces the comparison table from bench_*.jsonl

## Results

Single-prompt bench (`"The capital of France is"`, 7-token prefill,
greedy decode, top_k=8 manifest default). The substring evaluator
matches `"paris"` in lowercased output.

| W   | mt  | tok/s   | wall_s  | quality (substr) | qualitative
|-----|-----|---------|---------|------------------|------------
| 0   | 64  | 0.1253  | 511.0   | 1/1              | coherent
| 32  | 64  | 0.1585  | 403.8   | 1/1              | coherent ~25 tok, then "Question Question?" loop
| 0   | 128 | 0.1192  | 1074.0  | 1/1              | coherent
| 32  | 128 | 0.2150  | 595.2   | 1/1              | coherent ~30 tok, then "?? && .. .. &." garbage
| 128 | 128 | 0.1235  | 1036.2  | 1/1              | coherent

### Speedup

- **W=32 vs W=0 at mt=64**: 0.1585 / 0.1253 = **+26.5% tok/s** (window
  engages around decode token 25, ~38 of 64 decoded tokens benefit).
- **W=32 vs W=0 at mt=128**: 0.2150 / 0.1192 = **+80.4% tok/s** (window
  engages around decode token 25, ~102 of 128 decoded tokens benefit).
- **W=128 vs W=0 at mt=128**: 0.1235 / 0.1192 = **+3.6% tok/s** (window
  engages only on last ~6 tokens — within noise; matches the user's
  pre-bench expectation that W=128 at mt=128 is "barely active").

### Quality

Substring eval passes for every run because the first
sentence (" Paris.") is always emitted before the window-induced
divergence point. Inspecting the full completions:

- W=0 (both mt=64, mt=128) and W=128 mt=128 are coherent throughout.
- **W=32 quality cliff is severe.** The first ~25-30 generated
  tokens match the W=0 trajectory verbatim ("Paris. The capital of
  Germany is Berlin. ..."); from the point past_seq_len exceeds the
  window (decode token ≈ 25), output degenerates first into a
  repetitive "Question?" loop (mt=64) and then into pure garbage
  ("??? ?? ?? ... & & & . . .") (mt=128). This is the K2.6-specific
  failure mode flagged in the 029 impl commit: K2.6 has no per-layer
  attention_type so the cap is uniform across all 60 MoE layers + the
  dense layer-0, and W=32 strips out too much of the early prompt
  context for the model to keep its reasoning chain.

### Verdict

- **F5 throughput claim holds.** At mt=128 with W=32 we measure
  +80.4% tok/s on a single-prompt single-stage decode — the largest
  per-token speedup in the 029 batch (vs A8's bf16 KV and C1's expert
  prefetch).
- **F5 is not usable at small W on this model.** W=32 hits the
  quality cliff before token 30 of decode. Useful operating regime
  is W >= 256 or so on K2.6 chat workloads (untested here; needs a
  long-context bench to measure both the quality recovery and the
  decreasing throughput win as W grows).
- Substring evals pass at every W because they only check the first
  sentence; **a stricter eval that ran past the first ~30 tokens
  would have flagged W=32 as quality-fail.**

