# c31: AWQ INT4 export — TIMED OUT

## Setup
Llama 3.1 8B src (FP16) → INT4 with AWQ + scale-estimation, group=128,
wikitext2 calibration. Run on alpha B390 (CPU-bound export).

```
optimum-cli export openvino \
  -m C:\cascadia\models\llama-3.1-8b-src \
  --weight-format int4 --group-size 128 \
  --awq --scale-estimation --dataset wikitext2 \
  --task text-generation-with-past \
  llama-3.1-8b-int4-awq
```

## Result
**TIMED OUT** at 71+ min CPU time. Per LOOP.md the time-box for any
single experiment is 30 min wallclock; AWQ blew through that and we
killed the process.

Phases observed:
1. Model load (FP16 from .safetensors): ~5 min.
2. Tracing + export: ~10 min.
3. Statistics collection (128 calibration samples): ~17 min.
4. AWQ + scale estimation: started but did not complete in 40+ minutes.

The tracing + statistics phases produced expected output. The AWQ phase
did not generate progress messages, making it hard to estimate ETA.

## What was successfully measured before timeout

The export actually completed enough work to print the bitwidth distribution:

```
| int8_asym, per-channel    | 13% (2 / 226)               | 0% (0 / 224)                           |
| int4_asym, group size 128 | 87% (224 / 226)             | 100% (224 / 224)                       |
```

So the model was correctly weight-format-decided (87% INT4, 13% INT8 for
sensitive layers). The remaining work was applying AWQ scale fitting
across all weight tensors, which is iterative and CPU-heavy.

## Next attempt (open follow-up)

Either:
1. Pull a HF-published AWQ model directly: e.g.,
   `OpenVINO/Meta-Llama-3.1-8B-Instruct-int4-cw-ov` (column-wise INT4)
   or `huggingface/llama-3.1-8b-int4-AWQ-OV` if exists.
2. Re-run AWQ on charlie if its CPU is faster.
3. Skip `--scale-estimation` (which is the slow part) and try plain `--awq`.
4. Use `--awq` without `--dataset` (data-free AWQ, much faster).

For autolab purposes: time investment / value ratio is too high for
explicit AWQ export. We'll use the pre-quantized HF variants if accuracy
matters; default INT4 (data-free) is good enough for perf benchmarking.
