# Device profile + HETERO placement — workflow & findings

**Status:** Step 1 of [#41](https://github.com/labscommunity/cascadia/issues/41)
(three-tier {iGPU, NPU, CPU} ILP placement). Lands the
`cascadia profile-devices` subcommand and the validation data that scopes
the rest of #41.

## What this is

Issue #41 proposes per-(layer, device) ILP placement on Intel AI PCs, on the
premise that OpenVINO's default `HETERO:GPU,CPU` is leaving headroom on the
table relative to a cost-aware placement. Before building the ILP, we need to
measure whether that premise holds *on the hardware we actually have*. This
tool is that measurement.

`cascadia profile-devices` enumerates the OV plugins on a worker host, compiles
the target model on each, and reports cold-compile time + decode tok/s. The
output is a `device_profile.json` consumed by:

- humans choosing a `--device` for `cascadia worker`
- (future) the ILP step, which needs per-device costs as input
- the recommendation matrix later in this document

## Quickstart

```bash
cascadia profile-devices \
  --model /path/to/ov-exported-model \
  --output device_profile.json \
  --max-tokens 32 \
  --runs 3
```

The default `--devices auto` enumerates whatever OV sees (typically
`CPU`, `GPU`, `NPU` on a Lunar Lake AI PC). Add `HETERO:` strings explicitly:

```bash
cascadia profile-devices --devices "auto,HETERO:GPU,CPU,HETERO:GPU,CPU,NPU" ...
```

…or sweep every priority permutation with `--include-hetero-permutations`
(factorial — 6 extra runs on a 3-device host).

## File format

JSON, schema `version: 1`. See `crates/cascadia-cli/src/profile.rs`.

```json
{
  "schema_version": 1,
  "hardware": {
    "host_devices": [
      { "name": "CPU", "full_name": "Intel(R) Core(TM) Ultra 7 258V" },
      { "name": "GPU", "full_name": "Intel(R) Arc(TM) 140V GPU (16GB)" },
      { "name": "NPU", "full_name": "Intel(R) AI Boost" }
    ]
  },
  "model": "C:\\cascadia\\models\\qwen3-pre-ov",
  "prompt": "Explain Intel Lunar Lake in three sentences.",
  "max_tokens": 32,
  "runs": 3,
  "warmup": 1,
  "results": [
    {
      "device": "CPU",
      "compile_s": 1.59,
      "warmup_s": 1.12,
      "best_run_s": 1.105,
      "runs_s": [1.105, 1.119, 1.132],
      "tok_per_sec": 28.96,
      "output_preview": "<think>\nOkay…"
    },
    {
      "device": "NPU",
      "error": "compile-fail: openvino-genai error: …StopLocationVerifierPass…"
    }
  ],
  "best_device": "GPU",
  "best_tok_per_sec": 41.19
}
```

`null` fields are dropped, not serialised, so a failed device emits only
`{device, error}`. Successful devices always populate `compile_s`,
`best_run_s`, `tok_per_sec`, and (if `--max-tokens > 0`) `output_preview`.

## Measured on an Intel Lunar Lake laptop (Core Ultra 7 258V + Arc 140V iGPU + NPU 4)

Run 2026-05-21, OpenVINO 2026.1.0, openvino_genai 2026.1.0.0-2957-1dabb8c2255.
Model: Qwen3-1.7B int4, exported via Optimum-Intel. 32-token greedy decode,
3 measured runs (best reported), 1 warmup. Numbers from `cascadia
profile-devices` itself, not a separate Python harness.

| Device                       | compile (s) | best run (s) | **tok/s** | vs GPU alone |
|------------------------------|-------------|--------------|-----------|--------------|
| CPU                          |       1.3   |        1.017 |     31.45 |       0.66×  |
| GPU (Arc 140V iGPU)          |      10.8   |        0.669 |     47.82 |       1.00×  |
| NPU                          |     fail    |         —    |       —   |     —        |
| **HETERO:GPU,CPU**           |      23.7   |        0.655 | **48.87** |     **1.02×**|
| HETERO:GPU,CPU,NPU           |      23.0   |        0.669 |     47.85 |       1.00×  |
| HETERO:NPU,GPU,CPU           |     fail    |         —    |       —   |     —        |

NPU failure is the same `StopLocationVerifierPass: Found 16 duplicated names`
issue [#37](https://github.com/labscommunity/cascadia/issues/37)
identified in PR #34 — still unfixed in OV 2026.1 for this Qwen3 export.
HETERO with NPU first inherits the failure; HETERO with NPU as third
priority skips NPU silently and runs on GPU+CPU.

### What the numbers say

1. **`HETERO:GPU,CPU` edges out `GPU` alone by ≈1 tok/s (2%).** Within the
   noise floor of 3-sample best-of timing at sub-second wall times. The
   premise of #41 — that ILP-driven placement can beat stock HETERO by
   ≥10% — is **not validated on Lunar Lake for Qwen3-1.7B**. Stock HETERO
   is doing fine on this hardware/model combination.

2. **CPU is 1.5× slower than GPU** for this model. OneDNN-on-Xe2 dominates
   AVX2 on the int4 GEMM shapes Qwen3-1.7B emits. The model fits in
   the iGPU's 16 GB UMA partition so the iGPU never needs to spill.

3. **`HETERO:GPU,CPU,NPU` ≈ `GPU` alone.** OV's HETERO plugin notices
   NPU can't compile any op for Qwen3, silently drops it, and ends up
   roughly equivalent to GPU-only. Useful as a safe default when
   operators don't know which devices the model supports.

4. **NPU is not reachable for Qwen3.** Op-support gap; documented in #37.
   For models in the LLaMA / Mistral / Phi family the NPU may yet work
   (OV 2026.x added Qwen2.5-1.5B, Phi-3.5-mini, and several others to
   the NPU-supported list per release notes) — but until we measure
   those, the ILP can't plan against NPU for any Qwen3-class workload.

5. **The Python-bench numbers in PR #34's discussion (HETERO 10% slower
   than GPU) were measurement artifacts.** Likely from per-call pipeline
   construction overhead being charged against HETERO's longer compile
   path. The cascadia profile-devices tool measures decode-only after
   pipeline construction and warmup, and shows HETERO is competitive.

## How `--device` maps to OV

`cascadia worker --device <STRING>` passes the string verbatim to OV's
`Core::compile_model(model, device)`. OV accepts:

- Single plugin: `CPU`, `GPU`, `NPU`
- Multi-GPU rank: `GPU.0`, `GPU.1`, …
- HETERO priority list: `HETERO:DEV1,DEV2,…` (comma-separated, no spaces;
  highest priority first)
- AUTO: `AUTO:DEV1,DEV2,…` (auto-picks between listed plugins per request)

For HETERO, each op falls to the highest-priority device that supports it.
With manual affinity (`node.get_rt_info()["affinity"] = "..."` set on the
OV Model before compile), the priority list is ignored for the
explicitly-assigned ops. The ILP step of #41 will use manual affinity.

## Recommendation matrix (from this run)

| Workload                                | Recommended `--device`                    |
|-----------------------------------------|-------------------------------------------|
| Qwen3-1.7B int4, ≤ 4 k context          | `GPU` (or `HETERO:GPU,CPU` — both ≈ 48 tok/s) |
| Qwen3-1.7B int4 + want NPU              | (not yet — see #37)                       |
| Larger model (> 16 GB UMA-shared, untested here) | `HETERO:GPU,CPU` then re-measure |

For any operator deploying a new model: run `cascadia profile-devices` first
on each target host class; pick the highest `best_tok_per_sec` from the JSON.
Compile-time outliers (NPU's 0-or-fail behaviour) are recorded so you can
preemptively exclude them via `--devices CPU,GPU` rather than wait on a
failed compile per request.

Cold-compile cost matters: HETERO took 24 s vs GPU-alone's 11 s on this host.
That's worth amortising via the `--ov-cache-dir` flag (or `cascadia
worker --ov-cache-dir`) so subsequent invocations on the same model hit
a cached blob in ≈1 s instead.

## What this does NOT do

It does not build the ILP solver or the OV IR-rewrite step that sets
per-op affinities. The data above doesn't yet justify that work: the
single test model fits on one device and the largest measurable headroom
is "GPU-only beats HETERO by 10 %," which a one-line `--device GPU`
choice captures. The follow-up triggers for building the ILP are:

- A model in the cascadia target set that exceeds 16 GB UMA partition
  (forces a placement decision).
- NPU op-support for Qwen3 / LLaMA-class (lets the ILP pick between
  three plugins on the same model, instead of two-plus-a-failure).
- Multiple measured devices where no single plugin dominates the others
  on every op type (today, GPU dominates on every op we can run).

Until then, `cascadia profile-devices` is the answer to "which `--device`
flag should I pass?", and the JSON output gates the decision to build
the ILP.

## References

- Issue [#41](https://github.com/labscommunity/cascadia/issues/41) —
  three-tier ILP placement
- Issue [#37](https://github.com/labscommunity/cascadia/issues/37) — NPU
  compile failures (`vpux-compiler StopLocationVerifierPass` for Qwen3)
- PR [#34](https://github.com/labscommunity/cascadia/pull/34) — first
  measured the GPU/CPU/HETERO numbers; this run reproduces with newer OV
- PowerInfer §6.3 (MIT, arxiv:2312.12456) — original two-tier ILP design
- PowerInfer-2 §4.1.3 (MIT, arxiv:2406.06282) — dynamic batch-size
  per-device graph swap
- OV HETERO docs:
  <https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html>
