# NPUW weights-bank sharing probe

**Question:** can two *user-level* NPU compilations (our decode + chunked-
prefill IR variants — byte-identical weights, different static shapes) share
one weight allocation through OpenVINO's NPUW weights bank, cutting an
all-NPU phase split from 2× to ~1× resident weights?

**Why it might work:** NPUW internally shares one bank across its own
prefill/generate/head submodels; the bank manager is process-global, keyed by
the `NPUW_WEIGHTS_BANK` string, and bank storage dedups tensors per device.
If two `compile_model` calls with the same bank name dedup identical
constants, we get the sharing for free.

**Why it might not:** `NPU_USE_NPUW=YES` routes the whole compile through
NPUW's partitioner, which is built for single whole models (it does its own
prefill/generate split for LLM-flagged stateful graphs) — our graphs are
already sharded, stateless, and static, a shape NPUW was not designed to
ingest. This is an undocumented surface; treat any behavior as unsupported.

## Method

`crates/cascadia-engine-openvino/tests/npuw_bank_probe.rs` (env-gated):
compiles BOTH variants on `NPU` (`--prefill-device NPU` equivalent), runs an
8-token sanity generation, then holds the process alive while
`a sysfoot wrapper (run_npuw_probe.ps1 — NOT committed; lost in a session restart; re-derive: poll `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory` at 5 s intervals around the PROBE-HOLD window)` samples peak working set. Two modes:

- `baseline` — plain NPU compiles (the shipping all-NPU config).
- `npuw` — adds `NPU_USE_NPUW=YES` + `NPUW_WEIGHTS_BANK=cascadia-bank-probe`
  to both compiles.

Model: Llama-3.2-1B INT4 single-stage static export (+ seq=64 prefill
variant), Lunar Lake 258V (pawan-01), OV GenAI 2026.1 SDK.

## Results (pawan-01, LNL 258V, 2026-07-15, two complete runs)

Both modes **compiled and generated correct output** ("Paris. The capital of
Germany is Berlin") — NPUW accepts our pre-sharded stateless static graphs
and the shared bank name does not break execution.

| Mode | peak process WS | steady process WS |
|---|---|---|
| baseline (plain NPU ×2 compiles) | 9.39 GB | 0.04 GB |
| npuw (shared `NPUW_WEIGHTS_BANK`) | 6.32 GB | 4.32 GB |

**The memory verdict is UNRESOLVED — process working set cannot compare the
two paths.** The plain NPU plugin holds weights in driver-owned Level-Zero
allocations invisible to process WS (hence 0.04 GB "steady" with two models
loaded), while NPUW's bank allocates host-USM tensors that DO count (4.32 GB
— far above 2× the 0.64 GB INT4 weights, consistent with NPUW's DCOFF
decompressing INT4→f16 host-side). Peaks are dominated by compiler
transients. So: the bank likely dedups, but DCOFF expansion may make total
residency WORSE than plain-path INT4 for compressed models. A system-wide
free-memory-delta measurement (captures driver allocations) was scripted
(`a sysfoot wrapper (run_npuw_probe.ps1 — NOT committed; lost in a session restart; re-derive: poll `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory` at 5 s intervals around the PROBE-HOLD window)` sysfoot columns) but a bastion network outage
interrupted the run; re-run it when the fleet path is stable before drawing
a memory conclusion.

## Verdict

Functionally viable, memory-unproven, unsupported surface. Do not build on
it: prefer `--park-prefill` (shipped, measured) for residency relief today
and the in-place INT4 GEMV RFC (docs/rfcs/) for the principled 1× path.
Re-probe with the sysfoot measurement + accuracy sweep + multi-stage before
reconsidering.
