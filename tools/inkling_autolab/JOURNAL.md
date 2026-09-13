# Inkling / Panther Lake research journal

## 0 — orientation (2026-09-12)

User directive: autonomously maximize Inkling performance on the single authorized
Panther Lake host tate-07 (100.82.253.76), using ../autolab; commit as t8 without
coauthors. Work isolated on perf/inkling-panther-autolab from feat/inkling 9aaebff0.
The source worktree ../tahoma-inkling was clean. Latest progress: resident 975B
export runs at 1.5 model tokens/s on the Mac Pro, after concurrent experts,
residency-adaptive reads, pinning, and physical-core Rayon defaults.

Host inspection: Intel Core Ultra X7 358H, 16 cores/16 threads, Windows 11 Pro,
64 GB RAM (~40 GB available), C: only 4.07 GB free. No Inkling export found in the
model tree. The 512 GB export cannot be deployed with this storage capacity.
Full-model throughput remains unmeasured; synthetic production-size layer/kernel
rates are NOT model tokens/s and cannot establish a single-machine model record.
No existing model, cache, source, or service is removed by this research.

## 1 — hypothesis: thread/schedule tuning

Before measurement: on this 4P+8E+4LPE hybrid CPU, default 16-thread nested Rayon
execution may be slower than a smaller pool. Sweep threads, then expert schedule
and mmap read policy, against fixed synthetic full-size attention + eight selected
int4 experts. Preserve all output bits across scheduling changes. Use identical
inputs, warmup, repeated timing samples, and independent process runs. The bank
contains eight distinct full-size expert bins; 256 router rows select six fixed
routed experts plus two shared. This represents the resident active working set,
not the full expert population or paged-checkpoint behavior.

## Next hypotheses

- Process affinity can avoid scheduling work onto the low-power cores.
- Reusing each activation vector across several bf16 output rows may improve
  the attention GEMV while preserving each row's accumulation order.
- Independent attention projections may overlap through the same Rayon pool.

Stop a finite sweep after all configurations; confirm winners in paired repeated
runs and stop code exploration when tested alternatives no longer improve by 3%.
Do not describe a tested local optimum as a proven hardware maximum.

## 1 results — thread sweep

High priority removed the initial background scheduling variance (normal-priority
first scout: 47–64 ms; High priority at 16 threads: 45.97–46.51 ms). All seven
thread counts preserve output hash `4e89f0793afa1015`. Best 16 threads: median
46.358 ms/layer-token; 12: 47.679; 10: 48.407; 8: 50.047; 6: 56.187;
4: 62.134; 2: 90.707. The smaller-pool hypothesis is refuted at this schedule.

## 2 — expert schedule/read-policy hypothesis

Before measurement: Windows working-set residency may keep using whole-bin
copies despite a warm standby cache. Direct mmap reads should avoid copies in
this resident workload; concurrent versus serial experts may change contention.
Sweep both switches at 16 threads, requiring the reference hash.

## 2 results — direct mmap wins

16 threads, concurrent experts: default adaptive reads 46.602 ms, direct mmap
18.016 ms (2.59x). Serial experts: 49.108 ms adaptive, 20.082 ms direct.
All four runs preserve every output bit. The likely explanation is Windows
standby-cache versus process-working-set residency: explicit reads can leave
the mapping unvisited, so a working-set query keeps choosing a copy. This is
a hypothesis about the cause, not a traced conclusion. Keep the production
default for paged workloads until a real checkpoint can verify that regime.
For this resident configuration, set CASCADIA_INKLING_SEQ_READS=1.

## 3 — direct-read thread and affinity hypothesis

Before measurement: once copy traffic disappears the best thread count may
change. Repeat a smaller thread sweep, then compare explicit CPU bit masks.
Do not infer P/E/LPE identities solely from bit positions.

## Paused at user request — restart checkpoint

Stopped the controller with SIGINT during campaign 003, before the 8-thread
experiment could be recorded. Completed results: 16 threads 18.141694 ms;
12 threads 19.094159 ms, both hash `4e89f0793afa1015`. The raw partial export
is retained. Resume the same campaign/SQLite database; the first unrecorded
configuration is 8 threads, then 4, 6, 10. No kernel candidate was deployed.
The bf16 row-tiling candidate is local and untested: preserve as a patch and
do not promote it until x86 tests plus fixed-hash benchmarks pass.

## Resumed (2026-09-12)

The user resumed after restarting Codex. Worktree/SQLite database/candidate patch
intact; no benchmark/build processes on the host. Existing OVMS service retained.
Campaign 003 resumed from its first unrecorded configuration, 8 threads.

Before campaign 004: compare explicit affinity masks with a thread per allowed
CPU: 0xffff/16, 0x0fff/12, 0xfff0/12, 0x000f/4. These are bit masks, not asserted
P/E core identities. Same reference hash and five repeated timing samples.

## 5 — bf16 row reuse hypothesis (before measurement)

Candidate: share each activation load across two/four independent output rows
in the production bf16 GEMV, retaining each row's exact two FMA accumulators,
reduction and bf16 rounding. Default remains the reference one-row path during
experimentation. Require new x86 bit-exact tests, the Inkling fixture suite,
and the full synthetic-layer reference hash. Compare rows 1/2/4 in one binary
with direct mmap reads at 16 threads.

## Target clarified by user

The user is offline and explicitly requested continued autonomous work until
**25 tokens/s for the large Inkling model on this one PTL box**. That target is
full-model throughput, not the synthetic layer metric. Do not equate these.
The current export requires 512 GB storage and touches about 36.5 GB per decode
token; the resident batch-1 bandwidth arithmetic in INKLING_SCALING.md is far
below 25 tok/s. Investigate actual deployment capacity and throughput semantics
alongside engine optimizations; do not silently prune/change the model or claim
an estimated layer rate as attainment.

## 3–4 results

Direct reads remain fastest at 16 threads: 18.142 ms; 12: 19.094; 10: 20.143;
8: 21.485; 6: 23.710; 4: 27.427. Affinity campaign also favors all cores:
0xffff/16 18.658 ms; 0x0fff/12 19.045; 0xfff0/12 21.889; 0x000f/4 26.667.
All output hashes match. The affinity hypothesis is refuted for these masks.

## 6 — int4 row reuse hypothesis (before measurement)

The current AVX2 expert dot keeps one FMA dependency chain per row. A two/four
row tile can interleave independent chains and reuse activation loads, while
preserving the exact within-row sequence of four FMAs per quantization group.
Test separate bf16/int4 knobs and their composition. Exclude the AVX-512 dispatch
from this path because its reduction differs. Require bit-exact row tests over
all nibble values, varied scales, odd row counts and real input dimensions.

## 5 results — bf16 row reuse inconclusive

75 MSVC tests passed (new exact-row test + all 74 Inkling tests). Full-layer
reference hash remains exact at rows 1/2/4. Median ms: 17.953 / 17.823 /
18.083. The best change is <1%, below the 3% promotion threshold. Do not
change the default from this result. Retain the candidate as an experiment
while testing whether it composes with int4 row tiling.

## 7 — independent projection concurrency (before measurement)

Inkling's Q/K/V/relative-position projections all read the same hidden vector
and are independent. Run them concurrently through nested Rayon joins, preserving
the production GEMV and all output bits. Compare the new switch on/off after
selecting the int4 setting. Test fixtures with the switch enabled before timing.

## Deployment search result

Read-only inspection found one 1.024 TB physical SSD, no spare/unallocated disk,
no mapped network drives, and a 1 Gb/s physical Ethernet adapter. A bounded,
junction-safe scan of 20,095 non-system directories (including model and user
cache locations) found no Inkling manifest. The 100 Gb/s value reported for the
Tailscale tunnel is virtual-adapter metadata, not the physical link rate.
Result: there is still no full checkpoint to benchmark locally and no capacity
to copy the 512 GB export without reclaiming other projects' storage. The target
is not reached; all rates so far are synthetic resident-layer rates.

## 8 — iGPU backend hypothesis (before measurement)

Challenge the CPU-only implementation: use the existing exporter helpers to
build one 975B-sized expert as an OpenVINO graph on the bins' original u4/bf16
scale grid, preserving bf16 boundaries after each linear. Compare CPU and GPU
component latency and validate against independent f64 dots + bf16 rounding.
Numerical gate: RMS relative error <=0.5%, worst error <=3% of reference RMS;
this allows summation-order differences, not re-quantization. It is a component
probe, cannot establish model quality/throughput, and will not silently replace
the production backend. No recompilation/cache churn or full-population paging
is included in steady-state component timings.

## 6–7 results

All row combinations preserve the full output hash. Best combined bf16=2,
int4=4 is 17.114 ms versus 17.991 ms for 1/1 in campaign 006 (~5% speedup).
Campaign 007 independently repeats 17.116 ms for that combination. Concurrent
projections regress it to 17.518 ms; with int4=1 they regress 17.871 to 18.129.
Reject projection concurrency as production code; keep its patch/evidence.
Require interleaved independent process confirmation for the combined tiling
before promoting the optional optimized path.

## 9 — paired confirmation (before measurement)

Six independent process groups, each containing the original adaptive baseline,
original direct-read baseline, and bf16=2/int4=4 direct-read candidate. Rotate
order across groups so heat/cache/order does not always favor one arm. Seven
samples per process. Every run requires the same full-layer hash. Retain the
actual arm and binary SHA256 in the raw output, since the scheduled slot is
rotated by repetition number. This tests a local layer improvement only.

## 8 result and 10 hypothesis — GPU under the active working set

The single-expert OpenVINO probe passed its numerical oracle on both devices:
CPU 1.265 ms, GPU 0.483 ms. This compares two OpenVINO paths, not a Rust
whole-model run. Next test eight distinct experts (255 MB of original int4
weights) with fused, serial and asynchronous dispatch, on CPU and GPU. Validate
each expert independently at two inputs before timing; the same numerical
thresholds apply. Excludes dynamic routing, compilation churn, attention and
paging. This determines whether single-expert gains survive the active weight
set and dispatch overhead before considering a production GPU backend.

## 9 result — confirmed, optional profile accepted

Every one of 18 independent processes preserved the full output hash. Median
across six processes per arm: original adaptive 47.325608 ms; original direct
18.389891 ms; bf16=2/int4=4 direct 17.190217 ms. The combined improvement is
2.753x. Relative to direct reads alone the tiles give 1.070x, winning in all six
paired groups (1.062–1.075x), beyond the 3% promotion threshold. Keep the tiles
as opt-in shared kernels, with original defaults on untested platforms and
workloads. Reject and remove production projection concurrency; retain its patch.

## 11 hypothesis — Windows residency diagnosis

The adaptive path may repeatedly copy file-cache-hot weights because a buffered
read does not populate the mmap's process working set. Microsoft documents that
soft page faults can be satisfied from RAM outside the process working set:
https://learn.microsoft.com/en-us/windows/win32/memory/working-set
Probe QueryWorkingSetEx on a fresh mapping before/after three buffered reads,
prefetch, and a mapping page walk. Use only our synthetic bin; do not clear global
caches or change working-set limits. This diagnoses the resident-path gap; it
does not establish how an alternative policy performs with cold real weights.

## 10 result, validation, and 12 final confirmation hypothesis

Eight experts, CPU fused/serial/async: 10.052 / 12.332 / 9.541 ms.
GPU fused/serial/async: 3.777 / 4.239 / 3.268 ms. Every expert at both inputs
passed the independent numerical oracle. Production GPU integration and full
population/paging are unvalidated; retain this as a promising component result.

Final retained CPU source: 222 MSVC regression tests passed, including all
Inkling fixtures and shared DSV4/GLM math/expert tests. The new full-decode
harness reproduces all eight HF fixture greedy IDs across three repetitions,
with fixture-only metric, full_model=0 and hash 1f7cd0eb14a22662. Clippy completes
with existing library warnings and three benchmark iterator style suggestions.

Toolchain audit correction: an explicit query reports MSVC rustc 1.98.1,
LLVM 22.1.8. Earlier hand-entered 1.95 metadata did not establish the explicit
MSVC version and has been corrected, preserving a note. To remove ambiguity
from binary/compiler differences, campaign 012 repeats the rotating comparison
using the SAME final binary for all three settings (original algorithms at
rows 1/1, direct and adaptive, versus tiles 2/4). This is the final promotion
measurement; frozen earlier binaries/results remain intact.

## 11 result — diagnosis confirmed

All 64 sampled mmap pages remained invalid after each of three buffered reads
(8.79 / 8.18 / 8.04 ms for 31.85 MB) and a successful PrefetchVirtualMemory call.
A page walk of the mapping changed the same sample to 64/64 valid. This confirms
that the adaptive working-set gate can repeatedly take the bulk-copy path for
warm file-cache data on Windows. Corrected its source comment: it is a lower
bound on file-cache residency, not a true cache-miss count. Retained the explicit
direct-read resident profile. No automatic cold/paged policy change is justified
by this probe; it would need real-checkpoint paging measurements.

## 12 result — final profile and target status

All 18 runs completed, same final binary in every arm, all hashes exact. Median
across processes: adaptive 47.105705 ms, direct 18.229506 ms, tiled direct
17.455544 ms. Combined 2.699x; incremental tiles 1.044x. All six paired groups
favor tiles by 3.37–7.50%. Use these conservative final numbers as the retained
profile result; campaign 009's frozen-binary figures remain historical evidence.

The full-model launcher was exercised at the intended deployment path and
correctly failed because no checkpoint is present. 25 full-model tok/s is NOT
reached or measured. The complete-model benchmark is built and fixture-verified,
and the full campaign template plus target gate are ready for actual weights.
The resource blocker remains ~3.4 GB free disk versus a ~512 GB export. Do not
claim an ongoing background research agent after this session or burn repeated
component sweeps as a substitute for the missing full-model measurement.

The large resident-layer speedup is conditional: real-model prefill can touch the
mappings before decode and thereby avoid some repeated copies already. Full
checkpoint routing, prefill and paging must be measured before projecting this
factor onto user-visible generation. Likewise the GPU probe omits compilation
churn across the 16,512 MoE expert instances; it is not yet a deployment strategy.
