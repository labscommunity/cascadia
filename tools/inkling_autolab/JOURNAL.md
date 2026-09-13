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
