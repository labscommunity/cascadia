# Elastic-memory conformance checks

Runtime checks for the `--elastic` posture. They drive a real `cascadia run`
server and read the process's own memory split:

| check | method | pass condition |
|---|---|---|
| C2 output identity | stock vs `--elastic`, temperature 0 | byte-identical |
| C5 elasticity witness | `RssAnon` split (`/proc/<pid>/status`) | anon collapses vs stock while file-backed grows |
| C1 floor independence (opt-in, `--scale-model M2`) | settled committed memory at two model scales ≥8× apart (file-size ratio, fail-closed) | both floors within 1.5× of each other AND the large-model floor under 256 MB |
| C3 pressure survival (opt-in, `--pressure-mb N`) | serve under a cgroup `MemoryMax` cap, swap off | correct text, no OOM kill, **and the stock leg died at the same cap** (otherwise the check is recorded as inconclusive, not PASS) |
| C4 co-tenancy (opt-in, `--coten-models A,B,… --coten-budget-mb N`) | N co-resident servers under **one** fleet `MemoryMax`, budget above the sum of their elastic floors and far below naive N× provisioning | every tenant serves its solo-verified text |

Guards, so a pass means something:

- the elastic leg must log `elastic posture active` — hook off is invalid, not a pass;
- the backing dir must be disk-backed: on tmpfs the pages cannot be written
  back, so the posture gives no survival benefit even though `RssAnon` still
  collapses (measured: tmpfs backing dies at the same caps as stock);
- the interposer `.so` is selected by **mtime**, not lexicographic order: cargo
  leaves the previous build's `.so` beside the new one, and picking the wrong
  one silently measures an unpatched hook (this cost a whole C1 leg once);
- the pressure and co-tenancy legs verify `memory.max` was actually applied to
  the scope before trusting the result, and read `memory.events` for `oom_kill`;
- C1 refuses a model pair under 8× apart and C4 refuses a budget outside
  `(1.15×, 3×]` of the solo floor sum — both invalid instruments, not passes;
- capped legs drop the model files from the page cache first
  (`POSIX_FADV_DONTNEED`): cache charged by a previous leg is reparented when
  that leg's cgroup dies and would bias low caps toward survivable (an
  instrument bias measured during titration).

## Running

```bash
export INTEL_OPENVINO_DIR=/path/to/ov-genai-sdk     # or set LD_LIBRARY_PATH
python3 tests/conformance/test_elastic.py \
    --bin target/release/cascadia \
    --model /path/to/model-int4-ov \
    --elastic-dir /disk/backed/dir \
    [--pressure-mb 1024] \
    [--scale-model /path/to/model2] [--elastic-min-mb 16] \
    [--coten-models A,B --coten-budget-mb 1024] \
    [--json-out results.json]
```

Reference run (B70, Qwen2.5-1.5B-int4, CPU, 64 tok): C2 byte-identical;
C5 `RssAnon` 1106 → 170 MB, `RssFile` 918 → 1859 MB; C3 at 1024 MB — stock is
OOM-killed, elastic serves the identical text.

Opt-in legs, same box:

- **C4 co-tenancy — PASS**: Qwen2.5-1.5B + gemma-4-26b-a4b (int4) under one
  500 MB fleet `MemoryMax`; solo floors 82 + 172 MB (sum = 51% of the
  budget), both tenants served their solo text, `memory.max` verified on the
  scope. Naive 2× provisioning for these two is ~3 GB committed. (With the
  stock 1 MB threshold the same pair passes at 1600 MB — floors 178 + 528 MB.)
- **C1 floor independence — FAILS at ~3×; the residual is identified and is
  not reachable by allocator interposition.** Swept 2026-09-30 over
  Qwen2.5-1.5B vs Qwen3.8-27B (17.1× scale), B70, backing on ext4:

  | threshold | 1.5B | 27B | band |
  |---|---|---|---|
  | 16 MB | 883 MB | 4480 MB | 5.07× |
  | 1 MB | 178 MB | 597 MB | 3.35× |
  | 256 KB | 85 MB | 256 MB | 3.01× |
  | 64 KB | 62 MB | 212 MB | 3.41× |
  | 4 KB + `MALLOC_MMAP_THRESHOLD_` | 61 MB | 206 MB | 3.39× |
  | 64 KB + `MALLOC_MMAP_THRESHOLD_` | 67 MB | 224 MB | 3.36× |

  The band is pinned at ~3× whatever the threshold: the residual does not
  respond to the allocator. An `strace` of `mmap` during the 27B load closes
  it — of every mapping ≥1 MB, the ones that escape the hook are **all 69
  thread stacks** (`MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK`, 400 MB). The
  read-only anonymous mappings (70 of them, 7.3 GB) carry no resident pages,
  and forcing glibc off `brk` (`MALLOC_MMAP_THRESHOLD_`) changes nothing.
  Stacks are excluded deliberately: file-backing a stack breaks guard pages,
  growth and unwinding. That is memory no allocator interposer can or should
  redirect, so ≤1.5× is not reachable by this approach. The check is
  fail-closed and reports the band it measured — the correct behaviour.

  History, same box: gemma-26b vs 1.5B with `MALLOC_ARENA_MAX=1` +
  `ELASTIC_MIN_KB=256` — 172 vs 81 MB (2.11× across a 16.4× gap); stock 1 MB
  threshold 178 vs 531 MB (2.98×); Muse-Glimmer-30B-int4 at 16 MB —
  883 vs 7036 MB (7.97×). All well above criterion; same root cause.

## Large models and `--elastic-min-mb`

The 1 MB default now loads 27B-class exports. It did not before: the process
died with SIGSEGV during load because `big_alloc` ended each mapping flush
against the following unmapped page (`PAGE + round_up(size)`), so a request
whose size was an exact multiple of `PAGE` had zero slack behind it. Heap
allocators never have that shape — glibc's 16-byte header forces the block to
round up to a further page — and oneDNN's vectorised kernels read up to 64
bytes past the end of a buffer on the assumption that the slack is there.
dmesg gave the mechanism directly: `error 4` (read, page not present) at a
page-aligned address, with the faulting instruction `vmovups 0x40(%r10),%ymm6`
inside JIT-generated AVX2 code. `ELASTIC_MMAP=0` crashed identically, which
localised it to the malloc leg. Fixed with one page of tail slack
(`3438921`, mirrored in the Windows leg). `--elastic-min-mb 16` is still a
valid weights-only setting, but it is no longer needed to survive load. The
escape hatches (`--elastic-min-mb`, `--elastic-so`) remain env-driven with no
rebuild required.

## Platform scope

The checks are engine-independent: the interposer sits below the engine and
intercepts libc allocation, so they apply to any engine that allocates through
`malloc`. `test_elastic.py` (this file's sibling) is Linux-only as written —
`/proc` and systemd cgroups.

**Windows: `test_elastic_win.py`.** Same three questions, different witnesses:

| | Linux (`test_elastic.py`) | Windows (`test_elastic_win.py`) |
|---|---|---|
| memory witness | `RssAnon` from `/proc/<pid>/status` | `PagefileUsage` from `GetProcessMemoryInfo` (psapi) — the private commit |
| working set | `RssFile` for contrast | `WorkingSetSize`, **context only** — it does not drop without pressure (measured 3098 → 3114 MB) |
| pressure leg | cgroup v2 `MemoryMax` | Job Object `JOB_OBJECT_LIMIT_PROCESS_MEMORY` (opt-in, `--pressure-mb`) |
| hook install | `LD_PRELOAD` of the built `.so` | in-process Detours, compiled in when `DETOURS_DIR` was set at build |

Checks: **W1** gate line, **W2** output identity, **W3** elasticity witness on
the private commit, **W4** pressure survival (opt-in). A run without the gate
line is invalid, not a pass — and note that cargo caches the build fingerprint,
so after building Detours you need `cargo clean -p cascadia-elastic` or the hook
silently stays compiled out.

Reference run (2026-10-01, HunterLaptopSergio, Qwen2.5-1.5B-int8-ov, CPU):
**W1/W2/W3 PASS** — private commit 1646 → 171 MB (10% of stock), working set
3098 → 3114 MB, output byte-identical.
