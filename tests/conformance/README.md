# Elastic-memory conformance checks

Runtime checks for the `--elastic` posture. They drive a real `cascadia run`
server and read the process's own memory split:

| check | method | pass condition |
|---|---|---|
| C2 output identity | stock vs `--elastic`, temperature 0 | byte-identical |
| C5 elasticity witness | `RssAnon` split (`/proc/<pid>/status`) | anon collapses vs stock while file-backed grows |
| C1 floor independence (opt-in, `--scale-model M2`) | settled committed memory at two model scales ≥8× apart (file-size ratio, fail-closed) | both floors in one band |
| C3 pressure survival (opt-in, `--pressure-mb N`) | serve under a cgroup `MemoryMax` cap, swap off | correct text, no OOM kill |
| C4 co-tenancy (opt-in, `--coten-models A,B,… --coten-budget-mb N`) | N co-resident servers under **one** fleet `MemoryMax`, budget above the sum of their elastic floors and far below naive N× provisioning | every tenant serves its solo-verified text |

Guards, so a pass means something:

- the elastic leg must log `elastic posture active` — hook off is invalid, not a pass;
- the backing dir must be disk-backed: on tmpfs the pages cannot be written
  back, so the posture gives no survival benefit even though `RssAnon` still
  collapses (measured: tmpfs backing dies at the same caps as stock);
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
- **C1 floor independence — 2.1× band, criterion ≤1.5× not yet met**: with
  the interposer's mmap leg + fd-free pool, `MALLOC_ARENA_MAX=1` and
  `ELASTIC_MIN_KB=256`, the gemma-26b settled floor drops 527 → 172 MB and
  the 1.5B leg to 81 MB — band 2.11× across a 16.4× scale gap. History at
  the stock 1 MB threshold: 178 vs 531 MB (2.98×); other pairs:
  Qwen3.8-27B-int4 at `--elastic-min-mb 16` — 883 vs 4480 MB (5.07×);
  Muse-Glimmer-30B-int4 at 16 — 883 vs 7036 MB (7.97×). The residual after
  the mmap leg is model-width-scaling engine-internal state (oneDNN/MoE),
  below any allocator threshold. The check itself is fail-closed and
  reports the band it measured.

## Large models and `--elastic-min-mb`

Against multi-GB exports (27B-class) the `run` default threshold of 1 MB
file-backs some 1–16 MB engine-internal allocations too, and the process dies
with SIGSEGV during load (reproduced on B70, pool on and off alike; hook inerts
fine, `--elastic-min-mb 16` serves). `--elastic-min-mb 16` (weights-only
threshold) is the validated setting for C1's large leg today; the mechanism
itself is unaffected at small scales. The escape hatches (`--elastic-min-mb`,
`--elastic-so`) are env-driven and need no binary rebuild.

## Platform scope

The checks are engine-independent: the interposer sits below the engine and
intercepts libc allocation, so they apply to any engine that allocates through
`malloc`. This suite is Linux-only as written (`/proc`, systemd cgroups). The
Windows hook (PR #132, Detours) is verified separately — 1329 → 223 MB private
commit on Lunar Lake, gate line confirmed — but a Windows port of these checks
needs a private-bytes witness (psapi working-set counters) and a Job Object for
the pressure legs.
