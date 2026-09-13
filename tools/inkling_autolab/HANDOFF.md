# Restart handoff — Inkling / Panther Lake Autolab

**Paused at the user's explicit request on 2026-09-12 to restart Codex.**
Do not launch work until the user resumes the session. On resume, continue
this task autonomously; the user has already authorized testing tate-07 and
committing/pushing as t8, without coauthor trailers. Other agents share the
computer and repository. Do not touch their working trees or processes.

## Objective and scope

Find the other agent's Inkling progress, use sibling ../autolab to autonomously
optimize Inkling tokens/s on the single Panther Lake host tate-07, Tailscale
100.82.253.76. Discovery and setup are complete; optimization is in progress.
Full model benchmark is blocked by capacity: the 975B int4 export is 512 GB;
tate-07 has 64 GB RAM, ~4 GB free disk, and no Inkling checkpoint. We are
optimizing its **production-sized synthetic resident layer**, not claiming
whole-model tokens/s. Continue useful kernel work; do not delete others' files
to manufacture capacity. The user has been told this limitation.

## Locations and identities

- User's main checkout: `/Users/tatef/Workspaces/tahoma` on
  `perf/prefill-layer-streaming`, with other agents' dirty `glm5_run.rs` and
  untracked `ngram_sim.rs`; leave it untouched.
- Original Inkling worktree: `/Users/tatef/Workspaces/tahoma-inkling`, branch
  `feat/inkling`, clean at `9aaebff0`. Latest docs describe 1.5 full model tok/s
  on a resident 1.5 TB Mac Pro. PR #154 belongs to that workstream; don't rewrite.
- **Our worktree:** `/private/tmp/tahoma-inkling-panther-autolab`.
- **Our pushed branch:** `origin/perf/inkling-panther-autolab` on
  `https://github.com/labscommunity/cascadia.git`.
- Configured Git identity: `Tate Berenbaum <t8@users.noreply.github.com>`.
  No Co-Authored-By lines. Harness commit `da746136` already pushed. The
  checkpoint commit that contains this file adds the partial results and patch.
- Autolab: `/Users/tatef/Workspaces/autolab`, HEAD `3993e2c4`, with an unrelated
  pre-existing dirty `src/autolab/runners/ssh.py` (leave untouched).
- Python venv: `/private/tmp/inkling-autolab-venv`, installed editable Autolab
  plus pytest. No additional LLM/API key needed: this session is the research
  agent; Autolab handles campaign execution and SQLite persistence.
- Research files: `tools/inkling_autolab/{JOURNAL.md,README.md,research_plan.yaml,
  campaigns,results,results.db,run_campaign.py,run-bench.ps1,build.bat}`.
- Remote SSH: `ssh -o BatchMode=yes -o ConnectTimeout=10 cascadia-tate-07-ts`.
  Alias in `~/.ssh/config` is `devcloud@100.82.253.76`, key `~/.ssh/id_ed25519`.
  Hostname `pdx88-pa0794`, Windows 11 Pro / PowerShell default shell.
- Remote task-only root: `C:\Users\devcloud\inkling-autolab`.
  `repo` = exported baseline sources + benchmark example;
  `target` = isolated MSVC release build; `bin\baseline.exe` = frozen baseline;
  `synthetic-experts` = eight generated bins (~255 MB).
- Remote model/service/build directories from other agents are untouched.
  Existing `cascadia-swe-node` scheduled task and OVMS are not ours to stop.

## Verified results

All measured configurations have exact output hash `4e89f0793afa1015` for the
32-token, 16-warmup synthetic layer; each recorded process has five timing
samples. See committed raw JSON for all individual samples and configuration.

1. Normal-priority scout varied 47–64 ms/layer-token. Windows background
   scheduling was controlled by setting the benchmark child to High priority
   and applying an explicit process affinity mask. Five-sample variance then
   fell to ~1%. Use these controls for all further A/B comparisons.
2. Campaign 001, default adaptive reads / concurrent experts, all CPUs:
   16 threads 46.358 ms; 12 47.679; 10 48.407; 8 50.047; 6 56.187;
   4 62.134; 2 90.707. Smaller pools did not help.
3. Campaign 002, 16 threads:
   adaptive+parallel 46.602 ms; adaptive+serial 49.108;
   **direct mmap+parallel 18.016 ms**; direct+serial 20.082.
   `CASCADIA_INKLING_SEQ_READS=1` gives a 2.59x resident-layer gain.
   Do not change the paged-workload default solely from this synthetic result.
   Possible cause: Windows working-set residency remains false after explicit
   reads, so it repeatedly copies a warm standby-cached bin. Not yet traced.
4. Campaign 003 partially complete with direct mmap: 16 threads 18.141694 ms,
   12 threads 19.094159 ms. Interrupted during the 8-thread trial; it was NOT
   stored and should rerun. Then 4, 6, 10 remain.

Baseline binary SHA256:
`2a4292fa49def264b9efcb379d408abebf4a30935512ce0d760bbd6db423779d`.
Rust/MSVC 1.95, AVX2+FMA true, AVX512 false.
Autolab campaign/loop/SSH tests: 21 passed. Baseline MSVC build passed.
`cargo fmt --all -- --check` passed after formatting our files.
Full Inkling/x86 regression tests have not yet been run in this session.

## First actions when resumed

1. Read this file and JOURNAL.md, inspect `git status` in our worktree, and
   check the remote for competing inference/build jobs. The controller and our
   remote benchmark were stopped for the restart. Never run a benchmark and a
   build simultaneously on tate-07.
2. Continue the interrupted grid, from our worktree:

   ```sh
   /private/tmp/inkling-autolab-venv/bin/python -u \
     tools/inkling_autolab/run_campaign.py \
     tools/inkling_autolab/campaigns/003_direct_threads.yaml
   ```

   Network access may require the existing sandbox escalation. The approved
   prefix is the exact Python runner command above (through run_campaign.py).
   SSH and SCP also have saved approval rules. No new user task approval needed.
3. Design an affinity sweep under the best direct-read schedule. Current
   launcher accepts `-Mask` and `-Threads`; keep thread count aligned with CPUs
   available in each mask. Do not label bit positions P/E/LPE without probing.
4. **Unfinished SIMD candidate:** our local
   `crates/cascadia-engine-sparse-moe/src/dsv4/math.rs` is modified but NOT
   committed as production code. Its exact diff is saved and committed as
   `tools/inkling_autolab/candidates/004_bf16_rows.patch`. It adds 2-/4-row AVX2
   bf16 GEMV tiles that reuse activation loads, preserving each row's two FMA
   chains and final reduction. Knob `CASCADIA_BF16_GEMV_ROWS=1|2|4`, default 1
   while experimenting. Includes x86 bit-exact tail/unaligned/large-dimension
   unit coverage. This candidate is NOT compiled, tested, or deployed yet.
   If the current dirty math.rs already includes it, do not apply the patch
   again. In a recovered clean checkout, apply the saved patch once.
5. Upload only that file to the matching path under the remote `repo`, run
   `cmd /c C:\Users\devcloud\inkling-autolab\build.bat` via SSH, then save the
   new benchmark as `bin\bf16-rows.exe`. Preserve `baseline.exe`. Run the new
   tiled tests and Inkling tests on MSVC, then compare rows 1/2/4 using
   `run-bench.ps1 -Variant bf16-rows -Bf16Rows N -Reads 1 ...` with the reference
   hash required. Each candidate build needs its SHA256 recorded.
6. Further independent hypotheses: concurrent Q/K/V/R projection scheduling;
   AVX2 int4 row tiling to interleave independent accumulators without changing
   within-row FMA order; confirm wins in paired independent process runs and
   retune schedule if kernels improve. Quantization/numerical changes need real
   checkpoint quality validation, which this machine cannot currently supply.
7. Run relevant correctness tests, record failures as well as wins, commit and
   push verified changes as t8. Do not claim the proven hardware maximum or a
   whole-model token rate. User's intent remains maximum autonomous progress;
   stop at a measured local plateau only with remaining capacity limits clear.

## Recovery if /tmp was cleared

Create a fresh isolated worktree from `origin/perf/inkling-panther-autolab`.
Recreate the venv/install editable ../autolab. Reconstruct the SQLite database
from committed `results/001_threads.json`, `002_expert_schedule.json`, and
`003_direct_threads.partial.json` by calling
`autolab.metrics.db.ResultsDB(...).store_result(record)` for each record in
those arrays. Do not import `environment.json` as results. Apply the saved
candidate patch only in a clean checkout. The remote baseline/build/bins persist
under the task-owned directory. Resume campaign 003 without repeating completed
experiments.

## Operational details

- This session's `rg` binary hangs. It was tried first; use `git grep`,
  `git ls-files`, or bounded Python file searches instead. Initial hanging
  rg tool sessions were explicitly interrupted for this pause.
- Source archive `/private/tmp/inkling-autolab-source.tar.gz` (baseline 9aaebff0)
  also exists remotely at `C:\Users\devcloud\inkling-autolab-source.tar.gz`.
- The remote launcher sets High priority and mask on its own child, passes
  hashes through `--expect-hash`, and kills only that child if it fails.
- `run_campaign.py` serializes campaigns via flock, exports raw JSON on normal
  completion, and refuses promotion on failures/missing metrics/hash mismatch.
  Interrupted campaigns need a manual raw export (already done for 003).
- Root checkout pointer: `tmp/INKLING_AUTOLAB_HANDOFF.md`, for finding this
  checkpoint after the app restart.
