# Inkling on Panther Lake with Autolab

This project runs reproducible, sequential experiments against the production
Inkling engine on **tate-07**, `100.82.253.76` (SSH alias
`cascadia-tate-07-ts`, user `devcloud`). The controller runs on macOS/Linux;
the benchmark runs natively on Windows with MSVC. Source baseline: `9aaebff0`
on `feat/inkling`, from `/Users/tatef/Workspaces/tahoma-inkling`.

**Scope:** synthetic resident 975B-sized decoder layer, including attention,
convolutions, normalization, 256-row routing and six selected + two shared int4
experts. Eight distinct 31.85 MB bins are generated deterministically. The other
router entries are suppressed. The active weights exceed the CPU cache. This
measures production code at real dimensions, but omits checkpoint-dependent
routing, the full expert population, disk paging, embeddings, the head, and the
66-layer chain. `layer_tokens_per_s` is **not model tokens/s**. It must never be
reported as full-model throughput or multiplied into a model speed claim.

Inspection on 2026-09-12: Core Ultra X7 358H (16 cores/16 threads), 64 GB RAM,
Windows 11 Pro; ~4 GB disk free, no Inkling checkpoint in the model tree.
The current 512 GB export cannot be deployed with that storage capacity.
Full-model tokens/s remains an explicit blocked research question.

## Setup

Install the sibling Autolab checkout in an isolated Python environment:

```sh
python3 -m venv /tmp/inkling-autolab-venv
/tmp/inkling-autolab-venv/bin/pip install -e /path/to/autolab pytest
```

The current agent supplies the research decisions using `JOURNAL.md` and
`research_plan.yaml`; Autolab supplies campaign execution, SQLite persistence,
and resumption. This works in a Codex session without the Claude stop hook or
another LLM/API credential. `run_campaign.py` is the experiment executor, not a
background LLM agent. Continue the hypothesize → execute → analyze → revise loop
in the session; do not claim unattended agent work continues after it ends.

The remote source/target/binaries all live under the task-owned directory
`C:\Users\devcloud\inkling-autolab`. Upload a source archive, extract into
`repo`, upload `build.bat` and `run-bench.ps1`, then run `build.bat` over SSH.
Save the resulting `target\release\examples\inkling_bench.exe` as
`bin\baseline.exe` before compiling candidates. Preserve each compared binary
under its own name and record its SHA-256.

```sh
/tmp/inkling-autolab-venv/bin/python -u tools/inkling_autolab/run_campaign.py \
  tools/inkling_autolab/campaigns/001_threads.yaml
```

Campaigns use the SSH backend with **explicit `user: devcloud`**, and absolute
Windows command paths. Omit `runner.working_dir`: Autolab currently resolves
relative paths on the local controller, including Windows drive paths.

## Measurement and promotion rules

- Run only one campaign/build at a time on tate-07. The controller takes an
  exclusive lock. Check for other agents' inference/build jobs before starting.
- The launcher sets the benchmark's priority to High and explicitly applies
  its affinity mask, preventing Windows background scheduling from dominating.
  It changes only its own child process and cleans that process up on error.
- Warm up 16 positions, measure 32 positions, repeat five times, use median
  latency. Reset state between repeats. Record every sample, environment,
  dimensions, binary variant, and the full-output hash.
- Scheduling and SIMD changes must preserve the fixed reference output hash.
  Missing/non-finite metrics, nonzero exit, hash disagreement, or an incomplete
  sweep prevent promotion. Run the Inkling fixtures and the changed-kernel tests
  on the actual x86 host as well.
- Confirm winners with interleaved baseline/candidate process runs. Explore
  nearby thread counts/affinity and independent hypotheses. A <3% change within
  run-to-run spread is inconclusive; it is not a new performance record.
- Each finite grid disables Autolab's proximity-based early stopping by using a
  window larger than the grid. Close the campaign only after every configuration.
- Keep `results.db` local; portable raw records are exported to `results/*.json`.
  Keep failed trials as evidence, with a new campaign name for a revised retry.
- Never delete another agent's models/builds/caches, stop their workload, or
  change global machine settings. The full-checkpoint capacity problem is not
  permission to reclaim other projects' storage.

Autolab source used for this session: `3993e2c4`; its pre-existing local change
to `src/autolab/runners/ssh.py` was retained untouched. Campaign/loop/SSH tests:
21 passed before the campaign. No Autolab backend modification was needed.

Results and final launch recommendations are recorded in `JOURNAL.md`.
