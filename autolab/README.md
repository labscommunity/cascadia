# autolab/inkling-fleet-perf

> **Goal re-set by the user on 2026-09-20 (afternoon):** interactive speed for up to 15 concurrent streams, and the
> highest aggregate throughput at that concurrency: ideally 60 tok/s (4 tok/s per stream). The earlier targets
> (> 10 tok/s single stream, > 60 tok/s at any concurrency) are history; see PHYSICS.md "The 15-stream regime".

Autonomous performance research on the live 11-box Inkling fleet (975B MoE, 66 layers,
pipeline-parallel, 6 layers per Intel Panther Lake box: Core Ultra X7 358H, Arc B390 iGPU,
61 GiB LPDDR5x-8533, USB 1 GbE). Long-lived research branch; verified wins are cherry-picked
to `feat/inkling-multistream` (PR #159). This branch does not merge.

## Targets (set by the user, 2026-09-20)

| | start | target |
|---|---|---|
| single stream | 1.6 tok/s | **> 10 tok/s** |
| aggregate, many streams | 7-10 tok/s | **> 60 tok/s** |

on the iGPU path, across the 11 devices behind rank 0. `PHYSICS.md` says what each target
costs in memory traffic; both are beyond what tuning can reach, so the loop spends its time
on structural changes (see `MOONSHOTS.md`).

## The loop

1. **Research**: read the last results, derive the next hypothesis from first principles
   (`JOURNAL.md` records the reasoning, not only the numbers).
2. **Build**: batch changes into one binary, every variant behind an environment switch, so
   one fleet restart serves several experiments (`fleet-overrides.env` is sourced as shell
   with `RANK` set: per-rank A/B inside one run is possible and preferred).
3. **Publish** through the signed release channel (`bench/lab.py publish`), wait until 11
   workers serve one files version with steady restart counts.
4. **Gate**: greedy outputs against `bench/reference.json`. Garbage decodes at full speed
   (the f16 fused path printed `!!!!` at a fine tok/s), so nothing is timed before the gate.
5. **Measure**: timed phases plus every box's telemetry and stage profile
   (`bench/lab.py bench`, `bench/analyze.py`).
6. **Record**: `experiments/NNN_name/` (hypothesis, overrides, phases.json, telemetry,
   verdict), one line in `LEADERBOARD.md`, anything surprising in `DISCOVERIES.md`.

Rules: no phase waits longer than 15 minutes (enforced in `lab.py`); never wait for a person;
a failed rollout is rolled back from `~/inkling-release/baseline/`; composition is measured,
never assumed (features that win alone can lose together); parameter sweeps are calibration,
not research.

## Doors to the fleet (no shell anywhere)

- in: `~/inkling-release/bin/release.py publish` (Ed25519-signed; six file names only:
  `cascadia`, `fleet-overrides.env`, and with `--allow-infra` `run.sh`, `beacon.py`, `status.sh`;
  `updater.py` is never touched from the loop). Every `cascadia`/overrides release restarts all
  workers (about 2 minutes, caches cold).
- out: `http://localhost:18000` = rank 0 `:8000` (API, `/api/stats`, `/api/fleet/telemetry`)
  and `release.py status --json` (signed: fleet table, 30 lines of rank 0's worker log).
- `~/inkling-release/publisher.lock` marks this loop as the only publisher.

Never connect to relay ports 9100-9110. Prompts over 256 tokens crash the chain on binaries
before experiment 001.

## The experiment queue

`QUEUE.md` lists every experiment, finished, running and proposed, with its status; it is generated from one file
per item in `queue/items/`. Anyone (another agent, a teammate) adds an item with `bench/equeue.py add ...` or by
writing a file; the operator of the fleet runs them in priority order. Rules and fields: `queue/README.md`.
