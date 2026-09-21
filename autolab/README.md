# autolab/inkling-fleet-perf

> **Taking over, or about to publish anything to the fleet? Read [`OPERATING.md`](OPERATING.md) first.** The fleet has
> no shell; the release channel is the only door, and the document lists the releases that would close it.

> **Goal re-set by the user on 2026-09-20 (afternoon):** interactive speed for up to 15 concurrent streams, and the
> highest aggregate throughput at that concurrency: ideally 60 tok/s (4 tok/s per stream). The earlier targets
> (> 10 tok/s single stream, > 60 tok/s at any concurrency) are history; see PHYSICS.md "The 15-stream regime".

Autonomous performance research on the live 11-box Inkling fleet (975B MoE, 66 layers,
pipeline-parallel, 6 layers per Intel Panther Lake box: Core Ultra X7 358H, Arc B390 iGPU,
61 GiB LPDDR5x-8533, USB 1 GbE). Long-lived research branch; verified wins are cherry-picked
to `feat/inkling-multistream` (PR #159). This branch does not merge.

## Goals

**Current goal (set by the user on 2026-09-20, afternoon): interactive speed for up to 15 concurrent streams, and the
highest aggregate throughput at that concurrency, ideally 60 tok/s (4 tok/s per stream).** Test deployment, not
production. Earlier targets and where they ended:

| target | start | reached | note |
|---|---|---|---|
| aggregate > 60 tok/s at any concurrency | 7-10 | **64-70 tok/s at 176 streams** (011) | met in steady decode |
| single stream > 10 tok/s | 1.6 | 3.3-4.1 prose, 5-9 structured, 10-12 memorised | exact 10 on open prose is out of reach on this fleet (PHYSICS.md) |
| **15 streams: 60 tok/s, interactive** | 22.5-23.4, first token 31 s | **24-25 tok/s, first token 6-8 s (2 s alone)** | 60 is beyond the byte ceiling with exact int4 experts; ~32 is the ideal of this layout, 35-40 with speculation or lossy options |

## Status (2026-09-20, 23:30 CDT)

* **Running:** binary `cascadia-8baebd3c`, the role-swap `run.sh` and overrides of experiment 032 (exact outputs: both
  gates pass). Every MoE layer, the attention projections, rank 0's dense layers (as fused experts, 027) and the head
  run on the iGPUs; prompts travel as 8-row windows (024); a 0.6B draft model plus n-gram tables guess for a lone stream.
* **Role swap live (032):** the entry box (installed rank 0) lost all power three times on 2026-09-20. The box installed
  as rank 8 (no 25 W platform limit) now plays pipeline rank 0; the entry box plays rank 8, keeps the tunnel, the
  release poller and the file server, and relays :8000. It survived the load test afterwards, but it is 8-10 % slower
  and hungrier than its seven identical siblings at the same work: the unit or its power supply is suspect
  (brick/outlet not swapped yet). **Read `OPERATING.md` before publishing anything.**
* **What the evening's experiments said (026-029):** a frame of r rows costs a stage `16 + 19 r` ms and nothing on the
  device-call side moved those constants: the plugin's MoE decode kernels work (even on the exact group-32 weights,
  with a six-byte patch) and are no faster at one row; sharing head calls loses 2.5 %; a sleeping GPU wait costs
  2.3 ms a frame; rank 0's dense layers as fused experts save 8 ms on rank 0 and 0.5 % for the fleet, because the last
  rank (layers + 11.6 ms of head per frame) paces the ring.
* **Best remaining exact lever:** the model's own MTP head as the drafter (033, offline: right 0.73 of the time,
  0.63-0.69 on prose): one stream ~5-6 tok/s, ~2x per stream at 3-8 streams, about +7 % at 15. A multi-day build
  (queue). Lossy options (int4 attention, fewer experts) need the owner's decision.
* **Queue:** `QUEUE.md`. Anyone may add items (`queue/README.md`); one operator runs them.

## The loop

1. **Research**: read the last results, derive the next hypothesis from first principles
   (`JOURNAL.md` records the reasoning, not only the numbers).
2. **Build**: batch changes into one binary, every variant behind an environment switch, so
   one fleet restart serves several experiments (`fleet-overrides.env` is sourced as shell
   with `RANK` set: per-rank A/B inside one run is possible and preferred).
3. **Publish** through the signed release channel (`bench/lab.py publish`, after `--dry-run`), wait until 11
   workers serve one files version with steady restart counts (`SETTLED`). Nothing is sent before that.
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

Details, pitfalls and the releases that would break the setup: **`OPERATING.md`**.

- in: `bench/lab.py publish` -> `~/inkling-release/bin/release.py` (Ed25519-signed; six file names only: `cascadia`,
  `fleet-overrides.env`, and with `--allow-infra` `run.sh`, `beacon.py`, `status.sh`; `updater.py` is never touched).
  Every `cascadia` / `run.sh` / overrides release restarts all workers: 6-9 minutes. `lab.py publish` refuses the
  releases known to break the fleet and ends with `SETTLED` or `NOT SETTLED`: no traffic before `steady 3/3`.
- out: `http://localhost:18000` = the entry box's :8000, which relays to the box that plays rank 0 (API, dashboard,
  `/api/stats`, `/api/fleet/telemetry`), and `release.py status --json` (signed: fleet table by INSTALLED rank, 30
  lines of the entry box's worker log).
- `~/inkling-release/publisher.lock` marks the one publisher (`AUTOLAB_OPERATOR`).

Never connect to the pipeline ports 9100-9115.

## The experiment queue

`QUEUE.md` lists every experiment, finished, running and proposed, with its status; it is generated from one file
per item in `queue/items/`. Anyone (another agent, a teammate) adds an item with `bench/equeue.py add ...` or by
writing a file; the operator of the fleet runs them in priority order. Rules and fields: `queue/README.md`.
