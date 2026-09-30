# 032: another box plays rank 0; the entry box keeps the door and a middle rank

**Why.** The entry box lost power three times on 2026-09-20 (13:39, ~20:38, ~20:57 CDT), each time at a step change
in load, twice with its clock reset (a board that lost all power, not a crashed kernel). It is the box that draws
the most of the eight 25 W boxes (20.5-22.5 W package against 17-19 W) because rank 0 also runs the API, the
scheduler and the draft model. Three boxes of other hardware (installed as ranks 8-10, no such platform limit) have
never stopped. The user asked to move the rank 0 role to one of them, each box inheriting the other's role and data.

**What moves and what does not.** The PIPELINE role moves: rank, layers 0-5 with the embedding, the API + dashboard
+ scheduler, the draft model. Everything that belongs to the BOX stays: the beacon's installed rank and the names it
keeps (`<fleet>-rank-N`), so every box's updater still finds the release files where they are; the operator
tunnel, the release poller and the file server, whose keys were generated on that box and must not be copied. The
entry box then plays the other box's middle rank and relays its port 8000 to the box that plays rank 0, so
`localhost:18000`, the public path and Tailscale keep working unchanged (`fleet/api_relay.py`; it answers
`/api/fleet/telemetry` itself, because that file is written by the entry box's beacon).

**Two releases.**
1. `032a_role_sync.env` (overrides only, no role changes): the two boxes copy each other's role data over the LAN
   (`fleet/role_sync.py`: shells, experts, attention IRs, the embedding; sha256 per file; 60 MB/s; into the same model
   folder under other layer numbers; nothing deleted, nothing overwritten; refuses if the disk cannot hold the copy
   plus 70 GB for the IRs to regenerate). About 49 GB one way and 36 GB the other: ~15 minutes. When BOTH are
   verified each writes `role-swap/ready-<role>`. Watch `probe_read.py SW`: `ready=1` on both before step 2.
2. `032b_run.sh` + `032b_role_swap.env`: `run.sh` with `ROLE_SWAP="0 8"` (the repository's copy keeps it empty). A box
   of the pair changes role only if it holds its marker; rank 7 learns that rank 8 now lives on the entry box; run.sh
   regenerates the fused-MoE and dense IRs of the new layers before the worker starts (~15 s a layer).

**Tests before any of it touched the fleet.** run.sh's block for every kind of box, with and without the marker;
the sync between two fake boxes (bytes identical, resumable, refuses a taken name, refuses a small disk, the fleet
key not reachable, no listing, no writes); the relay (plain requests, local telemetry, SSE streamed event by event, 64
parallel streams, a clear 502); and both together in two containers named like the boxes on the miner: roles stay
until both markers exist, then flip, and a client of the entry box's :8000 is answered by the other box's API.

**Revert.** Publish the repository's `run.sh` (empty `ROLE_SWAP`) and the previous overrides: every box is back in
its installed role with its own data, which was never touched.

**Not covered.** If the entry box loses power again, the door closes and the pipeline stops (it still holds six
layers), exactly as today; the hope is that a middle rank's load no longer trips it. A second door on the new rank
0 box needs hands at its console (bundle outside the repository: `~/inkling-release/manual/asus-second-door/`).
Moving the release poller and file server as well means re-installing both boxes with swapped ranks from the SSD.

Prediction: rank 0's frame 44 -> ~39 ms on the faster box, the entry box at ~40 ms like ranks 1-7; 15 streams
unchanged or slightly better (rank 10's head still paces the ring); no more stops of the entry box.
Kill: a gate failure, the relay answering 502 for more than a minute after settle, any box not serving.
