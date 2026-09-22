# 032 verdict: the box installed as rank 8 plays rank 0; the entry box plays rank 8 and relays :8000. Kept.

Run on the user's go-ahead, 2026-09-20 21:30-22:07 CDT.

* **Blocked first by the entry box's clock**: every power loss resets it to the firmware date (July). `publish.py` stamps
  the fleet manifest with that clock and updaters refuse a manifest older than the one applied, so no release could
  roll out. Set by hand at the console once (`hwclock` is not installed there; not needed). Both 032 releases now
  carry a guarded self-heal for that box: if its clock is earlier than a release it already applied, take the time
  from an HTTPS Date header through the lab proxy, forward only.
* **032a (21:30, overrides only):** settled 11/11 in 8 min, no role changes. The two boxes hashed and exchanged their
  role data: 1,578 files / 51.7 GB to the entry box, the smaller rank-0 set the other way, ~62 MB/s (the throttle),
  ~14 min, every file sha256-checked, `err=0`; 1,257 GB and 834 GB free. Both wrote their markers
  ("ready: both boxes hold each other's role").
* **032b (21:58, run.sh with `ROLE_SWAP="0 8"` + overrides with the relay):** settled 11/11 in 8 min, steady 3/3.
  The dashboard's topology lists the entry box as `...-r8` on port 9108; `localhost:18000` answers through the relay
  (API, dashboard, and `/api/fleet/telemetry` with 11 ranks reporting); the new rank 0 fetched and verified the
  draft model through the proxy (P15L all zero); both gates pass exactly as before (2 exact, the third departs at
  character 81 as it always has). Gate speeds are lower for now (4.0 / 6.4 / 6.7 tok/s against 7.3 / 11.7 / 11.0):
  the table of memorised prompts stayed on the old box and is re-learned on the new one.

Lesson for the harness: the telemetry relay keeps ONE record per probe tag (the first), so a probe that reports
progress under a constant tag looks frozen (`SW0` showed zeros for 25 minutes while 51 GB moved). The entry box's own
log (signed status, last 30 lines) had the truth. Progress probes need a changing tag.

Open: whether the entry box still loses power as a middle rank (about 2-4 W less package power, but six expert
layers instead of four to reload). Its power brick/outlet has not been swapped yet. No load test was run after the
swap beyond the gates. Revert: publish the repository's `run.sh` and `030_clean_exact.env`.

## Load test after the swap (22:52-23:06 CDT, on the user's request)

| | before the swap | after |
|---|---|---|
| 15 streams x 128 tokens, steady tok/s | 24.8 | 23.6 / 24.6 (rank 10's head still paces the ring) |
| rank 0's stage per frame | 44.2 ms (entry box) | **39.5 ms** (the box without the platform limit) |
| one stream, SAME prompts: explain / story / code / arithmetic | 4.07 / 3.40 / 4.83 / 8.81 | 4.04 / 3.25 / 5.12 / 9.29 |
| one stream, same prompts: rewrite / true-false | 7.08 / 9.38 | 5.64 / 4.34 (their phrases lived in the memorised table, which stayed on the old box and is re-learned) |

The entry box survived all of it (restart counters unchanged, 11/11 serving): ~12 minutes of load including two
15-stream phases, the kind that stopped it at 20:57.

**The entry box is measurably unlike its seven identical siblings, doing the same work now:** 44.1 ms a frame against
38.8-41.3 (fused experts 27.5 ms against 23.9-25.1, iGPU 90 % busy against 75-83 %), 20.0 W of package power
against 17.3-19.4 W, 1.15 busy cores against ~0.96 (part of that is the door: relay, poller, tunnel, file server,
beacon aggregation). So the swap bought it less power margin than predicted (20.0 W, not 17-19), and the difference
is in the unit or its power supply, not in the role. It is now the slowest middle rank but still faster than rank 10
with its head (47.2 ms), so it costs the ring nothing.

`status.sh` (published 23:02, restarts nothing) now says which pipeline role a box plays; the fleet table keeps
listing boxes by INSTALLED rank, which is what the updater, the names and the tunnel depend on.
