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
