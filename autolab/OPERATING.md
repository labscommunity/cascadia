# Operating the Inkling fleet from this autolab: how to continue, how to deploy, what not to do

For any agent or person who takes over. Read this before you publish anything. The fleet has **no shell**: every
change goes through one signed release channel, and the machine that carries that channel has lost power three times.
A careless release cannot be fixed by logging in, because nobody can log in. Most rules below come from an incident;
the stories are in `JOURNAL.md` and in `experiments/NNN_*/verdict.md`.

Lab host names, addresses, the web proxy's name and the relay's address are deliberately absent from this repository.
They live in files on the operator's Mac that this document names.

## 1. The picture in one minute

* **Eleven Ubuntu boxes** (Panther Lake, Arc iGPU, 61 GiB) run Inkling (975B MoE, 66 layers) as a pipeline: six layers
  per rank, experts resident on each box's iGPU, activations passed box to box over 1 GbE. Rank 10 also holds the
  output head. Eight boxes have a 25 W platform limit; the three boxes installed as ranks 8-10 do not.
* **The entry box** is the box *installed* as rank 0. It carries the door: the operator tunnel (an outbound ssh client
  to a relay), the release poller, and the file server every box's updater pulls from (`<fleet>-rank-0:8088`). Their
  keys were generated on that box. None of this can be moved remotely; do not try.
* **A role swap is live (experiment 032).** The box installed as rank 8 *plays* pipeline rank 0 (API, scheduler, draft
  model, layers 0-5); the entry box plays pipeline rank 8 and relays its port 8000 to it. Host names, the updater's
  source, telemetry and `status.sh`'s table all go by **installed** rank; only the worker's role changed.
  `BOX_RANK` = installed rank, `RANK` = role played. Telemetry row `0` is the entry box (pipeline rank 8), row `8` is
  pipeline rank 0. Experiment 036 corrected `analyze.py`: role summaries use the inner profile rank and
  record the installed box separately; repeated/backlogged profile windows are de-duplicated.
* **Two doors, both on the operator's Mac:**
  * in: `autolab/bench/lab.py publish ...` -> `~/inkling-release/bin/release.py` -> poller on the entry box -> every
    box's updater. Only six file names exist: `cascadia`, `fleet-overrides.env`, `run.sh`, `status.sh`, `beacon.py`,
    `updater.py`.
  * out: `http://localhost:18000` (OpenAI API, dashboard, `/api/stats`, `/api/fleet/telemetry`, `/api/topology`) and
    `release.py status [--json]` (signed fleet table + the last 30 log lines of the **entry box's** worker).

## 2. Known-good releases and the current experiment

Latest serving release: **1790016660** (2026-09-21), binary
`~/inkling-release/builds/cascadia-639f0c02-streams`, with the existing
`040_counts_041_attn.env` overrides and 032b run.sh. This adds the Streams
dashboard to the already deployed engine. The owner corrected the entry
clock; all eleven workers settled and both correctness gates passed. The
fleet is available for dashboard testing; the temporary int4 serving canary
has not run. For this dashboard release, `cascadia-639f0c02` is the binary
rollback, with the same overrides and run.sh. See the latest JOURNAL entry.

As of 038 (2026-09-21), the kept binary is `~/inkling-release/builds/cascadia-bf6540ea`,
overrides `~/inkling-release/autolab-overrides/038_phrase_transfer.env`, with the same 032b run.sh below.
035 adds the full-chain readiness gate; 034's ten frames were negative, so keep eleven.
037's capture uses original f32 residuals (INKCAP02); raw states exceed f16 range.
Capture writes are off after the validated 36-prompt collection. 038 merged the old
entry box's phrase table into the new head, preserving both histories and a backup.
Check `~/inkling-release/autolab-state/continuation.json` and signed status for active experiments.
The table below remains the earlier rollback, with readiness and capture disabled.

| | file on the operator's Mac |
|---|---|
| binary | `~/inkling-release/builds/cascadia-8baebd3c` |
| run.sh | `~/inkling-release/autolab-overrides/032b_run.sh` (= `deploy/inkling-fleet/fleet/run.sh` with `ROLE_SWAP="0 8"`) |
| overrides | `~/inkling-release/autolab-overrides/032b_role_swap.env` |
| status.sh | `deploy/inkling-fleet/fleet/status.sh` |

* **Stay on the swap (normal):** keep the 032b run.sh and copy the latest published true overrides for each experiment; the table above is an earlier rollback.
* **Undo the swap:** `deploy/inkling-fleet/fleet/run.sh` (its `ROLE_SWAP` is empty) + `030_clean_exact.env`, with
  `--allow-role-change --allow-no-relay --allow-infra`. Every box returns to its installed role; its own data was
  never touched. This puts the API back on the box that loses power: only do it on the owner's word.

Measured on this configuration: one stream 3.3-4.1 tok/s on prose, 5-9 on structured tasks, first token 2.1 s;
15 streams 24-25 tok/s steady, first token 6-8 s; 176 streams ~65 tok/s. `LEADERBOARD.md` has the history,
`PHYSICS.md` why 60 tok/s at 15 streams is out of reach with exact int4 experts.

## 3. One iteration, step by step

1. **Pick the work**: `python3 autolab/bench/equeue.py next` (rules: `queue/README.md`). Anyone may add items; one
   operator runs them. Set the item `running`.
2. **Take the publisher lock** if it is not yours: read `~/inkling-release/publisher.lock/owner`, make sure its holder
   is finished (ask), write your own tag into it, and export `AUTOLAB_OPERATOR=<your tag>`. Two publishers at once
   have already cost one evening.
3. **Write the hypothesis first** (`experiments/NNN_name/hypothesis.md`): the arithmetic, a predicted number, a kill
   line. One change per release when the effect has to be attributed.
4. **Build** (only if the binary changes):
   `git archive --format=tar.gz -o /tmp/ms-src.tar.gz HEAD && scp /tmp/ms-src.tar.gz miner:inkling-build/ms-src.tar.gz`,
   `ssh miner 'bash -s' < autolab/bench/miner_build.sh`, then copy the binary to
   `~/inkling-release/builds/cascadia-<sha>`. Tests that need disk: `autolab/bench/miner_test.sh` (RAM disk).
5. **Make the overrides** by copying the *last published* file in `~/inkling-release/autolab-overrides/` and adding
   your block. Never start from a copy in `experiments/` (those are redacted) or from an old file (it lacks the relay,
   the clock self-heal, the host-cache cap).
6. **Dry run, then publish:**
   `lab.py publish --dry-run --note ... cascadia=... fleet-overrides.env=...` (guards only), then the same without
   `--dry-run`. `run.sh`, `status.sh`, `beacon.py` need `--allow-infra`.
7. **Wait for `SETTLED` / `steady 3/3`.** If `publish` says `NOT SETTLED`, run `lab.py settle` until it does.
   No request of any kind before that.
8. **Gate**: `lab.py gate` (greedy outputs against `bench/reference.json`: two prompts exact, the third departs at
   character 81 and always has). A gate failure = revert now. For anything that touches numerics also `lab.py quality`.
9. **Measure**: `lab.py bench NNN_name --phases warm1:1:24 fam0x:1:128 mix15a:15:128 mix15b:15:128`. Phase names pick
   the prompts: `famK*` one task family, `mix*` all twelve, `fresh*` unseen, `echo*` a copy task; the prompt texts
   depend on the experiment name, so compare like with like by re-running under the same name. "steady" = sum of the
   streams' own rates; "aggregate" includes admission.
   `analyze.py NNN_name --mode decode` shows where each box's time went.
10. **Write the verdict**, update the queue item (`equeue.py set ... status=done outcome=...`, `equeue.py render`),
    add a `LEADERBOARD.md` line if it is kept, commit, push.

Every `cascadia`, `run.sh` or `fleet-overrides.env` release restarts all eleven workers: 6-9 minutes without service,
and each restart is a 35-50 GB load on every box. `status.sh` restarts nothing. Batch your changes; the entry box has
died within a minute of such a reload.

## 4. Things that brick or break this setup

### Never
* **Never publish `updater.py`.** It is the only thing that can repair every other file. A broken updater is a fleet
  that can never be updated again.
* **Never publish `beacon.py` without the container rig** (`autolab/bench/beacon_rig.sh`, 53 checks on the build
  host). The beacon keeps the host names the pipeline and the updaters resolve, and on the entry box it keeps the
  file server alive.
* **Never touch the tunnel, the release poller, the file server, or any key**: `fleet.key` on the boxes, the signing
  key and operator key under `~/.ssh/inkling-release/` on the Mac, the tunnel keys on the entry box. Do not print,
  copy, move or publish them. Do not serve a box's install folder over HTTP (it holds `fleet.key`; `role_sync.py`
  serves `model/` only, on purpose).
* **Never install an ssh server on a box, never raise a power limit, never connect to a box's pipeline ports**
  (9100-9115). Ports in use: 8000 API (on the entry box: the relay), 8088 release files, 9099/udp beacon,
  9100+rank pipeline, 9115 reply link, 8099 loopback draft model, 9203 role sync while it runs.
* **Never use `--force`** in `lab.py run`: it kept hammering a broken fleet for twelve minutes (021).
* **Never send traffic to a fleet that is not settled.** Requests that arrive while the chain of ranks is assembling
  wedge it (streams admitted, no replies) until the next release (028's first rollout).
* **Never commit raw telemetry, host names, addresses, the proxy's or the relay's name.** `experiments/` is
  git-ignored: add single files with `git add -f`, never a folder (seven telemetry files once reached the public
  branch that way and the branch had to be rewritten). Raw telemetry lives in
  `~/inkling-release/autolab-telemetry/`. Overrides copied into the repository must have the proxy redacted
  (`sed -E 's#PX=http://[^ ;]+#PX=http://LAB-PROXY:PORT#; s#-x "http://[^"]+"#-x "http://LAB-PROXY:PORT"#'`).
* **Never push to `main`**, never add `Co-Authored-By` trailers in this repository, commit as the git-config
  identity. This branch (PR #162) is a laboratory and does not merge; verified wins are cherry-picked elsewhere.
* On the build host never write under `~/data_dir` (a live storage miner; the model export there is read-only for
  us), and its root disk is full: scratch goes to `/dev/shm`. This Mac has ~2 GB free: no test builds here.

### The traps `lab.py publish` now refuses (each one happened or nearly did)
| refused | what it would have done |
|---|---|
| a `run.sh` whose `ROLE_SWAP` differs from what the fleet runs (`~/inkling-release/autolab-state/role_swap`), e.g. the repository's own `run.sh` | silently moves rank 0 back onto the box that loses power, and repoints rank 7. Build release files with `autolab/bench/build_role_swap.py BASE.env OUT_DIR`; `--allow-role-change` when you mean it |
| overrides containing `LAB-PROXY:PORT` | a redacted repository copy: the draft model cannot be fetched and the entry box's clock self-heal breaks |
| overrides without the `api_relay.py` block while a swap is live | the entry box's :8000 goes dark: `localhost:18000`, the dashboard and the gates die (the release channel survives; republish to fix). `--allow-no-relay` when you mean it |
| a `run.sh` / overrides / `status.sh` that fails `bash -n` | workers that never start, on eleven boxes |
| a lock that is not yours | two publishers |

`lab.py publish` returns 0 only when the fleet settled, and prints `SETTLED` or `NOT SETTLED`; do not pipe its
output through `tail` and assume (that is how traffic reached a half-built chain).

### Writing overrides and `run.sh` (they run as root on every box at every worker start)
* `run.sh` runs under `set -u` and sources the overrides with `set -a`: an unset variable or a syntax error stops the
  worker on every box. Use `${VAR:-}`; put anything that may fail or take time in `( set +e +u; ... ) &`.
* Do not use `set --` at the top level of the overrides (it clobbers `run.sh`'s own arguments).
* Blocks keyed on `$RANK` follow the **role**; key on `${BOX_RANK:-$RANK}` for what belongs to a box (the relay, the
  clock self-heal, anything about the tunnel's machine).
* Work that needs RAM (IR generation needs ~10 GB) must finish **before** the worker starts; a serving box has 5-8 GB
  free and the OOM killer takes the worker (025).
* Keep these lines, they each closed an incident: `CASCADIA_INKLING_EXPERT_CACHE_MIB=1024` (a rank whose device calls
  fail must get slow, not OOM-killed), `vm.swappiness=1`, `CASCADIA_GPU_PAGES_GIB=52` (54 on the last rank),
  `CASCADIA_STREAMS_PREFILL_WINDOW=8`, the `cdc_ncm` timer block, `CASCADIA_FUSE_UP_SHIFT="8:4"`,
  `CASCADIA_STREAMS_HEAD_BATCH=1`, and leave `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD` alone (026, 029).
* New behaviour in the binary goes behind an environment variable that defaults to off, so the same binary can be
  switched by an overrides-only release and rolled back by one.

### Canaries (anything that can kill a worker or touch a system library)
One rank, never rank 0's role, never the last rank. The block must revert itself: a marker file, a check of the
unit's journal for signal deaths since the marker, and a start counter. **One release's restart cascade burns about
five starts**, so set the limit at 12, not 8. Enforce the load check (`CASCADIA_INKLING_OV_MOE_CHECK=enforce`).
Patching a library: verify the original's sha256, keep `<lib>.orig`, write atomically, pin the result's sha256, and
restore it in the release that ends the canary (022, 026, 029 are templates).

### Seeing inside a box without a shell
* **Integers**: print a line containing ` stage profile ` with `key=int` pairs from inside the worker's unit; the beacon
  relays it; `autolab/bench/probe_read.py PREFIX`. The relay keeps **one record per tag (the first)**: a progress probe
  under a constant tag looks frozen for ever (it cost 25 minutes of doubt during 032). Change the tag when the values
  change, or read the log instead.
* **Text**: only the **entry box's** worker journal reaches the Mac (`release.py status --json`, 30 lines). Another
  box serves a small file for a few minutes on a spare port, the entry box curls it and pages it into its own journal
  ten lines at a time with a prefix; `autolab/bench/collect_pages.py PREFIX OUT` collects the pages. Key that block on
  `BOX_RANK`, not `RANK`.
* Rank 10's clock is a day ahead. Clocks across the fleet disagree; trust "rt" (receive time) and your own Mac's
  clock. The 036 analyzer now translates profile windows by receive age and excludes duplicate records.

## 5. The entry box: what happens when it dies, and what to do

It lost all power three times on 2026-09-20 (its clock resets to the firmware date each time, and the kernel log of
the previous boot simply ends). After the swap it does a middle rank's work and is still 8-10 % slower and 1-2 W
hungrier than its seven identical siblings: the unit or its power supply is suspect. Its brick and outlet had not
been swapped when this was written.

* **Symptoms**: `localhost:18000` stops answering (HTTP 000) while the Mac's tunnel process is alive;
  `release.py status` says `STALE` with a growing age. Everything stops: the pipeline needs all eleven boxes, and the
  door is on that box. Only a person on site can power-cycle it. Stop all traffic and tell the owner; do not retry
  in a loop.
  A future-dated report can have a negative apparent age while already stale (seen again during 040).
  Check whether its signed timestamp advances across observations; a negative age is not proof of freshness.
  The harness now stops after an incomplete/failed phase or warmup and after a serial gate transport error;
  it still waits for active requests to hit their timeout unless the operator stops the process.
* **After it is back**: the workers re-assemble by themselves. Check `release.py status`: if the age is absurd
  (weeks) or negative, the box's clock was reset. `publish.py` stamps the fleet's manifest with that clock and every
  updater refuses a manifest older than the last one it applied, so **no release rolls out until the clock is right**.
  The published overrides heal this at worker start (time from an HTTPS `Date` header through the proxy, forward only,
  only when the clock is earlier than a release that box already applied). If that failed, someone types
  `sudo date -u -s "YYYY-MM-DD HH:MM:SS"` at its console (there is no `hwclock` on the boxes, and none is needed).
* **Do not** try to move the tunnel, the poller or the file server to another box through the release channel: the
  updaters find their source by the installed-rank name, and a half-moved door is a closed door. The hands-on route
  (re-install both boxes with swapped ranks from the SSD kit, then the tunnel setup on the new box) and a smaller
  "second door" for the dashboard only are described in `~/inkling-release/manual/asus-second-door/README.txt` on the
  operator's Mac. Both need a person at the consoles.

## 6. Where things are

| what | where |
|---|---|
| this lab | `autolab/`: `README.md`, `PHYSICS.md` (what the hardware allows, with the measured percentages), `MOONSHOTS.md`, `JOURNAL.md`, `LEADERBOARD.md`, `QUEUE.md` (generated), `queue/`, `experiments/` (git-ignored, force-add single files) |
| harness | `autolab/bench/`: `lab.py` (status, publish, settle, gate, quality, bench, run), `analyze.py`, `probe_read.py`, `collect_pages.py`, `equeue.py`, `build_role_swap.py`, `miner_build.sh`, `miner_test.sh`, `collect.py` + `drafter_study.py` (offline drafter studies), `nettest.py`, `patch_gpu_plugin*.py` |
| what the boxes run | `deploy/inkling-fleet/fleet/`: `run.sh` (carries the IR generator; `ROLE_SWAP` empty here), `status.sh`, `role_sync.py`, `api_relay.py`, `beacon.py` and `updater.py` (read, do not publish) |
| engine | `crates/cascadia-engine-sparse-moe/src/`: `engine.rs` (scheduler, streams, speculation), `inkling/` (loader, fused-MoE / attention / dense / head device backends), `lm_draft.rs`, `ngram_draft.rs` |
| operator's Mac, outside the repository | `~/inkling-release/`: `bin/release.py`, `builds/`, `autolab-overrides/` (the REAL overrides, with the proxy's name), `autolab-state/role_swap`, `autolab-telemetry/`, `autolab-notes/`, `manual/`, `publisher.lock/`, `baseline/` |
| build host (`ssh miner`) | `~/inkling-build` (release builds; do not delete its `target`), `/dev/shm` (scratch, RAM), the read-only export and checkpoint incl. the shipped MTP head |
| offline studies | `autolab/research/mtp_offline/` (scripts + result tables of the MTP-head and logit-lens study, 033), `crates/cascadia-engine-sparse-moe/examples/inkling_spec_dump.rs` (hidden-state dump) |
| where the model runs for studies | **The fleet.** Text, acceptance rates, expert usage and hidden states are measured on the fleet's own numerics (f16 fused experts, int8 attention): its text departs from any other path's after 7-70 tokens, and the heads and drafters will see ITS states. Hidden states come from the fleet state capture (queue item); scoring and training run on the build host. |
| the Mac Pro (`ssh pro`, whole int4 export resident on the CPU path, ~0.7 s per token) | Only for what the fleet cannot do: (1) a bit-exact CPU REFERENCE of the whole model in one process, for parity work and for looking at ANY tensor of any layer (the fleet only exposes what crosses a rank boundary unless every rank is instrumented and redeployed); (2) work that must go on while the fleet is down or must not be restarted. `~/inkling-spec-offline/` there holds 033's dump. Do not plan new measurements on it otherwise. |

## 7. Before every publish

1. The lock is mine, and nobody else is benchmarking.
2. I started from the last published overrides; my block uses `${VAR:-}` and backgrounds anything slow.
3. `bash -n` passes; new shell logic ran in a sandbox (the build host has Docker: two containers named like two boxes
   tested the role swap end to end).
4. The binary's new behaviour is off by default and its tests passed (`miner_test.sh`).
5. I know the revert: which three files, already on disk.
6. `lab.py publish --dry-run` passed.
7. After publishing: `SETTLED`, then `lab.py gate`, then load. If the gate fails or a box is not serving: revert first,
   investigate afterwards.
