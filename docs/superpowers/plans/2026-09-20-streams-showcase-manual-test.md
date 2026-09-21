# Streams showcase — manual test checklist

**Spec:** `docs/superpowers/specs/2026-09-20-streams-showcase-design.md`
**Plan:** `docs/superpowers/plans/2026-09-20-streams-showcase.md`
**Branch:** `7i7o/streams-showcase`

No automated tests ship with this feature (decided in the spec). This is the
acceptance checklist. Part A runs on a laptop against the mock engine and
exercises every state the runner can be in. Part B runs against the Inkling
fleet and checks the things the mock cannot show: pacing, real throughput,
legibility at scale, and an hour of unattended running.

Tick every box. Anything that does not match is a bug in the implementation
or an error in the spec; write it down either way.

---

## Part A — local, mock engine

### A0. Setup

```bash
# terminal 1, repo root — stub build, then serve the mock with a LOW cap so 503s happen
cargo build --release -p cascadia
CASCADIA_API_MAX_CONCURRENT=4 ./target/release/cascadia run mock-model --engine mock

# terminal 2
cd crates/cascadia-dashboard/web && npm ci && npm run dev
```

Open `http://localhost:5173/`. The mock engine echoes the prompt's words back
one per token with no delay, so tiles cycle fast; that is expected.

- [ ] `curl -s localhost:8000/api/stats` shows `"max_concurrent":4`.
- [ ] `/chat` streams an echo of whatever you send (baseline: the API works).

### A1. Routing and theme scoping

- [ ] Nav has three entries: Cluster, Chat, Streams.
- [ ] `/` (Cluster) and `/chat` look exactly as before: light, 1280 px container, footer present. Compare against `main` in a second tab if unsure.
- [ ] `/streams`: near-black page, dark nav, logo visible, Streams pill highlighted in mint, **no footer**, **no page scrollbar** at 1080p.
- [ ] Switching Streams → Cluster → Streams flips the theme cleanly with no flash of the wrong colours and no layout shift in the nav.
- [ ] Narrow the window to ~700 px: no horizontal scrollbar on `/streams`; the toolbar stat cells scroll horizontally inside the toolbar instead.
- [ ] Browser console: no errors or warnings on any of the three routes.

### A2. Initial state

- [ ] 16 tiles in a 4 × 4 grid filling the viewport under the toolbar.
- [ ] Each tile: `#01` … `#16`, grey dot, `idle`, body `press Play`, footer `0 tok · TTFT — · 0.0 s`.
- [ ] Toolbar: model name in the picker, Play button enabled (mint), cells `Active 0 · Queued 0 · Agg tok/s — · Tokens 0 · Replies 0 · Server 0 / 4`.
- [ ] Stop the mock server, reload `/streams`: picker shows the `/v1/models` error, Play is disabled with a `waiting for a model` tooltip, tiles say `waiting for a model`, `Server —`. Restart the server, reload: back to normal.

### A3. Play, run, Stop

- [ ] Play → tiles start one at a time, roughly 300 ms apart (watch the first second).
- [ ] With cap 4: some tiles show amber `queued` with `waiting for a slot… (attempt N)`, N climbing; `Server` reads up to `4 / 4`; `Queued` in the toolbar matches the count of amber tiles.
- [ ] Queued tiles recover on their own and stream; nothing needs clicking.
- [ ] A streaming tile shows `> prompt` in grey, the reply in white, a blinking mint block cursor while streaming, `streaming` in mint.
- [ ] `cooldown` follows each reply for 1–2 s (grey dot), then a new prompt begins.
- [ ] No two tiles show the same prompt at the same time (watch for a minute).
- [ ] After a tile's second prompt, the first exchange sits above it dimmed. After the fourth, only the last three earlier exchanges remain; the body auto-scrolls to the bottom.
- [ ] `Tokens`, `Replies` climb; `Agg tok/s` shows a number; per-tile footers show tokens, a TTFT in ms, and elapsed seconds.
- [ ] Stop → every tile goes `idle` immediately, text and scrollback stay, `Agg tok/s` freezes at its last value, `Server` returns to `0 / 4` within a second, `Active 0 · Queued 0`.
- [ ] Play again → tiles archive their last reply into scrollback and start fresh prompts. `Tokens` and `Replies` reset to 0.

### A4. Per-tile controls

- [ ] Hovering a tile reveals Pause and Skip in its header; moving off hides them.
- [ ] Pause on a **streaming** tile → status reads `pausing` until the reply ends, then `paused`; the tile stops cycling. Hover: the button now reads Resume.
- [ ] Resume → the tile picks a new prompt and runs.
- [ ] Pause on a **queued** or **cooldown** tile → `paused` at once; no request is fired for it afterwards (its `Server` contribution disappears).
- [ ] Skip on a **cooldown** or **streaming** tile → it starts a new prompt immediately, no cooldown; the cut-short reply (if any text) goes into scrollback.
- [ ] Skip is disabled (dimmed) while the runner is stopped and on paused tiles; Pause is available while stopped.
- [ ] Stop, Pause one idle tile, Play → that tile stays `paused` while the other 15 run. Resume it → it joins.

### A5. Leaving the page

- [ ] With streams running, click Chat. `Server` on return (or `curl localhost:8000/api/stats`) shows `requests_in_flight: 0` within a second of leaving.
- [ ] Back on Streams: 16 fresh `idle` tiles, `press Play`, counters at zero.
- [ ] Browser back/forward between the routes behaves the same.

### A6. Fullscreen

- [ ] Fullscreen button → only the toolbar and grid fill the screen (no nav); background stays near-black, not the browser's black backdrop with a gap.
- [ ] Button label changes to Exit fullscreen; Esc exits; label restores.
- [ ] Streams keep running through the transition.

### A7. Settings drawer

- [ ] Gear → 320 px panel on the right with Streams 16, Max tokens 160, Cooldown min 1, Cooldown max 2, Temperature 0, the server-cap line in amber (`Server admits 4 … Streams above the cap will queue.`), and Reset to defaults.
- [ ] Esc closes it; gear toggles it; × closes it.
- [ ] Streams → 20 while running: four tiles appended and started; grid re-flows to 5 × 4; the others never blink.
- [ ] Streams → 6 while running: tiles #07+ disappear; `Server` in-flight drops.
- [ ] Streams → 3: server-cap line turns grey (below the cap).
- [ ] Clamping: 0 → 1; 99 → 64 (grid becomes scrollable once tiles would be under 160 px); Cooldown min 5 → max snaps to 5; Temperature 3 → 1.5; Max tokens 5 → 16.
- [ ] Max tokens → 16 while running: each tile's *next* echo is cut at 16 words; the reply in flight finishes at its old length.
- [ ] Cooldown min/max → 5/5: every tile now pauses ~5 s between prompts.
- [ ] Persistence: set Streams 9, reload → 9 tiles; DevTools → Local Storage → `cascadia.streams.settings.v1` holds the JSON. Reset to defaults → 16 tiles and default JSON.
- [ ] Bad storage: in the console run
  `localStorage.setItem("cascadia.streams.settings.v1", '{"streamCount":500,"cooldownMaxS":-3}')` then reload → 64 tiles, Cooldown max shows 1, no console error. Then `localStorage.setItem("cascadia.streams.settings.v1", "not json")`, reload → defaults, no console error. Reset to defaults afterwards.

### A8. Server failure and recovery

- [ ] Kill the mock server while running: tiles turn red `error` with `server unreachable`; the network tab shows retries at roughly 1, 2, 4, 8, 10, 10 … seconds per tile.
- [ ] Restart the server: tiles recover on their next retry; no click needed; counters resume.
- [ ] Restart the server with `CASCADIA_API_MAX_CONCURRENT=64`: with 16 streams nothing ever queues and `Server` shows `16 / 64` at peak; the drawer line is grey.

### A9. Build gates

```bash
cd crates/cascadia-dashboard/web && npm run typecheck && npm run build
cd ../../.. && cargo build -p cascadia-dashboard --features embed-spa && cargo test -p cascadia-dashboard --features embed-spa
```

- [ ] All four commands succeed.
- [ ] `git status --porcelain` is empty; `git log --oneline feat/inkling-multistream..HEAD` shows the doc commit(s) plus the seven feature commits.

---

## Part B — fleet, Inkling (no release needed)

Nothing in Part B requires shipping a binary. The Vite dev server proxies
`/api`, `/v1` (including SSE) and `/health` to whatever `VITE_API_PROXY`
names, so point it at rank 0's API through the operator tunnel and the whole
screen runs against the live fleet from your laptop. Ship (Part C) only after
Part B passes. Keep `deploy/inkling-fleet/bench.py` handy for cross-checks.

### B0. Run against the live fleet through the tunnel

```bash
# the operator tunnel exposes rank 0's :8000 as localhost:18000
curl -s http://localhost:18000/api/stats          # expect JSON with max_concurrent
cd crates/cascadia-dashboard/web
VITE_API_PROXY=http://localhost:18000 npm run dev
```

- [ ] `http://localhost:5173/streams` shows the fleet's model in the picker (`inkling` or whatever `/v1/models` returns) and `Server 0 / <cap>` with the fleet's real cap (the default is 16 unless `fleet-overrides.env` on rank 0 sets `CASCADIA_API_MAX_CONCURRENT`).
- [ ] Note the cap. `fleet.env` has `STREAMS=16` per rank; the UI default of 16 streams matches. Anything above the cap will queue, which is fine and is part of the demo.
- [ ] Keep the fleet-side Cluster page (`http://localhost:18000/`) open in a second window during B1–B3 to watch `In flight` and `Tokens` move while the streams run.

### B1. Pacing and legibility

- [ ] Default 16 streams, Play: tiles fill in over ~5 s; TTFT per tile is seconds, not milliseconds; replies stream visibly word by word.
- [ ] At 1080p, 16 and 20 tiles are readable from two metres: prompt line, reply text and the footer numbers.
- [ ] 32 tiles: still fills the screen; text small but legible up close. 64: the grid scrolls; document that this is the expected behaviour.
- [ ] Fullscreen on the presentation display: no nav, no browser chrome, near-black edge to edge.

### B2. Throughput plausibility

- [ ] Note `Agg tok/s` after two minutes with 16 streams. Run
  `python3 deploy/inkling-fleet/bench.py http://<rank0>:8000 --streams 16 --tokens 64`
  from a machine on the LAN with the UI **stopped**; its `aggregate=` figure should be in the same ballpark (within ~25 %; the UI's replies are longer and its cooldowns lower duty cycle, so expect the UI slightly under bench).
- [ ] Per-tile tok/s roughly equals bench's `stream_tok_s`.
- [ ] Raise Streams past the server's `CASCADIA_API_MAX_CONCURRENT` (see `fleet.env`): the surplus queue in amber, `Server` pins at `cap / cap`, aggregate does not collapse.
- [ ] Temperature 0.7: replies vary between tiles that happen to share a prompt family; no errors. Back to 0 afterwards.

### B3. Soak

- [ ] Leave 16–20 streams running for 60 minutes.
- [ ] Tile count unchanged; no tile stuck in `prefill`, `queued` or `error` for more than a couple of minutes; `Replies` keeps climbing.
- [ ] Browser task manager: the tab's memory is flat after the first few minutes (scrollback is capped at three exchanges per tile).
- [ ] Console: no errors accumulated.
- [ ] Stop → every tile idles within a second; `curl <rank0>:8000/api/stats` shows `requests_in_flight: 0`.

### B4. Cross-tab moment (for the demo script)

- [ ] With streams running, switching to Cluster kills them (by design — spec decision). Document in the demo script that the multi-stream wall and the Cluster counters are shown one after the other, not side by side, or open Cluster in a second window.

---

## Part C — ship to the fleet (binary release)

Rank 0 serves the SPA from inside the `cascadia` binary; there is no
standalone frontend process and no runtime static-file override. A frontend
change rides a binary release, which restarts all 11 workers and takes 5–8
minutes to settle. Do this only after Parts A and B pass, and only when nobody
is mid-experiment on the fleet.

### C0. Pre-flight: make sure the new tarball drops nothing

> **Done for this ship on 2026-09-20.** The tarball on the build machine
> (`~/inkling-build/dash-dist.tar.gz`, dated Sep 19 21:44, top-level entry
> `dist/`) lists `index-DUJ_Gm7o.js` and `index-DCDzzC-l.css`, which is
> exactly what a clean build of `feat/inkling-multistream`'s `web/` produces
> (Node 22, current lockfile). Every remote branch shares that web tree. The
> `tahoma-dashboard` worktree on the Mac mini is on the stale `feat/dashboard`
> branch (upstream deleted), one commit behind `main`, with nothing
> uncommitted or unpushed under `web/`. Skip to C1 unless the tarball's date
> or names have changed since. The steps stay here for the next SPA release.

The build machine's script untars `~/inkling-build/dash-dist.tar.gz` over
`crates/cascadia-dashboard/web` before building, so the *tarball* is the SPA
that ships, whatever branch the checkout is on. Vite's asset names are
content hashes and are reproducible for identical sources and lockfile, so:

- [ ] On the build machine: `tar -tzf ~/inkling-build/dash-dist.tar.gz | grep assets/` and note the `index-<hash>.js` / `index-<hash>.css` names.
- [ ] On your laptop, on a clean checkout of `feat/inkling-multistream`: `cd crates/cascadia-dashboard/web && npm ci && npm run build && ls dist/assets`. Same names → the deployed SPA has no unpushed changes and it is safe to replace the tarball. Different names → someone built the deployed tarball from a web tree that is not on origin. The operator's notes name a `tahoma-dashboard` worktree on a `feat/dashboard` branch; `tahoma` is the Cascadia workspace on the team's Mac mini, not the miner, so look there. Diff its `web/` against this branch and merge before shipping.
- [ ] Confirm the tarball layout you are about to ship matches the old one's top-level entry (`dist/`): `tar -tzf ~/dash-dist.tar.gz | head -3`.
- [ ] Confirm nobody holds the publisher lock (`~/inkling-release/publisher.lock` on the build machine) or is mid-experiment on the fleet.

### C1. Build

- [ ] Back up the old tarball: `cp ~/inkling-build/dash-dist.tar.gz ~/inkling-build/dash-dist.tar.gz.before-streams`.
- [ ] Copy the new one from Task 7 into place as `~/inkling-build/dash-dist.tar.gz`.
- [ ] Run the build machine's usual script (it does `tar -xzf … -C repo/crates/cascadia-dashboard/web` then `cargo build --release -p cascadia --features openvino,dashboard-embed`). No branch change is needed: this feature has no Rust changes, so the checkout can stay where it is, including on `autolab/inkling-fleet-perf` with its `/api/fleet/telemetry` route.
- [ ] Prove the new SPA is inside the binary before publishing: `strings target/release/cascadia | grep -c 'waiting for a slot'` prints at least 1 (a string only the streams tile contains). If it prints 0 the stale-tarball trap bit you; check which tarball was untarred.

### C2. Publish and verify

- [ ] Publish through the fleet's signed channel the usual way (the operator wrapper, or `publish.py` on rank 0 with the new `cascadia` in the served folder). Every box's updater installs it and restarts its worker; wait 5–8 minutes for the pipeline to re-form.
- [ ] Through the tunnel: `curl -s http://localhost:18000/streams | grep -o 'assets/index-[A-Za-z0-9_-]*\.js'` shows the new hash from C0.
- [ ] `http://localhost:18000/streams` renders the dark wall; run B1 quickly (Play, 16 streams, two minutes).
- [ ] `http://localhost:18000/` and `/chat` still look and work as before.
- [ ] `curl -s http://localhost:18000/api/fleet/telemetry | head -c 200` still answers (the Rust side of the build was not regressed by the branch the machine built from).
- [ ] `curl -s http://localhost:18000/api/stats` shows the pipeline serving again (`requests_in_flight` moves when you Play).
- [ ] Rollback if needed: restore `dash-dist.tar.gz.before-streams`, rebuild, republish.
