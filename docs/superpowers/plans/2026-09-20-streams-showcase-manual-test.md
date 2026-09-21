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

## Part B — fleet, Inkling

Deploy the branch to rank 0 the usual way (`deploy/inkling-fleet/apply-update.sh`;
the SPA must be built and the binary built with `--features dashboard-embed`).
Open the rank-0 dashboard from the presentation machine. Keep
`deploy/inkling-fleet/bench.py` handy for cross-checking numbers.

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
