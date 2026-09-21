# Streams showcase screen — design

**Date:** 2026-09-20
**Status:** approved design, ready for an implementation plan
**Scope:** `crates/cascadia-dashboard/web` (the Vite/React SPA). No Rust changes.

## 1. Goal

Add a third screen to the Cascadia dashboard that runs many autonomous
inference streams at once and shows them all live, terminal-style, on one
dark full-bleed page. Each stream picks a random prompt from a built-in pool,
streams the model's reply over the existing `/v1/chat/completions` SSE
endpoint, waits one to two seconds, and picks another. The point is to make
the fleet's multi-stream capability visible: twenty tiles each ticking along
at a couple of tokens per second reads as a busy machine, even when a single
stream alone would look slow.

The existing screens (Cluster at `/`, Chat at `/chat`) are untouched apart
from the nav gaining a third entry and a dark variant.

## 2. Decisions already made

| Topic | Decision |
|---|---|
| Visual treatment | Dark theme scoped to this route only. Full-bleed layout (no 1280 px container, no footer). Fullscreen button. |
| Server capacity (HTTP 503) | Retry with jittered backoff; tile shows a `queued` state. Any stream count is allowed; the UI shows the server's cap. |
| Leaving the page | Streams stop. The runner is owned by the page and aborted on unmount. |
| Automated tests | None. A manual test checklist is delivered instead (see §11). |
| Settings | Inline drawer on the streams page, persisted in `localStorage`. |
| Prompt pool | Built-in TypeScript list of ~150 short prompts, six families. |
| Controls | Global Play / Stop; per-tile Pause / Resume and Skip. |
| Runner architecture | Plain TypeScript class outside React, consumed via `useSyncExternalStore`, snapshots coalesced to one per animation frame. |
| Branch | `7i7o/streams-showcase`, branched from `feat/inkling-multistream`. |
| Docs | This spec and the plan live in `docs/superpowers/` and are committed on the branch. |

## 3. Non-goals

- No global dark mode or theme toggle. Cluster and Chat stay light.
- No new backend routes. Everything the screen needs already exists:
  `/v1/models`, `/v1/chat/completions` (SSE), `/api/stats`.
- No persistence of stream output or history across reloads.
- No thinking/reasoning rendering beyond what Chat does today (the reply
  text is shown as the server streams it).
- No keyboard shortcuts, no presets, no prompt editing UI.

## 4. Route and shell

- **Route:** `/streams`. Registered in `main.tsx` inside the existing
  `Layout` route. The SPA fallback in `spa.rs` already serves the shell for
  any extension-less path, so no Rust change is needed.
- **Nav:** third `NavItem` labelled `Streams`, after `Chat`.
- **Layout on this route.** `Layout` reads `useLocation()`. When the path
  starts with `/streams` it:
  - sets `data-theme="dark"` on its root `div`;
  - swaps the root's `min-h-screen` for `h-screen overflow-hidden`, so the
    grid scrolls inside its container instead of growing the page;
  - gives `<main>` `flex flex-col min-h-0` so the page can fill it
    (`flex-1 min-h-0`);
  - omits the footer.
  Everywhere else the layout renders exactly as today.
- **Page fill.** The page root is `flex-1 flex flex-col min-h-0` under the
  56 px sticky nav, so toolbar plus grid always fill the viewport.
- **Fullscreen.** A toolbar button calls `requestFullscreen()` on the page
  root element (not `documentElement`), so the nav is excluded while
  presenting. The button label follows `document.fullscreenElement` via a
  `fullscreenchange` listener. `Esc` exits (browser default). The page root
  carries its own dark background so it does not show the black `::backdrop`.

## 5. Theming

### 5.1 Mechanism

- `tailwind.config.ts`: `darkMode: ["selector", '[data-theme="dark"]']`.
- **Tailwind v3 caveat:** the selector strategy compiles `dark:` utilities
  to `:is([data-theme="dark"] *)`, i.e. *descendants only*. The element that
  carries the attribute does not match its own `dark:` classes. Therefore the
  Layout root's own colours come from a plain CSS rule, and everything inside
  it uses `dark:` utilities:

  ```css
  [data-theme="dark"] {
    color-scheme: dark;
    background-color: theme(colors.night);
    color: theme(colors.night-ink);
  }
  ```
- `index.html` keeps `<meta name="color-scheme" content="light">`; the rule
  above overrides it for the dark subtree (form controls, scrollbars).
- The stylesheet's "light mode only by intent" comment gets one added
  sentence naming the streams route as the deliberate exception. Nothing
  else in that comment changes.

### 5.2 Tokens (added to `theme.extend.colors`)

| Token | Hex | Use |
|---|---|---|
| `night` | `#0B0F14` | page background |
| `night-2` | `#111821` | tile / drawer surface |
| `night-3` | `#182230` | tile header, hover surface |
| `night-rule` | `#1F2A37` | borders |
| `night-ink` | `#E6EBF0` | primary text on dark |
| `night-dim` | `#9AA7B4` | secondary text |
| `night-low` | `#5C6875` | labels, dimmed scrollback |

Brand accents `mint`, `mint-bright`, `celadon`, `persian` are reused on dark
for status and highlights (`mint-bright` is the terminal accent). The light
`state-*` colours are too dark for the night surfaces, so on dark the tiles
use Tailwind's default palette, which stays available under `extend`:
ok = `mint-bright`, warm = `amber-400`, error = `red-400`.

### 5.3 Components that gain `dark:` variants

- `Nav`: header background/border, active pill, hover colour, "live" label.
- `ModelPicker`: select background, border, text (it is reused in the
  streams toolbar).
- Logo: add `public/logo-dark.svg`, a copy of `logo.svg` with the ink fill
  `#13233A` swapped to `#F9FAF7` (the gradient mark is left as is). `Nav`
  picks the source from the theme. Verify visually; if the gradient mark
  reads badly on dark, a `dark:brightness-110` on the `img` is acceptable.

## 6. Screen layout

### 6.1 Toolbar (single 48 px row, `border-b border-night-rule`)

Left to right:

1. `label-mono` text `Streams` plus a pulse dot when running.
2. `ModelPicker` (dark variant).
3. **Play** / **Stop** button (one button, label and colour flip with state).
   Disabled with a hint when no model is loaded.
4. Aggregate stat cells (mono, tabular): `Active` (streams currently in
   `prefill`/`streaming`), `Queued`, `Agg tok/s`, `Tokens`, `Replies`,
   `Server` (`in_flight / max_concurrent` from `/api/stats`, polled every
   second with the existing `useStats` hook; `—` while unreachable).
5. Right-aligned: gear button (opens settings drawer), fullscreen button.

### 6.2 Grid

- Container: `flex-1 min-h-0 overflow-auto p-3`.
- Columns are computed, not auto-fit, so the last row is not ragged:

  ```ts
  /** Pick a column count that leaves the fewest empty cells, then the
   *  tile aspect closest to 1.6. Candidates are within ±1 of sqrt(n·w/h). */
  export function chooseColumns(n: number, width: number, height: number): number
  ```
  Candidates: `floor(sqrt(n·w/h)) - 1 … ceil(sqrt(n·w/h)) + 1`, clamped to
  `[1, n]`. Score 1: `cols·ceil(n/cols) - n` (empty cells, lower is better).
  Score 2 (tie-break): `|tileAspect - 1.6|` where
  `tileAspect = (w/cols) / (h/rows)`. Width and height come from a
  `ResizeObserver` on the grid container.
- Rows: `grid-template-rows: repeat(rows, minmax(160px, 1fr))`. In a
  definite-height container `1fr` shares the height; when
  `rows × 160 px` exceeds it the container scrolls instead.
- Gap `12px`.

### 6.3 Tile (`StreamTile`)

Terminal look: `bg-night-2 border border-night-rule rounded-md`, mono font,
`text-[12.5px] leading-snug`, `overflow-hidden flex flex-col`.

- **Header** (`bg-night-3`, 28 px): `#07` index, status dot + status word,
  live `tok/s` on the right. Status colours: `idle` night-low, `queued`
  state-warm, `prefill` celadon (pulsing), `streaming` mint-bright,
  `cooldown` night-dim, `paused` night-low, `error` state-error.
- **Body** (`flex-1 min-h-0 overflow-y-auto px-3 py-2`): scrollback of the
  last **3** finished exchanges rendered dimmed (`text-night-low`), then the
  current exchange: `> {prompt}` in `text-night-dim`, then the streamed reply
  in `text-night-ink whitespace-pre-wrap`, then a blinking block cursor
  (`▌`, 1 s step animation) while `prefill`/`streaming`. Auto-scrolls to the
  bottom on every update using `scrollTop = scrollHeight` (same technique as
  `ChatSurface`).
  - `queued`: reply area shows `waiting for a slot… (attempt N)`.
  - `error`: reply area shows the error text in `state-error`.
- **Footer** (24 px, `border-t border-night-rule`, `label-mono` scale):
  `tokens · TTFT ms · tok/s · elapsed s`.
- **Hover controls** (top-right of the header, `opacity-0 group-hover:opacity-100`):
  `Pause` / `Resume` and `Skip`. Disabled while the runner is stopped.
- Tiles are `React.memo`'d and receive an immutable per-stream state object,
  so a token on stream 3 re-renders only tile 3.

### 6.4 Settings drawer (`StreamSettingsDrawer`)

Right-anchored panel, 320 px, `bg-night-2`, slides over the grid, closes on
the gear, an `×`, or `Esc`. Fields (number inputs with the ranges enforced
by `clampSettings`):

| Field | Range | Default | Applies |
|---|---|---|---|
| Streams | 1 – 64 | 16 | live: tiles added/removed immediately |
| Max tokens per reply | 16 – 1024 | 160 | next prompt per stream |
| Cooldown min (s) | 0 – 30 | 1 | next cooldown |
| Cooldown max (s) | min – 30 | 2 | next cooldown |
| Temperature | 0 – 1.5 (step 0.1) | 0 | next prompt per stream |

Below the fields: a read-only line `Server admits N concurrent requests
(CASCADIA_API_MAX_CONCURRENT)` from `/api/stats`, turning `state-warm` when
`Streams > N` with the note `streams above the cap will queue`.

A `Reset to defaults` link restores the table above.

### 6.5 Empty / degraded states

- No model yet: Play disabled, tiles render with `idle` and the body text
  `waiting for a model`.
- `/v1/models` failed: `ModelPicker` already renders the error inline; Play
  stays disabled.
- Runner stopped with previous output: tiles keep their last text and
  scrollback; status `idle`.

## 7. Runner

### 7.1 Files

- `src/lib/streamRunner.ts` — the class, pure TS, no React import.
- `src/lib/streamSettings.ts` — `StreamSettings`, `DEFAULT_SETTINGS`,
  `clampSettings`, `loadSettings`, `saveSettings` (key
  `cascadia.streams.settings.v1`; a parse failure returns defaults).
- `src/lib/prompts.ts` — `PROMPTS: readonly string[]`.
- `src/lib/gridLayout.ts` — `chooseColumns`.
- `src/hooks/useStreamRunner.ts` — creates one runner per page mount
  (`useRef`), subscribes with `useSyncExternalStore`, calls `runner.stop()`
  in the unmount cleanup, returns `{ snapshot, runner }`.

### 7.2 Types

```ts
export type StreamStatus =
  | "idle" | "queued" | "prefill" | "streaming" | "cooldown" | "paused" | "error";

export type Exchange = {
  prompt: string;
  reply: string;
  tokens: number;
  ttftMs: number | null;
  tokPerSec: number | null;
};

export type StreamState = {
  id: number;                 // 0-based, stable while the stream exists
  status: StreamStatus;
  prompt: string | null;      // current prompt
  reply: string;              // current streamed text
  tokens: number;
  ttftMs: number | null;
  elapsedMs: number;
  tokPerSec: number | null;   // decode rate, same math as ChatSurface
  attempt: number;            // 503 retries for the current prompt
  error: string | null;
  pauseRequested: boolean;    // Pause pressed mid-reply; idles after it
  history: Exchange[];        // last 3 finished exchanges, oldest first
};

export type Aggregate = {
  running: boolean;
  startedAt: number | null;   // performance.now() at Play
  activeStreams: number;      // prefill + streaming
  queuedStreams: number;
  tokensPerSecond: number | null; // rolling 10 s window, all streams
  totalTokens: number;        // since Play
  completedReplies: number;   // since Play
};

export type RunnerSnapshot = {
  settings: StreamSettings;
  model: string;
  aggregate: Aggregate;
  streams: StreamState[];
};
```

### 7.3 Public API

```ts
class StreamRunner {
  constructor(settings: StreamSettings, model: string);
  getSnapshot(): RunnerSnapshot;          // stable reference until something changes
  subscribe(cb: () => void): () => void;  // useSyncExternalStore contract
  setModel(id: string): void;             // applies to each stream's next prompt
  applySettings(next: StreamSettings): void;
  start(): void;
  stop(): void;                           // aborts all, keeps last text, statuses -> idle
  pauseStream(id: number): void;
  resumeStream(id: number): void;
  skipStream(id: number): void;
}
```

### 7.4 Lifecycle

- **start():** `running = true`, reset `totalTokens`, `completedReplies`,
  the token ring and `startedAt`. Launch stream `i` after `i × 300 ms`
  (stagger). Streams that were `paused` stay paused.
- **Per-stream loop** (an `async` function per stream holding its own
  `AbortController` and pending timer handle):
  1. If `pauseRequested` → status `paused`, clear the flag, exit the loop.
  2. Pick a prompt (§7.5). Status `prefill`, `attempt = 0`, reset per-reply
     metrics, `reply = ""`.
  3. Call `chatStream({ model, messages: [{ role: "user", content: prompt }],
     max_tokens, temperature }, signal)`.
     - `HttpError` with status **503** → status `queued`, `attempt++`, wait
       `backoff(attempt)`, go to 3 (same prompt).
     - Other `HttpError` or a mid-stream `{object:"error"}` chunk → status
       `error` with the message, wait 3 s, go to 1.
     - Network failure (`TypeError` from `fetch`) → status `error`
       (`server unreachable`), `attempt++`, wait `min(10 s, 1 s × 2^attempt)`,
       go to 3. (`attempt` counts both 503 and network retries for the
       current prompt; it resets when a new prompt is picked.)
     - `AbortError` → exit the loop silently (stop/skip handle state).
  4. On the first chunk: status `streaming`, record TTFT. Per chunk: append
     `delta.content`, `tokens += n_tokens ?? 1`, push `(now, n)` to the
     shared token ring, recompute `elapsedMs` and `tokPerSec`.
  5. On `[DONE]`: push the exchange onto `history` (cap 3),
     `completedReplies++`, status `cooldown`, wait a uniform random
     duration in `[cooldownMinS, cooldownMaxS]` seconds, go to 1.
- **backoff(attempt)** = `min(5000, 500 + 250 × attempt) × U(0.8, 1.2)` ms.
  Unlimited attempts while running.
- **stop():** `running = false`; abort every controller, clear every timer,
  set every non-paused stream to `idle` (keep `prompt`, `reply`, `history`,
  metrics on screen), `pauseRequested = false` everywhere.
- **pauseStream(id):** if `streaming`/`prefill` → `pauseRequested = true`
  (finishes the reply, then `paused`). If `queued`/`cooldown` → cancel the
  pending timer/retry, status `paused` now. If `idle` (runner stopped) →
  status `paused` so it stays out on the next Play.
- **resumeStream(id):** status `idle`; if running, launch its loop.
- **skipStream(id):** abort the current request or timer and relaunch the
  loop immediately (no cooldown). No-op when the runner is stopped.
- **applySettings(next):** store clamped settings. If `streamCount` grew,
  append streams (launched staggered if running). If it shrank, abort and
  drop the highest-index streams. Other fields are read at the moment a
  stream picks its next prompt / cooldown.
- **setModel(id):** stored; used by the next request each stream makes.

### 7.5 Prompt selection

Uniform random over `PROMPTS` excluding (a) prompts currently held by any
other stream and (b) the stream's own last prompt. If the exclusion empties
the pool (more streams than prompts), fall back to excluding only (b).

### 7.6 Metrics

- Per stream, identical to `ChatSurface`: `ttft = firstChunkAt - startedAt`;
  `tokPerSec = (tokens - 1) / (elapsed - ttft)` once `tokens > 1` and the
  decode window exceeds 50 ms, else `null`.
- Aggregate `tokensPerSecond`: a ring of `(timestampMs, nTokens)` events
  shared by all streams. On each push prune entries older than 10 s.
  Value = `sum(n) / min(10, secondsSinceStart)`; `null` before the first
  token. Recomputed on the same frame tick as the snapshot.

### 7.7 Snapshot emission

Any mutation marks the runner dirty and schedules one
`requestAnimationFrame` (if none is pending). The frame handler rebuilds the
snapshot **immutably**: a new `streams` array containing new objects only
for streams that changed since the last frame, a new `aggregate`, then
notifies subscribers. `getSnapshot()` returns the last built object, so
`useSyncExternalStore` sees a stable reference between frames. When the
document is hidden (`document.hidden`), fall back to a 250 ms
`setTimeout` so metrics keep updating in a background tab.

## 8. SSE client change

`src/lib/sse.ts` gains

```ts
export class HttpError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "HttpError";
  }
}
```

and `chatStream` throws `new HttpError(r.status, message)` instead of a
plain `Error`, where `message` is the same `HTTP <status>: <text>` string
it builds today. `ChatSurface` only reads
`.message`, so its behaviour is unchanged. Nothing else in the file moves.

## 9. Prompt pool

`src/lib/prompts.ts` exports `PROMPTS` as a single `readonly string[]` of
about 150 entries, grouped under comment headers by family, roughly 25 each:

1. **Facts** — one-word or one-sentence factual answers.
2. **Explain simply** — "explain X to a ten-year-old in two sentences".
3. **Short creative** — a haiku, a two-line rhyme, a one-sentence story.
4. **Tiny code** — "write a three-line Python function that…", one language
   per prompt, answer must fit in a few lines.
5. **Lists** — "name three…, one phrase each".
6. **Reasoning** — small arithmetic or logic with a brief explanation.

Rules for every prompt: plain English, one request, an explicit length bound
("in two sentences", "in one word", "three items"), no trailing whitespace,
no duplicates, nothing that invites long output or refusals. The 16 prompts
in `deploy/inkling-fleet/bench.py` are included verbatim as seeds. Target
reply length 40 – 150 tokens so tiles keep cycling at low tok/s; the default
`max_tokens = 160` caps outliers.

## 10. Files

**New**

- `web/public/logo-dark.svg`
- `web/src/pages/Streams.tsx`
- `web/src/components/streams/StreamToolbar.tsx`
- `web/src/components/streams/StreamTile.tsx`
- `web/src/components/streams/StreamSettingsDrawer.tsx`
- `web/src/hooks/useStreamRunner.ts`
- `web/src/lib/streamRunner.ts`
- `web/src/lib/streamSettings.ts`
- `web/src/lib/gridLayout.ts`
- `web/src/lib/prompts.ts`

**Modified**

- `web/src/main.tsx` — register `/streams`.
- `web/src/components/Nav.tsx` — third item, dark variants, dark logo.
- `web/src/components/Layout.tsx` — route-aware theme attribute, main flex,
  footer omission.
- `web/src/components/ModelPicker.tsx` — dark variants only.
- `web/src/lib/sse.ts` — `HttpError`.
- `web/tailwind.config.ts` — `darkMode` selector, night tokens.
- `web/src/styles/globals.css` — dark root rule, cursor blink keyframes,
  one-sentence amendment to the light-only comment.
- `README.md` — the "Web dashboard" paragraph lists the streams showcase.

**Not touched:** anything under `crates/*/src` Rust, `Dashboard.tsx`,
`Chat.tsx`, `ChatSurface.tsx`.

## 11. Verification

No automated tests (decided). The implementation plan ships with
`docs/superpowers/plans/2026-09-20-streams-showcase-manual-test.md`, a
checklist covering:

- **Local, mock engine:** `cascadia run mock-model --engine mock` with
  `CASCADIA_API_MAX_CONCURRENT=4` and `npm run dev`; verifies routing, theme
  scoping, Play/Stop, 503 queueing and recovery, per-tile Pause/Skip, live
  stream-count changes, settings persistence, fullscreen, unmount-stops,
  and that Cluster/Chat are visually unchanged. Note the mock engine echoes
  the prompt's words instantly, so pacing is not observable locally.
- **Fleet, Inkling:** pacing, aggregate tok/s plausibility against
  `bench.py`, 20-tile legibility on a 1080p screen, an hour-long soak with
  no growth in tile count or memory.

`npm run typecheck` and `npm run build` must pass; `cargo build --release
-p cascadia --features dashboard-embed` must embed the result.

## 12. Assumptions

- Node 20+ and `npm ci` are available on the implementing machine
  (`node_modules` is not present in this clone yet). Any new dependency must
  satisfy the machine's 14-day package age policy; this design adds none.
- `temperature = 0` is the known-good setting on the fleet. Raising it is
  a user choice in the drawer, not a default.
- The server's `n_tokens` per chunk is honoured for token counts, as Chat
  does today.
