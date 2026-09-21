# Streams Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/streams` screen to the Cascadia dashboard SPA that runs many autonomous inference streams at once and shows them live, terminal-style, on a dark full-bleed page.

**Architecture:** A plain-TypeScript `StreamRunner` class owns N async stream loops, their abort controllers and timers, and publishes one immutable snapshot per animation frame that the page reads through `useSyncExternalStore`. The page is owned by the route, so leaving it stops every stream. A dark theme is scoped to this route by a `data-theme="dark"` attribute on the layout root that Tailwind's `dark:` variant keys on.

**Tech Stack:** Vite 5, React 19, react-router-dom 7, Tailwind 3.4, TypeScript 5.7 (strict, `noUnusedLocals`). No new dependencies. No Rust changes.

**Spec:** `docs/superpowers/specs/2026-09-20-streams-showcase-design.md` — read it first; every task below cites the section it implements.

## Global Constraints

- All work happens in `crates/cascadia-dashboard/web/`. Paths below are relative to that directory unless they start with `crates/`, `docs/` or `README.md`.
- Branch: `7i7o/streams-showcase` (already exists, branched from `feat/inkling-multistream`). Do not push; the user pushes.
- Commit messages: `type(dashboard): lowercase summary` in the repo's style (see `git log`). No `Co-Authored-By`, no `Claude-Session`, no AI attribution of any kind, in commits, code, comments or docs.
- No new npm dependencies. `npm ci` only installs what `package-lock.json` already pins.
- No automated tests (decided in the spec). Every task ends with `npm run typecheck` passing and, where the task has visible behaviour, a manual browser check listed in that task.
- `tsconfig.json` has `noUnusedLocals` and `noUnusedParameters` on: an unused import or variable fails typecheck. Import only what you use.
- Light mode stays the default everywhere except `/streams`. Do not touch `Dashboard.tsx`, `Chat.tsx`, `ChatSurface.tsx`, or anything under `crates/*/src`.
- Tailwind v3 `dark:` caveat (spec §5.1): the element carrying `data-theme="dark"` does not match its own `dark:` classes; only descendants do. The layout root's own colours come from a CSS rule.
- Design tokens (spec §5.2): `night #0B0F14`, `night-2 #111821`, `night-3 #182230`, `night-rule #1F2A37`, `night-ink #E6EBF0`, `night-dim #9AA7B4`, `night-low #5C6875`. Dark status colours: ok `mint-bright`, warm `amber-400`, error `red-400` (Tailwind defaults, still available under `extend`).
- Settings defaults and ranges (spec §6.4): streams 16 (1–64), max tokens 160 (16–1024), cooldown 1–2 s (each 0–30, max ≥ min), temperature 0 (0–1.5). localStorage key `cascadia.streams.settings.v1`.
- Runner constants (spec §7.4): stagger 300 ms, history cap 3, aggregate window 10 s, error hold 3 s, capacity backoff `min(5000, 500 + 250·attempt) × U(0.8, 1.2)` ms, network backoff `min(10000, 1000·2^attempt)` ms.

---

## File structure

| File | Responsibility |
|---|---|
| `tailwind.config.ts` | `darkMode` selector strategy + night tokens |
| `src/styles/globals.css` | `[data-theme="dark"]` root rule, cursor blink, comment amendment |
| `public/logo-dark.svg` | logo with the ink wordmark swapped to paper |
| `src/components/Layout.tsx` | route-aware shell: theme attribute, viewport pin, footer omission |
| `src/components/Nav.tsx` | third nav item, dark variants, dark logo |
| `src/components/ModelPicker.tsx` | dark variants only |
| `src/lib/sse.ts` | `HttpError` so callers can read the status |
| `src/lib/streamSettings.ts` | settings type, defaults, clamp, load/save |
| `src/lib/gridLayout.ts` | `chooseColumns` |
| `src/lib/prompts.ts` | the prompt pool |
| `src/lib/streamRunner.ts` | the runner: loops, backoff, metrics, snapshots |
| `src/hooks/useStreamRunner.ts` | one runner per page mount, `useSyncExternalStore`, stop on unmount |
| `src/components/streams/StreamTile.tsx` | one terminal tile |
| `src/components/streams/StreamToolbar.tsx` | model picker, Play/Stop, aggregate cells, gear, fullscreen |
| `src/components/streams/StreamSettingsDrawer.tsx` | settings panel |
| `src/pages/Streams.tsx` | page: grid sizing, fullscreen, drawer wiring |
| `src/main.tsx` | route registration |
| `README.md` | one clause in the "Web dashboard" paragraph |

---

### Task 0: Environment and baseline

**Files:** none modified.

- [ ] **Step 1: Install and confirm the untouched SPA builds**

```bash
cd crates/cascadia-dashboard/web
npm ci
npm run typecheck
npm run build
```
Expected: `npm ci` completes (this clone has no `node_modules` yet), typecheck prints nothing, build ends with `✓ built in …`. If `npm ci` fails on the machine's 14-day package age policy, stop and report the blocked package and version; do not change the policy.

- [ ] **Step 2: Build the stub binary and start a mock-engine server with a low cap**

From the repo root, in its own terminal (leave it running for the whole plan):
```bash
cargo build --release -p cascadia
CASCADIA_API_MAX_CONCURRENT=4 ./target/release/cascadia run mock-model --engine mock
```
Expected: the log advertises the API and dashboard on `:8000`. `curl -s localhost:8000/api/stats` returns JSON with `"max_concurrent":4`. The mock engine echoes the prompt's words back one per token, instantly; that is enough to exercise every runner state, just not pacing.

- [ ] **Step 3: Start the Vite dev server**

In another terminal, leave running:
```bash
cd crates/cascadia-dashboard/web && npm run dev
```
Expected: `http://localhost:5173/` shows the light Cluster page and `/chat` works against the mock (send "hello world" → the reply echoes it).

- [ ] **Step 4: Confirm the branch**

```bash
git branch --show-current   # 7i7o/streams-showcase
git status --porcelain      # empty
```

---

### Task 1: Dark shell, nav entry, route stub

Implements spec §4 and §5. After this task `/streams` renders a dark, full-bleed, footer-less page (with a placeholder body), and the other two routes are pixel-identical to before.

**Files:**
- Modify: `tailwind.config.ts:8-9` (add `darkMode`) and `:34` (append tokens)
- Modify: `src/styles/globals.css:13` (comment), `:16-29` (`@layer base`), `:47` (`@layer components`)
- Create: `public/logo-dark.svg`
- Modify: `src/components/Layout.tsx` (whole file)
- Modify: `src/components/Nav.tsx` (whole file)
- Create: `src/pages/Streams.tsx` (stub, replaced in Task 5)
- Modify: `src/main.tsx:13-14, 24-25`

**Interfaces:**
- Produces: `Nav` takes a `dark: boolean` prop. Tailwind tokens `night`, `night-2`, `night-3`, `night-rule`, `night-ink`, `night-dim`, `night-low`. CSS class `cursor-blink`.

- [ ] **Step 1: Tailwind dark-mode selector and tokens**

In `tailwind.config.ts`, insert directly after `export default {` (line 8):
```ts
  // The streams showcase (/streams) is the one dark route. `dark:` utilities
  // key on this attribute, which Layout sets on its root for that route only.
  // v3 selector strategy = descendants only; the root styles itself in
  // globals.css.
  darkMode: ["selector", '[data-theme="dark"]'],
```
Insert directly after the `"state-error": "#B42318",` line (line 34, inside `colors`):
```ts
        // Night palette for the streams showcase (route-scoped dark theme).
        // Brand accents (mint, mint-bright, celadon, persian) are reused on
        // top of these; the light state-* colours lack contrast on them, so
        // dark status text uses Tailwind's amber-400 / red-400 instead.
        night: "#0B0F14",
        "night-2": "#111821",
        "night-3": "#182230",
        "night-rule": "#1F2A37",
        "night-ink": "#E6EBF0",
        "night-dim": "#9AA7B4",
        "night-low": "#5C6875",
```

- [ ] **Step 2: Global CSS — dark root rule, cursor blink, comment amendment**

In `src/styles/globals.css`, change line 13 from
```
 * Note we deliberately do NOT match exo's dark "command center" theme.
```
to
```
 * Note we deliberately do NOT match exo's dark "command center" theme.
 * The one exception is the streams showcase at /streams, which scopes a
 * dark theme to its own route via `[data-theme="dark"]` below; every other
 * screen stays light.
```
Inside `@layer base { … }`, directly after the closing `}` of the `:root { … }` block (after line 28), add:
```css

  /* Streams showcase root. Tailwind's `dark:` variant matches descendants of
   * this attribute (see tailwind.config.ts), so the element carrying it
   * styles itself here. `color-scheme: dark` flips form controls and
   * scrollbars for the subtree despite the page-level <meta> saying light. */
  [data-theme="dark"] {
    color-scheme: dark;
    background-color: theme(colors.night);
    color: theme(colors.night-ink);
  }
```
Inside `@layer components { … }`, directly before the `@keyframes pulse {` block (line 79), add:
```css
  /* Terminal caret for a tile that is waiting on or receiving tokens. */
  .cursor-blink {
    animation: cursor-blink 1s steps(2, start) infinite;
  }

  @keyframes cursor-blink {
    to {
      visibility: hidden;
    }
  }

```

- [ ] **Step 3: Dark logo**

```bash
cd crates/cascadia-dashboard/web
sed 's/#13233A/#F9FAF7/g' public/logo.svg > public/logo-dark.svg
grep -c '#F9FAF7' public/logo-dark.svg   # expect 1
```

- [ ] **Step 4: Layout — route-aware shell**

Replace `src/components/Layout.tsx` with:
```tsx
import { Outlet, useLocation } from "react-router-dom";

import { Nav } from "./Nav";

export function Layout() {
  const { pathname } = useLocation();
  // The streams showcase is the one dark, full-bleed route. It sets the
  // attribute Tailwind's `dark:` variant keys on, pins the shell to the
  // viewport so the tile grid scrolls inside its own box instead of growing
  // the page, and drops the footer. Every other route renders as before.
  const showcase = pathname.startsWith("/streams");

  return (
    <div
      data-theme={showcase ? "dark" : undefined}
      className={showcase ? "h-screen overflow-hidden flex flex-col" : "min-h-screen flex flex-col"}
    >
      <Nav dark={showcase} />
      <main className={showcase ? "flex-1 min-h-0 flex flex-col" : "flex-1"}>
        <Outlet />
      </main>
      {showcase ? null : (
        <footer className="border-t border-rule-2 mt-16">
          <div className="max-w-container mx-auto px-6 py-6 flex items-center justify-between label-mono">
            <span>cascadia · cluster</span>
            <span className="text-ink-low/70">distributed LLM inference for Intel hardware</span>
          </div>
        </footer>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Nav — third item, dark variants, dark logo**

Replace `src/components/Nav.tsx` with:
```tsx
import { NavLink } from "react-router-dom";

export function Nav({ dark }: { dark: boolean }) {
  return (
    <header className="sticky top-0 z-10 bg-white/85 backdrop-blur border-b border-rule dark:bg-night/85 dark:border-night-rule">
      <div className="max-w-container mx-auto px-6 h-14 flex items-center gap-8">
        <NavLink
          to="/"
          aria-label="Cascadia"
          className="shrink-0 hover:opacity-80 transition-opacity"
        >
          <img
            src={dark ? "/logo-dark.svg" : "/logo.svg"}
            alt="Cascadia"
            width={140}
            height={30}
            className="h-[26px] w-auto"
          />
        </NavLink>
        <nav className="flex items-center gap-1">
          <NavItem to="/" end label="Cluster" />
          <NavItem to="/chat" label="Chat" />
          <NavItem to="/streams" label="Streams" />
        </nav>
        <div className="ml-auto flex items-center gap-2 label-mono dark:text-night-dim">
          <span className="pulse-dot" aria-hidden />
          <span>live</span>
        </div>
      </div>
    </header>
  );
}

function NavItem({ to, label, end }: { to: string; label: string; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `px-3.5 py-1.5 rounded-full label-mono transition-colors ${
          isActive
            ? "text-pine bg-mint-wash dark:text-mint-bright dark:bg-night-3"
            : "hover:text-persian dark:hover:text-mint"
        }`
      }
    >
      {label}
    </NavLink>
  );
}
```

- [ ] **Step 6: Route stub**

Create `src/pages/Streams.tsx`:
```tsx
// Placeholder so the dark shell can be verified before the grid exists.
// Replaced wholesale in Task 5.
export function Streams() {
  return (
    <div className="flex-1 min-h-0 flex flex-col bg-night text-night-ink">
      <div className="flex h-12 shrink-0 items-center border-b border-night-rule px-4 label-mono text-night-dim">
        Streams
      </div>
      <div className="flex-1 min-h-0 p-3 font-mono text-[12.5px] text-night-low">
        showcase grid lands in Task 5
      </div>
    </div>
  );
}
```
In `src/main.tsx`, add the import after line 14 (`import { Dashboard } …`):
```tsx
import { Streams } from "@/pages/Streams";
```
and the route after line 25 (`<Route path="/chat" …`):
```tsx
          <Route path="/streams" element={<Streams />} />
```

- [ ] **Step 7: Verify**

```bash
npm run typecheck
```
Expected: no output. In the browser at `http://localhost:5173/streams`: near-black page, dark nav with a visible logo, "Streams" pill highlighted in mint, no footer, no page scrollbar. Click "Cluster": light page, footer present, exactly as before. Click "Chat": same. Back to "Streams": dark again. Resize the window narrow: still no horizontal scrollbar.

- [ ] **Step 8: Commit**

```bash
git add tailwind.config.ts src/styles/globals.css public/logo-dark.svg src/components/Layout.tsx src/components/Nav.tsx src/pages/Streams.tsx src/main.tsx
git commit -m "feat(dashboard): dark full-bleed shell and nav entry for the streams route"
```

---

### Task 2: Typed HTTP error in the SSE client

Implements spec §8. Lets the runner distinguish a 503 from other failures without parsing the message. `ChatSurface` only reads `.message`, so it is unaffected.

**Files:**
- Modify: `src/lib/sse.ts:37` (insert before `export type ChatStreamArgs`), `:62` (the throw)

**Interfaces:**
- Produces: `export class HttpError extends Error { readonly status: number }`, thrown by `chatStream` for any non-2xx response, with the same message text as before.

- [ ] **Step 1: Add the class**

Insert directly before `export type ChatStreamArgs = {` (line 37):
```ts
/**
 * Non-2xx response from `/v1/chat/completions`. Carries the status so a
 * caller can treat 503 (server at its `max_concurrent` cap; retry with
 * backoff) differently from a 4xx it must not retry. The message is the
 * same `HTTP <status>: <body>` text callers already display.
 */
export class HttpError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

```

- [ ] **Step 2: Throw it**

Change line 62 from
```ts
    throw new Error(`HTTP ${r.status}: ${text || r.statusText}`);
```
to
```ts
    throw new HttpError(r.status, `HTTP ${r.status}: ${text || r.statusText}`);
```

- [ ] **Step 3: Verify**

```bash
npm run typecheck
```
Expected: no output. In the browser, `/chat` still streams a reply from the mock. Then, to see the error path unchanged: stop the mock server, send a chat message, expect `[stream error: …]` appended to the bubble as before; restart the mock server afterwards.

- [ ] **Step 4: Commit**

```bash
git add src/lib/sse.ts
git commit -m "feat(dashboard): sse client throws a typed HttpError with the status"
```

---

### Task 3: Pure modules — settings, grid layout, prompt pool

Implements spec §6.2 (`chooseColumns`), §6.4 (settings ranges/persistence), §9 (prompt pool). No React, no DOM beyond `localStorage`.

**Files:**
- Create: `src/lib/streamSettings.ts`
- Create: `src/lib/gridLayout.ts`
- Create: `src/lib/prompts.ts`

**Interfaces:**
- Produces:
  - `type StreamSettings = { streamCount: number; maxTokens: number; cooldownMinS: number; cooldownMaxS: number; temperature: number }`
  - `DEFAULT_SETTINGS: StreamSettings`
  - `clampSettings(s: Partial<StreamSettings> | null | undefined): StreamSettings`
  - `loadSettings(): StreamSettings`, `saveSettings(s: StreamSettings): void`
  - `chooseColumns(n: number, width: number, height: number): number`
  - `PROMPTS: readonly string[]`

- [ ] **Step 1: Settings module**

Create `src/lib/streamSettings.ts`:
```ts
// Settings for the streams showcase: defaults, hard ranges, and the
// localStorage round-trip. Everything that reaches the runner or the UI goes
// through `clampSettings`, so a hand-edited or stale stored value can never
// produce 0 streams, a negative cooldown, or max < min.

export type StreamSettings = {
  /** Number of concurrent autonomous streams (tiles). */
  streamCount: number;
  /** `max_tokens` sent with every request. */
  maxTokens: number;
  /** Idle time between one reply finishing and the next prompt, in seconds. */
  cooldownMinS: number;
  cooldownMaxS: number;
  /** Sampling temperature; 0 matches Chat and the fleet bench. */
  temperature: number;
};

export const DEFAULT_SETTINGS: StreamSettings = {
  streamCount: 16,
  maxTokens: 160,
  cooldownMinS: 1,
  cooldownMaxS: 2,
  temperature: 0,
};

export const LIMITS = {
  streamCount: { min: 1, max: 64 },
  maxTokens: { min: 16, max: 1024 },
  cooldownS: { min: 0, max: 30 },
  temperature: { min: 0, max: 1.5 },
} as const;

const STORAGE_KEY = "cascadia.streams.settings.v1";

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, n));
}

export function clampSettings(s: Partial<StreamSettings> | null | undefined): StreamSettings {
  const src = s ?? {};
  const d = DEFAULT_SETTINGS;
  const streamCount = Math.round(
    clampNumber(src.streamCount, LIMITS.streamCount.min, LIMITS.streamCount.max, d.streamCount),
  );
  const maxTokens = Math.round(
    clampNumber(src.maxTokens, LIMITS.maxTokens.min, LIMITS.maxTokens.max, d.maxTokens),
  );
  const cooldownMinS = clampNumber(
    src.cooldownMinS,
    LIMITS.cooldownS.min,
    LIMITS.cooldownS.max,
    d.cooldownMinS,
  );
  // Max is clamped to the same range, then never allowed below min.
  const cooldownMaxS = Math.max(
    cooldownMinS,
    clampNumber(src.cooldownMaxS, LIMITS.cooldownS.min, LIMITS.cooldownS.max, d.cooldownMaxS),
  );
  const temperature = clampNumber(
    src.temperature,
    LIMITS.temperature.min,
    LIMITS.temperature.max,
    d.temperature,
  );
  return { streamCount, maxTokens, cooldownMinS, cooldownMaxS, temperature };
}

/** Stored settings, or defaults when nothing is stored or it fails to parse. */
export function loadSettings(): StreamSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return clampSettings(raw ? (JSON.parse(raw) as Partial<StreamSettings>) : null);
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: StreamSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // Storage unavailable (private mode, quota): the demo still runs, the
    // values just don't survive a reload.
  }
}
```

- [ ] **Step 2: Grid layout module**

Create `src/lib/gridLayout.ts`:
```ts
/**
 * Column count for `n` tiles in a `width × height` box.
 *
 * `auto-fit` grids leave a ragged last row (6, 6, 6, 2), which looks broken
 * on a wall of terminals. Instead: take the ideal square-ish count
 * `sqrt(n · width / height)`, try the integers within ±1 of it, and keep the
 * one that leaves the fewest empty cells; on a tie prefer the tile aspect
 * ratio closest to 1.6 (a comfortable terminal shape). With 20 tiles on a
 * 16:9 screen this yields 5 × 4; with 16, 4 × 4; with 10, 5 × 2.
 */
export function chooseColumns(n: number, width: number, height: number): number {
  if (n <= 1) return 1;
  if (width <= 0 || height <= 0) return Math.ceil(Math.sqrt(n));

  const ideal = Math.sqrt((n * width) / height);
  const lo = Math.min(n, Math.max(1, Math.floor(ideal) - 1));
  const hi = Math.min(n, Math.max(lo, Math.ceil(ideal) + 1));

  let best = lo;
  let bestEmpty = Number.POSITIVE_INFINITY;
  let bestAspect = Number.POSITIVE_INFINITY;
  for (let cols = lo; cols <= hi; cols++) {
    const rows = Math.ceil(n / cols);
    const empty = cols * rows - n;
    const aspect = Math.abs(width / cols / (height / rows) - 1.6);
    if (empty < bestEmpty || (empty === bestEmpty && aspect < bestAspect)) {
      best = cols;
      bestEmpty = empty;
      bestAspect = aspect;
    }
  }
  return best;
}
```

- [ ] **Step 3: Prompt pool**

Create `src/lib/prompts.ts`. Every entry asks for a bounded short answer so replies land around 40–150 tokens (spec §9). The 16 entries marked `// bench` are copied verbatim from `deploy/inkling-fleet/bench.py`.
```ts
// Prompt pool for the streams showcase. Six families, ~25 each. Rules: plain
// English, one request per prompt, an explicit length bound, no inner double
// quotes, no duplicates, nothing that invites a long answer or a refusal.
// Entries marked `// bench` are verbatim from deploy/inkling-fleet/bench.py.

export const PROMPTS: readonly string[] = [
  // ---- Facts -------------------------------------------------------------
  "What is the capital of France? Answer in one word.", // bench
  "What is the boiling point of water? Explain.", // bench
  "Name a famous painting and its painter.", // bench
  "How many continents are there? Answer in one sentence.",
  "Which planet is closest to the Sun? One sentence.",
  "What is the largest mammal on Earth? One sentence.",
  "In which year did humans first land on the Moon? One sentence.",
  "What is the chemical symbol for gold? Answer in one word.",
  "Which language has the most native speakers? One sentence.",
  "What is the tallest mountain on Earth? One sentence.",
  "Who wrote Romeo and Juliet? Answer in a few words.",
  "What is the smallest prime number? One sentence.",
  "How many legs does a spider have? One sentence.",
  "What is the hardest natural substance? One sentence.",
  "Which ocean is the deepest? One sentence.",
  "What gas do plants absorb from the air? One sentence.",
  "What is the longest river in Africa? One sentence.",
  "How many bones are in the adult human body? One sentence.",
  "What is the freezing point of water in Fahrenheit? One sentence.",
  "Which metal is liquid at room temperature? One sentence.",
  "What is the currency of Japan? Answer in one word.",
  "Name the four seasons in order, starting with spring.",
  "What does DNA stand for? One sentence.",
  "Which instrument has 88 keys? One sentence.",
  "Roughly how fast is light in kilometres per second? One sentence.",

  // ---- Explain simply ---------------------------------------------------
  "Explain in three sentences why the sky is blue.", // bench
  "Why is the ocean salty? Two sentences.", // bench
  "Summarise photosynthesis in two sentences.", // bench
  "What does a compiler do? Two sentences.", // bench
  "Explain gravity to a child in two sentences.", // bench
  "Give one tip for sleeping better, in two sentences.", // bench
  "Explain how a rainbow forms, in two sentences.",
  "Explain what a black hole is to a ten-year-old, in two sentences.",
  "Why do we have leap years? Two sentences.",
  "Explain how a refrigerator keeps food cold, in two sentences.",
  "Why does ice float on water? Two sentences.",
  "Explain what the internet is to a ten-year-old, in two sentences.",
  "How does a battery store energy? Two sentences.",
  "Why do leaves change colour in autumn? Two sentences.",
  "Explain what inflation means, in two sentences.",
  "How do vaccines work? Two sentences, simple words.",
  "Why does bread rise? Two sentences.",
  "Explain what an algorithm is to a ten-year-old, in two sentences.",
  "Why is the sea blue but a glass of water clear? Two sentences.",
  "How do aeroplanes stay in the air? Two sentences.",
  "Explain what a mixture-of-experts model is, in two sentences.",
  "Why do cats purr? Two sentences.",
  "Explain what a GPU does, in two sentences.",
  "How does a thermostat work? Two sentences.",
  "Why does the Moon have phases? Two sentences.",
  "Explain what encryption is to a ten-year-old, in two sentences.",

  // ---- Short creative ----------------------------------------------------
  "Write two sentences about the Pacific Ocean.", // bench
  "Describe a cat in two sentences.", // bench
  "Describe rain in one sentence.", // bench
  "What is a haiku? Give one.", // bench
  "Write a haiku about a mountain at dawn.",
  "Write a haiku about a busy train station.",
  "Write a two-line rhyme about coffee.",
  "Write a one-sentence story about a lost umbrella.",
  "Describe a thunderstorm in two sentences.",
  "Write a haiku about autumn leaves.",
  "Describe a lighthouse at night in two sentences.",
  "Write a two-line rhyme about a sleepy dog.",
  "Write a one-sentence story about a robot learning to paint.",
  "Describe the smell of fresh bread in one sentence.",
  "Write a haiku about the first snow of winter.",
  "Describe a desert at noon in two sentences.",
  "Write a two-line rhyme about the sea.",
  "Write a one-sentence story about a key that opens nothing.",
  "Describe a city waking up in two sentences.",
  "Write a haiku about a river in summer.",
  "Describe an old library in two sentences.",
  "Write a two-line rhyme about the Moon.",
  "Write a one-sentence story about a letter that arrived fifty years late.",
  "Describe a forest after rain in two sentences.",
  "Write a haiku about a cup of tea.",

  // ---- Tiny code ---------------------------------------------------------
  "Write a three-line Python function that returns the square of a number.",
  "Write a one-line Python expression that reverses a string called s.",
  "Write a JavaScript function that checks whether a number is even, in three lines.",
  "Write a Python function that returns the larger of two numbers, in three lines.",
  "Write a SQL query that counts the rows in a table called users.",
  "Write a Bash one-liner that counts the lines in a file called log.txt.",
  "Write a Python list comprehension that squares the numbers 1 to 5.",
  "Write a Rust function that adds two i32 values, in three lines.",
  "Write a JavaScript arrow function that doubles a number, in one line.",
  "Write a Python function that returns True if a string is a palindrome, in three lines.",
  "Write a Go function that returns the length of a slice of ints, in three lines.",
  "Write a one-line Python expression that sums a list called nums.",
  "Write a CSS rule that centres the text in a class called title.",
  "Write a Python dictionary with three fruits as keys and their colours as values.",
  "Write a JavaScript function that returns the last element of an array, in one line.",
  "Write a SQL query that selects the name column from a table called cities, sorted alphabetically.",
  "Write a Python function that converts Celsius to Fahrenheit, in two lines.",
  "Write a Bash command that lists only the directories in the current folder.",
  "Write a TypeScript type for a point with x and y numbers.",
  "Write a Python function that returns the first n Fibonacci numbers, in five lines or fewer.",
  "Write a regular expression that matches a four-digit year.",
  "Write a one-line Python expression that checks whether 7 is in a list called xs.",
  "Write a C function that returns the maximum of two ints, in three lines.",
  "Write a JSON object describing a book with title, author and year.",
  "Write a Python function that counts the vowels in a string, in three lines.",

  // ---- Lists ---------------------------------------------------------------
  "Name two planets and one fact about each.", // bench
  "Name three primary colours, one word each.",
  "Name three programming languages, one word each.",
  "List three things you need to bake bread, one phrase each.",
  "Name three countries in South America, one word each.",
  "List three benefits of walking, one phrase each.",
  "Name three musical instruments with strings, one word each.",
  "List three uses for a paper clip, one phrase each.",
  "Name three famous scientists, one name each.",
  "List three things to pack for a beach day, one phrase each.",
  "Name three types of cloud, one word each.",
  "List three ways to save water at home, one phrase each.",
  "Name three board games, one name each.",
  "List three ingredients in a basic tomato sauce, one word each.",
  "Name three rivers in Europe, one name each.",
  "List three reasons to learn a second language, one phrase each.",
  "Name three kinds of renewable energy, one phrase each.",
  "List three things a good teacher does, one phrase each.",
  "Name three noble gases, one word each.",
  "List three tips for a good night of sleep, one phrase each.",
  "Name three shapes with four sides, one word each.",
  "List three things found in a kitchen drawer, one word each.",
  "Name three species of big cat, one word each.",
  "List three steps to make a cup of tea, one phrase each.",
  "Name three moons in the solar system, one word each.",

  // ---- Reasoning -----------------------------------------------------------
  "List three prime numbers and say why they are prime.", // bench
  "What is 6 times 7? Explain briefly.", // bench
  "If a train travels 60 km in one hour, how far does it go in two and a half hours? Show the calculation in one line.",
  "What is 15 percent of 200? Show one line of working.",
  "A dozen eggs cost 3 dollars. How much do 4 eggs cost? One sentence.",
  "Is 51 a prime number? Answer and explain in one sentence.",
  "What is the next number in the sequence 2, 4, 8, 16? One sentence.",
  "If today is Monday, what day is it in ten days? One sentence.",
  "What is 9 squared minus 1? One line of working.",
  "A rectangle is 4 by 6. What is its area and its perimeter? One sentence each.",
  "How many minutes are in a day? Show the calculation.",
  "Which is heavier, a kilogram of feathers or a kilogram of steel? One sentence.",
  "If you have 3 apples and eat one, how many are left? One sentence.",
  "What is half of a quarter? One sentence.",
  "Sort these numbers from smallest to largest: 17, 3, 42, 8. One line.",
  "What is 1000 divided by 8? One line of working.",
  "A clock shows 3:45. How many minutes until 4:30? One sentence.",
  "What is the sum of the first five odd numbers? Show the working in one line.",
  "If five pens cost 10 dollars, how much do eight pens cost? One sentence.",
  "Which is larger, 2 to the power of 10, or 1000? One sentence.",
  "How many sides do two hexagons have in total? One sentence.",
  "What is 12 times 12? Answer in one sentence.",
  "A car uses 5 litres per 100 km. How much fuel for 250 km? One line of working.",
  "Is the number 144 a perfect square? Answer and explain in one sentence.",
  "What is 7 times 8 plus 4? One line of working.",
];
```

- [ ] **Step 4: Verify**

```bash
npm run typecheck
node -e '
const src = require("fs").readFileSync("src/lib/prompts.ts", "utf8");
const items = [...src.matchAll(/^\s+"((?:[^"\\]|\\.)*)",/gm)].map((m) => m[1]);
const dupes = items.filter((p, i) => items.indexOf(p) !== i);
const bench = (src.match(/\/\/ bench/g) || []).length;
console.log({ prompts: items.length, dupes, bench });
if (items.length < 140 || dupes.length || bench !== 16) process.exit(1);
'
```
Expected: typecheck silent; the node line prints `{ prompts: 151, dupes: [], bench: 16 }` (count may differ by one or two if you fix a prompt; must stay ≥ 140 with zero dupes and exactly 16 bench entries).

- [ ] **Step 5: Commit**

```bash
git add src/lib/streamSettings.ts src/lib/gridLayout.ts src/lib/prompts.ts
git commit -m "feat(dashboard): settings, grid sizing and prompt pool for the streams showcase"
```

---

### Task 4: The runner and its hook

Implements spec §7 (all subsections). Pure TypeScript class plus a thin React hook. Nothing renders yet; Task 5 puts it on screen.

**Files:**
- Create: `src/lib/streamRunner.ts`
- Create: `src/hooks/useStreamRunner.ts`

**Interfaces:**
- Consumes: `chatStream`, `HttpError` from `@/lib/sse` (Task 2); `PROMPTS` (Task 3); `StreamSettings`, `DEFAULT_SETTINGS`, `clampSettings`, `loadSettings` (Task 3).
- Produces (used by Tasks 5 and 6):
  - types `StreamStatus`, `Exchange`, `StreamState`, `Aggregate`, `RunnerSnapshot`
  - `class StreamRunner` with `getSnapshot`, `subscribe`, `setModel(id)`, `applySettings(next)`, `start()`, `stop()`, `pauseStream(id)`, `resumeStream(id)`, `skipStream(id)`
  - `useStreamRunner(): { snapshot: RunnerSnapshot; runner: StreamRunner }`

- [ ] **Step 1: The runner**

Create `src/lib/streamRunner.ts`:
```ts
// Autonomous multi-stream runner behind the /streams showcase.
//
// Plain TypeScript, no React. It owns N stream loops (one async function
// each), their AbortControllers and timers, and a shared token ring for the
// aggregate rate. The page reads it through `useSyncExternalStore`: every
// mutation marks the runner dirty, and one requestAnimationFrame later a new
// immutable snapshot is published. Twenty streams at token rate therefore
// cost one React commit per frame, not one per token.
//
// Concurrency model: each slot carries a `generation` counter. A loop
// captures the generation it was launched with and exits the moment it
// resumes from any await and finds the counter moved on. Stop, Skip, Pause
// and shrinking the stream count all just bump the counter, abort the
// in-flight request and wake the pending sleep; no flags to keep in sync.

import { PROMPTS } from "@/lib/prompts";
import { chatStream, HttpError } from "@/lib/sse";
import { clampSettings, DEFAULT_SETTINGS, type StreamSettings } from "@/lib/streamSettings";

export type StreamStatus =
  | "idle"
  | "queued"
  | "prefill"
  | "streaming"
  | "cooldown"
  | "paused"
  | "error";

/** A past prompt/reply pair kept as scrollback in the tile. */
export type Exchange = {
  prompt: string;
  reply: string;
  tokens: number;
  ttftMs: number | null;
  tokPerSec: number | null;
};

export type StreamState = {
  /** 0-based, equals the slot's index; stable while the stream exists. */
  id: number;
  status: StreamStatus;
  /** Current (or, after Stop, last) prompt. */
  prompt: string | null;
  /** Current streamed reply text. */
  reply: string;
  tokens: number;
  /** Time to first chunk of the successful attempt, ms. */
  ttftMs: number | null;
  /** Wall time since the current attempt was sent, ms. */
  elapsedMs: number;
  /** Decode tok/s, same rule as ChatSurface. */
  tokPerSec: number | null;
  /** Retries (503 or network) for the current prompt; resets on a new prompt. */
  attempt: number;
  error: string | null;
  /** Pause pressed mid-reply: finish it, then park. */
  pauseRequested: boolean;
  /** Up to HISTORY_CAP earlier exchanges, oldest first. */
  history: Exchange[];
};

export type Aggregate = {
  running: boolean;
  /** performance.now() at Play, null before the first Play. */
  startedAt: number | null;
  /** Streams in prefill or streaming. */
  activeStreams: number;
  queuedStreams: number;
  /** Tokens across all streams over the last 10 s; null before the first token. Frozen after Stop. */
  tokensPerSecond: number | null;
  totalTokens: number;
  completedReplies: number;
};

export type RunnerSnapshot = {
  settings: StreamSettings;
  model: string;
  aggregate: Aggregate;
  streams: StreamState[];
};

/** Gap between consecutive stream launches, so twenty prompts don't all hit prefill at once. */
const STAGGER_MS = 300;
const HISTORY_CAP = 3;
const AGG_WINDOW_MS = 10_000;
/** How long a tile shows a non-capacity error before moving to a new prompt. */
const ERROR_HOLD_MS = 3_000;
const CAPACITY_BACKOFF_CAP_MS = 5_000;
const NETWORK_BACKOFF_CAP_MS = 10_000;
/** Elapsed-time tick for in-flight tiles; 10 fps is plenty for a clock. */
const TICK_MS = 100;
/** rAF does not fire in a hidden tab; a timer keeps a background tab's numbers current. */
const HIDDEN_FLUSH_MS = 250;

/** 500 ms + 250 ms per attempt, capped at 5 s, ±20 % jitter. */
export function capacityBackoffMs(attempt: number): number {
  const base = Math.min(CAPACITY_BACKOFF_CAP_MS, 500 + 250 * attempt);
  return base * (0.8 + Math.random() * 0.4);
}

/** 1 s doubling per attempt, capped at 10 s, so a dead server is not hammered. */
export function networkBackoffMs(attempt: number): number {
  return Math.min(NETWORK_BACKOFF_CAP_MS, 1000 * 2 ** attempt);
}

/**
 * Decode tok/s: tokens after the first over the post-TTFT window, once that
 * window exceeds 50 ms. Identical to the rule in ChatSurface so the two
 * screens agree.
 */
export function decodeRate(tokens: number, elapsedMs: number, ttftMs: number | null): number | null {
  if (ttftMs === null || tokens <= 1) return null;
  const windowS = (elapsedMs - ttftMs) / 1000;
  return windowS > 0.05 ? (tokens - 1) / windowS : null;
}

type RequestOutcome =
  | { kind: "done" }
  | { kind: "capacity" }
  | { kind: "network" }
  | { kind: "error"; message: string }
  | { kind: "aborted" };

type Slot = {
  /** Mutable working copy; the snapshot gets a spread of it when dirty. */
  state: StreamState;
  dirty: boolean;
  /** Bumped on every launch and cancel; a loop exits when its captured value is stale. */
  generation: number;
  controller: AbortController | null;
  /** Resolves the loop's pending sleep early (skip / pause / stop). */
  wake: (() => void) | null;
  /** performance.now() when the current request attempt was sent; null when not in flight. */
  startedAt: number | null;
};

function newState(id: number): StreamState {
  return {
    id,
    status: "idle",
    prompt: null,
    reply: "",
    tokens: 0,
    ttftMs: null,
    elapsedMs: 0,
    tokPerSec: null,
    attempt: 0,
    error: null,
    pauseRequested: false,
    history: [],
  };
}

function newSlot(id: number): Slot {
  return {
    state: newState(id),
    dirty: true,
    generation: 0,
    controller: null,
    wake: null,
    startedAt: null,
  };
}

const EMPTY_AGGREGATE: Aggregate = {
  running: false,
  startedAt: null,
  activeStreams: 0,
  queuedStreams: 0,
  tokensPerSecond: null,
  totalTokens: 0,
  completedReplies: 0,
};

export class StreamRunner {
  private settings: StreamSettings;
  private model: string;
  private slots: Slot[] = [];
  private running = false;
  private startedAt: number | null = null;
  private totalTokens = 0;
  private completedReplies = 0;
  /** (timestamp, tokens) events from every stream, pruned to the last AGG_WINDOW_MS. */
  private ring: { t: number; n: number }[] = [];
  private lastRate: number | null = null;
  private tickTimer: number | null = null;
  private flushScheduled = false;
  private listeners = new Set<() => void>();
  private snapshot: RunnerSnapshot = {
    settings: DEFAULT_SETTINGS,
    model: "",
    aggregate: EMPTY_AGGREGATE,
    streams: [],
  };

  constructor(settings: StreamSettings, model: string) {
    this.settings = clampSettings(settings);
    this.model = model;
    for (let i = 0; i < this.settings.streamCount; i++) this.slots.push(newSlot(i));
    this.snapshot = this.buildSnapshot(performance.now());
  }

  // ---- useSyncExternalStore contract -------------------------------------
  // Arrow properties so the hook can pass them unbound.

  getSnapshot = (): RunnerSnapshot => this.snapshot;

  subscribe = (cb: () => void): (() => void) => {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  };

  // ---- public controls -------------------------------------------------------

  /** Used by each stream's next request; in-flight requests keep the model they started with. */
  setModel(id: string): void {
    if (id === this.model) return;
    this.model = id;
    this.mark();
  }

  /**
   * Stream count applies immediately: extra tiles are appended (and launched,
   * staggered, if running), surplus tiles are cancelled and dropped from the
   * top index down. Every other field is read when a stream next picks a
   * prompt or a cooldown.
   */
  applySettings(next: StreamSettings): void {
    const clamped = clampSettings(next);
    this.settings = clamped;
    while (this.slots.length > clamped.streamCount) {
      const slot = this.slots.pop();
      if (slot) this.cancel(slot);
    }
    let added = 0;
    while (this.slots.length < clamped.streamCount) {
      const slot = newSlot(this.slots.length);
      this.slots.push(slot);
      if (this.running) this.launch(slot, STAGGER_MS * added++);
    }
    this.mark();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.startedAt = performance.now();
    this.totalTokens = 0;
    this.completedReplies = 0;
    this.ring = [];
    this.lastRate = null;
    this.tickTimer = window.setInterval(() => this.tick(), TICK_MS);
    let i = 0;
    for (const slot of this.slots) {
      if (slot.state.status === "paused") continue;
      this.launch(slot, STAGGER_MS * i++);
    }
    this.mark();
  }

  /** Abort everything now. Tiles keep their last prompt, text, scrollback and numbers on screen. */
  stop(): void {
    if (!this.running) return;
    this.running = false;
    if (this.tickTimer !== null) {
      window.clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
    for (const slot of this.slots) {
      this.cancel(slot);
      const s = slot.state;
      s.pauseRequested = false;
      if (s.status !== "paused") {
        s.status = "idle";
        s.error = null;
      }
      slot.dirty = true;
    }
    this.mark();
  }

  /**
   * Mid-reply: let the reply finish, then park. Anywhere else (queued,
   * cooldown, error, idle): park now, cancelling any pending retry so a
   * paused tile never fires a request.
   */
  pauseStream(id: number): void {
    const slot = this.slots[id];
    if (!slot) return;
    const s = slot.state;
    if (s.status === "paused") return;
    if (s.status === "prefill" || s.status === "streaming") {
      s.pauseRequested = true;
    } else {
      this.cancel(slot);
      s.status = "paused";
      s.pauseRequested = false;
      s.error = null;
    }
    this.mark(slot);
  }

  resumeStream(id: number): void {
    const slot = this.slots[id];
    if (!slot || slot.state.status !== "paused") return;
    slot.state.status = "idle";
    slot.state.pauseRequested = false;
    if (this.running) this.launch(slot, 0);
    this.mark(slot);
  }

  /** Abort the current prompt (or its wait) and start a fresh one with no cooldown. */
  skipStream(id: number): void {
    const slot = this.slots[id];
    if (!slot || !this.running || slot.state.status === "paused") return;
    this.launch(slot, 0);
  }

  // ---- loop machinery --------------------------------------------------------

  /**
   * Kill whatever the slot's loop is doing: the in-flight request, or the
   * sleep it is parked in. The bumped generation makes the old loop return
   * as soon as it resumes.
   */
  private cancel(slot: Slot): void {
    slot.generation++;
    slot.controller?.abort();
    slot.controller = null;
    slot.wake?.();
    slot.startedAt = null;
  }

  private launch(slot: Slot, delayMs: number): void {
    this.cancel(slot);
    void this.loop(slot, slot.generation, delayMs);
  }

  private alive(slot: Slot, gen: number): boolean {
    return this.running && slot.generation === gen;
  }

  private sleep(slot: Slot, ms: number): Promise<void> {
    return new Promise((resolve) => {
      const timer = window.setTimeout(() => {
        slot.wake = null;
        resolve();
      }, ms);
      slot.wake = () => {
        window.clearTimeout(timer);
        slot.wake = null;
        resolve();
      };
    });
  }

  private async loop(slot: Slot, gen: number, initialDelayMs: number): Promise<void> {
    const s = slot.state;
    if (initialDelayMs > 0) {
      await this.sleep(slot, initialDelayMs);
      if (!this.alive(slot, gen)) return;
    }
    while (this.alive(slot, gen)) {
      if (s.pauseRequested) {
        this.park(slot);
        return;
      }
      this.beginPrompt(slot, this.pickPrompt(slot));

      // One prompt, retried in place on capacity / network trouble; given up
      // (new prompt) on any other error.
      let done = false;
      for (;;) {
        const outcome = await this.request(slot, gen);
        if (!this.alive(slot, gen)) return;
        if (outcome.kind === "done") {
          done = true;
          break;
        }
        if (outcome.kind === "aborted") return;
        // A pause asked for mid-reply parks now rather than retrying.
        if (s.pauseRequested) {
          this.park(slot);
          return;
        }
        s.attempt++;
        if (outcome.kind === "capacity") {
          s.status = "queued";
          this.mark(slot);
          await this.sleep(slot, capacityBackoffMs(s.attempt));
        } else if (outcome.kind === "network") {
          s.status = "error";
          s.error = "server unreachable";
          this.mark(slot);
          await this.sleep(slot, networkBackoffMs(s.attempt));
        } else {
          s.status = "error";
          s.error = outcome.message;
          this.mark(slot);
          await this.sleep(slot, ERROR_HOLD_MS);
          break;
        }
        if (!this.alive(slot, gen)) return;
        s.status = "prefill";
        s.error = null;
        this.mark(slot);
      }
      if (!this.alive(slot, gen)) return;
      if (!done) continue;

      this.completedReplies++;
      this.mark(slot);
      if (s.pauseRequested) {
        this.park(slot);
        return;
      }
      s.status = "cooldown";
      this.mark(slot);
      const { cooldownMinS, cooldownMaxS } = this.settings;
      await this.sleep(slot, (cooldownMinS + Math.random() * (cooldownMaxS - cooldownMinS)) * 1000);
    }
  }

  private park(slot: Slot): void {
    const s = slot.state;
    s.pauseRequested = false;
    s.status = "paused";
    slot.startedAt = null;
    this.mark(slot);
  }

  /**
   * Move the previous exchange (finished, or cut short by Skip/Stop) into
   * scrollback and reset the per-prompt fields. Archiving here rather than
   * at reply end keeps a finished reply bright through its cooldown and
   * dims it only when the next prompt starts.
   */
  private beginPrompt(slot: Slot, prompt: string): void {
    const s = slot.state;
    if (s.prompt !== null && s.reply.length > 0) {
      const previous: Exchange = {
        prompt: s.prompt,
        reply: s.reply,
        tokens: s.tokens,
        ttftMs: s.ttftMs,
        tokPerSec: s.tokPerSec,
      };
      s.history = [...s.history, previous].slice(-HISTORY_CAP);
    }
    s.prompt = prompt;
    s.reply = "";
    s.tokens = 0;
    s.ttftMs = null;
    s.elapsedMs = 0;
    s.tokPerSec = null;
    s.attempt = 0;
    s.error = null;
    s.status = "prefill";
    this.mark(slot);
  }

  /** Uniform over prompts not held by another stream and not this stream's last one (spec §7.5). */
  private pickPrompt(slot: Slot): string {
    const held = new Set<string>();
    for (const other of this.slots) {
      if (other !== slot && other.state.prompt !== null) held.add(other.state.prompt);
    }
    const last = slot.state.prompt;
    let pool = PROMPTS.filter((p) => p !== last && !held.has(p));
    if (pool.length === 0) pool = PROMPTS.filter((p) => p !== last);
    if (pool.length === 0) pool = [...PROMPTS];
    return pool[Math.floor(Math.random() * pool.length)];
  }

  private async request(slot: Slot, gen: number): Promise<RequestOutcome> {
    const s = slot.state;
    const controller = new AbortController();
    slot.controller = controller;
    const startedAt = performance.now();
    slot.startedAt = startedAt;
    let completed = false;
    try {
      const stream = chatStream(
        {
          model: this.model,
          messages: [{ role: "user", content: s.prompt ?? "" }],
          max_tokens: this.settings.maxTokens,
          temperature: this.settings.temperature,
        },
        controller.signal,
      );
      for await (const chunk of stream) {
        if (!this.alive(slot, gen)) return { kind: "aborted" };
        // Mid-stream engine failure: {object:"error"} with no `choices`.
        if (chunk.object === "error") return { kind: "error", message: chunk.error.message };
        const now = performance.now();
        if (s.ttftMs === null) {
          s.ttftMs = now - startedAt;
          s.status = "streaming";
        }
        const n = chunk.n_tokens ?? 1;
        s.tokens += n;
        s.reply += chunk.choices[0]?.delta.content ?? "";
        s.elapsedMs = now - startedAt;
        s.tokPerSec = decodeRate(s.tokens, s.elapsedMs, s.ttftMs);
        this.ring.push({ t: now, n });
        this.totalTokens += n;
        this.mark(slot);
      }
      s.elapsedMs = performance.now() - startedAt;
      completed = true;
      return { kind: "done" };
    } catch (err) {
      if ((err as Error).name === "AbortError") return { kind: "aborted" };
      if (err instanceof HttpError) {
        return err.status === 503 ? { kind: "capacity" } : { kind: "error", message: err.message };
      }
      // fetch() rejects with a TypeError when the server is unreachable or
      // the connection drops mid-body.
      if (err instanceof TypeError) return { kind: "network" };
      return { kind: "error", message: (err as Error).message };
    } finally {
      // Close the connection if we left the body early (error chunk, stale
      // generation). Harmless after a normal end.
      if (!completed) controller.abort();
      // Only clear our own bookkeeping: a newer attempt may already own the slot.
      if (slot.controller === controller) {
        slot.controller = null;
        slot.startedAt = null;
      }
    }
  }

  // ---- metrics and snapshots -----------------------------------------------

  /** 10 fps: advance elapsed on in-flight tiles and let the aggregate window slide. */
  private tick(): void {
    const now = performance.now();
    for (const slot of this.slots) {
      if (slot.startedAt === null) continue;
      const st = slot.state.status;
      if (st !== "prefill" && st !== "streaming") continue;
      slot.state.elapsedMs = now - slot.startedAt;
      slot.dirty = true;
    }
    this.mark();
  }

  private mark(slot?: Slot): void {
    if (slot) slot.dirty = true;
    if (this.flushScheduled) return;
    this.flushScheduled = true;
    if (typeof document !== "undefined" && document.hidden) {
      window.setTimeout(() => this.flush(), HIDDEN_FLUSH_MS);
    } else {
      window.requestAnimationFrame(() => this.flush());
    }
  }

  private flush(): void {
    this.flushScheduled = false;
    this.snapshot = this.buildSnapshot(performance.now());
    for (const cb of this.listeners) cb();
  }

  /** New snapshot object; unchanged streams keep their previous object so memoised tiles skip. */
  private buildSnapshot(now: number): RunnerSnapshot {
    const prev = this.snapshot.streams;
    const streams = this.slots.map((slot, i) => {
      const cached = prev[i];
      if (!slot.dirty && cached && cached.id === slot.state.id) return cached;
      slot.dirty = false;
      return { ...slot.state };
    });
    return {
      settings: this.settings,
      model: this.model,
      aggregate: this.aggregate(now),
      streams,
    };
  }

  private aggregate(now: number): Aggregate {
    let active = 0;
    let queued = 0;
    for (const { state } of this.slots) {
      if (state.status === "prefill" || state.status === "streaming") active++;
      else if (state.status === "queued") queued++;
    }
    // Frozen at its last value after Stop, rather than decaying to zero.
    if (this.running) this.lastRate = this.windowRate(now);
    return {
      running: this.running,
      startedAt: this.startedAt,
      activeStreams: active,
      queuedStreams: queued,
      tokensPerSecond: this.lastRate,
      totalTokens: this.totalTokens,
      completedReplies: this.completedReplies,
    };
  }

  /** Tokens in the last 10 s over min(10 s, time since Play). */
  private windowRate(now: number): number | null {
    const cutoff = now - AGG_WINDOW_MS;
    let drop = 0;
    while (drop < this.ring.length && this.ring[drop].t < cutoff) drop++;
    if (drop > 0) this.ring.splice(0, drop);
    if (this.totalTokens === 0 || this.startedAt === null) return null;
    const windowS = Math.min(AGG_WINDOW_MS, now - this.startedAt) / 1000;
    if (windowS <= 0) return null;
    let sum = 0;
    for (const e of this.ring) sum += e.n;
    return sum / windowS;
  }
}
```

- [ ] **Step 2: The hook**

Create `src/hooks/useStreamRunner.ts`:
```ts
import { useEffect, useRef, useSyncExternalStore } from "react";

import { type RunnerSnapshot, StreamRunner } from "@/lib/streamRunner";
import { loadSettings } from "@/lib/streamSettings";

/**
 * One runner per page mount. Leaving the page stops every stream (decided
 * in the spec): the unmount cleanup aborts them. `stop()` on a runner that
 * is not running is a no-op, so StrictMode's mount/unmount/mount in dev is
 * harmless.
 */
export function useStreamRunner(): { snapshot: RunnerSnapshot; runner: StreamRunner } {
  const ref = useRef<StreamRunner | null>(null);
  if (ref.current === null) ref.current = new StreamRunner(loadSettings(), "");
  const runner = ref.current;

  const snapshot = useSyncExternalStore(runner.subscribe, runner.getSnapshot);

  useEffect(() => () => runner.stop(), [runner]);

  return { snapshot, runner };
}
```

- [ ] **Step 3: Verify**

```bash
npm run typecheck
```
Expected: no output. (The runner is exercised in the browser in Task 5.)

- [ ] **Step 4: Commit**

```bash
git add src/lib/streamRunner.ts src/hooks/useStreamRunner.ts
git commit -m "feat(dashboard): autonomous multi-stream runner with 503 queueing and frame-coalesced snapshots"
```

---

### Task 5: Tiles, toolbar and the page

Implements spec §6.1, §6.2, §6.3, §6.5, the fullscreen part of §4, and the `ModelPicker` line of §5.3. After this task the showcase runs end to end against the mock engine; only the settings drawer (Task 6) is missing.

**Files:**
- Create: `src/components/streams/StreamTile.tsx`
- Create: `src/components/streams/StreamToolbar.tsx`
- Modify: `src/components/ModelPicker.tsx:40, 59`
- Modify: `src/pages/Streams.tsx` (replace the Task 1 stub)

**Interfaces:**
- Consumes: `StreamRunner`, `RunnerSnapshot`, `StreamState`, `StreamStatus` (Task 4); `useStreamRunner` (Task 4); `chooseColumns` (Task 3); `useStats` and `Stats` (existing); `ModelPicker` (existing).
- Produces:
  - `StreamTile({ stream: StreamState; running: boolean; emptyText: string; runner: StreamRunner })` (memoised)
  - `StreamToolbar({ snapshot: RunnerSnapshot; runner: StreamRunner; serverStats: Stats | null; fullscreen: boolean; onToggleFullscreen: () => void; onToggleSettings: () => void })`

- [ ] **Step 1: The tile**

Create `src/components/streams/StreamTile.tsx`:
```tsx
import { memo, useEffect, useRef } from "react";

import type { StreamRunner, StreamState, StreamStatus } from "@/lib/streamRunner";

type Props = {
  stream: StreamState;
  running: boolean;
  /** Body text for a tile that has never held a prompt. */
  emptyText: string;
  runner: StreamRunner;
};

const DOT: Record<StreamStatus, string> = {
  idle: "bg-night-low",
  queued: "bg-amber-400",
  prefill: "bg-celadon animate-pulse",
  streaming: "bg-mint-bright",
  cooldown: "bg-night-dim",
  paused: "bg-night-low",
  error: "bg-red-400",
};

const WORD: Record<StreamStatus, string> = {
  idle: "text-night-low",
  queued: "text-amber-400",
  prefill: "text-celadon",
  streaming: "text-mint-bright",
  cooldown: "text-night-dim",
  paused: "text-night-low",
  error: "text-red-400",
};

/**
 * One terminal tile. Memoised: the runner hands out a new `stream` object
 * only for streams that changed since the last frame, so a token on stream
 * 3 re-renders tile 3 alone. `runner` and `emptyText` are stable props.
 */
export const StreamTile = memo(function StreamTile({ stream, running, emptyText, runner }: Props) {
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Keep the newest text in view. scrollTop rather than scrollIntoView so
  // the page itself never jumps.
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [stream.reply, stream.history, stream.status]);

  const inFlight = stream.status === "prefill" || stream.status === "streaming";
  const paused = stream.status === "paused";
  const statusWord = stream.pauseRequested ? "pausing" : stream.status;

  return (
    <div className="group flex min-h-0 flex-col overflow-hidden rounded-md border border-night-rule bg-night-2 font-mono text-[12.5px] leading-snug">
      <header className="flex h-7 shrink-0 items-center gap-2 bg-night-3 px-3">
        <span className="tabular-nums text-night-low">#{String(stream.id + 1).padStart(2, "0")}</span>
        <span className={`h-1.5 w-1.5 rounded-full ${DOT[stream.status]}`} aria-hidden />
        <span className={`label-mono ${WORD[stream.status]}`}>{statusWord}</span>
        <span className="ml-auto flex items-center gap-2">
          <span className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
            {paused ? (
              <TileButton onClick={() => runner.resumeStream(stream.id)}>Resume</TileButton>
            ) : (
              <TileButton onClick={() => runner.pauseStream(stream.id)}>Pause</TileButton>
            )}
            <TileButton onClick={() => runner.skipStream(stream.id)} disabled={!running || paused}>
              Skip
            </TileButton>
          </span>
          <span className="tabular-nums text-night-dim">
            {stream.tokPerSec != null ? `${stream.tokPerSec.toFixed(1)} tok/s` : "—"}
          </span>
        </span>
      </header>

      <div ref={bodyRef} className="min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-2">
        {stream.history.map((ex, i) => (
          <div key={i} className="whitespace-pre-wrap text-night-low">
            <div>&gt; {ex.prompt}</div>
            <div>{ex.reply}</div>
          </div>
        ))}
        {stream.prompt !== null ? (
          <div className="whitespace-pre-wrap">
            <div className="text-night-dim">&gt; {stream.prompt}</div>
            {stream.status === "queued" ? (
              <div className="text-amber-400">waiting for a slot… (attempt {stream.attempt})</div>
            ) : null}
            {stream.status === "error" ? <div className="text-red-400">{stream.error}</div> : null}
            <div className="text-night-ink">
              {stream.reply}
              {inFlight ? <span className="cursor-blink text-mint-bright">▌</span> : null}
            </div>
          </div>
        ) : (
          <div className="text-night-low">{emptyText}</div>
        )}
      </div>

      <footer className="label-mono flex h-6 shrink-0 items-center gap-3 border-t border-night-rule px-3 tabular-nums">
        <span>{stream.tokens} tok</span>
        <span>TTFT {stream.ttftMs != null ? `${stream.ttftMs.toFixed(0)} ms` : "—"}</span>
        <span className="ml-auto">{(stream.elapsedMs / 1000).toFixed(1)} s</span>
      </footer>
    </div>
  );
});

function TileButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  children: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-sm border border-night-rule px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-night-dim transition-colors hover:border-mint hover:text-mint disabled:opacity-30 disabled:hover:border-night-rule disabled:hover:text-night-dim"
    >
      {children}
    </button>
  );
}
```

- [ ] **Step 2: The toolbar**

Create `src/components/streams/StreamToolbar.tsx`:
```tsx
import { ModelPicker } from "@/components/ModelPicker";
import type { Stats } from "@/lib/api";
import type { RunnerSnapshot, StreamRunner } from "@/lib/streamRunner";

type Props = {
  snapshot: RunnerSnapshot;
  runner: StreamRunner;
  /** Latest /api/stats, or null while unreachable. */
  serverStats: Stats | null;
  fullscreen: boolean;
  onToggleFullscreen: () => void;
  onToggleSettings: () => void;
};

export function StreamToolbar({
  snapshot,
  runner,
  serverStats,
  fullscreen,
  onToggleFullscreen,
  onToggleSettings,
}: Props) {
  const { aggregate: agg, model } = snapshot;
  const cells: { label: string; value: string }[] = [
    { label: "Active", value: String(agg.activeStreams) },
    { label: "Queued", value: String(agg.queuedStreams) },
    { label: "Agg tok/s", value: agg.tokensPerSecond != null ? agg.tokensPerSecond.toFixed(1) : "—" },
    { label: "Tokens", value: agg.totalTokens.toLocaleString() },
    { label: "Replies", value: String(agg.completedReplies) },
    {
      label: "Server",
      value: serverStats ? `${serverStats.requests_in_flight} / ${serverStats.max_concurrent}` : "—",
    },
  ];

  return (
    <div className="flex h-12 shrink-0 items-center gap-5 border-b border-night-rule px-4">
      <span className="label-mono flex items-center gap-2 text-night-dim">
        {agg.running ? <span className="pulse-dot" aria-hidden /> : null}
        Streams
      </span>
      <ModelPicker value={model} onChange={(id) => runner.setModel(id)} />
      <button
        onClick={() => (agg.running ? runner.stop() : runner.start())}
        disabled={!model}
        title={model ? undefined : "waiting for a model"}
        className={`rounded-sm px-4 py-1 font-mono text-[11px] uppercase tracking-wider transition-colors disabled:opacity-40 ${
          agg.running
            ? "bg-red-400/15 text-red-400 hover:bg-red-400/25"
            : "bg-mint-bright text-night hover:bg-mint"
        }`}
      >
        {agg.running ? "Stop" : "Play"}
      </button>
      <div className="flex items-center gap-5 overflow-x-auto">
        {cells.map((c) => (
          <div key={c.label} className="flex items-baseline gap-1.5 whitespace-nowrap">
            <span className="label-mono text-night-low">{c.label}</span>
            <span className="font-mono text-[15px] tabular-nums text-night-ink">{c.value}</span>
          </div>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-1">
        <IconButton onClick={onToggleSettings} label="Settings">
          ⚙
        </IconButton>
        <IconButton onClick={onToggleFullscreen} label={fullscreen ? "Exit fullscreen" : "Fullscreen"}>
          {fullscreen ? "⤡" : "⤢"}
        </IconButton>
      </div>
    </div>
  );
}

function IconButton({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className="h-8 w-8 rounded-sm text-[15px] text-night-dim transition-colors hover:bg-night-3 hover:text-night-ink"
    >
      {children}
    </button>
  );
}
```

- [ ] **Step 3: ModelPicker dark variants**

In `src/components/ModelPicker.tsx`, line 40, change
```tsx
    return <span className="label-mono text-state-error">/v1/models {error}</span>;
```
to
```tsx
    return <span className="label-mono text-state-error dark:text-red-400">/v1/models {error}</span>;
```
and on line 59 append the dark classes to the `<select>`'s `className` string so it reads:
```tsx
        className="font-mono text-[13px] text-ink bg-paper border border-rule rounded-sm px-2 py-1 focus:outline-none focus:border-persian hover:border-pine transition-colors dark:bg-night-2 dark:border-night-rule dark:text-night-ink dark:hover:border-mint"
```
Nothing else in the file changes.

- [ ] **Step 4: The page**

Replace `src/pages/Streams.tsx` with:
```tsx
import { useEffect, useRef, useState } from "react";

import { StreamTile } from "@/components/streams/StreamTile";
import { StreamToolbar } from "@/components/streams/StreamToolbar";
import { useStats } from "@/hooks/useStats";
import { useStreamRunner } from "@/hooks/useStreamRunner";
import { chooseColumns } from "@/lib/gridLayout";

/**
 * The streams showcase: many autonomous prompts streaming at once on a dark,
 * full-bleed wall of terminal tiles. The runner is owned by this page, so
 * navigating away stops every stream (spec §2).
 */
export function Streams() {
  const { snapshot, runner } = useStreamRunner();
  const stats = useStats();
  const pageRef = useRef<HTMLDivElement | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 });
  const [fullscreen, setFullscreen] = useState(false);

  // The column count depends on the grid's box (content box; padding
  // excluded), so measure it and re-measure on resize.
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setGridSize({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === pageRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // Fullscreen the page element, not the document, so the nav drops out
  // while presenting. Esc exits (browser default).
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void pageRef.current?.requestFullscreen();
  };

  const n = snapshot.streams.length;
  const cols = chooseColumns(n, gridSize.width, gridSize.height);
  const rows = Math.max(1, Math.ceil(n / cols));
  const emptyText = snapshot.model
    ? snapshot.aggregate.running
      ? "starting…"
      : "press Play"
    : "waiting for a model";

  return (
    <div ref={pageRef} className="relative flex min-h-0 flex-1 flex-col bg-night text-night-ink">
      <StreamToolbar
        snapshot={snapshot}
        runner={runner}
        serverStats={stats.kind === "ready" ? stats.data : null}
        fullscreen={fullscreen}
        onToggleFullscreen={toggleFullscreen}
        onToggleSettings={() => undefined /* drawer lands in Task 6 */}
      />
      <div ref={gridRef} className="min-h-0 flex-1 overflow-auto p-3">
        <div
          className="grid h-full gap-3"
          style={{
            gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${rows}, minmax(160px, 1fr))`,
          }}
        >
          {snapshot.streams.map((s) => (
            <StreamTile
              key={s.id}
              stream={s}
              running={snapshot.aggregate.running}
              emptyText={emptyText}
              runner={runner}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Verify in the browser**

```bash
npm run typecheck
```
Expected: no output. Then at `http://localhost:5173/streams` with the mock server from Task 0 (cap 4) running:

1. **Layout.** 16 tiles in a 4 × 4 grid filling the viewport under the toolbar, no page scrollbar, each tile reads `#01`–`#16`, `idle`, body `press Play`. The toolbar shows the model name from the picker and `Server 0 / 4`.
2. **Play.** Click Play. Tiles light up one every ~300 ms. Because the mock replies instantly, most tiles cycle prefill → streaming → cooldown quickly; with cap 4 you will see `queued` tiles in amber with `waiting for a slot… (attempt N)`. `Server` shows up to `4 / 4`. `Active`, `Replies` and `Tokens` climb. `Agg tok/s` shows a number after the first tokens.
3. **Scrollback.** After a tile's second prompt, the first exchange appears above it dimmed; after four, only the last three are kept.
4. **Stop.** Click Stop. Every tile goes `idle` immediately, text stays on screen, `Agg tok/s` freezes at its last value, `Server` returns to `0 / 4` within a second.
5. **Pause / Skip.** Play again. Hover a tile: Pause and Skip appear top-right. Click Pause on a streaming tile → `pausing`, then `paused` after the reply; the tile no longer cycles. Hover it: the button reads Resume; click it → it cycles again. Click Skip on a cooldown tile → it starts a new prompt with no wait.
6. **Pause while stopped.** Stop, hover a tile, click Pause → `paused`. Play → that tile stays `paused` while the others run. Resume it.
7. **Unmount stops.** With streams running, click Chat in the nav, then Streams again. Tiles are back to `idle` with fresh state; `Server` drops to `0 / 4` right after leaving.
8. **Fullscreen.** Click the fullscreen button: the tile wall fills the screen with no nav; press Esc: back to normal, button label restored.
9. **Server down.** Stop the mock server while streams run. Tiles show `error` with `server unreachable`, retrying with growing gaps (watch the network tab: ~1 s, 2 s, 4 s, 8 s, 10 s, 10 s…). Restart the server: tiles recover on their next retry without pressing anything.
10. **Console.** No errors or warnings in the browser console during any of the above.

- [ ] **Step 6: Commit**

```bash
git add src/components/streams/StreamTile.tsx src/components/streams/StreamToolbar.tsx src/components/ModelPicker.tsx src/pages/Streams.tsx
git commit -m "feat(dashboard): streams showcase page with terminal tiles, toolbar and fullscreen"
```

---

### Task 6: Settings drawer

Implements spec §6.4. Live stream-count changes, per-prompt application of the other fields, localStorage persistence, server-cap hint.

**Files:**
- Create: `src/components/streams/StreamSettingsDrawer.tsx`
- Modify: `src/pages/Streams.tsx` (replace whole file with the version below)

**Interfaces:**
- Consumes: `StreamSettings`, `DEFAULT_SETTINGS`, `LIMITS`, `clampSettings`, `saveSettings` (Task 3); `runner.applySettings` (Task 4); `StreamToolbar`, `StreamTile` (Task 5).
- Produces: `StreamSettingsDrawer({ open: boolean; settings: StreamSettings; serverMax: number | null; onChange: (next: StreamSettings) => void; onClose: () => void })`

- [ ] **Step 1: The drawer**

Create `src/components/streams/StreamSettingsDrawer.tsx`:
```tsx
import { useEffect } from "react";

import { clampSettings, DEFAULT_SETTINGS, LIMITS, type StreamSettings } from "@/lib/streamSettings";

type Props = {
  open: boolean;
  settings: StreamSettings;
  /** The server's admission cap from /api/stats, or null while unreachable. */
  serverMax: number | null;
  onChange: (next: StreamSettings) => void;
  onClose: () => void;
};

/**
 * Right-anchored panel over the grid. Every edit goes through
 * `clampSettings`, so the parent always receives a valid settings object
 * and the inputs snap back into range as you type.
 */
export function StreamSettingsDrawer({ open, settings, serverMax, onChange, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const set = (patch: Partial<StreamSettings>) => onChange(clampSettings({ ...settings, ...patch }));
  const aboveCap = serverMax != null && settings.streamCount > serverMax;

  return (
    <aside
      role="dialog"
      aria-label="Stream settings"
      className="absolute inset-y-0 right-0 z-20 flex w-80 flex-col gap-5 border-l border-night-rule bg-night-2 p-5 shadow-elev"
    >
      <div className="flex items-center justify-between">
        <span className="label-mono text-night-dim">Settings</span>
        <button
          onClick={onClose}
          aria-label="Close settings"
          className="text-[18px] leading-none text-night-dim hover:text-night-ink"
        >
          ×
        </button>
      </div>

      <Field
        label="Streams"
        value={settings.streamCount}
        min={LIMITS.streamCount.min}
        max={LIMITS.streamCount.max}
        step={1}
        onChange={(v) => set({ streamCount: v })}
      />
      <Field
        label="Max tokens per reply"
        value={settings.maxTokens}
        min={LIMITS.maxTokens.min}
        max={LIMITS.maxTokens.max}
        step={8}
        onChange={(v) => set({ maxTokens: v })}
      />
      <Field
        label="Cooldown min (s)"
        value={settings.cooldownMinS}
        min={LIMITS.cooldownS.min}
        max={LIMITS.cooldownS.max}
        step={0.5}
        onChange={(v) => set({ cooldownMinS: v })}
      />
      <Field
        label="Cooldown max (s)"
        value={settings.cooldownMaxS}
        min={settings.cooldownMinS}
        max={LIMITS.cooldownS.max}
        step={0.5}
        onChange={(v) => set({ cooldownMaxS: v })}
      />
      <Field
        label="Temperature"
        value={settings.temperature}
        min={LIMITS.temperature.min}
        max={LIMITS.temperature.max}
        step={0.1}
        onChange={(v) => set({ temperature: v })}
      />

      <p className={`font-mono text-[11px] leading-relaxed ${aboveCap ? "text-amber-400" : "text-night-low"}`}>
        {serverMax != null
          ? `Server admits ${serverMax} concurrent requests (CASCADIA_API_MAX_CONCURRENT).${
              aboveCap ? " Streams above the cap will queue." : ""
            }`
          : "Server cap unknown (/api/stats unreachable)."}
      </p>

      <button
        onClick={() => onChange({ ...DEFAULT_SETTINGS })}
        className="label-mono self-start text-night-dim transition-colors hover:text-mint-bright"
      >
        Reset to defaults
      </button>
    </aside>
  );
}

function Field({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label-mono text-night-dim">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = e.target.valueAsNumber;
          if (Number.isFinite(v)) onChange(v);
        }}
        className="rounded-sm border border-night-rule bg-night px-2 py-1 font-mono text-[13px] text-night-ink focus:border-persian focus:outline-none"
      />
    </label>
  );
}
```

- [ ] **Step 2: Wire it into the page**

Replace `src/pages/Streams.tsx` with:
```tsx
import { useEffect, useRef, useState } from "react";

import { StreamSettingsDrawer } from "@/components/streams/StreamSettingsDrawer";
import { StreamTile } from "@/components/streams/StreamTile";
import { StreamToolbar } from "@/components/streams/StreamToolbar";
import { useStats } from "@/hooks/useStats";
import { useStreamRunner } from "@/hooks/useStreamRunner";
import { chooseColumns } from "@/lib/gridLayout";
import { clampSettings, saveSettings, type StreamSettings } from "@/lib/streamSettings";

/**
 * The streams showcase: many autonomous prompts streaming at once on a dark,
 * full-bleed wall of terminal tiles. The runner is owned by this page, so
 * navigating away stops every stream (spec §2).
 */
export function Streams() {
  const { snapshot, runner } = useStreamRunner();
  const stats = useStats();
  const pageRef = useRef<HTMLDivElement | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 });
  const [fullscreen, setFullscreen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // The column count depends on the grid's box (content box; padding
  // excluded), so measure it and re-measure on resize.
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setGridSize({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === pageRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // Fullscreen the page element, not the document, so the nav drops out
  // while presenting. Esc exits (browser default).
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void pageRef.current?.requestFullscreen();
  };

  // The runner publishes its snapshot on the next frame, so persist the
  // clamped value we hand it rather than reading it back.
  const onSettingsChange = (next: StreamSettings) => {
    const clamped = clampSettings(next);
    runner.applySettings(clamped);
    saveSettings(clamped);
  };

  const n = snapshot.streams.length;
  const cols = chooseColumns(n, gridSize.width, gridSize.height);
  const rows = Math.max(1, Math.ceil(n / cols));
  const emptyText = snapshot.model
    ? snapshot.aggregate.running
      ? "starting…"
      : "press Play"
    : "waiting for a model";
  const serverStats = stats.kind === "ready" ? stats.data : null;

  return (
    <div ref={pageRef} className="relative flex min-h-0 flex-1 flex-col bg-night text-night-ink">
      <StreamToolbar
        snapshot={snapshot}
        runner={runner}
        serverStats={serverStats}
        fullscreen={fullscreen}
        onToggleFullscreen={toggleFullscreen}
        onToggleSettings={() => setSettingsOpen((v) => !v)}
      />
      <div ref={gridRef} className="min-h-0 flex-1 overflow-auto p-3">
        <div
          className="grid h-full gap-3"
          style={{
            gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${rows}, minmax(160px, 1fr))`,
          }}
        >
          {snapshot.streams.map((s) => (
            <StreamTile
              key={s.id}
              stream={s}
              running={snapshot.aggregate.running}
              emptyText={emptyText}
              runner={runner}
            />
          ))}
        </div>
      </div>
      <StreamSettingsDrawer
        open={settingsOpen}
        settings={snapshot.settings}
        serverMax={serverStats ? serverStats.max_concurrent : null}
        onChange={onSettingsChange}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
```

- [ ] **Step 3: Verify in the browser**

```bash
npm run typecheck
```
Expected: no output. Then at `/streams`:

1. **Open / close.** Click the gear: a 320 px panel slides over the right of the grid with five fields showing 16 / 160 / 1 / 2 / 0 and the line `Server admits 4 concurrent requests (CASCADIA_API_MAX_CONCURRENT). Streams above the cap will queue.` in amber. Press Esc: it closes. Click the gear twice: opens, closes.
2. **Live stream count.** With streams running, set Streams to 20: four new tiles appear and start (staggered) while the others keep going; the grid re-flows to 5 × 4. Set it to 6: the top-index tiles vanish mid-stream and `Server` drops accordingly. Set it to 3: the amber line turns grey (below the cap of 4).
3. **Clamping.** Type 0 in Streams → snaps to 1. Type 99 → 64 (the grid becomes scrollable once tiles would be under 160 px tall). Set Cooldown min to 5 → Cooldown max snaps up to 5. Temperature 3 → 1.5.
4. **Per-prompt fields.** Set Max tokens to 16 while running: on each tile's *next* prompt the mock echo is cut at 16 words; the in-flight reply finishes at its old length.
5. **Persistence.** Set Streams to 9, reload the page: 9 tiles. Open DevTools → Application → Local Storage: key `cascadia.streams.settings.v1` holds the JSON. Click Reset to defaults: 16 tiles, stored JSON back to defaults.
6. **Stale storage.** In the console run `localStorage.setItem("cascadia.streams.settings.v1", "{\"streamCount\":500,\"cooldownMaxS\":-3}")` and reload: 64 tiles, cooldown max shows 1 (clamped up to min), no console error. Run `localStorage.setItem("cascadia.streams.settings.v1", "not json")` and reload: defaults, no console error. Click Reset to defaults afterwards.

- [ ] **Step 4: Commit**

```bash
git add src/components/streams/StreamSettingsDrawer.tsx src/pages/Streams.tsx
git commit -m "feat(dashboard): streams settings drawer with live count changes and persistence"
```

---

### Task 7: README, production build, embed check

Implements spec §10 (README) and §11 (build gates).

**Files:**
- Modify: `README.md:164`

- [ ] **Step 1: README clause**

On `README.md` line 164, change
```
cluster topology with per-link latency/bandwidth, live request/token counters, and a chat surface.
```
to
```
cluster topology with per-link latency/bandwidth, live request/token counters, a chat surface, and a streams showcase that runs many autonomous prompts at once.
```
Nothing else on the line changes.

- [ ] **Step 2: Production build**

```bash
cd crates/cascadia-dashboard/web
npm run typecheck
npm run build
```
Expected: typecheck silent; build ends `✓ built in …` with `dist/index.html` and hashed assets. Serve it once to make sure the production bundle (not just dev) works: `npm run preview` then open the printed URL + `/streams` — note the preview server has no API proxy, so the picker shows an error and Play stays disabled, but the dark shell, 16 idle tiles and the settings drawer must all render. Ctrl-C the preview.

Then package the built SPA the way the fleet's build machine expects it. Its
build script untars this file over `crates/cascadia-dashboard/web`, so the
tarball's top-level entry must be `dist/` (confirmed 2026-09-20 against the
tarball on the build machine, which lists `dist/index.html`, `dist/assets/…`):
```bash
cd crates/cascadia-dashboard/web
tar -czf "$HOME/dash-dist.tar.gz" dist
tar -tzf "$HOME/dash-dist.tar.gz" | head -3    # dist/  dist/index.html  dist/assets/…
```
Do not commit the tarball or `dist/`; `dist/` is gitignored and the tarball
lives outside the repo. Tell the user where it is.

- [ ] **Step 3: Embed check**

From the repo root:
```bash
cargo build -p cascadia-dashboard --features embed-spa
cargo test -p cascadia-dashboard --features embed-spa
```
Expected: both succeed (the crate's existing tests cover `/chat` resolving to the shell; `/streams` takes the same extension-less fallback path). The full binary with the UI baked in is `cargo build --release -p cascadia --features dashboard-embed`; run it only if you intend to deploy from this machine, it takes a while.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: mention the streams showcase in the dashboard section"
```

- [ ] **Step 5: Final state**

```bash
git status --porcelain     # empty
git log --oneline feat/inkling-multistream..HEAD
```
Expected: the spec/plan docs commit(s) plus seven feature commits (Tasks 1–7), nothing unstaged. Do not push. Hand over to the manual test checklist at `docs/superpowers/plans/2026-09-20-streams-showcase-manual-test.md`: Part A runs locally, Part B runs against the live fleet through the operator tunnel with no release, Part C ships the `~/dash-dist.tar.gz` from Step 2 above.

---

## Post-plan additions (implemented 2026-09-21, after Task 7)

Two commits landed on top of the plan at the user's request; the spec was
updated to match (§2, §6.3, §6.4, §7.1, §7.4, §8).

- `45f20a75 feat(dashboard): streams tile display toggles, font size, and full-response mode`
  - `src/lib/streamSettings.ts`: `showTileHeader`, `showTileFooter`,
    `fontSizePx` (9–20), `streamResponses`; `clampBool` for the booleans.
  - `src/lib/sse.ts`: `chatComplete()` (`stream:false`), same `HttpError`
    contract as `chatStream`.
  - `src/lib/streamRunner.ts`: `request()` branches on
    `settings.streamResponses`; the non-streaming branch keeps the tile in
    `prefill`, leaves TTFT null and rates the whole reply over the wait.
  - `src/components/streams/StreamTile.tsx`: `showHeader`, `showFooter`,
    `fontSizePx` props; header and footer render conditionally.
  - `src/components/streams/StreamSettingsDrawer.tsx`: font-size field and
    three checkboxes (with a hint on the streaming one); drawer scrolls.
  - `src/pages/Streams.tsx`: passes the three tile props from settings.
- `2b21634a feat(dashboard): default streams tiles to hidden header/footer and 10px font`
  - `DEFAULT_SETTINGS`: `showTileHeader: false`, `showTileFooter: false`,
    `fontSizePx: 10`.

The shipped bundle for this state is `index-Ba7ZRF0b.js` (see the manual
test checklist, Part C).
