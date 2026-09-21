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
import { chatComplete, chatStream, type ChatStreamArgs, HttpError } from "@/lib/sse";
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
    const args: ChatStreamArgs = {
      model: this.model,
      messages: [{ role: "user", content: s.prompt ?? "" }],
      max_tokens: this.settings.maxTokens,
      temperature: this.settings.temperature,
    };
    try {
      if (this.settings.streamResponses) {
        const stream = chatStream(args, controller.signal);
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
      } else {
        // Full reply in one request: there is no first-token event, so TTFT
        // stays null and the rate is the whole reply over the whole wait. The
        // tile sits in prefill (prompt + caret) until the reply lands.
        const { text, tokens } = await chatComplete(args, controller.signal);
        if (!this.alive(slot, gen)) return { kind: "aborted" };
        const now = performance.now();
        s.reply = text;
        s.tokens = tokens;
        s.elapsedMs = now - startedAt;
        s.tokPerSec = s.elapsedMs > 50 ? tokens / (s.elapsedMs / 1000) : null;
        this.ring.push({ t: now, n: tokens });
        this.totalTokens += tokens;
        this.mark(slot);
      }
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
