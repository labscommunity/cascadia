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
  /** Show each tile's top bar (index, status, tok/s, hover Pause/Skip). */
  showTileHeader: boolean;
  /** Show each tile's bottom metrics bar (tokens, TTFT, elapsed). */
  showTileFooter: boolean;
  /** Body text size in px for the prompt/reply/scrollback. */
  fontSizePx: number;
  /** Stream tokens over SSE (true) or fetch the full reply at once (false). */
  streamResponses: boolean;
};

export const DEFAULT_SETTINGS: StreamSettings = {
  streamCount: 16,
  maxTokens: 160,
  cooldownMinS: 1,
  cooldownMaxS: 2,
  temperature: 0,
  showTileHeader: false,
  showTileFooter: false,
  fontSizePx: 10,
  streamResponses: true,
};

export const LIMITS = {
  streamCount: { min: 1, max: 64 },
  maxTokens: { min: 16, max: 1024 },
  cooldownS: { min: 0, max: 30 },
  temperature: { min: 0, max: 1.5 },
  fontSizePx: { min: 9, max: 20 },
} as const;

const STORAGE_KEY = "cascadia.streams.settings.v1";

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, n));
}

function clampBool(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
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
  const fontSizePx = clampNumber(
    src.fontSizePx,
    LIMITS.fontSizePx.min,
    LIMITS.fontSizePx.max,
    d.fontSizePx,
  );
  const showTileHeader = clampBool(src.showTileHeader, d.showTileHeader);
  const showTileFooter = clampBool(src.showTileFooter, d.showTileFooter);
  const streamResponses = clampBool(src.streamResponses, d.streamResponses);
  return {
    streamCount,
    maxTokens,
    cooldownMinS,
    cooldownMaxS,
    temperature,
    showTileHeader,
    showTileFooter,
    fontSizePx,
    streamResponses,
  };
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
