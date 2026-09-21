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
