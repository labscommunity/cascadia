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
      className="absolute inset-y-0 right-0 z-20 flex w-80 flex-col gap-5 overflow-y-auto border-l border-night-rule bg-night-2 p-5 shadow-elev"
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
      <Field
        label="Terminal font size (px)"
        value={settings.fontSizePx}
        min={LIMITS.fontSizePx.min}
        max={LIMITS.fontSizePx.max}
        step={0.5}
        onChange={(v) => set({ fontSizePx: v })}
      />

      <Checkbox
        label="Show tile header"
        checked={settings.showTileHeader}
        onChange={(v) => set({ showTileHeader: v })}
      />
      <Checkbox
        label="Show tile footer"
        checked={settings.showTileFooter}
        onChange={(v) => set({ showTileFooter: v })}
      />
      <Checkbox
        label="Stream tokens as they arrive"
        checked={settings.streamResponses}
        onChange={(v) => set({ streamResponses: v })}
        hint="Off = fetch the whole reply in one request; the tile shows it once the model finishes."
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

function Checkbox({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <label className="flex cursor-pointer flex-col gap-1">
      <span className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 rounded-sm border border-night-rule bg-night accent-mint-bright"
        />
        <span className="label-mono text-night-dim">{label}</span>
      </span>
      {hint ? (
        <span className="pl-6 font-mono text-[11px] leading-relaxed text-night-low">{hint}</span>
      ) : null}
    </label>
  );
}
