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
