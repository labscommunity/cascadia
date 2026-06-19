export type ChatStats = {
  tokens: number;
  /** Seconds from request send to the first streamed chunk. */
  ttftSeconds: number | null;
  /** Wall-clock elapsed since request send, in seconds. */
  elapsedSeconds: number;
  /** Decode tokens/sec (excludes TTFT — measures decode throughput). */
  tokensPerSecond: number | null;
};

/**
 * Bottom-of-card stat bar — modelled on tahoma-demo's four-cell footer.
 * Tabular numerics so the values don't jitter as digits change width.
 */
export function StatBar({ stats }: { stats: ChatStats | null }) {
  const cells: { label: string; value: string }[] = [
    {
      label: "Tok/s",
      value: stats?.tokensPerSecond != null ? stats.tokensPerSecond.toFixed(1) : "—",
    },
    {
      label: "Tokens",
      value: stats ? stats.tokens.toString() : "—",
    },
    {
      label: "TTFT",
      value: stats?.ttftSeconds != null ? `${(stats.ttftSeconds * 1000).toFixed(0)} ms` : "—",
    },
    {
      label: "Elapsed",
      value: stats ? `${stats.elapsedSeconds.toFixed(2)} s` : "—",
    },
  ];

  return (
    <div className="border-t border-rule-2 px-5 py-3 grid grid-cols-4 gap-4">
      {cells.map((c) => (
        <div key={c.label}>
          <div className="label-mono">{c.label}</div>
          <div className="font-mono text-[18px] tabular-nums text-ink mt-0.5">{c.value}</div>
        </div>
      ))}
    </div>
  );
}
