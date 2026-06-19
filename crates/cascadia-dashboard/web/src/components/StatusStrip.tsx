import type { Edge, NodeInfo, Stats } from "@/lib/api";
import { formatLatency } from "@/lib/format";

type Props = {
  nodes: NodeInfo[];
  edges: Edge[];
  stats: Stats | null;
};

/**
 * Editorial header strip — five mono stats laid out in a row. Borrowed
 * structurally from cascadia-website's "proof strip" pattern: large
 * tabular numbers, tiny uppercase mono labels, separated by hairlines
 * rather than card edges so the eye reads across.
 */
export function StatusStrip({ nodes, edges, stats }: Props) {
  const deviceTallies = nodes.reduce<Record<string, number>>((acc, n) => {
    acc[n.device] = (acc[n.device] ?? 0) + 1;
    return acc;
  }, {});
  const deviceLabel = Object.entries(deviceTallies)
    .sort((a, b) => b[1] - a[1])
    .map(([dev, n]) => `${n} ${dev}`)
    .join(" · ") || "—";

  // Median round-trip latency across measured edges — gives one number
  // that summarises whether the cluster's interconnect is healthy. NaN
  // when no edges yet (probe loop hasn't populated any).
  const sortedLatencies = edges.map((e) => e.latency_ms).sort((a, b) => a - b);
  const medianLatency = sortedLatencies.length
    ? sortedLatencies[Math.floor(sortedLatencies.length / 2)]
    : null;

  const cells: { label: string; value: string }[] = [
    { label: "Nodes", value: nodes.length.toString() },
    { label: "Devices", value: deviceLabel },
    {
      label: "Median RTT",
      value: medianLatency != null ? formatLatency(medianLatency) : "—",
    },
    {
      label: "In flight",
      value: stats ? stats.requests_in_flight.toString() : "—",
    },
    {
      label: "Tokens",
      value: stats ? stats.tokens_total.toLocaleString() : "—",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-y-6 gap-x-8 border-y border-rule py-6">
      {cells.map((c) => (
        <div key={c.label} className="border-l border-rule-2 pl-4 first:border-l-0 first:pl-0">
          <div className="label-mono">{c.label}</div>
          <div className="font-mono text-[24px] tabular-nums text-ink mt-1">{c.value}</div>
        </div>
      ))}
    </div>
  );
}
