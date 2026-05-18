import { useEffect, useMemo, useState } from "react";

import { LatencyMatrix } from "@/components/LatencyMatrix";
import { NodeCard } from "@/components/NodeCard";
import { StatusStrip } from "@/components/StatusStrip";
import { useStats } from "@/hooks/useStats";
import { useTopology } from "@/hooks/useTopology";

export function Dashboard() {
  const topology = useTopology();
  const stats = useStats();

  // Refresh the "freshness clock" once per second so stale-node detection
  // and "last seen Xs ago" decay smoothly between topology fetches.
  const now = useNow(1000);

  const { nodes, edges } = useMemo(() => {
    if (topology.kind !== "ready") return { nodes: [], edges: [] };
    return topology.data;
  }, [topology]);

  return (
    <div className="max-w-container mx-auto px-6 py-12 space-y-12">
      <header className="space-y-3 max-w-3xl">
        <div className="label-mono">Cluster overview</div>
        <h1 className="font-display text-display text-ink">Nodes &amp; latencies</h1>
        <p className="text-ink-dim text-[17px] leading-relaxed">
          Live view of every Intel device this Tahoma coordinator has seen.
          Latency and bandwidth come from measured probes between peers — not
          declared specs — so placement decisions can lean on what the network
          actually does.
        </p>
      </header>

      <StatusStrip nodes={nodes} stats={stats.kind === "ready" ? stats.data : null} now={now} />

      <section className="space-y-4">
        <SectionHeader label="Nodes" hint={`${nodes.length} discovered`} />
        {topology.kind === "loading" ? (
          <SurfaceMessage>contacting coordinator…</SurfaceMessage>
        ) : topology.kind === "error" ? (
          <SurfaceMessage tone="error">
            /api/topology unreachable — is `tahoma worker --api` running?
          </SurfaceMessage>
        ) : nodes.length === 0 ? (
          <SurfaceMessage>
            No peers have registered yet. Discovery is mDNS-based; nodes appear
            here within ~2 s of joining the network.
          </SurfaceMessage>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...nodes]
              .sort((a, b) => a.node_id.localeCompare(b.node_id))
              .map((n) => (
                <NodeCard key={n.node_id} node={n} now={now} />
              ))}
          </div>
        )}
      </section>

      <section className="space-y-4">
        <SectionHeader
          label="Latency matrix"
          hint="measured · per directed edge"
        />
        <div className="surface p-6">
          <LatencyMatrix nodes={nodes} edges={edges} />
        </div>
      </section>
    </div>
  );
}

function SectionHeader({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <h2 className="font-display text-h3 text-ink">{label}</h2>
      {hint ? <span className="label-mono">{hint}</span> : null}
    </div>
  );
}

function SurfaceMessage({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "error";
}) {
  const toneClass = tone === "error" ? "text-state-error" : "text-ink-low";
  return (
    <div className={`surface p-8 font-mono text-[13px] ${toneClass}`}>{children}</div>
  );
}

function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}
