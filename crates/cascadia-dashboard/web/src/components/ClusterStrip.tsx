import { useTopology } from "@/hooks/useTopology";

import { DeviceChip } from "./DeviceChip";

/**
 * Horizontal strip of node chips above the chat surface, so the user
 * has situational awareness about which nodes are part of the cluster
 * serving their request. Today every node in the topology is shown
 * (we don't track pipeline-stage assignment yet); when we do, this is
 * the natural place to highlight the active path.
 */
export function ClusterStrip() {
  const topology = useTopology(3000);

  if (topology.kind !== "ready") {
    return (
      <div className="border-b border-rule-2 px-5 py-2.5 label-mono">
        Cluster: {topology.kind === "loading" ? "loading…" : "unreachable"}
      </div>
    );
  }

  const nodes = topology.data.nodes;
  if (nodes.length === 0) {
    return (
      <div className="border-b border-rule-2 px-5 py-2.5 label-mono">
        Cluster: no peers discovered
      </div>
    );
  }

  return (
    <div className="border-b border-rule-2 px-5 py-2.5 flex items-center gap-3 overflow-x-auto">
      <span className="label-mono shrink-0">Cluster</span>
      <div className="flex items-center gap-2 flex-nowrap">
        {[...nodes]
          .sort((a, b) => a.node_id.localeCompare(b.node_id))
          .map((n) => (
            <span
              key={n.node_id}
              className="flex items-center gap-2 px-2 py-1 rounded-sm bg-paper-2 border border-rule-2 shrink-0"
              title={`${n.host}:${n.port} · engines: ${n.engines.join(", ") || "—"}`}
            >
              <span className="pulse-dot" aria-hidden />
              <span className="font-mono text-[11px] text-ink">{n.node_id}</span>
              <DeviceChip device={n.device} />
            </span>
          ))}
      </div>
    </div>
  );
}
