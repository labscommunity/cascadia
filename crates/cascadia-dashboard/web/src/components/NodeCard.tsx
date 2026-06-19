import type { NodeInfo } from "@/lib/api";
import { humanizeMemory, relativeAge } from "@/lib/format";

import { DeviceChip } from "./DeviceChip";

export function NodeCard({ node, now }: { node: NodeInfo; now: number }) {
  // Presence in topology is the liveness signal — the mDNS daemon
  // removes peers from this list when TTL expires, so anything we render
  // is by definition currently being heard from. The `last_seen` field
  // is now decorative ("when we last got a fresh resolution event")
  // because mdns-sd suppresses ServiceResolved for already-known peers,
  // making it unreliable as a freshness proxy.
  return (
    <article className="surface p-5 flex flex-col gap-4 hover:shadow-elev transition-shadow">
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-mono text-[15px] text-ink truncate" title={node.node_id}>
            {node.node_id}
          </h3>
          <p className="font-mono text-[12px] text-ink-low truncate" title={`${node.host}:${node.port}`}>
            {node.host}:{node.port}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="pulse-dot" aria-label="live" />
          <DeviceChip device={node.device} />
        </div>
      </header>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
        <div className="col-span-2 min-w-0">
          <dt className="label-mono">CPU</dt>
          <dd className="font-mono text-[13px] text-ink mt-0.5 truncate" title={node.cpu_model || undefined}>
            {node.cpu_model || "—"}
            {node.cpu_cores ? <span className="text-ink-low"> · {node.cpu_cores} cores</span> : null}
          </dd>
        </div>
        <div>
          <dt className="label-mono">Memory</dt>
          <dd className="font-mono text-[13px] text-ink mt-0.5">{humanizeMemory(node.memory_mb)}</dd>
        </div>
        <div className="min-w-0">
          <dt className="label-mono">OS</dt>
          <dd className="font-mono text-[13px] text-ink mt-0.5 truncate" title={node.os || undefined}>
            {node.os || "—"}
          </dd>
        </div>
        <div className="col-span-2">
          <dt className="label-mono">Namespace</dt>
          <dd className="font-mono text-[13px] text-ink mt-0.5">{node.namespace}</dd>
        </div>
      </dl>

      <div>
        <div className="label-mono mb-1.5">Engines</div>
        {node.engines.length === 0 ? (
          <div className="font-mono text-[12px] text-ink-low">—</div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {node.engines.map((e) => (
              <span
                key={e}
                className="inline-flex px-1.5 py-0.5 rounded-sm bg-paper-2 font-mono text-[11px] text-ink-dim border border-rule-2"
              >
                {e}
              </span>
            ))}
          </div>
        )}
      </div>

      <footer className="border-t border-rule-2 pt-2 -mx-1 px-1 flex items-center justify-between label-mono">
        <span>last seen</span>
        <span className="text-ink-dim">{relativeAge(node.last_seen, now)}</span>
      </footer>
    </article>
  );
}
