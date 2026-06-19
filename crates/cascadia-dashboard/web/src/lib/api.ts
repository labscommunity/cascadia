// Typed client for the dashboard + OpenAI-compat endpoints exposed by the
// cascadia binary. Wire shapes mirror `crates/cascadia-dashboard/src/lib.rs` and
// `crates/cascadia-api/src/lib.rs`; bump these together when either side
// changes.

export type NodeInfo = {
  node_id: string;
  host: string;
  port: number;
  namespace: string;
  device: string;
  memory_mb: number;
  /** CPU brand string, e.g. "Intel(R) Core(TM) Ultra 7 258V". "" if unknown. */
  cpu_model: string;
  /** Logical CPU count. 0 if unknown. */
  cpu_cores: number;
  /** OS label, e.g. "Windows 11" / "Darwin 25.5". "" if unknown. */
  os: string;
  engines: string[];
  /** Unix seconds, fractional. Compared against Date.now()/1000 for freshness. */
  last_seen: number;
};

export type Edge = {
  src: string;
  dst: string;
  latency_ms: number;
  bandwidth_mbps: number;
  last_measured: number;
};

export type TopologyResponse = {
  nodes: NodeInfo[];
  edges: Edge[];
};

export type Stats = {
  requests_total: number;
  requests_in_flight: number;
  tokens_total: number;
  max_concurrent: number;
};

export type ModelEntry = {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
};

export type ModelsResponse = {
  object: "list";
  data: ModelEntry[];
};

async function json<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(`${path} returned ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  topology: () => json<TopologyResponse>("/api/topology"),
  stats: () => json<Stats>("/api/stats"),
  models: () => json<ModelsResponse>("/v1/models"),
};
