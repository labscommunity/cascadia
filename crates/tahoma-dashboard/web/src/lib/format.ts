// Small presentational helpers shared by the dashboard surfaces.

export function humanizeMemory(mb: number): string {
  if (mb <= 0) return "—";
  if (mb >= 1024) {
    const gb = mb / 1024;
    return gb >= 100 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;
  }
  return `${mb} MB`;
}

/** "5s ago", "2m ago", "1h ago". Returns "now" for < 1s. */
export function relativeAge(unixSeconds: number, nowSeconds: number = Date.now() / 1000): string {
  const delta = Math.max(0, nowSeconds - unixSeconds);
  if (delta < 1) return "now";
  if (delta < 60) return `${delta.toFixed(0)}s ago`;
  if (delta < 3600) return `${(delta / 60).toFixed(0)}m ago`;
  return `${(delta / 3600).toFixed(0)}h ago`;
}

/** Considered "live" if we've heard from the node in the last 10 seconds. */
export const FRESH_THRESHOLD_S = 10;

export function isFresh(lastSeen: number, nowSeconds: number = Date.now() / 1000): boolean {
  return nowSeconds - lastSeen < FRESH_THRESHOLD_S;
}

/**
 * Color band for a measured edge latency. Tuned for Intel AI PC fleets:
 * within-switch links are sub-millisecond, across-switch a few ms, and
 * anything past ~20 ms is a WAN hop that hurts pipeline throughput (per
 * rainier's experiments — 50 ms WAN hop drops throughput ~65%).
 */
export type LatencyBand = "great" | "good" | "warm" | "bad";

export function latencyBand(latencyMs: number): LatencyBand {
  if (latencyMs < 1) return "great";
  if (latencyMs < 5) return "good";
  if (latencyMs < 20) return "warm";
  return "bad";
}

export function formatLatency(latencyMs: number): string {
  if (latencyMs < 1) return `${(latencyMs * 1000).toFixed(0)} µs`;
  if (latencyMs < 10) return `${latencyMs.toFixed(2)} ms`;
  return `${latencyMs.toFixed(1)} ms`;
}

export function formatBandwidth(mbps: number): string {
  if (mbps <= 0) return "—";
  if (mbps >= 1000) return `${(mbps / 1000).toFixed(1)} Gbps`;
  return `${mbps.toFixed(0)} Mbps`;
}
