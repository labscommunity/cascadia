import { useEffect, useState } from "react";

import { api, type TopologyResponse } from "@/lib/api";

type State =
  | { kind: "loading" }
  | { kind: "ready"; data: TopologyResponse; fetchedAt: number }
  | { kind: "error"; error: Error };

/**
 * Polls `/api/topology` on an interval. SSE delta updates will replace
 * polling in a follow-up commit; until then 2 s is a reasonable trade
 * between cluster-event latency and request load (the endpoint is
 * read-only and reads a HashMap behind an RwLock — cheap).
 */
export function useTopology(intervalMs = 2000) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const data = await api.topology();
        if (!alive) return;
        setState({ kind: "ready", data, fetchedAt: Date.now() });
      } catch (e) {
        if (!alive) return;
        setState({ kind: "error", error: e as Error });
      }
    };
    void tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return state;
}
