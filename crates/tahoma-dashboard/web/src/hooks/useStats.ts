import { useEffect, useState } from "react";

import { api, type Stats } from "@/lib/api";

type State =
  | { kind: "loading" }
  | { kind: "ready"; data: Stats }
  | { kind: "error"; error: Error };

/** Polls `/api/stats`. Cheap atomic counters; 1 s is fine. */
export function useStats(intervalMs = 1000) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const data = await api.stats();
        if (!alive) return;
        setState({ kind: "ready", data });
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
