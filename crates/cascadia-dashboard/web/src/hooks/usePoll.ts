import { useEffect, useRef, useState } from "react";

export type PollState<T> =
  | { kind: "loading" }
  | { kind: "ready"; data: T; fetchedAt: number }
  | { kind: "error"; error: Error };

/**
 * Poll an async fetcher on an interval, exposing a discriminated state.
 * Shared by `useTopology` / `useStats` (and `ClusterStrip`) so the
 * loading/ready/error + alive-flag + interval-cleanup logic lives in one
 * place. The fetcher is held in a ref so a new fetcher identity each
 * render doesn't re-subscribe the interval; only `intervalMs` does.
 */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number): PollState<T> {
  const [state, setState] = useState<PollState<T>>({ kind: "loading" });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const data = await fetcherRef.current();
        if (alive) setState({ kind: "ready", data, fetchedAt: Date.now() });
      } catch (e) {
        if (alive) setState({ kind: "error", error: e as Error });
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
