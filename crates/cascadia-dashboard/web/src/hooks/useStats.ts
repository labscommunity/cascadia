import { api } from "@/lib/api";

import { usePoll } from "./usePoll";

/** Polls `/api/stats`. Cheap atomic counters; 1 s is fine. */
export function useStats(intervalMs = 1000) {
  return usePoll(api.stats, intervalMs);
}
