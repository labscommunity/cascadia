import { api } from "@/lib/api";

import { usePoll } from "./usePoll";

/**
 * Polls `/api/topology` on an interval. SSE delta updates will replace
 * polling in a follow-up commit; until then 2 s is a reasonable trade
 * between cluster-event latency and request load (the endpoint is
 * read-only and reads a HashMap behind an RwLock — cheap).
 */
export function useTopology(intervalMs = 2000) {
  return usePoll(api.topology, intervalMs);
}
