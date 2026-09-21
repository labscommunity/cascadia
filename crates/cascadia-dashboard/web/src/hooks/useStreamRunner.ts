import { useEffect, useRef, useSyncExternalStore } from "react";

import { type RunnerSnapshot, StreamRunner } from "@/lib/streamRunner";
import { loadSettings } from "@/lib/streamSettings";

/**
 * One runner per page mount. Leaving the page stops every stream (decided
 * in the spec): the unmount cleanup aborts them. `stop()` on a runner that
 * is not running is a no-op, so StrictMode's mount/unmount/mount in dev is
 * harmless.
 */
export function useStreamRunner(): { snapshot: RunnerSnapshot; runner: StreamRunner } {
  const ref = useRef<StreamRunner | null>(null);
  if (ref.current === null) ref.current = new StreamRunner(loadSettings(), "");
  const runner = ref.current;

  const snapshot = useSyncExternalStore(runner.subscribe, runner.getSnapshot);

  useEffect(() => () => runner.stop(), [runner]);

  return { snapshot, runner };
}
