/**
 * Column count for `n` tiles in a `width × height` box.
 *
 * `auto-fit` grids leave a ragged last row (6, 6, 6, 2), which looks broken
 * on a wall of terminals. Instead: take the ideal square-ish count
 * `sqrt(n · width / height)`, try the integers within ±1 of it, and keep the
 * one that leaves the fewest empty cells; on a tie prefer the tile aspect
 * ratio closest to 1.6 (a comfortable terminal shape). With 20 tiles on a
 * 16:9 screen this yields 5 × 4; with 16, 4 × 4; with 10, 5 × 2.
 */
export function chooseColumns(n: number, width: number, height: number): number {
  if (n <= 1) return 1;
  if (width <= 0 || height <= 0) return Math.ceil(Math.sqrt(n));

  const ideal = Math.sqrt((n * width) / height);
  const lo = Math.min(n, Math.max(1, Math.floor(ideal) - 1));
  const hi = Math.min(n, Math.max(lo, Math.ceil(ideal) + 1));

  let best = lo;
  let bestEmpty = Number.POSITIVE_INFINITY;
  let bestAspect = Number.POSITIVE_INFINITY;
  for (let cols = lo; cols <= hi; cols++) {
    const rows = Math.ceil(n / cols);
    const empty = cols * rows - n;
    const aspect = Math.abs(width / cols / (height / rows) - 1.6);
    if (empty < bestEmpty || (empty === bestEmpty && aspect < bestAspect)) {
      best = cols;
      bestEmpty = empty;
      bestAspect = aspect;
    }
  }
  return best;
}
