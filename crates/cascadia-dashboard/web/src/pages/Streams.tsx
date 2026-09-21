import { useEffect, useRef, useState } from "react";

import { StreamSettingsDrawer } from "@/components/streams/StreamSettingsDrawer";
import { StreamTile } from "@/components/streams/StreamTile";
import { StreamToolbar } from "@/components/streams/StreamToolbar";
import { useStats } from "@/hooks/useStats";
import { useStreamRunner } from "@/hooks/useStreamRunner";
import { chooseColumns } from "@/lib/gridLayout";
import { clampSettings, saveSettings, type StreamSettings } from "@/lib/streamSettings";

/**
 * The streams showcase: many autonomous prompts streaming at once on a dark,
 * full-bleed wall of terminal tiles. The runner is owned by this page, so
 * navigating away stops every stream (spec §2).
 */
export function Streams() {
  const { snapshot, runner } = useStreamRunner();
  const stats = useStats();
  const pageRef = useRef<HTMLDivElement | null>(null);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 });
  const [fullscreen, setFullscreen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // The column count depends on the grid's box (content box; padding
  // excluded), so measure it and re-measure on resize.
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setGridSize({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === pageRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // Fullscreen the page element, not the document, so the nav drops out
  // while presenting. Esc exits (browser default).
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void pageRef.current?.requestFullscreen();
  };

  // The runner publishes its snapshot on the next frame, so persist the
  // clamped value we hand it rather than reading it back.
  const onSettingsChange = (next: StreamSettings) => {
    const clamped = clampSettings(next);
    runner.applySettings(clamped);
    saveSettings(clamped);
  };

  const n = snapshot.streams.length;
  const cols = chooseColumns(n, gridSize.width, gridSize.height);
  const rows = Math.max(1, Math.ceil(n / cols));
  const emptyText = snapshot.model
    ? snapshot.aggregate.running
      ? "starting…"
      : "press Play"
    : "waiting for a model";
  const serverStats = stats.kind === "ready" ? stats.data : null;

  return (
    <div ref={pageRef} className="relative flex min-h-0 flex-1 flex-col bg-night text-night-ink">
      <StreamToolbar
        snapshot={snapshot}
        runner={runner}
        serverStats={serverStats}
        fullscreen={fullscreen}
        onToggleFullscreen={toggleFullscreen}
        onToggleSettings={() => setSettingsOpen((v) => !v)}
      />
      <div ref={gridRef} className="min-h-0 flex-1 overflow-auto p-3">
        <div
          className="grid h-full gap-3"
          style={{
            gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${rows}, minmax(160px, 1fr))`,
          }}
        >
          {snapshot.streams.map((s) => (
            <StreamTile
              key={s.id}
              stream={s}
              running={snapshot.aggregate.running}
              emptyText={emptyText}
              runner={runner}
            />
          ))}
        </div>
      </div>
      <StreamSettingsDrawer
        open={settingsOpen}
        settings={snapshot.settings}
        serverMax={serverStats ? serverStats.max_concurrent : null}
        onChange={onSettingsChange}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
