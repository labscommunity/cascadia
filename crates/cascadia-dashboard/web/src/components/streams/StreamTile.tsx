import { memo, useEffect, useRef } from "react";

import type { StreamRunner, StreamState, StreamStatus } from "@/lib/streamRunner";

type Props = {
  stream: StreamState;
  running: boolean;
  /** Body text for a tile that has never held a prompt. */
  emptyText: string;
  /** Show the top bar (index, status, tok/s, hover Pause/Skip). */
  showHeader: boolean;
  /** Show the bottom metrics bar (tokens, TTFT, elapsed). */
  showFooter: boolean;
  /** Body text size in px (prompt/reply/scrollback). */
  fontSizePx: number;
  runner: StreamRunner;
};

const DOT: Record<StreamStatus, string> = {
  idle: "bg-night-low",
  queued: "bg-amber-400",
  prefill: "bg-celadon animate-pulse",
  streaming: "bg-mint-bright",
  cooldown: "bg-night-dim",
  paused: "bg-night-low",
  error: "bg-red-400",
};

const WORD: Record<StreamStatus, string> = {
  idle: "text-night-low",
  queued: "text-amber-400",
  prefill: "text-celadon",
  streaming: "text-mint-bright",
  cooldown: "text-night-dim",
  paused: "text-night-low",
  error: "text-red-400",
};

/**
 * One terminal tile. Memoised: the runner hands out a new `stream` object
 * only for streams that changed since the last frame, so a token on stream
 * 3 re-renders tile 3 alone. `runner` and `emptyText` are stable props.
 */
export const StreamTile = memo(function StreamTile({
  stream,
  running,
  emptyText,
  showHeader,
  showFooter,
  fontSizePx,
  runner,
}: Props) {
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Keep the newest text in view. scrollTop rather than scrollIntoView so
  // the page itself never jumps.
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [stream.reply, stream.history, stream.status]);

  const inFlight = stream.status === "prefill" || stream.status === "streaming";
  const paused = stream.status === "paused";
  const statusWord = stream.pauseRequested ? "pausing" : stream.status;

  return (
    <div className="group flex min-h-0 flex-col overflow-hidden rounded-md border border-night-rule bg-night-2 font-mono text-[12.5px] leading-snug">
      {showHeader ? (
        <header className="flex h-7 shrink-0 items-center gap-2 bg-night-3 px-3">
          <span className="tabular-nums text-night-low">#{String(stream.id + 1).padStart(2, "0")}</span>
          <span className={`h-1.5 w-1.5 rounded-full ${DOT[stream.status]}`} aria-hidden />
          <span className={`label-mono ${WORD[stream.status]}`}>{statusWord}</span>
          <span className="ml-auto flex items-center gap-2">
            <span className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
              {paused ? (
                <TileButton onClick={() => runner.resumeStream(stream.id)}>Resume</TileButton>
              ) : (
                <TileButton onClick={() => runner.pauseStream(stream.id)}>Pause</TileButton>
              )}
              <TileButton onClick={() => runner.skipStream(stream.id)} disabled={!running || paused}>
                Skip
              </TileButton>
            </span>
            <span className="tabular-nums text-night-dim">
              {stream.tokPerSec != null ? `${stream.tokPerSec.toFixed(1)} tok/s` : "—"}
            </span>
          </span>
        </header>
      ) : null}

      <div
        ref={bodyRef}
        className="min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-2"
        style={{ fontSize: `${fontSizePx}px` }}
      >
        {stream.history.map((ex, i) => (
          <div key={i} className="whitespace-pre-wrap text-night-low">
            <div>&gt; {ex.prompt}</div>
            <div>{ex.reply}</div>
          </div>
        ))}
        {stream.prompt !== null ? (
          <div className="whitespace-pre-wrap">
            <div className="text-night-dim">&gt; {stream.prompt}</div>
            {stream.status === "queued" ? (
              <div className="text-amber-400">waiting for a slot… (attempt {stream.attempt})</div>
            ) : null}
            {stream.status === "error" ? <div className="text-red-400">{stream.error}</div> : null}
            <div className="text-night-ink">
              {stream.reply}
              {inFlight ? <span className="cursor-blink text-mint-bright">▌</span> : null}
            </div>
          </div>
        ) : (
          <div className="text-night-low">{emptyText}</div>
        )}
      </div>

      {showFooter ? (
        <footer className="label-mono flex h-6 shrink-0 items-center gap-3 border-t border-night-rule px-3 tabular-nums">
          <span>{stream.tokens} tok</span>
          <span>TTFT {stream.ttftMs != null ? `${stream.ttftMs.toFixed(0)} ms` : "—"}</span>
          <span className="ml-auto">{(stream.elapsedMs / 1000).toFixed(1)} s</span>
        </footer>
      ) : null}
    </div>
  );
});

function TileButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  children: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-sm border border-night-rule px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-night-dim transition-colors hover:border-mint hover:text-mint disabled:opacity-30 disabled:hover:border-night-rule disabled:hover:text-night-dim"
    >
      {children}
    </button>
  );
}
