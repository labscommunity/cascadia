export function Chat() {
  return (
    <div className="max-w-container mx-auto px-6 py-12 space-y-12">
      <header className="space-y-3 max-w-3xl">
        <div className="label-mono">Playground</div>
        <h1 className="font-display text-display text-ink">Chat</h1>
        <p className="text-ink-dim text-[17px] leading-relaxed">
          Send a prompt to the model the coordinator currently has loaded. The
          response streams over SSE; a stat bar at the bottom surfaces
          tokens/sec, time-to-first-token, and end-to-end elapsed time.
        </p>
      </header>

      <section className="surface p-8">
        <div className="font-mono text-sm text-ink-low">
          chat interface renders in the next commit
        </div>
      </section>
    </div>
  );
}
