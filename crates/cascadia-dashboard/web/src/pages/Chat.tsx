import { ChatSurface } from "@/components/ChatSurface";

export function Chat() {
  return (
    <div className="max-w-container mx-auto px-6 py-12 space-y-10">
      <header className="space-y-3 max-w-3xl">
        <div className="label-mono">Playground</div>
        <h1 className="font-display text-display text-ink">Chat</h1>
        <p className="text-ink-dim text-[17px] leading-relaxed">
          Send a prompt to the model the coordinator currently has loaded. The
          response streams over SSE chunk-by-chunk; the footer surfaces decode
          tokens/sec, time-to-first-token, and end-to-end elapsed time so you
          can tell at a glance whether the cluster is hitting its target.
        </p>
      </header>

      <ChatSurface />
    </div>
  );
}
