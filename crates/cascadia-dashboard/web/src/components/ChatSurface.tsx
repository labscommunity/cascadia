import { useEffect, useRef, useState } from "react";

import { type ChatMessage, chatStream } from "@/lib/sse";

import { ClusterStrip } from "./ClusterStrip";
import { ModelPicker } from "./ModelPicker";
import { StatBar, type ChatStats } from "./StatBar";

export function ChatSurface() {
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [stats, setStats] = useState<ChatStats | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll the message column to the bottom when new tokens arrive
  // or a new turn starts. Done with scrollTop rather than ref-into-view
  // so it doesn't yank focus off the textarea.
  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Cancel any in-flight stream if this surface unmounts mid-generation.
  useEffect(() => () => abortRef.current?.abort(), []);

  const send = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || streaming || !model) return;

    const userTurn: ChatMessage = { role: "user", content: trimmed };
    const history = [...messages, userTurn];
    // Append the user turn and a placeholder assistant turn we'll fill in
    // as chunks arrive. Done in one setMessages so React commits both.
    setMessages([...history, { role: "assistant", content: "" }]);
    setPrompt("");
    setStreaming(true);
    setStats({ tokens: 0, ttftSeconds: null, elapsedSeconds: 0, tokensPerSecond: null });

    const startedAt = performance.now();
    let firstChunkAt: number | null = null;
    let totalTokens = 0;
    let assistantText = "";

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const chunk of chatStream({ model, messages: history }, controller.signal)) {
        // A mid-stream engine failure arrives as {object:"error", error:{…}}
        // with NO `choices`. Surface the engine's real reason (the catch
        // below renders it) instead of dereferencing a missing `choices`.
        if (chunk.object === "error") {
          throw new Error(chunk.error.message);
        }
        if (firstChunkAt === null) firstChunkAt = performance.now();
        totalTokens += chunk.n_tokens ?? 1;
        const delta = chunk.choices[0]?.delta.content ?? "";
        assistantText += delta;

        setMessages((prev) => {
          const next = prev.slice();
          next[next.length - 1] = { role: "assistant", content: assistantText };
          return next;
        });

        const now = performance.now();
        const elapsed = (now - startedAt) / 1000;
        const ttft = firstChunkAt != null ? (firstChunkAt - startedAt) / 1000 : null;
        // Decode tok/s — tokens AFTER the first (the first token's time is
        // TTFT, not decode) over the post-TTFT window. Hold off until >1
        // token AND >50 ms have elapsed so a tiny initial window can't
        // divide into an absurd thousands-of-tok/s reading.
        const decodeWindow = ttft != null ? elapsed - ttft : null;
        const decodeTokens = totalTokens - 1;
        const tokensPerSecond =
          decodeWindow != null && decodeWindow > 0.05 && decodeTokens > 0
            ? decodeTokens / decodeWindow
            : null;
        setStats({
          tokens: totalTokens,
          ttftSeconds: ttft,
          elapsedSeconds: elapsed,
          tokensPerSecond,
        });
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setMessages((prev) => {
        const next = prev.slice();
        const last = next[next.length - 1];
        const errLine = `\n\n[stream error: ${(err as Error).message}]`;
        if (last && last.role === "assistant") {
          next[next.length - 1] = { role: "assistant", content: last.content + errLine };
        }
        return next;
      });
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  };

  const cancel = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="surface flex flex-col">
      <header className="border-b border-rule-2 px-5 py-3 flex items-center justify-between gap-4">
        <ModelPicker value={model} onChange={setModel} />
        {streaming ? (
          <span className="flex items-center gap-2 label-mono text-pine">
            <span className="pulse-dot" aria-hidden />
            streaming
          </span>
        ) : (
          <span className="label-mono">idle</span>
        )}
      </header>

      <ClusterStrip />

      <div
        ref={scrollerRef}
        className="px-5 py-6 min-h-[420px] max-h-[60vh] overflow-y-auto"
      >
        {messages.length === 0 ? (
          <EmptyState modelReady={Boolean(model)} />
        ) : (
          <div className="space-y-5">
            {messages.map((m, i) => (
              <Bubble key={i} message={m} pending={i === messages.length - 1 && streaming} />
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-rule-2 px-5 py-3">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder={model ? "Ask anything…" : "waiting for a model"}
          rows={3}
          disabled={!model}
          className="w-full resize-none bg-transparent font-sans text-[15px] text-ink placeholder:text-ink-low focus:outline-none disabled:opacity-50"
        />
        <div className="mt-2 flex items-center justify-end gap-2">
          {streaming ? (
            <button
              onClick={cancel}
              className="px-4 py-1.5 rounded-sm border border-rule font-mono text-[11px] uppercase tracking-wider text-ink-dim hover:text-state-error hover:border-state-error transition-colors"
            >
              Cancel
            </button>
          ) : null}
          <button
            onClick={() => void send()}
            disabled={streaming || !model || !prompt.trim()}
            className="px-4 py-1.5 rounded-sm bg-ink text-paper font-mono text-[11px] uppercase tracking-wider hover:bg-pine disabled:bg-rule disabled:text-ink-low transition-colors"
          >
            Send
          </button>
        </div>
      </div>

      <StatBar stats={stats} />
    </div>
  );
}

function EmptyState({ modelReady }: { modelReady: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center text-center pt-12 pb-4 gap-3">
      <p className="font-display text-h4 text-ink-dim">
        {modelReady ? "Send a prompt to begin." : "Waiting for a model."}
      </p>
      <p className="font-mono text-[12px] text-ink-low max-w-md">
        Each turn is rendered live as the engine streams tokens. The stat bar
        below tracks decode throughput, time-to-first-token, and end-to-end
        elapsed time.
      </p>
    </div>
  );
}

function Bubble({ message, pending }: { message: ChatMessage; pending: boolean }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[72%] rounded-md bg-mint-wash text-ink px-4 py-2 font-sans text-[15px] whitespace-pre-wrap leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1">
      <div className="label-mono">Assistant</div>
      <div className="font-sans text-[15px] text-ink whitespace-pre-wrap leading-relaxed">
        {message.content ? (
          message.content
        ) : pending ? (
          <span className="text-ink-low">thinking…</span>
        ) : (
          <span className="text-ink-low italic">(empty)</span>
        )}
      </div>
    </div>
  );
}
