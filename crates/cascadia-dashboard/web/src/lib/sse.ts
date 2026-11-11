// Streaming consumer for cascadia-api's /v1/chat/completions SSE response.
//
// We hand-roll the parser rather than use EventSource because EventSource
// is GET-only — and OpenAI-shaped completions are POST. The wire format
// is "data: <json>\n\n", terminated by "data: [DONE]\n\n". Token counts come
// from `usage` (requested with `continuous_usage_stats`) because SSE event
// boundaries are not token boundaries: speculative engines emit several
// tokens per chunk.

export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ChatUsage = {
  prompt_tokens: number;
  /**
   * Running count while streaming; the last content chunk and the final
   * usage chunk carry the authoritative total, which can be lower.
   */
  completion_tokens: number;
  total_tokens: number;
};

export type ChatCompletionChunk = {
  id: string;
  object: "chat.completion.chunk";
  system_fingerprint?: string | null;
  /** Present on every chunk when `continuous_usage_stats` is requested. */
  usage?: ChatUsage | null;
  /** Empty on the final usage-only chunk. */
  choices: {
    index: number;
    /** `role` is only on the first delta. */
    delta: { role?: string; content?: string | null };
    logprobs?: unknown;
    finish_reason: string | null;
  }[];
};

/** Emitted by the server when a task fails mid-stream — note: NO `choices`. */
export type ChatErrorChunk = {
  id: string;
  object: "error";
  error: {
    message: string;
    type?: string;
    param?: string | null;
    code?: string | null;
  };
};

export type ChatStreamChunk = ChatCompletionChunk | ChatErrorChunk;

export type TokenCount = {
  /** Completion tokens so far; the last usage seen wins. */
  total: number;
  /**
   * The engine does not report per-chunk counts (a content chunk left the
   * running count unchanged, or no usage arrived), so chunk timing says
   * nothing about token timing.
   */
  unknownTokenBoundaries: boolean;
};

export const initialTokenCount: TokenCount = { total: 0, unknownTokenBoundaries: false };

/** Fold one streamed chunk into the running completion-token count. */
export function countTokens(prev: TokenCount, chunk: ChatCompletionChunk): TokenCount {
  const choice = chunk.choices[0];
  const hasContent = Boolean(choice?.delta.content);
  const usage = chunk.usage;
  if (usage == null || !Number.isFinite(usage.completion_tokens)) {
    // No count from the server: one per content chunk is all we can say.
    return hasContent
      ? { total: prev.total + 1, unknownTokenBoundaries: true }
      : prev;
  }
  const total = Math.max(0, usage.completion_tokens);
  // A finishing chunk carries the authoritative total, which may equal or
  // undercut the running count, so only mid-stream chunks reveal an engine
  // that reports no per-chunk counts.
  const isFinal = choice === undefined || choice.finish_reason != null;
  const unchanged = hasContent && !isFinal && total === prev.total;
  // A whole reply in one finishing chunk, with no count before it, says
  // nothing about token timing either.
  const onlyChunk = hasContent && isFinal && prev.total === 0;
  return {
    total,
    unknownTokenBoundaries: prev.unknownTokenBoundaries || unchanged || onlyChunk,
  };
}

export type ChatStreamArgs = {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
};

export async function* chatStream(
  args: ChatStreamArgs,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamChunk> {
  const r = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: args.model,
      messages: args.messages,
      max_tokens: args.max_tokens ?? 256,
      temperature: args.temperature ?? 0,
      stream: true,
      stream_options: { include_usage: true, continuous_usage_stats: true },
    }),
    signal,
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`HTTP ${r.status}: ${text || r.statusText}`);
  }
  if (!r.body) throw new Error("response had no body");

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        if (!frame.startsWith("data:")) continue;
        const payload = frame.slice(5).trim();
        if (!payload) continue;
        if (payload === "[DONE]") return;
        try {
          yield JSON.parse(payload) as ChatStreamChunk;
        } catch {
          // Skip malformed frame rather than abort the stream — better
          // UX for one bad chunk in the middle of a long generation.
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
