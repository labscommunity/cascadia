// Streaming consumer for cascadia-api's /v1/chat/completions SSE response.
//
// We hand-roll the parser rather than use EventSource because EventSource
// is GET-only — and OpenAI-shaped completions are POST. The wire format
// is "data: <json>\n\n", terminated by "data: [DONE]\n\n". Cascadia adds a
// non-standard `n_tokens` field per chunk so spec-decode chunks count
// correctly (see cascadia-api lib.rs line ~470).

export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ChatCompletionChunk = {
  id: string;
  object: "chat.completion.chunk";
  /** Cascadia extension: tokens condensed into this chunk (spec-decode emits 1..K+1). */
  n_tokens?: number;
  choices: {
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: "stop" | null;
  }[];
};

/** Emitted by the server when a task fails mid-stream — note: NO `choices`. */
export type ChatErrorChunk = {
  id: string;
  object: "error";
  error: { message: string; type?: string };
};

export type ChatStreamChunk = ChatCompletionChunk | ChatErrorChunk;

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
