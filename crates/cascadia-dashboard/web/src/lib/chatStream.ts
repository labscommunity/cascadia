// Pure accumulation of a /v1/chat/completions stream: answer text,
// reasoning text, token count, and the timing the stat bar shows.
//
// A thinking model's reasoning arrives as `delta.reasoning_content`, apart
// from the answer in `delta.content`. Both are generated tokens, so both
// count, and the first of either kind is the time to first token. A server
// started with `--reasoning-format none` keeps `<think>…</think>` inside
// `content`; that text is passed through as answer text, as before.

import type { ChatCompletionChunk } from "./sse";

export type StreamTally = {
  content: string;
  reasoning: string;
  /** Model tokens so far: per-chunk `n_tokens`, then the final `completion_tokens`. */
  tokens: number;
  /** `performance.now()` of the first chunk carrying text of either kind. */
  firstTokenAt: number | null;
  /** `tokens` as it stood right after that first chunk. */
  tokensAtFirst: number;
  /** A text chunk reported `n_tokens: 0`: chunk boundaries are not token boundaries. */
  unknownTokenBoundaries: boolean;
};

export function emptyTally(): StreamTally {
  return {
    content: "",
    reasoning: "",
    tokens: 0,
    firstTokenAt: null,
    tokensAtFirst: 0,
    unknownTokenBoundaries: false,
  };
}

export function applyChunk(t: StreamTally, chunk: ChatCompletionChunk, now: number): StreamTally {
  const delta = chunk.choices[0]?.delta;
  const content = delta?.content ?? "";
  const reasoning = delta?.reasoning_content ?? "";
  const hasText = Boolean(content || reasoning);
  const tokens =
    chunk.completion_tokens != null
      ? chunk.completion_tokens
      : t.tokens + (chunk.n_tokens ?? (hasText ? 1 : 0));
  const first = t.firstTokenAt === null && hasText;
  return {
    content: t.content + content,
    reasoning: t.reasoning + reasoning,
    tokens,
    firstTokenAt: first ? now : t.firstTokenAt,
    tokensAtFirst: first ? tokens : t.tokensAtFirst,
    unknownTokenBoundaries: t.unknownTokenBoundaries || (hasText && chunk.n_tokens === 0),
  };
}

export type StreamRates = {
  ttftSeconds: number | null;
  elapsedSeconds: number;
  tokensPerSecond: number | null;
};

export function rates(t: StreamTally, startedAt: number, now: number): StreamRates {
  const elapsed = (now - startedAt) / 1000;
  const ttft = t.firstTokenAt != null ? (t.firstTokenAt - startedAt) / 1000 : null;
  if (t.unknownTokenBoundaries) {
    // Only the final total is known, so use whole-request throughput
    // rather than inventing token timing.
    return {
      ttftSeconds: ttft,
      elapsedSeconds: elapsed,
      tokensPerSecond: elapsed > 0.05 && t.tokens > 0 ? t.tokens / elapsed : null,
    };
  }
  // Decode rate: the tokens generated after the first chunk, over the time
  // since that chunk. Reasoning and answer tokens count alike.
  const window = ttft != null ? elapsed - ttft : null;
  const decodeTokens = t.tokens - t.tokensAtFirst;
  return {
    ttftSeconds: ttft,
    elapsedSeconds: elapsed,
    tokensPerSecond:
      window != null && window > 0.05 && decodeTokens > 0 ? decodeTokens / window : null,
  };
}
