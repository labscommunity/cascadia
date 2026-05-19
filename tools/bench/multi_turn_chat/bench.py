#!/usr/bin/env python3
"""Multi-turn chat bench for KV-cache features (iter 060/072/084).

Drives /v1/chat/completions against a running tahoma worker and emits
per-request JSONL with prompt size, completion tokens, total time, and
finish reason. Designed to reveal the static-prompt and per-session KV
cache wins from iter 060 and iter 072.

Modes
-----
single_turn_repeat
    N identical single-turn requests against the same prompt. With a
    static prompt cache enabled, requests 2..N should skip prefill and
    therefore have lower total time than request 1.

multi_turn_chat
    M conversations of N turns each. Each turn includes ALL prior
    user/assistant pairs in the messages array. With a per-session KV
    cache enabled (X-Session-Id header set), turns 2..N for one
    conversation prefill only the NEW user message rather than the
    entire history.

Each request is sent with max_tokens=1 to isolate the prefill cost
from decode, then a second batch is sent with max_tokens controlled by
--max-tokens-decode to measure end-to-end. The bench is single-threaded
to avoid concurrency-induced variance; the worker should be otherwise
idle.

Output is JSONL, one line per request. Use --summary to also print
an aggregate table at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestRecord:
    """One HTTP request worth of timing + bookkeeping."""

    mode: str
    workload: str
    conversation_id: str
    turn_index: int
    session_id: Optional[str]
    max_tokens: int
    prompt_chars: int
    prompt_messages: int
    started_at: float
    finished_at: float
    completion_tokens: int
    finish_reason: str
    status: str  # "ok" or "error:<class>"
    error: Optional[str] = None

    def total_seconds(self) -> float:
        return self.finished_at - self.started_at

    def as_jsonl(self) -> str:
        d = {
            "mode": self.mode,
            "workload": self.workload,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "session_id": self.session_id,
            "max_tokens": self.max_tokens,
            "prompt_chars": self.prompt_chars,
            "prompt_messages": self.prompt_messages,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_seconds": self.total_seconds(),
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "status": self.status,
        }
        if self.error is not None:
            d["error"] = self.error
        return json.dumps(d, separators=(",", ":"))


def chat_completion(
    base_url: str,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    session_id: Optional[str],
    timeout: float,
) -> tuple[float, dict | None, str | None]:
    """POST /v1/chat/completions, return (elapsed_seconds, body_dict, error)."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    if session_id is not None:
        req.add_header("X-Session-Id", session_id)

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return time.monotonic() - start, body, None
    except urllib.error.HTTPError as e:
        return time.monotonic() - start, None, f"http:{e.code} {e.reason}"
    except urllib.error.URLError as e:
        return time.monotonic() - start, None, f"url:{e.reason}"
    except (TimeoutError, OSError) as e:
        return time.monotonic() - start, None, f"io:{type(e).__name__} {e}"


def record_from_response(
    *,
    mode: str,
    workload: str,
    conversation_id: str,
    turn_index: int,
    session_id: Optional[str],
    max_tokens: int,
    messages: list[dict],
    started_at: float,
    elapsed: float,
    body: dict | None,
    err: str | None,
) -> RequestRecord:
    prompt_chars = sum(len(m["content"]) for m in messages)
    if body is None:
        return RequestRecord(
            mode=mode,
            workload=workload,
            conversation_id=conversation_id,
            turn_index=turn_index,
            session_id=session_id,
            max_tokens=max_tokens,
            prompt_chars=prompt_chars,
            prompt_messages=len(messages),
            started_at=started_at,
            finished_at=started_at + elapsed,
            completion_tokens=0,
            finish_reason="error",
            status="error",
            error=err,
        )
    usage = body.get("usage") or {}
    choice = (body.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason") or "unknown"
    return RequestRecord(
        mode=mode,
        workload=workload,
        conversation_id=conversation_id,
        turn_index=turn_index,
        session_id=session_id,
        max_tokens=max_tokens,
        prompt_chars=prompt_chars,
        prompt_messages=len(messages),
        started_at=started_at,
        finished_at=started_at + elapsed,
        completion_tokens=int(usage.get("completion_tokens") or 0),
        finish_reason=finish_reason,
        status="ok",
    )


def run_single_turn_repeat(
    *,
    args: argparse.Namespace,
    out,
    records: list[RequestRecord],
) -> None:
    """Hit the same prompt N times; static prompt cache should make 2..N cheap."""
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    workload = "single_turn_repeat"
    convo_id = f"convo-{uuid.uuid4().hex[:8]}"
    batches = [("prefill", 1)]
    if not args.skip_e2e:
        batches.append(("e2e", args.max_tokens_decode))
    for batch, max_tokens in batches:
        for i in range(args.repeats):
            started = time.time()
            elapsed, body, err = chat_completion(
                args.url,
                model=args.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=args.temperature,
                session_id=None,  # 060 keys on prompt prefix, not session
                timeout=args.timeout,
            )
            rec = record_from_response(
                mode=batch,
                workload=workload,
                conversation_id=convo_id,
                turn_index=i,
                session_id=None,
                max_tokens=max_tokens,
                messages=messages,
                started_at=started,
                elapsed=elapsed,
                body=body,
                err=err,
            )
            records.append(rec)
            out.write(rec.as_jsonl() + "\n")
            out.flush()


def run_multi_turn_chat(
    *,
    args: argparse.Namespace,
    out,
    records: list[RequestRecord],
) -> None:
    """Per-session multi-turn: accumulate messages each turn. Session cache wins on turn 2+."""
    workload = "multi_turn_chat"
    canned_user = [
        "Hello, can you help me understand prime numbers?",
        "What's the difference between prime and composite numbers?",
        "Give me the first ten primes.",
        "Why isn't one considered prime?",
        "Are there infinitely many primes?",
        "What's the largest known prime?",
        "Where would I learn more about number theory?",
        "Thanks, that was helpful.",
    ]
    canned_assistant_filler = (
        "Sure, I can walk you through that. Let me know if you want more detail."
    )

    for c in range(args.conversations):
        convo_id = f"convo-{uuid.uuid4().hex[:8]}"
        session_id = convo_id if args.use_session_id else None
        messages: list[dict] = [
            {"role": "system", "content": args.system_prompt},
        ]
        batches = [("prefill", 1)]
        if not args.skip_e2e:
            batches.append(("e2e", args.max_tokens_decode))
        for t in range(args.turns):
            messages.append(
                {"role": "user", "content": canned_user[t % len(canned_user)]}
            )
            for batch, max_tokens in batches:
                started = time.time()
                elapsed, body, err = chat_completion(
                    args.url,
                    model=args.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                    session_id=session_id,
                    timeout=args.timeout,
                )
                rec = record_from_response(
                    mode=batch,
                    workload=workload,
                    conversation_id=convo_id,
                    turn_index=t,
                    session_id=session_id,
                    max_tokens=max_tokens,
                    messages=messages,
                    started_at=started,
                    elapsed=elapsed,
                    body=body,
                    err=err,
                )
                records.append(rec)
                out.write(rec.as_jsonl() + "\n")
                out.flush()
            # Use the canned filler instead of replaying actual model output so
            # all conversations share identical message structure across runs.
            messages.append(
                {"role": "assistant", "content": canned_assistant_filler}
            )


def summarize(records: list[RequestRecord]) -> str:
    """Aggregate by (workload, mode, turn_index) and print a compact table."""

    @dataclass
    class Agg:
        n: int = 0
        total_seconds_sum: float = 0.0
        completion_tokens_sum: int = 0
        errors: int = 0
        per_request_seconds: list[float] = field(default_factory=list)

    buckets: dict[tuple[str, str, int], Agg] = {}
    for r in records:
        key = (r.workload, r.mode, r.turn_index)
        agg = buckets.setdefault(key, Agg())
        agg.n += 1
        if r.status == "ok":
            agg.total_seconds_sum += r.total_seconds()
            agg.completion_tokens_sum += r.completion_tokens
            agg.per_request_seconds.append(r.total_seconds())
        else:
            agg.errors += 1

    rows = []
    rows.append(
        f"{'workload':<22} {'mode':<8} {'turn':>4} {'n':>3} "
        f"{'mean_s':>8} {'min_s':>8} {'max_s':>8} "
        f"{'tok_sum':>8} {'tok/s':>8} {'err':>4}"
    )
    rows.append("-" * len(rows[0]))
    for (workload, mode, turn), agg in sorted(buckets.items()):
        ok_n = max(1, agg.n - agg.errors)
        mean_s = agg.total_seconds_sum / ok_n
        per = agg.per_request_seconds
        min_s = min(per) if per else 0.0
        max_s = max(per) if per else 0.0
        tps = (agg.completion_tokens_sum / agg.total_seconds_sum) if agg.total_seconds_sum > 0 else 0.0
        rows.append(
            f"{workload:<22} {mode:<8} {turn:>4} {agg.n:>3} "
            f"{mean_s:>8.3f} {min_s:>8.3f} {max_s:>8.3f} "
            f"{agg.completion_tokens_sum:>8} {tps:>8.3f} {agg.errors:>4}"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-turn chat bench for KV-cache features")
    ap.add_argument("--url", required=True, help="Worker base URL, e.g. http://miner:18000")
    ap.add_argument("--model", default="K2.6")
    ap.add_argument("--mode", choices=["single_turn_repeat", "multi_turn_chat", "both"], default="both")
    ap.add_argument("--repeats", type=int, default=5, help="single_turn_repeat: requests per workload")
    ap.add_argument("--conversations", type=int, default=3, help="multi_turn_chat: number of conversations")
    ap.add_argument("--turns", type=int, default=5, help="multi_turn_chat: turns per conversation")
    ap.add_argument("--use-session-id", action="store_true", help="multi_turn_chat: send X-Session-Id header")
    ap.add_argument("--max-tokens-decode", type=int, default=32, help="end-to-end batch max_tokens")
    ap.add_argument("--skip-e2e", action="store_true", help="skip max_tokens=N batch; only measure max_tokens=1 (prefill cost)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful assistant. Answer concisely and accurately. "
            "When the user asks a clarifying question, refer back to prior "
            "context in this conversation."
        ),
    )
    ap.add_argument(
        "--user-prompt",
        default="Explain what a binary tree is in two sentences.",
        help="single_turn_repeat: prompt body",
    )
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--summary", action="store_true", help="print aggregate table to stderr at end")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    records: list[RequestRecord] = []
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "_run": {
                        "url": args.url,
                        "model": args.model,
                        "mode": args.mode,
                        "repeats": args.repeats,
                        "conversations": args.conversations,
                        "turns": args.turns,
                        "use_session_id": args.use_session_id,
                        "max_tokens_decode": args.max_tokens_decode,
                        "temperature": args.temperature,
                        "started_at": time.time(),
                    }
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        if args.mode in ("single_turn_repeat", "both"):
            run_single_turn_repeat(args=args, out=out, records=records)
        if args.mode in ("multi_turn_chat", "both"):
            run_multi_turn_chat(args=args, out=out, records=records)
    if args.summary:
        print(summarize(records), file=sys.stderr)
    if any(r.status != "ok" for r in records):
        print(f"WARN: {sum(1 for r in records if r.status != 'ok')} of {len(records)} requests failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
