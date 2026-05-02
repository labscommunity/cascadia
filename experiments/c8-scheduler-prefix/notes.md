# c8: SchedulerConfig + prefix caching

## Setup

- Model: Llama 3.1 8B INT4 on alpha B390 GPU.
- Two-turn chat: identical 708-char system prompt + different user
  questions. 32 generated tokens per turn. Greedy.
- Compared `enable_prefix_caching=True` against `False`.

## Results

| Config | Load (s) | Turn 1 (s) | Turn 2 (s) | Notes |
|---|---|---|---|---|
| prefix_caching=False | 21.65 | 1.92 | 1.57 | Turn 2 faster from OS-level / OV-kernel cache |
| prefix_caching=True | 21.88 | 2.02 | 1.80 | **No win — slightly slower than False** |

## Why no win

The synthesis (`literature/_intel_synthesis.md` #3) cited "≥85% TTFT reduction
on turn-N chat with shared system prompts" — we didn't see it. Likely causes:

1. **We didn't call `pipe.start_chat()`** — the synthesis snippet shows
   that `pipe.start_chat()` is what activates the chat-mode prefix
   reuse. We treated each turn as an independent `pipe.generate()` call
   with the system prefix re-tokenized each time.
2. **System prefix may be too short** — 708 chars (~200 tokens) is in
   the noise band. Prefix caching wins are typically reported on 1000+
   token prefixes where the prefill fraction dominates.
3. **`SchedulerConfig` itself adds path overhead** — when prefix caching
   doesn't engage, you're paying for the scheduler's bookkeeping with no
   savings.

## Follow-up to try

- c8-3 (later): same harness but with `pipe.start_chat()` between the warmup
  and turn 1, and `pipe.finish_chat()` after.
- c8-4 (later): longer system prompt (~2000 tokens of canned context) so
  the prefill fraction is large enough that caching it pays off.

For NOW: prefix caching is a chat-quality-of-experience knob, not a
bench-mode win. Documented as no-op in our standard one-shot bench.
