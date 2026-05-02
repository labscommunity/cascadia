# c11: chat-mode prefix caching with start_chat()

## Setup

- Model: Llama 3.1 8B INT4 on alpha B390 GPU.
- Three-turn chat: 9282-char shared context (~2300 tokens) + different
  user questions per turn. 32 generated tokens per turn. Greedy.
- Tested with vs without `pipe.start_chat()` and `enable_prefix_caching`.

## Results

| Config | Turn 1 (s) | Turn 2 (s) | Turn 3 (s) |
|---|---|---|---|
| no prefix, no start_chat | 1.48 | 1.46 | 1.18 |
| prefix on, start_chat on | 1.45 | 1.53 | 1.76 |

## Conclusion

Even with `pipe.start_chat()` AND `enable_prefix_caching=True` AND a
~2300-token shared prefix, **prefix caching gave NO win** — turn 2 and
turn 3 were actually slightly slower than the no-prefix baseline.

The synthesis predicted ~85% TTFT reduction. We're not seeing it.
Possible reasons:
- We're calling `pipe.generate(SYSTEM + USER_QUESTION)` with the system
  prepended each call, instead of using a chat-API form like
  `pipe.generate(USER_QUESTION)` after `start_chat()` (which is supposed
  to maintain history internally).
- The cache_size=4 GB may not be enough for the prefix to stay resident
  across all turns.
- This LLMPipeline version (2026.1.0) may have a different prefix-cache
  activation flow than the synthesis was based on.

Not pursuing further. Documented as "needs deeper API investigation"
and moving on.
