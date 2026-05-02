# c12: CacheEvictionConfig for long-gen

## Setup
8B INT4 on alpha B390 GPU, 256 generated tokens, creative prompt.
Compared no eviction vs `max_cache_size=512`.

## Results

| Config | tok/s |
|---|---|
| no eviction | 21.82 |
| max_cache_size=512 | 21.76 |

Identical (within noise). Eviction didn't trigger because our actual
sequence (8 prompt + 256 gen = 264 tokens) is well under 512.

To actually test eviction's effect we'd need `max_cache_size < 264`
(e.g. 192 with start_size=32 + recent_size=128 leaves only 32 mid-tokens
for eviction). Quality risk high; deferred to a follow-up.

## Takeaway
Eviction is a tool for *bounded memory*, not for *speeding up decode at
moderate context*. The 8B's 256-tok degradation isn't fixable by
eviction alone.
