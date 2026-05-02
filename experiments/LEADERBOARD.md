# Leaderboard

Best measured tok/s per (model, hardware, engine, prompt-class) combination. Rows are replaced when a higher number lands; the previous record stays in the linked experiment dir.

## Llama 3.1 8B Instruct INT4 — single node

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| alpha B390 GPU | ov-optimum | 8.89 | c0-baselines/c0-1b |
| **charlie 140V GPU** | **ov-optimum** | **10.33** | c0-baselines/c0-2 |
| alpha B390 GPU | ov-spec K=4 + 1B INT4 draft | 13.83 | c0-baselines/c0-3 |

(charlie + spec not yet measured — open thread)

## Llama 3.1 8B Instruct INT4 — distributed (alpha + charlie via Thunderbolt 4)

| Engine | Config | tok/s | Source |
|---|---|---|---|
| **ov-dist-spec** | K=4, v5 shards | **17.59** | c0-baselines/c0-5 |

(ov-runtime currently broken on v5 — see c0-4)

## Notes

- All numbers above are decode-only tok/s (load + warmup excluded), 64 generated tokens, prompt = "What is the capital of France?".
- Today's numbers are roughly half of what `main`'s `baselines.md` claimed for the single-node engines. We treat today's numbers as the comparison reference; the gap-vs-main is itself an experimental thread (campaigns to come).
