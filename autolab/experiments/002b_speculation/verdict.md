# 002b verdict: KEEP. Speculation is exact and pays what the model predicts; drafts are the limit.

Gate: PASS, 3/3 character-identical with guesses flying (a guess never chooses a token).

| request | tok/s without (002a) | with speculation | guesses sent / right / wrong |
|---|---|---|---|
| gate prompts (32 tokens each) | 1.60-1.68 | 1.77, 1.91, 2.13 | - |
| "why is the sky blue", 64 tokens | 1.52 | 1.60, then 1.77 | 30 / 3 / 5 and 29 / 3 / 3 |
| "repeat this paragraph twice", 96 tokens | ~1.6 | **2.79** | 67 / 47 / 2 |

The copy task had 47 of 96 tokens arrive one stage time after the previous one instead of a
round trip later. Model check: a = 0.49, T = 58.5 ms, D = 11 gives
1 / (T (a + (1 - a) D)) = 2.8 tok/s. Measured 2.79.

On free-form reasoning the per-request n-gram table rarely has anything to propose (30 guesses in
64 tokens, 3 right): +5-15 %. What a better drafter is worth on this fleet, from the same model:
a = 0.3 -> 2.2 tok/s, 0.5 -> 2.9, 0.7 -> 4.6, 0.9 -> 8.6 (T = 58 ms; the 25 W limit is in T).
Next: the cross-request table (binary 3), then decide whether a draft model is reachable.
