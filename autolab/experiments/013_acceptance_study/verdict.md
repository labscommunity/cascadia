# 013 verdict: a small drafter model is right 54-61 % of the time on this model's text, the n-gram tables 30-38 %. No drafter gets near 85 %.

192 greedy responses of the fleet (30,673 tokens, 12 prompt families), teacher-forced, scored at the model's own
token boundaries (3000 positions per neural drafter; `study.json` holds the per-position hit vectors).

| drafter | right | note |
|---|---|---|
| n-gram, request history only | 10.5 % | proposes at 17 % of positions, 60 % of those right |
| n-gram, history + table learned from the other 191 responses | 37.9 % | 32.1 % with 95 responses: grows with seen traffic of the same kind |
| Llama 3.2 1B / 3B, Llama 3.1 8B (Q4_K_M) | 49.4 / 52.6 / 53.9 % | |
| **Qwen3 0.6B** / 1.7B / 4B (Q4_K_M) | **53.6** / 58.2 / 60.8 % | a reasoning model drafts a reasoning model's `<think>` text best |
| oracle: either of two drafters right | <= 66 % | the best pair (Qwen3-4B + Llama-8B) |

By task, Qwen3-0.6B (4B in brackets): arithmetic 0.78 (0.82), rewriting 0.71 (0.76), translation 0.61 (0.73), code
0.59 (0.65), true/false 0.55 (0.64), tables 0.51 (0.63), how-to 0.48, lists 0.47, facts 0.47, poem 0.45,
**explain 0.41 (0.49), story 0.39 (0.45)**. Seven times more parameters buy 6-8 points.

What it means for `1 / (a*T + (1-a)*L)` with L ~ 410 ms, T ~ 50 ms:

- open prose: a ~ 0.42 -> ~3.8 tok/s; arithmetic / rewriting: a ~ 0.75 -> ~7 tok/s. Measured in 015c: 3.1-3.7 and
  5.9-8.2 (10.9 on a true/false prompt, where the tables had seen the family).
- **10 tok/s needs a >= 0.85 at this L. No drafter reaches that on open text** (two drafters with an oracle: 0.66).
  The remaining factor has to come from L: several memory buses on one token's layer (expert parallelism, now
  that a LAN round trip is 0.25 ms, see 014), fewer bytes (int4 attention), or an acceptance rule that is not exact.
- Size is the wrong axis for the drafter: 0.6B at ~25 ms/token keeps ahead of a 50 ms frame on rank 0's idle CPU
  cores; 1.7B would cost rank 0's bus more than its 4.6 points return. Fit is the right axis: a drafter distilled
  on this model's own outputs (the fleet writes ~200k tokens an hour) is the next step for a.
