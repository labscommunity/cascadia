# 010 verdict: seeding the cross-request drafter with 30k tokens of varied traffic does not move unseen prompts

240 different prompts x 128 tokens (42 tok/s, 12 minutes), then four unseen prompts, single stream:
3.54, 2.81, 3.59, 3.12 tok/s (mean 3.27; before seeding 3.09 and 3.59). Right guesses per 64 tokens:
24, 12, 27, 20 (before: 10-19). The time model says the same: a ~ 0.3, T ~ 40 ms, D = 11 ->
0.3 x 40 + 0.7 x 440 = 320 ms per token. A table of word n-grams saturates around a third; past
that the draft has to understand the text. Closed: more traffic is not the lever.
