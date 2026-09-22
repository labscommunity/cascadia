# 013: how good can a drafter be on this model's text? (offline, no fleet change)

**Question.** Single-stream speed is `1 / (a*T + (1-a)*L)`. 012 measured a = 0.28-0.32 for the n-gram drafter.
What `a` would other drafters reach on text this model really writes?

**Method.** `bench/collect.py` records 192 greedy responses (160 tokens each, 12 prompt families: explanations,
code, arithmetic, stories, lists, tables, rewriting, poems, how-tos, translation, true/false). `bench/drafter_study.py`
replays drafters over them, teacher-forced, at the model's own token boundaries (o200k_base): a hit = the
drafter's greedy continuation of the conversation so far starts with the text of the model's next token. That
is exactly what a drafter with a different vocabulary can deliver, and it is the conditional acceptance a
pipelined chain of guesses sees.

Drafters: the fleet's n-gram drafter (history, history + leave-one-out table), and small open LMs through
llama.cpp on the Mac: Qwen3 0.6B / 1.7B / 4B, Llama 3.2 1B / 3B, Llama 3.1 8B (all Q4_K_M).

**Prediction.** Small LMs agree with a large model on the "easy" half of tokens: a = 0.45-0.6, rising slowly
with size. If even the 8B stays under 0.7, a drafter alone cannot deliver 10 tok/s (needs 0.85 at L = 440 ms).
