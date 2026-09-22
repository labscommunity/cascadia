# 016: does a drafter tuned on this model's own text guess it better? (data collection + offline study)

013: Qwen3-0.6B names the model's next token 41 % of the time on explanations and 78 % on arithmetic. The gap is
style (how this model phrases its `<think>` text), which a small model can learn from a few hundred thousand of the
model's own tokens. Literature (DistillSpec and successors): +10 to +45 % relative acceptance.

**Step 1 (this folder):** collect the fleet's greedy responses to real instructions (databricks-dolly-15k, shuffled,
prompt + context under 700 characters), 256 tokens each, at a gentle 32 streams so that a person using the API
still gets a slot within seconds.
**Step 2:** fine-tune Qwen3-0.6B on (instruction -> response) with plain next-token loss on the response, on the
miner's GPU; hold out 10 % of the responses; score with `bench/drafter_study.py` against the untuned model on the
held-out set AND on 013's corpus (different prompts).
**Decision:** worth shipping if a on prose rises by >= 0.06 (about +10 % single-stream speed). Shipping a tuned
GGUF to rank 0 needs a place to download it from (the release channel carries six named files only).
