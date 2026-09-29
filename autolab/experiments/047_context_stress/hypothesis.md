# 047: the context stress test (prepared 2026-09-29; NOT run until the owner says so)

**Owner's ask.** Step the context up and measure at every step, comprehensively, for the paper.

**What runs** (`bench/run_context_stress.sh`, `bench/context_stress.py`): for one stream, then four streams at once,
prompts of 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k tokens of natural text with a needle near the start and a
question at the end; 3 repeats up to 16k, 2 up to 64k, 1 above; 96 new tokens each. Per request: exact prompt
tokens, first token (prefill), prefill tok/s, decode tok/s at that context, tokens produced, needle found, and
every box's CPU, GPU, package power, free memory and stage profile sampled every 5 s. Results are on disk after
every request (`context_stress.json`, `context_stress.csv`), the harness predicts each request's time from the
rates it measured and stops a pass when the budget cannot hold the next one, and the runner reverts at the end.

**Fixes this needs (binary `cascadia-6430d70e-ctx2` = the deployed engine + 034's probe and API caps +):**
1. A prompt goes down one window per engine step; a step that sends a window returned nothing, and the runner
   closes a task after three such steps. Any prompt longer than ~22 windows was being closed as "no progress"
   (176 tokens at the production 8-row window). The feed now emits a progress chunk per window, which the API
   does not send and nobody counts (`Chunk::progress`). Test: a ~400-token prompt at a 4-row window, with the
   runner's rule enforced, decodes the same tokens as the single-stage path.
2. The first device call at a new row count compiles that shape's kernels on every layer of every rank (034: 34 s
   to the first token of a 1k prompt at a 64-row window). `CASCADIA_INKLING_OV_WARM_ROWS=64` compiles them at load.

**Predictions.** Prefill at 64-row windows: 120-160 tok/s at 1k-8k (the fused-experts prefill path at 64 rows shares
expert reads), falling as the CPU attention over the growing context grows with the square of the prompt:
~17 TFLOP per box for a 32k prompt, ~70 at 64k, ~280 at 128k, ~1,100 at 256k. First token: ~10 s / 20 / 40 / 80 s,
~3-4 min at 32k, 10-15 min at 64k, 45-60 min at 128k, 3-4 h at 256k (the budget decides where it stops). Decode
per token at N: the probe's ~45 ms + 2.9 ms per 1k per box, times eleven in series: 3.5 tok/s at 1k, ~2.6 at 16k,
~1.5 at 64k, ~0.7 at 128k, ~0.35 at 256k. Four streams: aggregate ~4x the single rate at small contexts (the
pipeline has idle room), memory 4x per box (1.1 GB per box at 4 x 32k). Needle: found at every size if the global
layers' attention beyond the relative bias' 1,024 positions works; a miss at large sizes would be a finding.

**Time.** Publish + settle ~10 min, gates 1 min, the stress pass as long as `BUDGET_S` (default 4 h), revert ~8 min.
Single-stream pass through 128k with repeats: ~2 h; the 4-stream pass through 32k-64k: ~1.5 h; 256k alone: ~3-4 h.

**Kill / safety.** No traffic before `steady 3/3`; gate failure = revert; the harness stops at the first request
error; every request is capped; every number is on disk when measured; the runner always ends with the revert,
which applies by itself if the entry box dies and comes back.
