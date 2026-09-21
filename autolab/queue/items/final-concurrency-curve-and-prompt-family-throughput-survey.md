---
id: final-concurrency-curve-and-prompt-family-throughput-survey
title: Final concurrency curve and prompt-family throughput survey
status: blocked
outcome: 
priority: 1
target: single stream, multi stream, prompt families
exact: yes
needs: Owner explicitly paused all tests; wait for owner to say resume. 18 completed phases retained, API idle.
proposed_by: owner request 2026-09-21
owner: autolab-continuation-20260921
experiment: 046_final_performance
created: 2026-09-21
updated: 2026-09-21
---

# 046: final concurrency and prompt-family performance survey

Owner requested autonomous measurements and charts on the verified Streams
release 1790016660, with int4 experts and int8 attention. No deployment or
model-numerical change is part of this survey. Expert counters and bounded
f32 capture remain as in the deployed overrides; telemetry is collected.

## Prediction

Single-stream speed varies strongly with prompt predictability and phrase
history. Aggregate decode throughput should rise from single digits to
roughly 60–70 tok/s, with little gain beyond 128–256 streams. Fifteen streams
should remain near 24–25 tok/s, about 1.6 per stream. Higher concurrency
should increase first-token latency and reduce tok/s per stream.

## Method

Two sweeps at 1, 2, 4, 6, 8, 11, 15, 22, 32, 48, 64, 96, 128, 176 and 256
streams, forward then reverse, 128 output tokens per request. Extend to 352
if 256 improves on all lower points by more than 5%. For small concurrency,
run enough complete cohorts to cover at least twelve requests. Twelve prompt
families rotate evenly where cohort size allows, with identical deterministic
prompts across concurrency levels and passes. This is a study first pass and
repeat pass, not a claim that the model has never seen these prompts before.
The fleet retains its learned phrase history throughout.

Then measure each family at 1, 15, 64 and the mixed sweep's best observed
aggregate-decode concurrency. Single-stream runs use three prompts and repeat
them; also repeat each family's high-concurrency point. Report maxima only
as best observed in this tested grid, not a theoretical/global maximum.
The families are explanation, code, arithmetic, story, tips, tables,
rewriting, facts, poetry, instructions, translation and true/false.

The client uses each SSE event's `n_tokens`, validates it against final API
usage, and cross-checks phase totals against server token/request counters.
It records per-event timestamps, prompt and completion token counts, TTFT,
output and finish reason. Tokens include reasoning and final-answer tokens.
Sustained throughput counts tokens during the shared decode interval from
the last stream's first token to the first stream's last token; per-stream
throughput divides this by concurrency. The interval and its duration are
retained. End-to-end throughput includes admission, prefill and drain.
Per-request rates exclude the first event's tokens from the timed decode
numerator. Report median/p95 TTFT and per-request variability separately.

Charts will show aggregate and per-stream throughput, startup latency,
prompt-family results and the throughput/latency tradeoff. Recommend settings
for aggregate throughput and for per-stream/TTFT targets; there is no single
optimum without a latency requirement. Error bars show repeated-run range,
not statistical confidence intervals. Raw telemetry/responses stay outside
the public repository; sanitized data, plots and the reproducible scripts
are committed individually.

## Kill

One publisher and no competing generation traffic. The already settled
release, file version and restart counts must remain unchanged. Stop and
cancel outstanding requests on an API/status failure, stale signed report,
worker restart or release change. Do not retry failed phases automatically.
Reject measurements if client token/request totals disagree with server
counters. Preserve completed phases and any interruption for resumption.
The readiness anchor is the successful 14:00 CDT settlement and gates;
empty beacon phase text is permitted only with those exact unchanged active
workers, because profiling ages startup text out of the journal window.
