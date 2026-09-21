# Inkling fleet performance survey

IN PROGRESS — completed phases only (capped at 176 streams)

Eleven Panther Lake boxes; release `1790016660`; int4 experts and int8 attention. All survey requests use temperature 0 and a 128-token output budget. Counts include reasoning and answer tokens.

Sustained decode is measured only while all requests in each cohort are decoding. Average per-stream speed divides that total by concurrency; it is not a lower bound for every individual request. End-to-end throughput includes admission, prefill and drain. The mixed workload cycles through twelve families; rounding to complete cohorts makes family proportions vary slightly with concurrency. The same deterministic prompts recur across sweep passes; learned phrase history remains enabled. Repeat ranges are observed variability, not confidence intervals. The fixed prompt pools are finite: high-concurrency family cohorts can repeat identical prompts. `unique_prompts` in the phase CSV records diversity; family maxima describe these pools, not a guarantee for arbitrary novel prompts.

At high concurrency, the engine’s 64-entry pending queue can reject a burst before the API’s 512-request limit is reached. Only explicit capacity rejections are retried with bounded backoff; retry counts are recorded and all queueing time remains included in TTFT and end-to-end throughput. Other errors stop the run.

The bounded diagnostic capture reached its storage budget during the first 22-stream attempt. That incomplete attempt was excluded and repeated. Capture writes were enabled for the ascending 1–15-stream points and disabled thereafter; the reverse sweep and family tests use the same capture-disabled state. No worker restarted and no model configuration changed. Repeat differences therefore include phrase learning, time/order effects and this instrumentation change.

Two initial 128-stream attempts were excluded: one received an admission 503, and the next received an engine no-progress error before generating tokens. The workers did not restart. A subsequent correctness gate passed; the verified-idle retry then completed all 128 requests without capacity retries. These interruptions are retained in [stress attempts](stress-attempts.json). Completed-run throughput is not an error-rate or reliability estimate.

The 256-stream attempt disconnected during admission. Outstanding requests were cancelled, the unchanged fleet returned to idle, and the correctness gate passed again. The remaining survey was capped at **176 streams**. The planned 256-stream repeats and conditional 352-stream extension were therefore not completed. This is an observed failure, not proof of a hard engine concurrency limit; the measured optimum is bounded by the tested range.

An **88-stream refinement** was added after the first sweep and the 176-stream repeat: it places eight rows in each of eleven pipeline groups. The observed slowdown above eight rows per group motivated this extra point. Both 88-stream runs occur after the original ascending pass; they are exploratory measurements, with the same token and health checks.

The owner paused and later resumed collection. Completed phases were retained and the interrupted 88-stream repeat was restarted. The serving release and worker restart counts were unchanged. Other generation occurred during the pause, so subsequent results also reflect any phrase-history learning from that traffic. It is excluded from all measured phase counters. See [pause history](pause-history.json).

An additional resumed 88-stream attempt was excluded after other generation raised the API count to 94 active requests. Completed phases are unaffected. The watchdog now stops immediately when active requests exceed the phase concurrency, in addition to the final token and request counter checks.

Best observed sustained aggregate: **60.29 tok/s at 88 streams**.
Smallest tested setting within 95% of that peak: **88 streams**.
Best observed throughput including startup/drain: **46.87 tok/s at 88 streams**.

These are different objectives from maximizing each user’s speed. Use the latency and per-stream columns to choose an operating point.

![Concurrency throughput](concurrency-throughput.png)

![First-token latency](concurrency-latency.png)

| Streams | Runs | Decode tok/s | Decode tok/s/stream | Including startup tok/s | TTFT median / p95 (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5.68 | 5.68 | 5.19 | 2.18 / 2.65 |
| 2 | 1 | 4.55 | 2.28 | 4.40 | 2.06 / 2.94 |
| 4 | 2 | 9.16 | 2.29 | 8.64 | 3.31 / 4.03 |
| 6 | 2 | 13.89 | 2.31 | 12.80 | 4.20 / 5.49 |
| 8 | 2 | 18.60 | 2.32 | 16.86 | 4.55 / 6.23 |
| 11 | 2 | 23.28 | 2.12 | 20.86 | 5.47 / 7.64 |
| 15 | 2 | 24.60 | 1.64 | 22.12 | 6.05 / 10.11 |
| 22 | 2 | 33.87 | 1.54 | 29.26 | 9.71 / 14.51 |
| 32 | 2 | 38.31 | 1.20 | 32.48 | 13.44 / 21.90 |
| 48 | 2 | 45.33 | 0.94 | 37.68 | 18.85 / 31.55 |
| 64 | 2 | 53.60 | 0.84 | 42.41 | 25.10 / 46.41 |
| 88 | 2 | 60.29 | 0.69 | 46.87 | 34.61 / 64.75 |
| 96 | 2 | 46.85 | 0.49 | 39.79 | 38.94 / 71.31 |
| 128 | 2 | 50.10 | 0.39 | 41.61 | 55.11 / 108.73 |
| 176 | 2 | 57.72 | 0.33 | 45.24 | 76.83 / 165.34 |

| Target average sustained tok/s/stream | Highest tested concurrency meeting it |
|---:|---:|
| 1 | 32 |
| 2 | 11 |
| 3 | 1 |
| 5 | 1 |
| 10 | None |

Data: [phase CSV](phase-results.csv), [per-request CSV](request-results.csv), [full sanitized phase measurements](measurements.json). Each chart is also available as SVG and PDF.

The exact prompts, generated text, per-event token timing and raw fleet telemetry are retained privately under the operator’s autolab-telemetry directory. No host names, addresses or raw telemetry are included here.

[Role diagnostics](role-diagnostics.csv) contain numeric compute, memory and fallback observations. Profile windows must fit entirely inside the shared decode interval and contain no admissions. These sampled counters exclude startup and drain; their correlations do not establish causes.

[Collection environment](collection-environment.json) records the release and collector hashes, client package versions and measurement settings.
