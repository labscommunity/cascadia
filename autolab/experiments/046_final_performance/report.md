# Inkling fleet performance survey

IN PROGRESS — completed phases only

Eleven Panther Lake boxes; release `1790016660`; int4 experts and int8 attention. All survey requests use temperature 0 and a 128-token output budget. Counts include reasoning and answer tokens.

Sustained decode is measured only while all requests in each cohort are decoding. End-to-end throughput includes admission, prefill and drain. Prompts are repeated across sweep passes; learned phrase history remains enabled. Repeat ranges are observed variability, not confidence intervals. The fixed prompt pools are finite: high-concurrency family cohorts can repeat identical prompts. `unique_prompts` in the phase CSV records diversity; family maxima describe these pools, not a guarantee for arbitrary novel prompts.

Best observed sustained aggregate: **13.94 tok/s at 6 streams**.
Smallest tested setting within 95% of that peak: **6 streams**.
Best observed throughput including startup/drain: **12.83 tok/s at 6 streams**.

These are different objectives from maximizing each user’s speed. Use the latency and per-stream columns to choose an operating point.

![Concurrency throughput](concurrency-throughput.png)

![First-token latency](concurrency-latency.png)

| Streams | Runs | Decode tok/s | Decode tok/s/stream | Including startup tok/s | TTFT median / p95 (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5.68 | 5.68 | 5.19 | 2.18 / 2.65 |
| 2 | 1 | 4.55 | 2.28 | 4.40 | 2.06 / 2.94 |
| 4 | 1 | 9.18 | 2.30 | 8.69 | 3.31 / 3.74 |
| 6 | 1 | 13.94 | 2.32 | 12.83 | 3.44 / 5.51 |

| Minimum sustained tok/s/stream | Highest tested concurrency meeting it |
|---:|---:|
| 1 | 6 |
| 2 | 6 |
| 3 | 1 |
| 5 | 1 |
| 10 | None |

Data: [phase CSV](phase-results.csv), [per-request CSV](request-results.csv), [full sanitized phase measurements](measurements.json). Each chart is also available as SVG and PDF.

The exact prompts, generated text, per-event token timing and raw fleet telemetry are retained privately under the operator’s autolab-telemetry directory. No host names, addresses or raw telemetry are included here.
