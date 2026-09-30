# 041: int4 attention projections are faster on this device

A bounded role-5 probe completed successfully before worker load. It built
candidate graphs in memory from existing shells, measured 41 iterations
following five warm-ups, and installed no IRs or serving-weight changes.
All eight measurements reached telemetry; the benchmark exited zero.

| Layer | Rows | Int8 QKVR + O, us | Int4 QKVR + O, us | Saving |
|---|---:|---:|---:|---:|
| 30 | 1 | 1298 | 792 | 39.0% |
| 30 | 2 | 1343 | 781 | 41.8% |
| 31 | 1 | 1338 | 808 | 39.6% |
| 31 | 2 | 1348 | 765 | 43.2% |

Prediction falsified: these projection shapes use int4 efficiently, unlike
the earlier dense/head FC measurements. Extrapolating the measured saving
across six layers suggests roughly 3.0–3.5 ms per frame, before interactions
with other work; this is not measured fleet speed.

Projection outputs differed from int8 by relative RMS 9.61–9.73% on the
random inputs. That is not a final-model quality metric. A next step would
be a reversible six-layer canary on role 5, separate IR directory, exact
comparison and quality tests, followed by immediate restoration of int8.
The queue's numerical policy requires the owner's decision for that serving
rollout. The current fleet continues to serve the original int8 attention.
