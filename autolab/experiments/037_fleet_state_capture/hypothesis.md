# 037: capture the residuals the fleet actually generates

The shipped MTP head must be scored on the fleet's f16 expert/int8 attention
states. Capture final or rank-boundary residuals in the Inkling runner,
behind CASCADIA_STREAMS_CAPTURE_FINAL / CAPTURE_BOUNDARY (off by default).
Use a 2048 MiB cumulative directory budget, f16 records, fixed positions,
rewind truncation, distinct stream-instance files, and a completion manifest.
The sampled next-token IDs come from the actual last-rank sampler.

Fetch completed files through a restricted read-only server and the entry
box's existing API relay. No new operator tunnel, keys or worker shell.
Prediction: capture off has no measurable cost; capture on adds under 1 ms
per frame. Failures stop capture without stopping inference. The unit tests
cover budget exhaustion and rewinds; the pipeline test covers prompt windows,
slot reuse, and capture/output-token agreement through the direct reply link.
Both normal output gates must pass before collecting the 36-prompt corpus.
