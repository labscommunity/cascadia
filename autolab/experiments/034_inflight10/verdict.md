# 034: negative — restore eleven frames

Ten frames did not deliver the predicted 5% gain. Same experiment tag and
phase names, same binary, both single and concurrent output gates passed:

| phase | eleven frames | ten frames | change |
|---|---:|---:|---:|
| fifteen streams A, steady tokens/s | 24.580 | 24.596 | +0.1% |
| fifteen streams B, steady tokens/s | 24.679 | 24.444 | -1.0% |
| 176 streams, steady tokens/s | 67.974 | 64.846 | -4.6% |

Mean first-token time at fifteen streams stayed about six seconds. The
single-stream repeat rose from 4.60 to 5.79 tokens/s, but one stream uses
only one group and the phrase drafter learned from the first run; this is
not evidence for ten groups. Every request completed, with no worker loss.
The round-time approximation was too optimistic about the longer frames.
Restore eleven groups in the next release (035), which adds the independent
readiness gate. No performance configuration from this experiment is kept.
