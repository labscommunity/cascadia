# 042: train a draft head on fleet states after shipped MTP misses its bar

039 measured original module-0 agreement 0.668 and deployment-format agreement
0.644, below the 0.70 qualification bar. The conditional queued training study
is now warranted. This is a head conditioned on hidden states for interactive
3–15-stream speculation, not the cancelled 016 text-only drafter distillation.

Collect 117 distinct prompts from the remaining 033 corpus, excluding every
rendered prompt duplicated in the original 36 held-out sequences. Use the
actual fleet at 15 streams and 160 generated tokens per prompt. Collection
also supplies the at-least-ten-minute twelve-family load for expert counters
040; counter instrumentation and temporary capture do not alter numerics.
Require finite f32 states, exact API-text matching and original token IDs.

Train on the build host with a fixed disjoint train/validation split inside
these 117 sequences; reserve all original 36 sequences for the final test.
The first bounded candidate is a small feature predictor conditioned on the
current post-final-norm residual and the embedding of the next verified token,
using the frozen output head. Also assess whether adapting the existing MTP
projection can retain its learned temporal context more economically. Any
training recipe and budget must be fixed before reading final test results.

The input idea follows EAGLE (Li et al., ICML 2024), whose original method
conditions feature extrapolation on a token advanced by one time step:
https://proceedings.mlr.press/v235/li24bt.html . Our final-layer feature and
Inkling-specific candidate are an experiment, not a reproduction of EAGLE.

Prediction: fleet-specific adaptation can recover at least five percentage
points from the shipped deployment grid. Qualification requires held-out
first-token agreement >=0.70 and a later bounded device cost study compatible
with rank-0 slack. Kill a bounded candidate below that bar, on train/test
leakage, non-finite state/loss, or excessive deployment cost. Never claim
training-set accuracy or an offline renewal model as measured fleet speed.
Keep candidate code and assets off the serving path until qualified.
