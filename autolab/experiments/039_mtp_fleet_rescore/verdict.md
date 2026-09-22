# 039: shipped MTP head misses the fleet acceptance bar

All 36 original rendered 033 prompts were collected from the actual fleet,
with 5,760 sampled tokens and 5,724 first-draft targets. All f32 captures were
finite, prompt rows matched original token IDs, and decoded full responses
matched the completion API exactly. No f16-overflow captures were used.

| Module-0 calculation | First-draft agreement |
|---|---:|
| 033 CPU-reference sequences, original full head | 0.726 |
| Fleet sequences, original full head | 0.6681 |
| Fleet sequences, original head restricted to 65,536 tokens | 0.6454 |
| Fleet sequences, int4 MLP + int8 projections + int8 65k head | 0.6436 |

The predeclared 0.70 bar is missed. Most deployment-format loss comes from
the vocabulary cutoff; quantizing the weights loses another 0.17 percentage
points. This offline quantized calculation uses f32 arithmetic, so it is not
GPU parity. Story is weakest (0.507 original / 0.482 deployed grids), while
code and arithmetic remain 0.75 or better on the deployed grids.

Teeth checks separate correct wiring from mistakes: swapped inputs 0.00035,
raw pre-final-norm input 0.0220, no temporal context 0.3529. Reapplying the
original bf16 output head agrees with actual fleet int8-head samples 0.9212;
the fleet's sampled IDs remain the ground truth, not this recomputation.

Eight-module expected accepted drafts are 2.128 with true-token context,
2.007 with self-fed context, and 0.769 without deeper-module context. These
are offline chain measurements, not runtime throughput. One exported module
cannot inherit the eight-module speed projection. Fifteen streams also have
little rank-0 slack and would pay extra expert rows to verify every guess.

Redirect the large wiring experiment to qualification of a head trained on
fleet states, as the queued conditional EAGLE-style study requests. Keep the
validated export, delivery helper and runtime design available. Do not deploy
unqualified MTP assets merely because the export is ready. A trained candidate
must pass a disjoint held-out acceptance test and cost measurement before
runtime deployment. This closes the re-score measurement, not the remaining
training or multi-stream runtime work.
