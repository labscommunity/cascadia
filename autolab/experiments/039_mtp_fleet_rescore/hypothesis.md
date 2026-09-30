# 039: score the shipped MTP head on the fleet's actual states

033 found first-draft acceptance 0.726 on 36 CPU-reference sequences. Its
inputs need to come from the serving fleet before committing to the large
wiring change. Collect the same 36 rendered prompts, greedily, with the
fleet's current experts, attention, and head. Record actual sampled IDs and
pre-final-norm residuals, and match each completed capture to its API text.

Score the original eight-module head with 033's protocols on the build host.
Prediction: module 0 remains near 0.7; wrong concatenation/normalization
variants remain far worse. Also score the planned int4 MLP, int8 projections
and 65k head weight grids to measure quantization loss before deployment.
The quantized offline calculation uses f32 arithmetic and is not a GPU
parity measurement. Compare family-balanced first-draft rates and prefix
acceptance; preserve input/output alignment checks.

Kill or redirect: module 0 below 0.7, a failed alignment check, or a deployment
format whose acceptance and measured cost cannot improve interactive speed.
Any runtime path must still verify every draft with the main model, preserve
per-stream KV/conv history and exact rewind behavior, and default to off.
