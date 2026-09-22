# 037: keep the capture tooling; disable writes after collection

The corrected INKCAP02 format stores the original f32 residuals. Both
fleet gates passed. The paired fifteen-stream runs were 24.427 / 24.485
tokens/s (035, capture off: 24.212 / 24.631); no material overhead was
measured. Every timed request completed. The unchanged model continues
to use its usual expert, attention and output-head backends.

Collected all 36 original rendered prompts for 039, three per family,
160 generated tokens each. Every prompt length matches the original 033
token IDs; every decoded capture matches its entire completion API text;
all captured residuals are finite. The raw completion endpoint skips
structural tokens, which the converter now mirrors explicitly. Actual
sampled IDs, including structural tokens, remain in the dataset. A sampled
file reached residual magnitude 4,272,968 before the final RMSNorm.

The original f16 format was rejected before corpus collection because
those large residuals overflowed. Its two phase results are retained as
`phases-f16.json`, not as usable state data. The f32 fix adds a large-value
regression; budget, rewind, slot reuse and direct-return token agreement
tests pass. Capture still fails closed on its own I/O or quota errors,
without failing inference.

The 2048 MiB directory budget and restricted read-only relay route remain.
038 disables new writes after the complete corpus was fetched. The raw
files and tensor dumps stay outside the repository, with scoring on the
build host. No worker shell, updater, key, tunnel or file-server changes.

Binary: cascadia-bf6540ea. Overrides: 037_fleet_state_capture.env.
