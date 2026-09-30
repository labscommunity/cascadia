# 044: price CPU attention/router overlap before a scheduler rewrite

The queued estimate leaves roughly 3.5 ms of CPU attention/router work in a
41 ms frame. 023 overlapped device calls only. This diagnostic measures real
CPU attention, norm and router kernels concurrently with an independent fused
expert call on one middle rank. It does not schedule two actual frames.

New behavior defaults off (CASCADIA_INKLING_CPU_OVERLAP_BENCH). At load, one
MoE layer measures one and two rows, contexts 128 and 512, 41 iterations after
three warm-ups: CPU alone, device alone, serial, and concurrent. A dedicated
thread and two barriers per iteration include synchronization cost. GPU output
must stay identical; a refused GPU call stops the diagnostic. Reports persist
seven seconds each for the beacon. Attention owns separate scratch KV and
convolution state, borrowing only weights. Model fixture tests check that
probing every layer leaves the live next-token result bit-identical.

Synthetic states and independent operands give an optimistic overlap bound.
The probe excludes attention-output/MLP-output conv and residual glue; it is
not the whole CPU frame. A positive result still needs a legal scheduler,
per-stream dependencies and exact rewind tests. A negative result can avoid
that rewrite. Prediction: at least 0.3 ms saved per layer at two rows; kill
if synchronized overlap saves less than 5% of the serial pair, or if GPU
outputs change. Restrict any fleet run to role 5 with a once-only marker and
12-start/signal-death cutoff. Fleet measurement waits for entry-box recovery.
