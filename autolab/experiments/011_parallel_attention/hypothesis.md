# 011 a frame's rows attend concurrently

Binary a94ca473 (sha b7e149a4), overrides unchanged from 009.
At 16 rows a frame's attention is 33-54 ms of 160-220, of which the device call (projections) is
14-15 ms; the rest is the per-row part, run row after row on CPUs that idle since the experts left.
Prediction: attention 33-54 -> ~20 ms per frame, slowest stage 221 -> ~195 ms:
176 streams 57.9 -> ~64 tok/s steady. Output must stay as in 009 (the change is bit-identical per row).
