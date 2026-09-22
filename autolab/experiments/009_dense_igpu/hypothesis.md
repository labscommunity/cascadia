# 009 dense layers on the iGPU, swappiness 1, 352 streams

Binary ea80c2a8 (sha fa0b6b79), run.sh with dense-IR generation, overrides = 008b + OV_DENSE + swappiness 1.

Predictions: rank 0's frame at 16 rows 230 -> ~180 ms (dense 60 ms on the CPU -> ~10 on the device),
which puts every rank at 160-200 ms: 176 streams 55.6 -> ~65 tok/s steady; 352 streams (32 rows per
frame) ~75. No more multi-hundred-ms frames from swap-ins. Single stream: rank 0's stage 47.7 -> ~40 ms.
The dense IR was checked against a numpy reference on the fixture (2.9e-7); on the fleet the gates and
the 12 answers decide.
