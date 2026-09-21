# 027: rank 0's dense layers through the fused-experts op

At 15 streams rank 0 is the stage everyone waits for: 51.9 ms a frame, 96 % busy, the others 35-42 ms and 23-35 %
idle (026). Its two dense MLPs (24,576 wide, 255 MB of int4 each) run as three compressed MatMuls that read at
~26-31 GB/s: 19 ms of the frame. A routed layer's eight experts are exactly as many bytes and take 3 ms through the
plugin's fused-experts op. So: cut the inner neurons into eight slices as wide as a routed expert, every row selects
all eight with weight 1 (`down(silu(gate x) * up x)` is a sum over inner neurons; gate/up are cut by rows, down by
columns, on the bins' own 32-weight groups: nothing is requantised). `run.sh` writes `dense_moe_ov/`; the engine
compares the new form with the old one at load and detaches it on disagreement (DM lines).

Prediction: rank 0 51.9 -> ~39 ms; the pace is then set by rank 10 (35 + 11.6 ms of head): +9-11 % at 15 streams.
Kill: a gate failure, DM cosine <= 0.9999, rank 0 slower.
