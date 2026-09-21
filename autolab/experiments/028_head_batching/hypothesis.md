# 028: the last rank shares one output-head call between frames that are already waiting

027 left rank 10 as the stage the ring waits for: 35.9 ms of layers + 11.6 ms of output head per 1.36-row frame,
96 % busy, everyone else 23-35 % idle. The head is one compressed FC over the whole vocabulary: 1.24 GB of int8 read
once per CALL whatever the row count (107 GB/s: nothing left in the kernel). Eleven frames per round = 13.6 GB per
round on rank 10's bus for fifteen rows that would fit in two or three calls.

So: on the last rank, when a decode frame's layers are done and the NEXT decode frame is already readable on the
upstream link, run that frame's layers first and make one head call for both (`CASCADIA_STREAMS_HEAD_BATCH=2`);
each frame still gets its own reply, a stream's rows reach its sampler in position order, anything that is not a
decode frame flushes what is owed first. It is self-limiting: frames only wait at rank 10 while rank 10 is the
slowest stage. Not for a lone stream (`..._MIN_STREAMS=4`): its reply must not wait for a guess frame's layers.
Also: 026's canary is removed (rank 5's plugin library restored, layer 30 back on moe_ov/).

Cost: the first frame's reply waits for the second frame's layers (~36 ms) on a ~540 ms round.
Prediction: rank 10 47.5 -> ~42 ms per frame; the round is then paced by rank 0 (43.7): 544 -> ~495 ms, **+8-10 % at
15 streams** (24.8 -> ~27), nothing at one stream, 176 streams unchanged or slightly better.
Kill: a gate failure; 15-stream steady below 24.5; single stream below 015c's numbers.
