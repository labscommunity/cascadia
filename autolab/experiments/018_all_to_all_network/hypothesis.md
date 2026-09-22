# 018: can this LAN carry expert-parallel traffic from every box at once? (overrides only, model idle)

**Why.** A symmetric expert-parallel layout (every box keeps its six layers' attention AND serves 1/11 of every
layer's experts) puts ~0.8 MB per token on every box's port in each direction: 50 MB/s at today's 65 tok/s, as
request/reply bursts to ten peers at once, 64 times per step. Whether eleven USB 1 GbE dongles and the venue switch
take that without loss (many-to-one bursts overflow small switch buffers; one retransmit timeout is ~200 ms) decides
whether that layout is worth building. The user asked for this test before anything else.

**Method.** `bench/nettest.py` on every box (from the overrides, 7 minutes after the worker starts, scheduled on rank
0's clock): each box echoes 100 kB messages with each of the ten others, paced to 1, 5, 15, 30, 50 MB/s transmitted
per box, 3 minutes per step. Per step and box: achieved rate (interface counters), message round trip p50 / p99 / max,
stalls over 50 and 200 ms, TCP retransmits, kernel complaints about USB / the NIC. Self-limiting: > 3 % retransmits
-> the box stops climbing; a kernel complaint -> it stops sending.

**Risk, stated before the run.** Rank 0 stopped dead two hours ago for a reason nobody has seen yet, and the NIC
path (USB dongle, `cdc_ncm` with its aggregation timer off) is one of three suspects. This test loads exactly that
path on all eleven boxes. It runs while someone is at the venue.

**Prediction.** Up to 30 MB/s: p99 under 10 ms, no retransmits. At 50 MB/s: some retransmits from many-to-one
bursts, p99 20-50 ms, a handful of 200 ms stalls. **Pass for building the layout:** at 50 MB/s, retransmits < 0.1 %
of segments and fewer than one 200 ms stall per box per minute.
