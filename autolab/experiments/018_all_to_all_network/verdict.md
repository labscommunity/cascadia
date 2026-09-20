# 018 verdict: this LAN carries expert-parallel traffic for ONE stream, not for the multi-stream load. FAIL at 50 MB/s.

All eleven boxes exchanged 100 kB request/reply messages with all ten others for 12.5 minutes, ramped on rank 0's
clock (`results.json`; ranks 1-8 restarted during the settle and joined at the 30 MB/s step, so the 5 and 15 MB/s
steps are a five-box subset at 1.5 and 6.3 MB/s). Model idle. No box aborted, no kernel complaint about USB or the
NIC anywhere (`kerr = 0`), all eleven workers still serving afterwards.

| target per box | achieved per box | message round trip p50 | p99 | max | > 50 ms | > 200 ms | TCP retransmits |
|---|---|---|---|---|---|---|---|
| (5 boxes) 1.5 MB/s | 1.5-2.2 | 3.1-4.7 ms | 3.9-6.0 ms | 15 ms | 0 | 0 | 0.03 % |
| (5 boxes) 6.3 MB/s | 6.3 | 2.8-5.1 ms | 4.5-7.7 ms | 8 ms | 0 | 0 | 0.01 % |
| **30 MB/s, all 11** | 27-32 | **4.8-14 ms** | **13-38 ms** | 43 ms | 0 | 0 | 0.005 % |
| **50 MB/s, all 11** | **41-45 (never reached 50)** | **38-50 ms** | **73-98 ms** | 226-557 ms | **38 % of messages** | 164 in 3 min | **0.14 %** (ranks 0-4: 0.24 %, 5-7: 0.10 %, 8-10: 0.01 %) |

Reading:

- Every port ran at ~36 % of line rate when the LAN congested, so the ports are not what filled up. The loss is
  graded by GROUP of ranks (0-4, 5-7, 8-10), and the throughput stalled where five boxes sending 6/10 of 42 MB/s
  to the other six adds up to ~126 MB/s: one gigabit. **The boxes are most likely on more than one switch, joined by
  1 GbE links** (to be confirmed by whoever cabled them). All-to-all traffic crosses those links; the pipeline's
  neighbour-to-neighbour traffic (2 MB/s) never noticed.
- Pass criterion was < 0.1 % retransmits and < 1 stall of 200 ms per box-minute at 50 MB/s: failed (0.14 %, up to 9
  per minute on ranks 0 and 2).
- 30 MB/s (what ~39 tok/s would put on the wire) is loss-free, but a symmetric expert-parallel step has 64
  all-to-all sync points, each as slow as its slowest reply: with a p99 of 13-38 ms per exchange a step would be
  around a second. Today's pipeline step at 16 rows is 0.17 s.
- A lone stream's traffic is ~0.3 MB/s of 12 kB messages: the unloaded numbers apply (0.5-0.7 ms per 12 kB round
  trip, 014). **Intra-layer partitioning remains viable as a single-stream mode; multi-stream stays on the
  pipeline, unless the eleven boxes are moved onto one non-blocking switch and this test is repeated.**
- Side result for the crash question: twelve minutes of the heaviest NIC traffic these boxes have seen, with
  `tx_timer_usecs = 0` on the eight boxes that have the knob, and nothing fell over. That is evidence against the
  NIC setting as the cause of rank 0's stop, not proof. (Ranks 8-10 are different hardware: no `cdc_ncm` knob.)
