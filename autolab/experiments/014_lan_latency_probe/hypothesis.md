# 014: what does a LAN round trip cost here, and why? (overrides only, fleet idle)

**Why it matters.** A lone token's trip through the fleet (L = 440 ms) is 26 GB read over ONE memory bus at a
time. Expert parallelism would let one layer's eight experts be read on eight buses at once, but it costs one
LAN round trip per layer (64 per token). At the 2-3.5 ms `ping` measured on this LAN that is 130-220 ms per token
and EP is pointless; at 0.2-0.3 ms it is 15-20 ms and EP would take ~190 ms off L. The user's `speedeth` project
(poll-mode I226 driver on Windows, 16 us RTT) makes the same point: the stack's waiting, not the wire, is the cost.

**Suspect.** The adapters are USB `cdc_ncm`. That driver holds a small frame back for up to
3 x `tx_timer_usecs` (default 400 us -> 1.2 ms) hoping to aggregate more into one USB transfer. Both directions
of a ping pay it: 2.4 ms. `tx_timer_usecs = 0` is documented ("disable aggregation") and is a plain sysfs write.
Second suspect: deep C-states (PM QoS `cpu_dma_latency = 0` holds cores in C0/C1).

**Method.** A background block in `fleet-overrides.env` on every rank, 7 minutes after the worker starts:
A = as is; B = `tx_timer_usecs = 0` on every box; C = B + PM QoS hold. In each phase the box pings its next rank:
idle (16 x 0.5 s apart), fast (300 x 2 ms apart), and 12 kB payloads (one f16 hidden state). Results leave as
integers (us) on a "stage profile" line that the beacon relays; `bench/probe_read.py` prints them. The timer is
restored at the end, by a watchdog if the neighbour goes silent, and at the next worker start if the probe was cut.

**Prediction.** A: ~2.5 ms. B: 0.3-0.6 ms. C: a little lower at idle, the same when pinging fast.
