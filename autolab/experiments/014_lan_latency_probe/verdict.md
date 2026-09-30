# 014 verdict: the LAN round trip was 3.3 ms because of a driver timer; it is 0.2-0.3 ms without it. KEEP (in 015).

The adapters are ASIX AX88179-family USB 3 GbE dongles in NCM mode (usb 0b95:1790, 5 Gb/s link, no PCI Ethernet in
the boxes, one Wi-Fi controller). `cdc_ncm` holds a small frame for up to 3 x `tx_timer_usecs` (400 us) waiting for
more to aggregate.

Rank 0 pinging rank 1 (014: only rank 0 switched its timer, the other ranks' probes were killed by the settle
restarts; 015 sets the timer at every worker start on every rank and reports 7 minutes later):

| round trip, us | default (both 400) | one side 0 | both sides 0 (015, ranks 1-7) |
|---|---|---|---|
| 300 pings 2 ms apart, avg (min) | 3281 (2258) | 1968 (1523) | **200-330** (78-220) |
| 12 kB payload, avg | 2994 | 1756 | **480-680** |
| 16 pings 0.5 s apart (link idle), avg | 3337 | 2041 | 570-800 |
| + PM QoS `cpu_dma_latency = 0` | - | 1761 (fast), 2001 (idle) | not needed: no gain worth the watts |

Consequences:

- The pipeline pays 11 hops per token: ~18 ms of the 440 ms trip were this timer. Rank 0's measured round trip fell
  from 575-607 ms to ~409 ms in 015b (together with rank 0 reading replies between guess frames).
- **Expert parallelism is no longer ruled out by latency** (0.25 ms per round trip x 64 layers = 16 ms per token, was
  210 ms). What remains against it is the 1 GbE wire: a lone row's hidden state is 12 kB (f16), so one layer's fan-out
  to six boxes is ~0.6 ms each way through the driver's port, 1.2-1.9 ms per layer all in, against 3.5 ms for reading
  the eight experts on one bus: about -110 ms per token (-27 % of L), at the price of re-sharding 43 GB per box and
  of the multi-stream mode (EP moves ~24 MB per token through one port: the documented 12 tok/s cap).
- `tx_timer_usecs = 0` costs nothing measurable at 176 streams (65.5 tok/s steady in 015b, 64.2 before).

The user's `speedeth` idea (poll-mode NIC driver, 16 us round trips on an I226) points the same way: the waiting is in
the stack, not the wire. These boxes have no PCIe NIC to bypass to, only the USB dongle, and its driver is the only
path to them (no remote hands), so the documented knob was the safe first step; it took 90 % of the latency out.
