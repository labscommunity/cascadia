# 024 verdict: KEEP. First token in 2 s instead of 5; a burst of 15 waits 7 s instead of 31. Config only.

`CASCADIA_STREAMS_PREFILL_WINDOW=8` (a prompt goes down as 8-row windows, one per group turn, pipelined across the
stages instead of one fat frame crossing them in series). Gates: exact, as before.

| | before (019/020) | windows of 8 |
|---|---|---|
| first token, one request alone | 4.8-6.0 s | **2.0-2.1 s** |
| first token, 15 requests arriving 1 s apart | - | **median 2.5 s, max 3.3 s** |
| first token, 15 requests at once | median 31 s, max 37-45 s | **median 6.9 s, max 10.0 s** |
| steady decode, 15 streams | 23.1-23.4 tok/s | 24.2-24.6 tok/s |
| whole phase, 15 streams x 128 tokens | 15.7-19.0 tok/s | **22.5 tok/s** |

Fat prefill frames (up to 8 prompts, 150-250 rows, ~1 s per stage) no longer sit in front of everyone's decode frames.

Side result (read-only page-out of rank 0's journal): in the boot that ended when rank 0 stopped dead (13:38:30 CDT)
the kernel logged NOTHING after 09:00 that morning: no GPU hang or reset, no USB / NIC error, no OOM, no thermal or
machine-check line, and the worker's own last lines are ordinary admissions. The box simply stopped, and came back
with its clock reset to July. That is what a power or firmware-level event looks like, not a software fault; it does
not prove the load did not provoke it.
