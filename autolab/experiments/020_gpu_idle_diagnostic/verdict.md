# 020 verdict: the attention calls are device-bound; input-tensor reuse is worth nothing; the MoE calls could not be measured this way, but the reason for their fixed cost was found in our own settings.

15 streams: 23.1-23.3 tok/s steady (as 019). Rank 5 with per-primitive profiling (it costs that rank ~3 ms a frame):

| per frame, 1.36 rows | wall | inside `infer()` | device execution |
|---|---|---|---|
| 12 attention calls | 13.0 ms (9.7 without profiling) | 11.8 | **8.0 ms** (0.78 GB: 97 GB/s) |
| 6 fused-MoE calls | 25.5 | 24.9 | 0.18 (the fused primitive does not report its kernels) |

- Attention: ~83 % of the call is the device executing. Host overhead is ~1.6 ms per frame, not the 4 ms guessed.
- Input reuse (ranks 1-4 vs 6-7, same hardware): 9.56-9.71 vs 9.71-9.72 ms attention, 20.7-21.4 vs 20.6-20.9 ms MoE per
  frame. **No difference. Closed; the flag stays off.**
- MoE: no device times. But reading our own notes on the plugin: the fleet runs with
  `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=0` (installer and engine default), because on the Windows driver the plugin's
  decode kernel (batched GEMV, 3 kernels, no host synchronisation) crashed. With 0, **every decode call takes the
  op's PREFILL path** (grouped GEMM with host waits, a CPU-side mask and blocking copies): the likely source of the
  ~1.5 ms fixed cost per MoE call, 9 ms of a 35 ms frame. It was never tried on the Linux driver. That is 021.
