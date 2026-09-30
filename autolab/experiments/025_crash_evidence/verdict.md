# 025 verdict: the decode kernel never crashed. It throws, the engine fell back to the CPU, and the box ran out of memory.

Read-only: rank 5 cut the minutes of 021 / 022 out of its own journal and served them on the LAN; rank 0 fetched the
text and paged it into its journal, which the signed status relays (a TEXT channel out of any rank, 22 lines a page).

- Both times the worker was killed by the **kernel's OOM killer** (`Out of memory: Killed process ... cascadia`,
  anon-rss 10.6 GB), not by a fault.
- With `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=32` every fused-MoE `infer()` throws, already in the warm-up
  (`ov_moe_warm ok=false`), with and without the 32-bit offset patch:
  `Exception from src/inference/src/cpp/infer_request.cpp:224: Unable to cast reference from base to derived type`.
- The engine treats a failed device call as transient and sends that call to the host kernels. Telemetry of 021 shows
  it: `ov_moe_calls 0, ov_moe_fallbacks 6-228` per window and the CPU expert cache growing to 9-14 GB on a box whose
  iGPU already owns 49 GB: memory available 250-500 MB, 8 GB of swap full, then the OOM kill, the chain re-forming,
  and the same again.

Two consequences. (1) The decode path is not dead: a failed cast is a software condition (most likely the type of
memory object behind one of this graph's inputs), being researched. (2) The engine's fallback is dangerous on a
fused rank: a layer whose device calls keep failing must be reported and the rank must refuse to serve, not quietly
fill the host with experts it has no room for. To do: latch after N consecutive failures and exit with a clear error.
