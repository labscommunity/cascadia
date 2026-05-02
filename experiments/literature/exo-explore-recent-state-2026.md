# exo-explore/exo: recent state and competitive position (2025-2026)

**Released:** Active dev; recent commits surveyed Apr 2026.
**What changed:** exo is the closest competitor to Tahoma — distributed LLM inference across heterogeneous consumer hardware. Recent commit pattern shows the project is in **maintenance + UX polish mode** rather than perf development:

- **rdma_ctl docs (PR #1977)**: Updated UI instructions for the macOS RDMA control flow — confirms exo's primary target is **Mac (Apple Silicon) clusters**, not Intel hardware.
- **macOS bug-report window improvements (PR #2003 / #2008)**: SwiftUI window for bug reports on EXO.app menubar.
- **HF rate-limit handling (PR #2009)**: Added retry-loop with HF's `t=` header parsing because exo "bursts ~200 HF Hub-API requests on every cold start, blowing past the anonymous 500-req/5-min budget." Indicates exo doesn't pre-cache models well — opportunity for Tahoma.
- **URLSession cache fix (PR #2005)**: ClusterStateService polling at 2 Hz was causing ~500-620 KB/sec of disk writes; fixed with ephemeral session.

**What's NOT in recent exo dev**: No new perf kernels, no new quantization support, no Intel GPU optimization, no PagedAttention work, no continuous batching, no speculative decoding. This is exo's strategic gap.

**Tahoma's positioning**:
- exo splits LAYERS across devices (pipeline parallelism). Slow for short generations.
- Tahoma can split TURNS across devices (Send turn N to device A while device B works on turn N+1).
- exo uses MLX on Apple, tinygrad on others; no native OV path; no Intel-XMX-aware kernels.
- Tahoma is OV-native on Intel + can dispatch to MLX on Apple = cross-platform with both vendors' best kernels.
- exo cold-start storms HF; Tahoma should ship pre-converted OV-IR models from `huggingface.co/OpenVINO/...` mirror.

**Headline perf claim (if any):** N/A from recent commits.
**How to use it from optimum-intel / OV runtime:** N/A — exo is a competitor.
**Intel GPU applicability:** LOW — exo treats Intel GPU as a fallback, not a target.
**Open hypothesis it generates for us:** Build a "head-to-head" microbench on alpha + charlie: same Llama-3-8B, same prompts, same total system memory. Run (a) exo native (b) Tahoma OV-GenAI (c) llama.cpp distributed mpi. Hypothesis: at batch=1 single-prompt latency, Tahoma is ≥1.8x faster than exo because of OV PagedAttention + KV-cache-on-GPU vs exo's no-cache-reuse pipeline parallelism.

Sources:
- https://github.com/exo-explore/exo (recent commits)
- exo PR #1977, #2003, #2005, #2008, #2009
