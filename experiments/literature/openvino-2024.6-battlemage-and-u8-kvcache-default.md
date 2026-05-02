# OpenVINO: Battlemage Arc B-series support + U8 KV cache default (2024.6.0)

**Released:** 2024-12 (2024-12-19)
**What changed:** First release with Intel Arc B-Series (Battlemage) GPU support. Device-memory-copy optimizations for Battlemage because B-series does not use L2 cache for host↔device transfers. KV cache flipped to *default* asymmetric U8 (was FP16/INT8 opt-in). NPU inference time + LLM perf improved. Lunar Lake (Core Ultra Series 2) on Windows.
**Headline perf claim (if any):** No %, but "reducing memory stress for LLMs and increasing their performance" via the U8 KV default.
**How to use it from optimum-intel / OV runtime:** U8 KV cache is now automatic; you can override:
```python
core.set_property("GPU", {"KV_CACHE_PRECISION": "f16"})  # opt out to FP16
```
For Battlemage device-memory copies, no env flag — improvements are in the plugin internally. To opt into Battlemage explicitly use `device="GPU.1"` if your host has both iGPU+dGPU.
**Intel GPU applicability:** HIGH for Arc B390 Battlemage — this is the release that introduced first-class B-series support. HIGH for Arc 140V Lunar Lake (Windows). The B-series device-copy optimization is specifically tailored to skipping the missing L2.
**Open hypothesis it generates for us:** On charlie (B390) compare 2024.5 vs 2024.6 OpenVINO host→device tensor transfer time for a 4K-token prompt. Hypothesis: 2024.6 reduces transfer overhead by ≥20% on B390 because of the no-L2 device-copy path.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2024.6.0
- https://www.phoronix.com/news/OpenVINO-2024.6-Released
