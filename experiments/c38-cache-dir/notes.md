# c38: CACHE_DIR cold-start vs warm-start measurement

## Setup
Llama 3.1 8B INT4 on alpha B390 GPU. LLMPipeline init with CACHE_DIR.
3 sequential runs:
1. Clear cache, then load (cold).
2. Load with cache populated (warm).
3. Load with cache populated (warm again).

## Results

| Run | Load time |
|-----|----------:|
| Cold (no cache) | 26.98s |
| Warm 1 | 10.20s |
| Warm 2 | 14.96s |

## Findings

- **CACHE_DIR saves 12-17s on warm load** vs cold (62-44% reduction).
- Cold load = 27s. Warm load = 10-15s. So setting CACHE_DIR essentially
  cuts model load time in half for repeat starts.
- The ~5s variance between warm loads is likely Windows file system
  cache effects (kernel cache files were just touched).

## Recommendation

`CACHE_DIR` should be set by default in the tahoma `ov-genai` engine.
The cost (a few hundred MB of disk per model+device combo) is well worth
the 17-second cold-start cut. Already wired in the engine via
`--ov-cache-dir`; the LOOP follow-up is to make this default-on.

Open: validate same on charlie 140V (Lunar Lake should benefit similarly
since the JIT compile is GPU-plugin level).
