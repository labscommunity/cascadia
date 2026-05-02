# c42: GGUF reader path — does LLMPipeline load .gguf directly?

## Setup
charlie 140V (Lunar Lake) GPU. Llama 3.2 1B Q4_K_M GGUF
(`hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF`,
`llama-3.2-1b-instruct-q4_k_m.gguf`, ~770 MB). LLMPipeline init with the
.gguf path. Compare to native OV IR (`srang992/Llama-3.2-1B-Instruct-ov-INT4`)
on same hardware.

## Results

| Format | Load time | Decode tok/s |
|--------|----------:|-------------:|
| Native OV INT4 IR (`srang992`) | ~5-10s | **211.39** |
| **GGUF Q4_K_M** | **30.7s** | **136.38** (**-35%**) |

## Findings

1. **GGUF reader path works in OV 2026.1**, no error, produces correct output.
2. **Load time is ~3-6× slower** for GGUF — the OV runtime converts GGUF
   tensors to its internal format on the fly. Should benefit from
   CACHE_DIR (untested).
3. **Decode throughput is 35% lower** than native OV INT4 IR. Likely
   reasons: GGUF Q4_K_M uses a different group structure (super-block
   based) that doesn't map perfectly to the XMX dynamic-quant code
   paths the native IR exercises.
4. **First text correct**: "The capital of France is Paris." — quality
   intact.

## Recommendation

**Native OV INT4 IR is the right production format on Intel GPU.** GGUF
is useful for:
- Prototyping (no `optimum-cli export` step required).
- Cross-engine compatibility (same .gguf works in llama.cpp, ollama,
  llama-cpp-python, ...).
- Quality regression testing when accuracy diverges (Q4_K_M and INT4
  use different quant strategies).

For tahoma: do not switch the default format to GGUF. Keep optimum-cli
INT4 export.

## Open
- Test GGUF load with CACHE_DIR — does it persist the conversion?
- Test 8B GGUF Q4_K_M on alpha B390 — does the relative penalty hold?
- Benchmark a few different GGUF quant types (Q5_K_M, Q8_0) for the
  perf/accuracy curve.
