# c37: KV cache precision sweep on alpha B390

## Setup
Llama 3.1 8B INT4 on alpha B390 GPU. LLMPipeline plain. 1024-token input,
128-token output. Prompt is a long passage + summarise instruction.
Best of 2 runs.

## Results

| KV precision | tok/s | vs u8 |
|--------------|------:|------:|
| u8 (default) | 74.86 |     — |
| u4           | 67.20 |  -10% |
| i8           | 74.35 |  -1%  |
| i4           | 65.72 |  -12% |
| bf16         | 20.03 |  **-73%** |

(`f8` failed compile — not supported on B390 in OV 2026.1.)

## Findings

1. **U8 (default) wins decisively.** Confirmed across the supported
   precisions on B390 in OV 2026.1.
2. **4-bit KV (u4 / i4) is 10-12% slower** than 8-bit, despite using half
   the memory. The decompression overhead per attention step exceeds the
   bandwidth savings at our context sizes (1K). At larger contexts (16K+)
   the bandwidth pressure may eventually flip the comparison.
3. **bf16 KV is catastrophic (-73%)** — likely forces a bf16 attention
   compute path the Battlemage XMX does not optimise well. Avoid
   completely.
4. Sign (u vs i) makes no difference for KV at the same bit width.

## Recommendation for tahoma

Do not expose `--ov-kv-precision` as a deployment knob. The default U8
is universally optimal on Battlemage. Keep the flag in the engine for
debugging only.

## Hypothesis to test (open follow-up)

At very long context (16K+) where memory bandwidth is more saturated,
u4 KV may eventually beat u8. Worth testing once we have a 16K-capable
test prompt.
