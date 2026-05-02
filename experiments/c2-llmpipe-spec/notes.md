# c2: LLMPipeline + draft_model — K-sweep

## Setup

- Target: Llama 3.1 8B INT4 at `C:\cascadia\models\llama-3.1-8b-int4`
- Draft: `srang992/Llama-3.2-1B-Instruct-ov-INT4` (HF-published OV INT4 1B)
  — used because our optimum-cli-exported 1B IR has `beam_idx`/`attention_mask`
  parameters that LLMPipeline rejects (c2-1 error).
- Hardware: alpha B390 GPU
- Prompt: "What is the capital of France?" (factual, short)
- Output: 64 tokens

## Results

| ID | K | Decode tok/s | Δ vs plain (96.4) |
|---|---|---|---|
| c2-1b | 5 | 87.63 | -9.1% |
| c2-2 | 3 | 89.82 | -6.8% |
| c2-3 | 7 | 100.26 | +4.0% |
| c2-4 | 10 | **100.90** | **+4.7%** |
| c2-5 | 15 | 100.09 | +3.8% |

For a 64-token short factual prompt on alpha B390, **spec decode through LLMPipeline gives +4-5% at best**. K=10 is the sweet spot. Lower K (3-5) is actively *harmful* — the per-spec-round cost (one draft pass + one verify pass) eats more time than the saved tokens give back when accept rate is moderate.

## Why the gain is so small

- Plain LLMPipeline already runs at ~10 ms/token on alpha B390. The draft model (1B) at GPU compute takes ~3-4 ms/round, plus round-trip overhead.
- For 64 tokens of factual output the prompt-prefill cost (~75 ms) dominates the difference; spec decode amortises poorly over only 64 decode steps.
- Compare to `c0-3` (ov-spec K=4, accept=0.50 → 13.83 tok/s) which is a totally different baseline (raw OV core, slower per-token compute → spec helps more).

## Next: c6 tests at 256 tokens, where spec decode should help more.
