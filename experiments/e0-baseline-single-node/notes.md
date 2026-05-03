# e0 — single-node baseline (alpha B390, ov-genai + FastDraft K=5)

**Hypothesis:** Re-establish the single-user single-node monolithic baseline with the current Rust port + Phase 14 hardening so all subsequent distributed campaigns have an honest target to beat.

**Setup:**
- alpha (Battlemage Arc B390 dGPU)
- Rust binary: post-PR-3 release build (Phase 14 security hardening)
- Engine: `ov-genai` (LLMPipeline) on GPU
- Model: `C:\cascadia\models\llama-3.1-8b-int4` (pre-existing INT4 IR)
- Draft: `C:\cascadia\models\fastdraft-150m-int8-ov` (Intel's prebuilt FastDraft companion)
- Spec K = 5
- Workload: 256-tok creative-writing prompt ("write a long detailed creative story about a curious robot named Atlas...") — picked because it doesn't EOS short like a factual prompt would (the 8-token Paris answer was useless for amortizing spec decode)
- 5 sequential trials, same prompt, fresh worker process each trial (cold start each time — but the OV cache on alpha keeps the kernel-JIT warm across runs)

## Procedure

`experiments/e0-baseline-single-node/logs/bench.ps1` runs 5 trials and parses each trial's `task done` log line for the engine-internal `tok_s` (which counts actual generated tokens / actual generate() elapsed, NOT wall-clock incl. load).

## Result

| Trial | Generated tokens | Engine elapsed (s) | tok/s |
|------:|-----------------:|-------------------:|------:|
| 1     | 257              | 11.144             | 23.06 |
| 2     | 257              | 11.191             | 22.96 |
| 3     | 257              | 11.168             | 23.01 |
| 4     | 257              | 11.156             | 23.04 |
| 5     | 257              | 11.268             | 22.81 |

**Median: 23.01 tok/s. Spread: 22.81 – 23.06 (1.1%).** Tight. No outliers.

## Conclusion

- Single-node `ov-genai + FastDraft K=5` on alpha for 256-tok creative output = **23.0 tok/s**, single-user, sequential.
- This is lower than the prior python-branch headline of ~27 tok/s — that headline was taken on shorter (64-tok) factual prompts where spec decode amortizes very favorably. For the long-form workload that distributed campaigns will be measured against, 23 is the honest baseline.
- **Bar = 23.0 × 1.20 = 27.6 tok/s sustained, distributed, on the same prompt + max_tokens setting.**

The previous python-autolab d3 distributed peak was 38.49 tok/s — but that was 256-tok output too with K=4 + FastDraft. We need to reverify that holds on the Rust port's `ov-dist-spec` engine in e1.
