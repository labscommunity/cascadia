# 007 verdict: KEEP. Fused f16 MoE on every rank: 45.7 tok/s steady, 3.0 tok/s on an unseen prompt, answers 12/12.

Gates: single PASS (2 of 3 prompts exact for 32 tokens; the third departs at character 81 with an
equivalent phrase: "simple arithmetic question: 6 x 7 = 42" -> "simple multiplication question.
6 x 7 = 42"), 8 side by side PASS. Half precision on the device is not bit-identical to the CPU
path, so an answer-level check was added: 12 questions with known answers, decoded side by side to
the final answer: **12/12 correct** (Paris, 42, 391, 2,3,5,7,11, Mars, Au, Shakespeare, Pacific,
366, 12, Buenos dias, carbon dioxide).

| phase | 006 (CPU experts) | 007 (3 of 6 MoE layers per rank on the iGPU, f16) |
|---|---|---|
| single stream, unseen prompts | 1.99 tok/s | **2.59, 3.04 tok/s** (TTFT 5.7-6.0 s) |
| gate prompts (memorised by the drafter) | 8.5 / 6.5 / 5.3 | **9.76 / 7.67 / 4.6** |
| 48 streams: steady / aggregate / TTFT mean | 26.4 / 18.7 / 44 s | **33.8 / 23.2 / 37 s** |
| 176 streams | 29.6 / 19.4 (005) | **44.6 / 26.8** |
| 264 streams | 34.9 / 20.6 | **45.7 / 29.1** (251 of 264 completed: tunnel resets) |

Per rank at 264 streams (19 rows/frame): 25 W ranks 316-376 ms/frame (16.3-19.1 ms/row), 60 W
ranks 200-245. Fleet utilization 84 %.

Two problems, both on rank 1 (layers 6-11, fused 6, 7, 8):
- 247 of 8979 fused calls (2.8 %) returned a non-finite value and fell back to the CPU path; every
  other rank had none. This is the known overflow INSIDE an expert on layer 8 (its shared expert's
  down projection reaches -94909 before any routing weight is applied); the weight rescale cannot
  reach it. The fallback kept the output right, but
- the fallbacks pulled layer 8's experts into the CPU cache (+5.7 GiB): rank 1 ended with 289 MiB
  available, swapping 492 pages/s, and became the slowest stage (376 ms/frame, 97.7 % busy).
Next: layer 8 back on the CPU (or regenerated with attenuated `up` scales), and the other three
layers of every rank onto the iGPU (008).
