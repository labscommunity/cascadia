# c24: Heterogeneous draft device — does NPU-as-draft + GPU-as-target win?

## Hypothesis
Lunar Lake's NPU (charlie) and Battlemage iGPU host's NPU (alpha) are both
idle while the GPU runs the target model. Putting the FastDraft 150M companion
on the NPU should give "free" draft compute since it's running on a separate
device — no GPU-context-switch cost between target and draft passes.

## Setup
- Llama 3.1 8B INT4 (target) on GPU.
- FastDraft 150M INT8 (draft) on {GPU, CPU, NPU}.
- K=5, 64-tok factual.

## Results
| Hardware | Target | Draft | tok/s |
|---|---|---|---|
| alpha (B390+host) | GPU | GPU | 131.29 (sanity check) |
| alpha (B390+host) | GPU | CPU | 132.40 |
| alpha (B390+host) | GPU | NPU | 33.21 (-75%) |
| charlie (140V+host) | GPU | CPU | 81.93 |
| charlie (140V+host) | GPU | NPU | 31.83 |

## Findings

1. **NPU draft is a 4× regression** on both targets. The NPU compiles the
   draft model successfully (no error, just `[function-outliner-vertical-fusion]
   warnings during compile`), but the cross-device synchronisation between
   GPU target and NPU draft kills throughput far worse than it saves.
2. **CPU draft basically ties GPU draft** on alpha (132.4 vs 131.3 — within
   noise). On charlie CPU draft costs ~15% (82 vs 96). This is interesting:
   on Battlemage with a beefier dGPU and host CPU, putting the small draft
   on the host CPU costs nothing because the GPU target runs in parallel.
3. **NPU is not a viable draft device** for spec decode on either platform
   in OV 2026.1 — at least with the GenAI HETERO compile we're getting.

## Why NPU loses
The 150M draft on NPU likely goes through an extra device bridge round-trip
per spec round (5 tokens × ~10 ms each vs ~1 ms on GPU). The amortised win
from "free compute on idle hardware" is dwarfed by the bridge cost.

## Action
Don't expose NPU as a draft device option in tahoma's `ov-genai` engine.
For systems with no GPU (CPU-only fleet), draft on CPU is fine. For
systems with GPU, draft on the same GPU is optimal.

## Open
- Could the NPU be useful for *prefill* of long prompts only? (TODO)
- Could the NPU be useful as a *target* for short prompts where the
  draft/spec overhead doesn't help anyway? (Tested in c26.)
