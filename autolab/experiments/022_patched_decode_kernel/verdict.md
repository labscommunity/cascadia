# 022 verdict: REVERTED BY ITSELF. The 32-bit expert offset is real, but fixing it is not enough.

The research agent found, in the 2026.3.1 source, that the decode kernels index an expert's weights with a 32-bit
product that overflows from expert id 228 at this model's sizes; the kernel text sits readable in the shipped
library, and a same-length patch (`int` -> `long`, 4 sites, sha256-checked, original kept) was applied on rank 5
together with the decode threshold. Result: `P22T patched=1 starts=1`, then `P22X reverted=1 died=1 starts=5`: the
worker still died by a signal on real frames; its next start restored the original library and the prefill path, the
chain re-formed, and the gate passed again (exact) with nobody touching it. Outage ~6 minutes.

So there is a second fault on that path for this graph or GPU (candidates: another 32-bit index, the scratch buffers
sized from the first shape seen, the hand-built graph's routing inputs). Finding it blind, at one fleet restart per
probe, costs more than it promises; the overflow itself is worth reporting upstream. **Closed for now.**

Kept from this: the self-reverting canary pattern (marker + journal check + restart count), `bench/patch_gpu_plugin.py`,
and in the engine (next build): device calls warmed and CHECKED at load on the HIGHEST expert ids against the host
kernels (a layer that disagrees is taken off the device), and configurable row buckets.
