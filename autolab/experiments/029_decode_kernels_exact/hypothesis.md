# 029: the decode kernels on the exact group-32 layers, by turning six 32s into 16s in the plugin

026 showed the GPU plugin's MoE decode kernels (batched GEMV) read experts at the bus limit (125-136 GB/s against
85-92 on the prefill path) and work once the weights are group 64, which is not exact. Why group 32 is refused:
`moe.cpp` sets the kernels' sub-group size with `info.arch >= gpu_arch::xe2 ? 32 : 16` and the GEMV kernels need
group >= 2 x sub-group. On every GPU before Xe2 the SAME code runs with sub-group 16, where group 32 is fine.

The choice is compiled into six places of `libopenvino_intel_gpu_plugin.so` 2026.3.1 (sha256 6f8bba74...), each
`cmp DWORD PTR [info+0xc0], 7|6; mov e?x, 0x20; mov e?x, 0x10; cmov..`: the gather, prefill-SwiGLU and GEMV JIT
constants, the group-size validation, and two dispatch sites (local work size). `bench/patch_gpu_plugin_sg16.py`
turns each `0x20` into `0x10` (originals verified byte for byte) and applies 022's 64-bit expert offset; patched
sha256 8c4bf294.... Found by disassembling the miner's byte-identical copy; no rebuild, nothing to ship but the
patcher inside the overrides.

Engine (binary with `CASCADIA_INKLING_OV_MOE_DECODE_LAYERS`): listed layers keep their ordinary IR, are compiled with
the decode threshold `CASCADIA_INKLING_OV_MOE_DECODE_ROWS` (1: the decode kernels share no expert between rows, 026)
and one-row calls go down UNPADDED; everything larger takes the prefill path as today.

Canary: rank 5, layers 30 and 31; layers 32-35 unchanged except that the prefill path's small kernels also run with
sub-group 16 now. Load check on the highest expert ids enforced at the exact bar (cosine > 0.995 against the host
kernels) for all six layers; LB lines time 1, 2, 3 rows per layer. Self-reverting (signal death or > 12 starts).

Prediction: one-row call 2.98 -> ~2.1-2.2 ms on layers 30-31, two and three rows unchanged, other layers unchanged.
If it holds: all layers of all ranks (030): -5 ms per one-row frame = +8 % at 15 streams, +13 % for a lone stream.
Kill: any check below 0.995, any restart of rank 5, gate failure, a slower prefill path on layers 32-35.
