#!/usr/bin/env python3
"""patch_sg16.py LIB [--write OUT]: OpenVINO 2026.3.1 GPU plugin, fused-MoE kernels on Xe2 and newer.

Two same-length patches of libopenvino_intel_gpu_plugin.so (sha256 6f8bba74...):
 1. the decode kernels' 32-bit expert offset (4 sites of embedded OpenCL text): int -> long (autolab 022);
 2. the MoE implementation's sub-group size, `info.arch >= gpu_arch::xe2 ? 32 : 16` (moe.cpp, six compiled sites:
    gather / prefill-swiglu / GEMV jit constants, the group-size validation, two dispatch sites): the 32 becomes 16,
    i.e. the configuration the same code runs with on every GPU before Xe2. With sub-group 16 the decode (batched
    GEMV) kernels accept int4 group 32 (they need group >= 2 x sub-group), which is how every Inkling expert is stored.
"""
import hashlib, sys
ORIG = "6f8bba74029e4fdf86fa8d57fabaae00c3a74ea880aaa8003270089763b1890e"
A = b"const int expert_wei_size=INTERMEDIATE_SIZE * HIDDEN_SIZE"
B = b"const long expert_wei_size=INTERMEDIATE_SIZE *HIDDEN_SIZE"
# (file offset of the instruction, original bytes, patched bytes): mov e?x, 0x20 -> mov e?x, 0x10
SITES = [
    (0x11953ed, "b820000000", "b810000000"),   # gather kernel jit: SUBGROUP_SIZE
    (0x11955f5, "b820000000", "b810000000"),   # GEMV group-size validation (sg)
    (0x119575a, "b820000000", "b810000000"),   # GEMV kernels jit: SUBGROUP_SIZE
    (0x1195ead, "b820000000", "b810000000"),   # prefill swiglu jit: SUBGROUP_SIZE
    (0x11a3df1, "b820000000", "b810000000"),   # dispatch (decode path): local size
    (0x11a905d, "ba20000000", "ba10000000"),   # dispatch (prefill path): local size
]
def patch(data):
    assert data.count(A) == 4 and data.count(B) == 0, (data.count(A), data.count(B))
    out = bytearray(data.replace(A, B))
    for off, old, new in SITES:
        old, new = bytes.fromhex(old), bytes.fromhex(new)
        assert bytes(out[off:off + len(old)]) == old, (hex(off), bytes(out[off:off + 8]).hex())
        out[off:off + len(new)] = new
    assert len(out) == len(data)
    return bytes(out)
if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    assert sha == ORIG, sha
    new = patch(data)
    print("patched sha256", hashlib.sha256(new).hexdigest(), "differing bytes", sum(a != b for a, b in zip(data, new)))
    if len(sys.argv) > 3 and sys.argv[2] == "--write":
        open(sys.argv[3], "wb").write(new)
