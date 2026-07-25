# RFC draft: in-place INT4 GEMV execution mode for the OpenVINO CPU plugin

**Status:** draft for upstream discussion (openvinotoolkit/openvino). NOT yet
filed — per repo policy, verify against the latest OV release and search for
prior art immediately before filing.

## Problem

Heterogeneous phase-split LLM serving on Intel AI PCs (prefill on NPU, decode
on CPU — the same split AMD ships as Ryzen AI hybrid mode and Intel's own NPUW
implements internally between its prefill/generate submodels) pays **~2×
resident weight memory**: every OpenVINO plugin repacks weights into its own
execution format at `compile_model` time.

- CPU: `prepareWeightsMemory` (`src/plugins/intel_cpu/src/nodes/executors/dnnl/dnnl_utils.cpp`)
  always allocates a plugin-owned buffer and reorders into the oneDNN-chosen
  blocked layout (`format_tag::any`); INT4 additionally gets nibble rewrites
  (i4→u4 +8). The `.bin`'s mmapped constants are never executed from.
- NPU: weights are memcpy'd into a transient Level-Zero buffer and transformed
  by an on-device init schedule into resident L0 weights
  (`weightless_graph.cpp`); ~1× resident after load.
- The only cross-model sharing (NPUW weights bank) is keyed **per device**
  (`weights_bank.hpp: unordered_map<std::string /*device*/, DeviceBank>`).

On a 32 GB Lunar Lake AI PC serving a pipeline stage, the duplicated CPU copy
is the difference between a model fitting or not.

## Proposal

An opt-in CPU-plugin execution mode (e.g.
`ov::intel_cpu::inplace_weights(true)` or an `ov::hint`) under which
**M=1 (GEMV) FullyConnected/MatMul with weights-decompression executes
directly from the model's original grouped-INT4 constant memory** (the
mmapped `.bin`), skipping the compile-time reorder for those primitives only.

Scope deliberately narrow:
- M=1 (single-token decode) only — GEMM keeps the blocked repack.
- Grouped u4/i4 + scales/zps in the canonical IR layout.
- Falls back to the current path when the ISA/kernel can't honor it.

## Why this is performance-viable

Decode GEMV is DRAM-bandwidth-bound, not layout-bound: Intel's own kernel
work (arXiv 2508.06753) streams grouped INT4 from the canonical layout at
"within 20-25% of theoretical peak" on Lunar Lake — i.e. ~0.7× DRAM peak, the
same ballpark the repacked path achieves, because the bottleneck is the
memory stream, not the register-blocking that the repack optimizes for GEMM.

Measured motivation (cascadia PR #107, Lunar Lake 258V, Llama-3.2-1B INT4):
NPU-prefill + CPU-decode phase split cuts TTFT 7.7-24× at unchanged decode
tok/s, with the shared host KV ring making the handoff free — the remaining
structural cost of the split is exactly the duplicated CPU weight copy this
RFC removes (steady-state residency would drop to NPU-resident weights + the
OS page cache of the shared .bin).

## Alternatives considered

- Cross-device weights bank: rejected — CPU and NPU execution formats differ;
  neither plugin can consume the other's buffer.
- Remote-tensor-wrapped constants: the per-plugin import/repack happens
  regardless of buffer ownership.
- Weightless blobs / mmap: shares the *source* pages (already the case) but
  not the resident execution copies.

## Open questions

1. Does oneDNN's `weights_decompression` brgemm path accept an external
   plain-layout u4 buffer for M=1, or is a new microkernel needed?
2. Interaction with `dynamic_quantization_group_size` (activation DQ forces
   symmetric weight conversion today — another rewrite).
3. Accuracy parity: the in-place path must be bit-compatible with the
   repacked path for the same primitive.
4. Should the mode be per-model (compile property) or per-primitive
   (automatic for M=1 shapes)?

## Prior-art search to redo at filing time

- `gh search issues --repo=openvinotoolkit/openvino "share weights devices"`
- `gh search prs --repo=openvinotoolkit/openvino "inplace weights"`
- pip index versions openvino (verify latest; re-run the residency repro).
