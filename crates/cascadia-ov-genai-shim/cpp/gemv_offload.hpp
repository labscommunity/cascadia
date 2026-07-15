// Internal C++ seam (NOT part of the C ABI): the CascadiaInt4Gemv custom-op
// offload used by cascadia_runtime_compile_gemv_offload in shim.cpp.
//
// offload_int4_gemv registers the op with the core (idempotent per core) and
// rewrites every NNCF sym-INT4 decompress→MatMul chain
//   Const(i4 [N,G,g]) → Convert(f16) → Multiply(scales f16 [N,G,1])
//     → Reshape([N,K]) → MatMul(act, ..., transpose_b=true)
// into a CascadiaInt4Gemv node that keeps the weight/scale Constants as op
// members: they never appear as graph constants, so the CPU plugin cannot
// repack them at compile — resident weights stay the read_model mmap pages
// of the original .bin. Returns the number of MatMuls offloaded.

#pragma once

#include <cstdint>
#include <memory>

namespace ov {
class Model;
class Core;
}  // namespace ov

namespace cascadia_gemv {

uint32_t offload_int4_gemv(ov::Core& core, const std::shared_ptr<ov::Model>& model);

}  // namespace cascadia_gemv
