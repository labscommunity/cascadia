// Internal C++ seam (NOT part of the C ABI): builds the standalone bf16
// round-trip canary graph used by cascadia_runtime_compile_bf16_canary in
// shim.cpp. See bf16_canary.cpp's header comment for what/why.

#pragma once

#include <cstddef>
#include <memory>

namespace ov {
class Model;
}  // namespace ov

namespace cascadia_bf16 {

// `x[n] f32 -> bf16-round-trip -> y[n] f32`. Input tensor name "x", output
// tensor name "y" (matches tools/glm5_attn_ov.py::_bf16_roundtrip_graph so
// the same battery-feeding code works against either).
std::shared_ptr<ov::Model> build_canary_model(size_t n);

}  // namespace cascadia_bf16
