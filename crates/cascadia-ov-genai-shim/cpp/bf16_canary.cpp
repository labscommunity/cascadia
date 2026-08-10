// C++ port of tools/glm5_attn_ov.py's `_bf16_roundtrip` / `_bf16_roundtrip_graph`
// (runtime repo, Task 2). Ported by hand, op for op — keep this in lockstep
// with the Python source; a divergence here would make the on-device canary
// (spec Sec9 gate 0 / glm5-attn-igpu-t3 Task 5 Step 0b) test a DIFFERENT
// subgraph than the one every exported attn_ov/layer_NN.xml actually embeds
// after each linear/RMSNorm/rope.
//
// NOT implemented as Convert(f32->bf16)->Convert(bf16->f32): that pair was
// observed silently elided as an identity by the CPU plugin under
// INFERENCE_PRECISION_HINT=f32 (the exact config every compile in this shim
// uses) — see the Python `_bf16_roundtrip` docstring for the full story.
// Implemented instead as exact arithmetic: bisect the power-of-two exponent
// bracketing |x| (comparisons/multiplies by powers of two are exact in
// IEEE754), then round(x / ulp, half_to_even) * ulp. Zero and NaN are
// special-cased (the bisection has no valid exponent for either).

#include "bf16_canary.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <openvino/core/model.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/equal.hpp>
// OpenVINO spells this `greater_eq.hpp` (cf. `less_eq.hpp`), not
// `greater_equal.hpp` — the latter does not exist in the include tree.
#include <openvino/op/greater_eq.hpp>
#include <openvino/op/is_nan.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/round.hpp>
#include <openvino/op/select.hpp>

namespace cascadia_bf16 {

namespace {

using ov::op::v0::Abs;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v1::Add;
using ov::op::v1::Divide;
using ov::op::v1::Equal;
using ov::op::v1::GreaterEqual;
using ov::op::v1::Multiply;
using ov::op::v1::Select;

std::shared_ptr<Constant> scalar_f32(float v) {
    return std::make_shared<Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{v});
}

// Strictly decreasing powers of two, sum=127 — IDENTICAL to Python's
// `_BF16_BISECT_STEPS`. The starting floor 2**-64 (not the true f32 floor
// 2**-127) is deliberate: Python's fix-round-1 note explains that starting at
// the actual subnormal boundary got silently flushed to zero during
// development (FTZ/DAZ). Keep these two lists in lockstep.
constexpr int kBisectSteps[] = {64, 32, 16, 8, 4, 2, 1};

}  // namespace

std::shared_ptr<ov::Model> build_canary_model(size_t n) {
    auto x = std::make_shared<Parameter>(ov::element::f32,
                                          ov::PartialShape{static_cast<int64_t>(n)});
    x->set_friendly_name("x");
    x->get_output_tensor(0).set_names({"x"});

    // std::ldexp(1.0f, k) == 2**k EXACTLY by definition (IEEE754 scales the
    // exponent field directly, no rounding) for any k in range -- unlike
    // std::exp2(float(k)), which routes through a transcendental
    // approximation and is not guaranteed exact even for integral inputs.
    // This bisection's correctness depends on every power-of-two constant
    // being bit-exact (see the module doc), so ldexp is not a style
    // preference here.
    auto ax = std::make_shared<Abs>(x);
    std::shared_ptr<ov::Node> e = scalar_f32(-64.0f);
    std::shared_ptr<ov::Node> p = scalar_f32(std::ldexp(1.0f, -64));
    for (int step : kBisectSteps) {
        auto cand_e = std::make_shared<Add>(e, scalar_f32(static_cast<float>(step)));
        auto cand_p = std::make_shared<Multiply>(p, scalar_f32(std::ldexp(1.0f, step)));
        auto take = std::make_shared<GreaterEqual>(ax, cand_p);
        e = std::make_shared<Select>(take, cand_e, e);
        p = std::make_shared<Select>(take, cand_p, p);
    }
    auto ulp = std::make_shared<Multiply>(p, scalar_f32(std::ldexp(1.0f, -7)));
    auto rounded = std::make_shared<ov::op::v5::Round>(
        std::make_shared<Divide>(x, ulp), ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    std::shared_ptr<ov::Node> result = std::make_shared<Multiply>(rounded, ulp);
    auto is_zero = std::make_shared<Equal>(x, scalar_f32(0.0f));
    result = std::make_shared<Select>(is_zero, x, result);  // preserve signed zero exactly

    // Canonical NaN (0x7FC00000), matching bf16_round's own doc note that
    // activations are never NaN in practice — payload/sign not preserved,
    // same limitation the Python reference documents.
    uint32_t nan_bits = 0x7FC00000u;
    float canonical_nan;
    std::memcpy(&canonical_nan, &nan_bits, sizeof(canonical_nan));
    auto nan_const = scalar_f32(canonical_nan);
    auto is_nan = std::make_shared<ov::op::v10::IsNaN>(x);
    auto y = std::make_shared<Select>(is_nan, nan_const, result);
    y->get_output_tensor(0).set_names({"y"});

    auto result_op = std::make_shared<Result>(y);
    return std::make_shared<ov::Model>(ov::ResultVector{result_op}, ov::ParameterVector{x},
                                        "bf16_roundtrip");
}

}  // namespace cascadia_bf16
