// CascadiaInt4Gemv: an OpenVINO extension op executing NNCF sym-INT4 grouped
// GEMV directly from the read_model mmap, so the CPU plugin never makes its
// own repacked resident copy of the weights (the 2x-residency cost of the
// hybrid prefill/decode split). Spike scope: decode (M=1..few) matmuls,
// symmetric INT4 group quantization, f16 activations/scales — exactly what
// `tools/export_shards.py --target npu` emits (verified: 113/113 MatMuls in
// a Llama-3.2-1B static stage match). Built on the public extension API —
// no OpenVINO changes.

#include "gemv_offload.hpp"

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include <openvino/core/parallel.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/op.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace cascadia_gemv {
namespace {

using ov::op::v0::Constant;

// Custom op holding the i4 weight + f16 scale Constants as MEMBERS (not
// graph inputs): the plugin sees a weightless node and falls back to
// evaluate(); the shared_ptrs keep the .bin mapping alive for the compiled
// model's lifetime.
class CascadiaInt4Gemv : public ov::op::Op {
public:
    OPENVINO_OP("CascadiaInt4Gemv", "cascadia");

    CascadiaInt4Gemv() = default;

    CascadiaInt4Gemv(const ov::Output<ov::Node>& act,
                     std::shared_ptr<Constant> weights_i4,
                     std::shared_ptr<Constant> scales_f16,
                     int64_t n, int64_t k, int64_t groups, int64_t group_size)
        : ov::op::Op({act}),
          m_w(std::move(weights_i4)),
          m_s(std::move(scales_f16)),
          m_n(n),
          m_k(k),
          m_groups(groups),
          m_gsize(group_size) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto shape = get_input_partial_shape(0);
        NODE_VALIDATION_CHECK(this, shape.rank().is_static() && shape.rank().get_length() >= 1,
                              "activation rank must be static");
        shape[shape.rank().get_length() - 1] = ov::Dimension(m_n);
        set_output_type(0, ov::element::f16, shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<CascadiaInt4Gemv>(args.at(0), m_w, m_s, m_n, m_k, m_groups,
                                                  m_gsize);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("n", m_n);
        visitor.on_attribute("k", m_k);
        visitor.on_attribute("groups", m_groups);
        visitor.on_attribute("group_size", m_gsize);
        return true;
    }

    bool has_evaluate() const override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        if (!m_w || !m_s) return false;
        const auto& act = inputs[0];
        auto& out = outputs[0];
        const size_t k = static_cast<size_t>(m_k);
        const size_t n = static_cast<size_t>(m_n);
        const size_t groups = static_cast<size_t>(m_groups);
        const size_t gsize = static_cast<size_t>(m_gsize);
        const size_t rows = act.get_size() / k;  // seq (decode: 1)
        if (act.get_size() != rows * k) return false;

        const auto* act16 = act.data<const ov::float16>();
        auto* out16 = out.data<ov::float16>();
        // i4 packed two-per-byte, element 0 in the LOW nibble; row-major
        // [N, G, g] => per-output-row stride k/2 bytes.
        const auto* wbytes = static_cast<const uint8_t*>(m_w->get_data_ptr());
        const auto* sc16 = static_cast<const ov::float16*>(m_s->get_data_ptr());

        // f32 scratch of the activation row: read once, reused across N.
        thread_local std::vector<float> act_f32;
        for (size_t r = 0; r < rows; ++r) {
            act_f32.resize(k);
            for (size_t i = 0; i < k; ++i) act_f32[i] = static_cast<float>(act16[r * k + i]);
            const float* a = act_f32.data();
            ov::parallel_for(n, [&](size_t row) {
                const uint8_t* wrow = wbytes + row * (k / 2);
                const ov::float16* srow = sc16 + row * groups;
                float acc = 0.f;
                for (size_t gi = 0; gi < groups; ++gi) {
                    const uint8_t* gw = wrow + gi * (gsize / 2);
                    const float* ga = a + gi * gsize;
                    float dot = 0.f;
                    for (size_t j = 0; j < gsize / 2; ++j) {
                        const uint8_t b = gw[j];
                        const int lo = static_cast<int8_t>(static_cast<uint8_t>(b << 4)) >> 4;
                        const int hi = static_cast<int8_t>(b) >> 4;
                        dot += static_cast<float>(lo) * ga[2 * j];
                        dot += static_cast<float>(hi) * ga[2 * j + 1];
                    }
                    acc += static_cast<float>(srow[gi]) * dot;
                }
                out16[r * n + row] = ov::float16(acc);
            });
        }
        return true;
    }

private:
    std::shared_ptr<Constant> m_w;
    std::shared_ptr<Constant> m_s;
    int64_t m_n = 0, m_k = 0, m_groups = 0, m_gsize = 0;
};

// Matcher: the exact NNCF sym-INT4 chain the exporter emits. Anything that
// deviates (asymmetric zp, non-f16, odd ranks) is left on the stock path.
class OffloadInt4GemvPass : public ov::pass::MatcherPass {
public:
    explicit OffloadInt4GemvPass(std::shared_ptr<std::atomic<uint32_t>> counter) {
        using namespace ov::pass::pattern;
        auto w = wrap_type<ov::op::v0::Constant>();
        auto cvt = wrap_type<ov::op::v0::Convert>({w});
        auto sc = wrap_type<ov::op::v0::Constant>();
        auto mul = wrap_type<ov::op::v1::Multiply>({cvt, sc});
        auto shp = wrap_type<ov::op::v0::Constant>();
        auto rsh = wrap_type<ov::op::v1::Reshape>({mul, shp});
        auto act = any_input();
        auto mm = wrap_type<ov::op::v0::MatMul>({act, rsh});

        auto callback = [=](ov::pass::pattern::Matcher& m) -> bool {
            const auto& map = m.get_pattern_value_map();
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(map.at(mm).get_node_shared_ptr());
            auto wconst = ov::as_type_ptr<Constant>(map.at(w).get_node_shared_ptr());
            auto sconst = ov::as_type_ptr<Constant>(map.at(sc).get_node_shared_ptr());
            if (!matmul || !wconst || !sconst) return false;
            if (matmul->get_transpose_a() || !matmul->get_transpose_b()) return false;
            if (wconst->get_element_type() != ov::element::i4) return false;
            if (sconst->get_element_type() != ov::element::f16) return false;
            const auto& ws = wconst->get_shape();
            const auto& ss = sconst->get_shape();
            if (ws.size() != 3 || ss.size() != 3) return false;
            if (ss[0] != ws[0] || ss[1] != ws[1] || ss[2] != 1) return false;
            const int64_t n = static_cast<int64_t>(ws[0]);
            const int64_t groups = static_cast<int64_t>(ws[1]);
            const int64_t gsize = static_cast<int64_t>(ws[2]);
            if (gsize % 2 != 0) return false;
            const int64_t k = groups * gsize;
            const auto& act_out = matmul->input_value(0);
            if (act_out.get_element_type() != ov::element::f16) return false;

            auto gemv = std::make_shared<CascadiaInt4Gemv>(act_out, wconst, sconst, n, k,
                                                           groups, gsize);
            gemv->set_friendly_name(matmul->get_friendly_name());
            ov::copy_runtime_info(matmul, gemv);
            ov::replace_node(matmul, gemv);
            counter->fetch_add(1, std::memory_order_relaxed);
            return true;
        };
        register_matcher(
            std::make_shared<ov::pass::pattern::Matcher>(mm, "CascadiaInt4GemvOffload"),
            callback);
    }
};

}  // namespace

uint32_t offload_int4_gemv(ov::Core& core, const std::shared_ptr<ov::Model>& model) {
    core.add_extension(std::make_shared<ov::OpExtension<CascadiaInt4Gemv>>());
    auto counter = std::make_shared<std::atomic<uint32_t>>(0);
    ov::pass::Manager manager;
    manager.register_pass<OffloadInt4GemvPass>(counter);
    manager.run_passes(model);
    return counter->load(std::memory_order_relaxed);
}

}  // namespace cascadia_gemv
