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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#define CASCADIA_GEMV_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#include <openvino/core/parallel.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace cascadia_gemv {
namespace {

using ov::op::v0::Constant;

#ifdef CASCADIA_GEMV_X86
bool cpu_has_avx2_fma() {
    static const bool ok = [] {
#if defined(_MSC_VER)
        int r[4] = {0};
        __cpuid(r, 1);
        const bool fma = (r[2] & (1 << 12)) != 0;
        const bool osxsave = (r[2] & (1 << 27)) != 0;
        __cpuidex(r, 7, 0);
        const bool avx2 = (r[1] & (1 << 5)) != 0;
        if (!(fma && avx2 && osxsave)) return false;
        return (_xgetbv(0) & 0x6) == 0x6;
#else
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    }();
    return ok;
}

// AVX2 grouped sym-INT4 dot: 16 packed bytes = 32 weights per iteration.
// Nibble decode via (x ^ 8) - 8 sign trick; _mm_unpacklo/hi_epi8(lo, hi)
// restores the low-nibble-first element order for free.
float dot_group_avx2(const uint8_t* gw, const float* ga, size_t gsize) {
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i bias = _mm_set1_epi8(8);
    size_t j = 0;
    for (; j + 16 <= gsize / 2; j += 16) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        const __m128i lo = _mm_and_si128(raw, mask4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
        // interleave -> element order e0,e1,e2,... then sign-extend 4->8 bit
        __m128i a = _mm_unpacklo_epi8(lo, hi);
        __m128i b = _mm_unpackhi_epi8(lo, hi);
        a = _mm_sub_epi8(_mm_xor_si128(a, bias), bias);
        b = _mm_sub_epi8(_mm_xor_si128(b, bias), bias);
        const float* base = ga + 2 * j;
        const __m256i w0 = _mm256_cvtepi8_epi32(a);
        const __m256i w1 = _mm256_cvtepi8_epi32(_mm_srli_si128(a, 8));
        const __m256i w2 = _mm256_cvtepi8_epi32(b);
        const __m256i w3 = _mm256_cvtepi8_epi32(_mm_srli_si128(b, 8));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w0), _mm256_loadu_ps(base + 0), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w1), _mm256_loadu_ps(base + 8), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w2), _mm256_loadu_ps(base + 16), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(w3), _mm256_loadu_ps(base + 24), acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float dot = _mm_cvtss_f32(s);
    for (; j < gsize / 2; ++j) {
        const uint8_t v = gw[j];
        const int lo = static_cast<int8_t>(static_cast<uint8_t>(v << 4)) >> 4;
        const int hi = static_cast<int8_t>(v) >> 4;
        dot += static_cast<float>(lo) * ga[2 * j];
        dot += static_cast<float>(hi) * ga[2 * j + 1];
    }
    return dot;
}
#endif  // CASCADIA_GEMV_X86

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
                     int64_t n, int64_t k, int64_t groups, int64_t group_size,
                     std::string tag)
        : ov::op::Op({act}),
          m_w(std::move(weights_i4)),
          m_s(std::move(scales_f16)),
          m_n(n),
          m_k(k),
          m_groups(groups),
          m_gsize(group_size),
          m_tag(std::move(tag)) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto shape = get_input_partial_shape(0);
        NODE_VALIDATION_CHECK(this, shape.rank().is_static() && shape.rank().get_length() >= 1,
                              "activation rank must be static");
        shape[shape.rank().get_length() - 1] = ov::Dimension(m_n);
        // Output element type FOLLOWS the activation: the plugin's precision
        // pipeline retypes the surrounding graph (f16 IR -> f32 execution on
        // CPU by default) but cannot retype a custom op — a fixed f16 output
        // would collide with retyped f32 consumers at plugin validation.
        auto et = get_input_element_type(0);
        NODE_VALIDATION_CHECK(
            this, et.is_dynamic() || et == ov::element::f16 || et == ov::element::f32,
            "activation must be f16 or f32");
        set_output_type(0, et.is_dynamic() ? ov::element::f16 : et, shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<CascadiaInt4Gemv>(args.at(0), m_w, m_s, m_n, m_k, m_groups,
                                                  m_gsize, m_tag);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("n", m_n);
        visitor.on_attribute("k", m_k);
        visitor.on_attribute("groups", m_groups);
        visitor.on_attribute("group_size", m_gsize);
        // The weights live in op MEMBERS, invisible to attribute-based node
        // comparison — without a distinguishing attribute, common-subexpression
        // elimination merges two ops with equal dims and the same input (e.g.
        // a layer's k_proj and v_proj), silently routing one projection's
        // output into the other. The per-instance tag keeps them distinct.
        visitor.on_attribute("weights_tag", m_tag);
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

        // The plugin may execute this in f16 (IR precision) or f32 (default
        // CPU execution precision) — handle both on each side.
        const bool in_f16 = act.get_element_type() == ov::element::f16;
        const bool out_f16 = out.get_element_type() == ov::element::f16;
        if (!in_f16 && act.get_element_type() != ov::element::f32) return false;
        if (!out_f16 && out.get_element_type() != ov::element::f32) return false;
        const auto* act16 = in_f16 ? act.data<const ov::float16>() : nullptr;
        const auto* act32 = in_f16 ? nullptr : act.data<const float>();
        auto* out16 = out_f16 ? out.data<ov::float16>() : nullptr;
        auto* out32 = out_f16 ? nullptr : out.data<float>();

        // i4 packed two-per-byte, element 0 in the LOW nibble; row-major
        // [N, G, g] => per-output-row stride k/2 bytes.
        const auto* wbytes = static_cast<const uint8_t*>(m_w->get_data_ptr());
        const auto* sc16 = static_cast<const ov::float16*>(m_s->get_data_ptr());

        // f32 scratch of the activation row: read once, reused across N.
        // Deliberately a per-call LOCAL, not thread_local: ov::parallel_for
        // blocks this thread and TBB work-steals other ready tasks onto it —
        // including another instance's evaluate(), which would clobber a
        // shared thread_local while our row lambdas still read it.
        std::vector<float> act_f32;
        for (size_t r = 0; r < rows; ++r) {
            act_f32.resize(k);
            for (size_t i = 0; i < k; ++i) {
                act_f32[i] = in_f16 ? static_cast<float>(act16[r * k + i]) : act32[r * k + i];
            }
            const float* a = act_f32.data();
#ifdef CASCADIA_GEMV_X86
            const bool use_avx2 = cpu_has_avx2_fma();
#else
            const bool use_avx2 = false;
#endif
            ov::parallel_for(n, [&](size_t row) {
                const uint8_t* wrow = wbytes + row * (k / 2);
                const ov::float16* srow = sc16 + row * groups;
                float acc = 0.f;
                for (size_t gi = 0; gi < groups; ++gi) {
                    const uint8_t* gw = wrow + gi * (gsize / 2);
                    const float* ga = a + gi * gsize;
                    float dot;
#ifdef CASCADIA_GEMV_X86
                    if (use_avx2) {
                        dot = dot_group_avx2(gw, ga, gsize);
                    } else
#endif
                    {
                        dot = 0.f;
                        for (size_t j = 0; j < gsize / 2; ++j) {
                            const uint8_t b = gw[j];
                            const int lo =
                                static_cast<int8_t>(static_cast<uint8_t>(b << 4)) >> 4;
                            const int hi = static_cast<int8_t>(b) >> 4;
                            dot += static_cast<float>(lo) * ga[2 * j];
                            dot += static_cast<float>(hi) * ga[2 * j + 1];
                        }
                    }
                    acc += static_cast<float>(srow[gi]) * dot;
                }
                if (out_f16) {
                    out16[r * n + row] = ov::float16(acc);
                } else {
                    out32[r * n + row] = acc;
                }
            });
        }
        return true;
    }

private:
    std::shared_ptr<Constant> m_w;
    std::shared_ptr<Constant> m_s;
    int64_t m_n = 0, m_k = 0, m_groups = 0, m_gsize = 0;
    std::string m_tag;
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

        auto match_ordinal = std::make_shared<std::atomic<uint32_t>>(0);
        auto callback = [=](ov::pass::pattern::Matcher& m) -> bool {
            // Debug bisection knobs (spike-only): CASCADIA_GEMV_MAX caps how
            // many MatMuls get offloaded; CASCADIA_GEMV_SKIP=i,j leaves the
            // i-th/j-th matches (in traversal order) on the stock path.
            static const long cap = [] {
                const char* v = std::getenv("CASCADIA_GEMV_MAX");
                return v ? std::atol(v) : -1;
            }();
            static const std::set<uint32_t> skip = [] {
                std::set<uint32_t> s;
                if (const char* v = std::getenv("CASCADIA_GEMV_SKIP")) {
                    const char* p = v;
                    while (*p) {
                        s.insert(static_cast<uint32_t>(std::atol(p)));
                        while (*p && *p != ',') ++p;
                        if (*p == ',') ++p;
                    }
                }
                return s;
            }();
            const uint32_t ordinal = match_ordinal->fetch_add(1, std::memory_order_relaxed);
            if (skip.count(ordinal)) {
                fprintf(stderr, "gemv-offload: skipping match %u\n", ordinal);
                return false;
            }
            if (cap >= 0 && counter->load(std::memory_order_relaxed) >=
                                static_cast<uint32_t>(cap)) {
                return false;
            }
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
                                                           groups, gsize,
                                                           matmul->get_friendly_name());
            gemv->set_friendly_name(matmul->get_friendly_name());
            ov::copy_runtime_info(matmul, gemv);
            ov::replace_node(matmul, gemv);
            const uint32_t idx = counter->fetch_add(1, std::memory_order_relaxed);
            if (cap >= 0) {
                // Bisection mode: name each rewrite so the first bad node is
                // identifiable from the run log. Spike-only.
                fprintf(stderr, "gemv-offload[%u]: %s N=%lld K=%lld act_rank=%zu\n", idx,
                        matmul->get_friendly_name().c_str(), static_cast<long long>(n),
                        static_cast<long long>(k),
                        act_out.get_partial_shape().rank().is_static()
                            ? static_cast<size_t>(act_out.get_partial_shape().rank().get_length())
                            : 0);
            }
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
