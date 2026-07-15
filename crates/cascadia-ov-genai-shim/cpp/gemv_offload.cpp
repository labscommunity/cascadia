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
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
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

// CASCADIA_GEMV_STATS=1: accumulate evaluate() wall time across all op
// instances and print a summary at process exit — splits kernel time from
// framework overhead when chasing the throughput gap. Spike-only.
struct GemvStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> ns{0};
    std::atomic<uint64_t> weight_bytes{0};
    bool enabled = false;
    GemvStats() {
        const char* v = std::getenv("CASCADIA_GEMV_STATS");
        enabled = v && *v == '1';
        if (enabled) std::atexit(&GemvStats::print);
    }
    static GemvStats& get() {
        static GemvStats s;
        return s;
    }
    static void print() {
        auto& s = get();
        const uint64_t c = s.calls.load(), n = s.ns.load(), b = s.weight_bytes.load();
        if (c == 0) return;
        fprintf(stderr,
                "gemv-stats: calls=%llu total_ms=%.1f avg_us=%.1f weight_GB=%.2f "
                "eff_GBps=%.1f\n",
                static_cast<unsigned long long>(c), n / 1e6, n / 1e3 / c, b / 1e9,
                b / (n / 1e9) / 1e9);
    }
};

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

bool cpu_has_avx_vnni() {
    static const bool ok = [] {
        if (!cpu_has_avx2_fma()) return false;
#if defined(_MSC_VER)
        int r[4] = {0};
        __cpuidex(r, 7, 1);  // leaf 7 subleaf 1: EAX bit 4 = AVX-VNNI
        return (r[0] & (1 << 4)) != 0;
#else
        return __builtin_cpu_supports("avxvnni");
#endif
    }();
    return ok;
}

// AVX-VNNI grouped dot with dynamically-quantized int8 activations.
// Weights become unsigned in ONE op (w_i4 + 8 == nibble ^ 8, since
// (x^8)-8 is the sign decode); dpbusd(u8 = w+8, s8 = act_q) accumulates
// Σ(w+8)·q exactly in i32 (4 products per lane, no i16 saturation stage),
// and the +8 bias is removed once per group via the precomputed Σq:
// Σw·q = acc − 8·Σq. Caller dequantizes with act_scale × weight_scale.
int32_t dot_group_vnni(const uint8_t* gw, const int8_t* qa, size_t gsize) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    const __m128i xor8 = _mm_set1_epi8(8);
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    size_t j = 0;
    for (; j + 16 <= gsize / 2; j += 16) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        const __m128i lo = _mm_and_si128(raw, mask4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
        const __m128i a = _mm_xor_si128(_mm_unpacklo_epi8(lo, hi), xor8);
        const __m128i b = _mm_xor_si128(_mm_unpackhi_epi8(lo, hi), xor8);
        const __m256i w = _mm256_set_m128i(b, a);  // 32 u8 weights, in order
        const __m256i q =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa + 2 * j));
        if ((j / 16) & 1) {
            acc1 = _mm256_dpbusd_avx_epi32(acc1, w, q);
        } else {
            acc0 = _mm256_dpbusd_avx_epi32(acc0, w, q);
        }
    }
    __m256i acc = _mm256_add_epi32(acc0, acc1);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t dot = _mm_cvtsi128_si32(s);
    for (; j < gsize / 2; ++j) {
        const uint8_t v = gw[j];
        const int lo = (v & 0xF) ^ 8;  // w + 8, matching the SIMD lanes
        const int hi = ((v >> 4) & 0xF) ^ 8;
        dot += lo * qa[2 * j] + hi * qa[2 * j + 1];
    }
    return dot;
}

// AVX2 grouped sym-INT4 dot. Nibble decode via (x ^ 8) - 8 sign trick;
// _mm_unpacklo/hi_epi8(lo, hi) restores the low-nibble-first element order
// for free. FOUR independent accumulator chains: a single accumulator
// serializes on FMA latency (~4-5 cycles) and was the dominant kernel stall;
// with group_size=128 (64 packed bytes) a group is exactly one unrolled
// iteration.
static inline __m256 dot16_lo(const __m128i raw, const __m128i mask4, const __m128i bias,
                              const float* base, __m256 acc, __m256 acc2, __m256* acc2_out) {
    const __m128i lo = _mm_and_si128(raw, mask4);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
    __m128i a = _mm_unpacklo_epi8(lo, hi);
    __m128i b = _mm_unpackhi_epi8(lo, hi);
    a = _mm_sub_epi8(_mm_xor_si128(a, bias), bias);
    b = _mm_sub_epi8(_mm_xor_si128(b, bias), bias);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a)),
                          _mm256_loadu_ps(base + 0), acc);
    acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a, 8))),
                           _mm256_loadu_ps(base + 8), acc2);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)),
                          _mm256_loadu_ps(base + 16), acc);
    acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8))),
                           _mm256_loadu_ps(base + 24), acc2);
    *acc2_out = acc2;
    return acc;
}

float dot_group_avx2(const uint8_t* gw, const float* ga, size_t gsize) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i bias = _mm_set1_epi8(8);
    size_t j = 0;
    for (; j + 32 <= gsize / 2; j += 32) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j + 16));
        acc0 = dot16_lo(r0, mask4, bias, ga + 2 * j, acc0, acc1, &acc1);
        acc2 = dot16_lo(r1, mask4, bias, ga + 2 * j + 32, acc2, acc3, &acc3);
    }
    for (; j + 16 <= gsize / 2; j += 16) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        acc0 = dot16_lo(r0, mask4, bias, ga + 2 * j, acc0, acc1, &acc1);
    }
    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
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
        auto& stats = GemvStats::get();
        const auto t0 = stats.enabled ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
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

        // Scratch (per-call LOCALS, not thread_local: ov::parallel_for
        // blocks this thread and TBB work-steals other ready tasks onto it —
        // including another instance's evaluate(), which would clobber a
        // shared thread_local while our row lambdas still read it):
        // f32 activation row, and — on the VNNI path — its per-group int8
        // quantization + (scale, Σq) pairs, all computed ONCE per row and
        // reused across all N outputs.
        std::vector<float> act_f32;
        std::vector<int8_t> act_q;
        std::vector<float> q_scale;
        std::vector<int32_t> q_sum;
#ifdef CASCADIA_GEMV_X86
        const bool use_avx2 = cpu_has_avx2_fma();
        // CASCADIA_GEMV_VNNI=1: dynamically-quantized int8 activations via
        // AVX-VNNI dpbusd (~3x per-core over the f32-convert chain). This is
        // dynamic quantization — output parity vs the stock f16/f32 kernel
        // must be re-validated whenever it's enabled. Spike knob.
        static const bool want_vnni = [] {
            const char* v = std::getenv("CASCADIA_GEMV_VNNI");
            return v && *v == '1';
        }();
        const bool use_vnni = want_vnni && cpu_has_avx_vnni();
#else
        const bool use_avx2 = false;
        const bool use_vnni = false;
#endif
        // CASCADIA_GEMV_SEQ=1: single-thread A/B knob (spike-only).
        static const bool sequential = [] {
            const char* v = std::getenv("CASCADIA_GEMV_SEQ");
            return v && *v == '1';
        }();
        for (size_t r = 0; r < rows; ++r) {
            act_f32.resize(k);
            for (size_t i = 0; i < k; ++i) {
                act_f32[i] = in_f16 ? static_cast<float>(act16[r * k + i]) : act32[r * k + i];
            }
            const float* a = act_f32.data();
            if (use_vnni) {
                act_q.resize(k);
                q_scale.resize(groups);
                q_sum.resize(groups);
                for (size_t gi = 0; gi < groups; ++gi) {
                    const float* ga = a + gi * gsize;
                    float amax = 0.f;
                    for (size_t i = 0; i < gsize; ++i) {
                        const float v = ga[i] < 0 ? -ga[i] : ga[i];
                        if (v > amax) amax = v;
                    }
                    const float s = amax > 0.f ? amax / 127.f : 1.f;
                    const float inv = 1.f / s;
                    int32_t sum = 0;
                    for (size_t i = 0; i < gsize; ++i) {
                        int q = static_cast<int>(ga[i] * inv + (ga[i] >= 0 ? 0.5f : -0.5f));
                        if (q > 127) q = 127;
                        if (q < -127) q = -127;
                        act_q[gi * gsize + i] = static_cast<int8_t>(q);
                        sum += q;
                    }
                    q_scale[gi] = s;
                    q_sum[gi] = sum;
                }
            }
            const std::function<void(size_t, size_t)> rows_fn = [&](size_t rb, size_t re) {
                for (size_t row = rb; row < re; ++row) {
                    const uint8_t* wrow = wbytes + row * (k / 2);
                    const ov::float16* srow = sc16 + row * groups;
                    float acc = 0.f;
                    for (size_t gi = 0; gi < groups; ++gi) {
                        const uint8_t* gw = wrow + gi * (gsize / 2);
                        float dot;
#ifdef CASCADIA_GEMV_X86
                        if (use_vnni) {
                            const int32_t dq =
                                dot_group_vnni(gw, act_q.data() + gi * gsize, gsize);
                            dot = q_scale[gi] * static_cast<float>(dq - 8 * q_sum[gi]);
                        } else if (use_avx2) {
                            dot = dot_group_avx2(gw, a + gi * gsize, gsize);
                        } else
#endif
                        {
                            const float* ga = a + gi * gsize;
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
                }
            };
            if (sequential) {
                rows_fn(0, n);
            } else {
                // Blocked ov::parallel_for: fewer, larger tasks than
                // per-row dispatch (measured better than both per-row and a
                // private TBB-independent pool, which loses to wake latency
                // + oversubscription against the plugin's own threads).
                const size_t block = 32;
                const size_t nblocks = (n + block - 1) / block;
                ov::parallel_for(nblocks, [&](size_t bi) {
                    const size_t b = bi * block;
                    const size_t e = b + block < n ? b + block : n;
                    rows_fn(b, e);
                });
            }
        }
        if (stats.enabled) {
            stats.calls.fetch_add(1, std::memory_order_relaxed);
            stats.ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count(),
                               std::memory_order_relaxed);
            stats.weight_bytes.fetch_add(rows * n * (k / 2), std::memory_order_relaxed);
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
