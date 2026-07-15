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
#include <map>
#include <memory>
#include <numeric>
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

#ifdef CASCADIA_HAVE_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#endif

#include <openvino/core/parallel.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace cascadia_gemv {
namespace {

using ov::op::v0::Constant;

#ifdef CASCADIA_HAVE_DNNL
// The endgame path: a dnnl matmul primitive with INT4 weights-decompression
// executing DIRECTLY over the op's mmapped weight bytes. Our packed-i4
// [N, K] rows are exactly dnnl's plain `ba` layout for weights dims [K, N]
// (zero weight copy); only the scales transpose into a small resident
// [G, N] f16 buffer (~4 MB across a 1B model). fpmath_mode(f16, apply_to_
// int=true) selects the same weights-decompression brgemm family the OV CPU
// plugin uses (measured ~51 GB/s vs our loop's ~22). TBB-runtime dnnl build
// shares the plugin's tbb12, so its threads come from the existing pool.
struct DnnlSegExec {
    dnnl::matmul prim;
    dnnl::memory w_mem, sc_mem, src_mem, dst_mem;
    size_t out_off = 0;
    size_t n = 0;
    std::vector<uint16_t> scales_gn;  // f16 bits, [G, N]
};
struct DnnlState {
    bool ok = false;
    dnnl::engine eng;
    dnnl::stream strm;
    std::vector<DnnlSegExec> segs;
};
#endif

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
    for (; j + 32 <= gsize / 2; j += 32) {
        _mm_prefetch(reinterpret_cast<const char*>(gw + j + 128), _MM_HINT_T0);
        const __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        const __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j + 16));
        const __m128i lo0 = _mm_and_si128(raw0, mask4);
        const __m128i hi0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask4);
        const __m128i lo1 = _mm_and_si128(raw1, mask4);
        const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask4);
        const __m256i w0 = _mm256_set_m128i(
            _mm_xor_si128(_mm_unpackhi_epi8(lo0, hi0), xor8),
            _mm_xor_si128(_mm_unpacklo_epi8(lo0, hi0), xor8));
        const __m256i w1 = _mm256_set_m128i(
            _mm_xor_si128(_mm_unpackhi_epi8(lo1, hi1), xor8),
            _mm_xor_si128(_mm_unpacklo_epi8(lo1, hi1), xor8));
        acc0 = _mm256_dpbusd_avx_epi32(
            acc0, w0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa + 2 * j)));
        acc1 = _mm256_dpbusd_avx_epi32(
            acc1, w1,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa + 2 * j + 32)));
    }
    for (; j + 16 <= gsize / 2; j += 16) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        const __m128i lo = _mm_and_si128(raw, mask4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
        const __m256i w = _mm256_set_m128i(
            _mm_xor_si128(_mm_unpackhi_epi8(lo, hi), xor8),
            _mm_xor_si128(_mm_unpacklo_epi8(lo, hi), xor8));
        acc0 = _mm256_dpbusd_avx_epi32(
            acc0, w, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa + 2 * j)));
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

// AVX2 grouped sym-INT4 dots — FLAT loops, no helper calls: MSVC refuses
// to inline SIMD helpers taking/returning __m256 by value and spills vector
// registers to stack per 16-byte step, capping a core at ~5.7 GB/s (the
// measured wall that VNNI, fusion, and 2-row blocking all failed to move).
// Nibble decode via (x ^ 8) - 8; unpacklo/hi restores low-first order.

#if defined(_MSC_VER)
#define CASCADIA_INLINE __forceinline
#else
#define CASCADIA_INLINE inline __attribute__((always_inline))
#endif

CASCADIA_INLINE float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

float dot_group_avx2(const uint8_t* gw, const float* ga, size_t gsize) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i bias = _mm_set1_epi8(8);
    size_t j = 0;
    for (; j + 16 <= gsize / 2; j += 16) {
        _mm_prefetch(reinterpret_cast<const char*>(gw + j + 128), _MM_HINT_T0);
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw + j));
        const __m128i lo = _mm_and_si128(raw, mask4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
        __m128i a = _mm_unpacklo_epi8(lo, hi);
        __m128i b = _mm_unpackhi_epi8(lo, hi);
        a = _mm_sub_epi8(_mm_xor_si128(a, bias), bias);
        b = _mm_sub_epi8(_mm_xor_si128(b, bias), bias);
        const float* base = ga + 2 * j;
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a)),
                               _mm256_loadu_ps(base + 0), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a, 8))),
            _mm256_loadu_ps(base + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)),
                               _mm256_loadu_ps(base + 16), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8))),
            _mm256_loadu_ps(base + 24), acc3);
    }
    float dot = hsum8(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; j < gsize / 2; ++j) {
        const uint8_t v = gw[j];
        const int lo = static_cast<int8_t>(static_cast<uint8_t>(v << 4)) >> 4;
        const int hi = static_cast<int8_t>(v) >> 4;
        dot += static_cast<float>(lo) * ga[2 * j];
        dot += static_cast<float>(hi) * ga[2 * j + 1];
    }
    return dot;
}

// Two rows per pass against one activation read; fully flat.
void dot_group_avx2_x2(const uint8_t* gw0, const uint8_t* gw1, const float* ga, size_t gsize,
                       float* d0, float* d1) {
    __m256 r0a = _mm256_setzero_ps(), r0b = _mm256_setzero_ps();
    __m256 r1a = _mm256_setzero_ps(), r1b = _mm256_setzero_ps();
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i bias = _mm_set1_epi8(8);
    size_t j = 0;
    for (; j + 16 <= gsize / 2; j += 16) {
        _mm_prefetch(reinterpret_cast<const char*>(gw0 + j + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(gw1 + j + 128), _MM_HINT_T0);
        const float* base = ga + 2 * j;
        const __m256 v0 = _mm256_loadu_ps(base + 0);
        const __m256 v1 = _mm256_loadu_ps(base + 8);
        const __m256 v2 = _mm256_loadu_ps(base + 16);
        const __m256 v3 = _mm256_loadu_ps(base + 24);

        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw0 + j));
        __m128i l0 = _mm_and_si128(q0, mask4);
        __m128i h0 = _mm_and_si128(_mm_srli_epi16(q0, 4), mask4);
        __m128i a0 = _mm_sub_epi8(_mm_xor_si128(_mm_unpacklo_epi8(l0, h0), bias), bias);
        __m128i b0 = _mm_sub_epi8(_mm_xor_si128(_mm_unpackhi_epi8(l0, h0), bias), bias);
        r0a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a0)), v0, r0a);
        r0b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a0, 8))), v1, r0b);
        r0a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), v2, r0a);
        r0b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0, 8))), v3, r0b);

        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gw1 + j));
        __m128i l1 = _mm_and_si128(q1, mask4);
        __m128i h1 = _mm_and_si128(_mm_srli_epi16(q1, 4), mask4);
        __m128i a1 = _mm_sub_epi8(_mm_xor_si128(_mm_unpacklo_epi8(l1, h1), bias), bias);
        __m128i b1 = _mm_sub_epi8(_mm_xor_si128(_mm_unpackhi_epi8(l1, h1), bias), bias);
        r1a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a1)), v0, r1a);
        r1b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a1, 8))), v1, r1b);
        r1a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), v2, r1a);
        r1b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8))), v3, r1b);
    }
    float f0 = hsum8(_mm256_add_ps(r0a, r0b));
    float f1 = hsum8(_mm256_add_ps(r1a, r1b));
    for (; j < gsize / 2; ++j) {
        const uint8_t v0 = gw0[j], v1 = gw1[j];
        f0 += static_cast<float>(static_cast<int8_t>(static_cast<uint8_t>(v0 << 4)) >> 4) *
                  ga[2 * j] +
              static_cast<float>(static_cast<int8_t>(v0) >> 4) * ga[2 * j + 1];
        f1 += static_cast<float>(static_cast<int8_t>(static_cast<uint8_t>(v1 << 4)) >> 4) *
                  ga[2 * j] +
              static_cast<float>(static_cast<int8_t>(v1) >> 4) * ga[2 * j + 1];
    }
    *d0 = f0;
    *d1 = f1;
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

    /// One fused GEMV over `segments` weight/scale pairs sharing the same
    /// activation and K/groups/group_size — output rows are the segments
    /// concatenated in order (a VariadicSplit downstream routes them back to
    /// the original consumers). A single-projection op is the 1-segment case.
    struct Segment {
        std::shared_ptr<Constant> w;
        std::shared_ptr<Constant> s;
        int64_t n = 0;
    };

    CascadiaInt4Gemv(const ov::Output<ov::Node>& act, std::vector<Segment> segments,
                     int64_t k, int64_t groups, int64_t group_size, std::string tag)
        : ov::op::Op({act}),
          m_segs(std::move(segments)),
          m_k(k),
          m_groups(groups),
          m_gsize(group_size),
          m_tag(std::move(tag)) {
        m_n = 0;
        for (const auto& seg : m_segs) m_n += seg.n;
        constructor_validate_and_infer_types();
    }

    const std::vector<Segment>& segments() const { return m_segs; }
    int64_t k() const { return m_k; }
    int64_t groups() const { return m_groups; }
    int64_t group_size() const { return m_gsize; }
    const std::string& tag() const { return m_tag; }

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
        return std::make_shared<CascadiaInt4Gemv>(args.at(0), m_segs, m_k, m_groups, m_gsize,
                                                  m_tag);
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
        if (m_segs.empty()) return false;
        for (const auto& seg : m_segs) {
            if (!seg.w || !seg.s) return false;
        }
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
        // [N, G, g] => per-output-row stride k/2 bytes. Global output row ->
        // (segment base, local row) via the prefix table below; blocks scan
        // it linearly (<= a handful of segments).
        struct SegPtrs {
            const uint8_t* w;
            const ov::float16* s;
            size_t start, end;
        };
        std::vector<SegPtrs> segp;
        segp.reserve(m_segs.size());
        {
            size_t off = 0;
            for (const auto& seg : m_segs) {
                SegPtrs sp;
                sp.w = static_cast<const uint8_t*>(seg.w->get_data_ptr());
                sp.s = static_cast<const ov::float16*>(seg.s->get_data_ptr());
                sp.start = off;
                off += static_cast<size_t>(seg.n);
                sp.end = off;
                segp.push_back(sp);
            }
        }

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
#ifdef CASCADIA_HAVE_DNNL
        std::vector<float> dnnl_dst;
#endif
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
#ifdef CASCADIA_HAVE_DNNL
            // CASCADIA_GEMV_DNNL=1: run this row through dnnl int4-
            // decompression matmuls over the SAME mmapped weights. Falls
            // back permanently to the built-in kernels on any failure.
            static const bool want_dnnl = [] {
                const char* v = std::getenv("CASCADIA_GEMV_DNNL");
                return v && (*v == '1' || *v == '2');
            }();
            if (want_dnnl) {
                auto st = dnnl_state();
                if (st->ok) {
                    bool done = true;
                    try {
                        if (out_f16) dnnl_dst.resize(n);
                        for (auto& ex : st->segs) {
                            ex.src_mem.set_data_handle(
                                const_cast<float*>(a));
                            float* dst_ptr = out_f16
                                                 ? dnnl_dst.data() + ex.out_off
                                                 : out32 + r * n + ex.out_off;
                            ex.dst_mem.set_data_handle(dst_ptr);
                            ex.prim.execute(
                                st->strm,
                                {{DNNL_ARG_SRC, ex.src_mem},
                                 {DNNL_ARG_WEIGHTS, ex.w_mem},
                                 {DNNL_ARG_DST, ex.dst_mem},
                                 {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                                  ex.sc_mem}});
                        }
                        st->strm.wait();
                        if (out_f16) {
                            for (size_t i = 0; i < n; ++i) {
                                out16[r * n + i] = ov::float16(dnnl_dst[i]);
                            }
                        }
                    } catch (const std::exception& e) {
                        fprintf(stderr,
                                "gemv-dnnl: execute failed (%s) — disabling\n",
                                e.what());
                        st->ok = false;
                        done = false;
                    }
                    if (done) continue;
                }
            }
#endif
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
            const auto emit = [&](size_t row, float acc) {
                if (out_f16) {
                    out16[r * n + row] = ov::float16(acc);
                } else {
                    out32[r * n + row] = acc;
                }
            };
            const std::function<void(size_t, size_t)> rows_fn = [&](size_t rb, size_t re) {
                size_t si = 0;
                while (segp[si].end <= rb) ++si;
                size_t row = rb;
                while (row < re) {
                    while (segp[si].end <= row) ++si;
#ifdef CASCADIA_GEMV_X86
                    // Fast path: TWO rows of the same segment per pass — one
                    // activation read feeds both weight streams (see
                    // dot_group_avx2_x2). f32 path only; VNNI keeps 1-row.
                    if (use_avx2 && !use_vnni && row + 1 < re && row + 1 < segp[si].end) {
                        const size_t local = row - segp[si].start;
                        const uint8_t* w0 = segp[si].w + local * (k / 2);
                        const uint8_t* w1 = w0 + (k / 2);
                        const ov::float16* s0 = segp[si].s + local * groups;
                        const ov::float16* s1 = s0 + groups;
                        float acc0 = 0.f, acc1 = 0.f;
                        for (size_t gi = 0; gi < groups; ++gi) {
                            float d0, d1;
                            dot_group_avx2_x2(w0 + gi * (gsize / 2), w1 + gi * (gsize / 2),
                                              a + gi * gsize, gsize, &d0, &d1);
                            acc0 += static_cast<float>(s0[gi]) * d0;
                            acc1 += static_cast<float>(s1[gi]) * d1;
                        }
                        emit(row, acc0);
                        emit(row + 1, acc1);
                        row += 2;
                        continue;
                    }
#endif
                    const size_t local = row - segp[si].start;
                    const uint8_t* wrow = segp[si].w + local * (k / 2);
                    const ov::float16* srow = segp[si].s + local * groups;
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
                    emit(row, acc);
                    ++row;
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
#ifdef CASCADIA_HAVE_DNNL
    // Lazily-built per executing instance (clones start empty). ok=false
    // after any failure -> permanent fallback to the built-in kernels.
    std::shared_ptr<DnnlState> dnnl_state() const {
        std::lock_guard<std::mutex> lk(m_dnnl_mu);
        if (m_dnnl) return m_dnnl;
        auto st = std::make_shared<DnnlState>();
        try {
            st->eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
            st->strm = dnnl::stream(st->eng);
            const int64_t k = m_k, groups = m_groups, gsize = m_gsize;
            size_t off = 0;
            for (const auto& seg : m_segs) {
                DnnlSegExec ex;
                ex.n = static_cast<size_t>(seg.n);
                ex.out_off = off;
                off += ex.n;
                // scales [N, G] f16 -> [G, N] (dnnl expects K-groups major)
                const auto* sc =
                    static_cast<const uint16_t*>(seg.s->get_data_ptr());
                ex.scales_gn.resize(static_cast<size_t>(groups) * ex.n);
                for (size_t n_i = 0; n_i < ex.n; ++n_i) {
                    for (int64_t g = 0; g < groups; ++g) {
                        ex.scales_gn[static_cast<size_t>(g) * ex.n + n_i] =
                            sc[n_i * static_cast<size_t>(groups) +
                               static_cast<size_t>(g)];
                    }
                }
                using dt = dnnl::memory::data_type;
                using tag = dnnl::memory::format_tag;
                // Mode 1 (CASCADIA_GEMV_DNNL=1): weights in OUR plain
                // ba layout, zero-copy from the mmap. Mode 2 (=2): let dnnl
                // pick its preferred layout (format_tag::any) and reorder
                // ONCE into a resident repacked buffer — measures upstream
                // brgemm's ceiling at the cost of the residency goal.
                static const bool pick_any = [] {
                    const char* v = std::getenv("CASCADIA_GEMV_DNNL");
                    return v && *v == '2';
                }();
                dnnl::memory::desc src_md({1, k}, dt::f32, tag::ab);
                dnnl::memory::desc w_plain({k, static_cast<int64_t>(ex.n)},
                                           dt::s4, tag::ba);
                dnnl::memory::desc w_md =
                    pick_any ? dnnl::memory::desc({k, static_cast<int64_t>(ex.n)},
                                                  dt::s4, tag::any)
                             : w_plain;
                dnnl::memory::desc dst_md({1, static_cast<int64_t>(ex.n)}, dt::f32,
                                          tag::ab);
                dnnl::primitive_attr attr;
                attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                                {gsize, 1}, dt::f16);
                dnnl::matmul::primitive_desc pd(st->eng, src_md, w_md, dst_md,
                                                attr);
                ex.prim = dnnl::matmul(pd);
                if (st->segs.empty()) {
                    fprintf(stderr, "gemv-dnnl impl: %s (mode %d)\n",
                            pd.impl_info_str(), pick_any ? 2 : 1);
                }
                if (pick_any) {
                    dnnl::memory plain_mem(
                        w_plain, st->eng,
                        const_cast<void*>(seg.w->get_data_ptr()));
                    ex.w_mem = dnnl::memory(pd.weights_desc(), st->eng);
                    dnnl::reorder(plain_mem, ex.w_mem)
                        .execute(st->strm, plain_mem, ex.w_mem);
                    st->strm.wait();
                } else {
                    ex.w_mem = dnnl::memory(
                        w_md, st->eng,
                        const_cast<void*>(seg.w->get_data_ptr()));
                }
                dnnl::memory::desc sc_md(
                    {groups, static_cast<int64_t>(ex.n)}, dt::f16, tag::ab);
                ex.sc_mem = dnnl::memory(sc_md, st->eng, ex.scales_gn.data());
                ex.src_mem = dnnl::memory(src_md, st->eng, DNNL_MEMORY_NONE);
                ex.dst_mem = dnnl::memory(dst_md, st->eng, DNNL_MEMORY_NONE);
                st->segs.push_back(std::move(ex));
            }
            st->ok = true;
        } catch (const std::exception& e) {
            fprintf(stderr,
                    "gemv-dnnl: primitive creation failed (%s) — falling back "
                    "to built-in kernels\n",
                    e.what());
            st->ok = false;
            st->segs.clear();
        }
        m_dnnl = st;
        return m_dnnl;
    }
    mutable std::shared_ptr<DnnlState> m_dnnl;
    mutable std::mutex m_dnnl_mu;
#endif
    std::vector<Segment> m_segs;
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

            std::vector<CascadiaInt4Gemv::Segment> segs;
            segs.push_back({wconst, sconst, n});
            auto gemv = std::make_shared<CascadiaInt4Gemv>(act_out, std::move(segs), k, groups,
                                                           gsize, matmul->get_friendly_name());
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

// Stage 2: fuse sibling GEMVs. Ops sharing the SAME activation output and
// (K, groups, gsize, seq-shape) — a layer's q/k/v, or gate/up — become ONE
// op whose output rows are the segments concatenated, plus a VariadicSplit
// routing each slice back to the original consumers. Halves the per-token
// node count (113 -> ~65 on Llama-1B) and doubles the mean GEMV size, which
// is what amortizes the measured per-node fork-join floor.
uint32_t fuse_sibling_gemvs(const std::shared_ptr<ov::Model>& model) {
    // Group in deterministic topological order.
    struct Group {
        std::vector<std::shared_ptr<CascadiaInt4Gemv>> ops;
    };
    std::map<std::string, Group> groups;
    std::vector<std::string> order;
    for (const auto& node : model->get_ordered_ops()) {
        auto gemv = ov::as_type_ptr<CascadiaInt4Gemv>(node);
        if (!gemv || gemv->get_output_target_inputs(0).empty()) continue;
        const auto& in = gemv->input_value(0);
        char key[256];
        snprintf(key, sizeof(key), "%p:%zu:%lld:%lld:%lld", // NOLINT
                 static_cast<void*>(in.get_node()), in.get_index(),
                 static_cast<long long>(gemv->k()), static_cast<long long>(gemv->groups()),
                 static_cast<long long>(gemv->group_size()));
        auto it = groups.find(key);
        if (it == groups.end()) order.push_back(key);
        groups[key].ops.push_back(gemv);
    }

    uint32_t fused = 0;
    for (const auto& key : order) {
        auto& grp = groups[key].ops;
        if (grp.size() < 2) continue;
        std::vector<CascadiaInt4Gemv::Segment> segs;
        std::vector<int64_t> lengths;
        std::string tag;
        for (const auto& op : grp) {
            for (const auto& seg : op->segments()) {
                segs.push_back(seg);
                lengths.push_back(seg.n);
            }
            if (!tag.empty()) tag += "+";
            tag += op->tag();
        }
        auto fused_op = std::make_shared<CascadiaInt4Gemv>(
            grp[0]->input_value(0), std::move(segs), grp[0]->k(), grp[0]->groups(),
            grp[0]->group_size(), tag);
        fused_op->set_friendly_name(grp[0]->get_friendly_name() + "_fused");
        auto axis = Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto lens = Constant::create(ov::element::i64, ov::Shape{lengths.size()}, lengths);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fused_op, axis, lens);
        split->set_friendly_name(fused_op->get_friendly_name() + "_split");
        for (size_t i = 0; i < grp.size(); ++i) {
            ov::copy_runtime_info(grp[i], {fused_op, split});
            grp[i]->output(0).replace(split->output(i));
        }
        ++fused;
        fprintf(stderr, "gemv-fuse: %zu siblings -> %s (N=%lld)\n", grp.size(),
                fused_op->get_friendly_name().c_str(),
                static_cast<long long>(
                    std::accumulate(lengths.begin(), lengths.end(), int64_t{0})));
    }
    return fused;
}

uint32_t offload_int4_gemv(ov::Core& core, const std::shared_ptr<ov::Model>& model) {
    core.add_extension(std::make_shared<ov::OpExtension<CascadiaInt4Gemv>>());
    auto counter = std::make_shared<std::atomic<uint32_t>>(0);
    ov::pass::Manager manager;
    manager.register_pass<OffloadInt4GemvPass>(counter);
    manager.run_passes(model);
    // CASCADIA_GEMV_NOFUSE=1: A/B knob for the sibling fusion (spike-only).
    static const bool nofuse = [] {
        const char* v = std::getenv("CASCADIA_GEMV_NOFUSE");
        return v && *v == '1';
    }();
    if (!nofuse && counter->load(std::memory_order_relaxed) > 0) {
        const uint32_t fused = fuse_sibling_gemvs(model);
        if (fused > 0) {
            model->validate_nodes_and_infer_types();
        }
    }
    return counter->load(std::memory_order_relaxed);
}

}  // namespace cascadia_gemv
