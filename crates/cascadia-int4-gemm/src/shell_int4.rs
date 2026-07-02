//! Int4-quantized shell variant. Re-quantizes the bf16 weights from the
//! safetensors shards into our in-memory int4 + bf16-scale format
//! (group_size=32, symmetric), then runs the standard int4 GEMV from
//! `kernel_avx512`. Net memory motion per shell call: 4.6 GB × 60
//! layers = ~5 GB / tok (vs 17.7 GB for bf16). Lets the OS keep more
//! expert pages hot in the 133 GB RAM budget.
//!
//! Quantization is one-shot at load time. The resulting buffers live in
//! a Rust-owned `Vec<u8>` so they're heap-resident (never evicted by
//! the page-cache pressure that would otherwise hit a mmap'd region).

use crate::format::f32_to_bf16_bits as f32_to_bf16_bits_local;
use crate::kernel_avx512::dequant_gemv_int4_auto;
use crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto;
use crate::kernel_avx512_multi_blocked::dequant_gemm_int4_multi_blocked_auto;
use crate::kernel_bf16::bf16_gemv_auto;
use crate::safetensors_source::SafetensorsShell;
use crate::shell::{
    self, ShellOutputs, HIDDEN, INTERMEDIATE_SHARED, KV_LORA_RANK, NUM_HEADS, N_ROUTED_EXPERTS,
    QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, Q_LORA_RANK, ROUTED_SCALING_FACTOR, TOPK,
    V_HEAD_DIM,
};
use rayon::prelude::*;

const GROUP_SIZE: usize = 32;

/// Per-shape SIMD dispatch for the batched int4 GEMM kernels.
///
/// **Why a per-shape table.** Iter 042's tile (`kernel_avx512_multi`)
/// wins over scalar on every K2.6 projection at seq>=2 (1.4-4.75x in
/// iter 042 microbench). Iter 046's row-blocked tile
/// (`kernel_avx512_multi_blocked`) wins +28-41% *over iter 042* on
/// the medium and large shapes, but the seq threshold where it
/// wins differs by shape size:
///
/// - 28 MB (oproj) / 7 MB+ aspect-ratio match (shared_down): blocked
///   wins consistently from seq>=4 onward.
/// - 7 MB wide-K shapes (shared_gate, shared_up): blocked wins at
///   seq>=8 (+28% at seq=16). At seq=4 the smaller shapes don't pay
///   off the RB=2 register pressure, so iter 042 holds.
/// - 2-5 MB (kvproj, qproj): iter 046 microbench showed wins at
///   seq>=8 (+62% / +118%) but flagged "variable seq=4 behavior" for
///   these shapes. **iter 075 keeps them on Generic (iter 042)** —
///   the win at seq>=8 is real but the engine-level decision needs a
///   dedicated bench against the actual call distribution (most
///   forward paths today are seq=1; spec-decode verify is the only
///   hot seq>=8 caller in flight). A follow-up iter can lift them
///   once the bench data lands.
///
/// **iter 075 dispatch rules:**
///
/// | Shape         | N     | K    | int4 MB | Kernel
/// |---------------|-------|------|---------|---------
/// | q_a_proj      |  1536 | 7168 |    5.5  | `Generic` (iter 042)
/// | q_b_proj      | 12288 | 1536 |    9.4  | `Generic` (iter 042)
/// | kv_a_proj     |   576 | 7168 |    2.1  | `Generic` (iter 042)
/// | kv_b_proj     | 16384 |  512 |    4.2  | `Generic` (iter 042)
/// | router        |   384 | 7168 |    1.4  | `Generic` (iter 042)
/// | shared_gate   |  2048 | 7168 |    7.3  | **`LargeShape`** (iter 042 < seq=8, iter 046 >= 8)
/// | shared_up     |  2048 | 7168 |    7.3  | **`LargeShape`** (iter 042 < seq=8, iter 046 >= 8)
/// | shared_down   |  7168 | 2048 |    7.3  | `SharedDown` (iter 046 >= seq=4 via blocked_auto)
/// | o_proj        |  7168 | 8192 |   28.0  | `Oproj` (iter 046 >= seq=4 via blocked_auto)
///
/// All four kernel paths are bit-identical per-cell (proved by the
/// `blocked_matches_iter042_multi_seq_8` test in
/// `kernel_avx512_multi_blocked`), so the dispatch decision is purely
/// performance — correctness is invariant.
///
/// Exposed at `pub(crate)` so `layer0_int4` can use the same dispatch
/// table — keeps the shape-→-kernel mapping a single source of truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProjShape {
    /// Largest shape: N=7168, K=8192. Iter 046 wins +41% at seq>=4.
    Oproj,
    /// 7 MB with the same N=7168 aspect: N=7168, K=2048. Iter 046
    /// wins from seq>=4 onward.
    SharedDown,
    /// **iter 075:** 7 MB wide-K shapes (shared_gate, shared_up:
    /// N=2048, K=7168). Iter 046 wins at seq>=8 (+28%); seq=4-7 stays
    /// on iter 042 because the smaller N=2048 doesn't pay off the
    /// blocked tile's RB=2 register pressure until xs reuse
    /// dominates.
    LargeShape,
    /// All other projections — iter 042 is consistently best or tied,
    /// or the iter 046 win is gated by "variable seq=4 behavior" that
    /// hasn't been re-benched at the engine level.
    Generic,
}

/// Single entry point all batched-projection callers go through.
///
/// At seq=1 the iter 042 wrapper itself routes to the single-token
/// kernel (preserving the seq=1 hot path that every K2.6 inference uses
/// today). At seq>=2 the wrapper picks AVX-512 multi tile when the host
/// supports it; for `Oproj` / `SharedDown` at seq>=4 we upgrade to the
/// row-blocked iter 046 tile via `dequant_gemm_int4_multi_blocked_auto`
/// (its dispatcher will fall back to iter 042 at seq=2-3 internally).
/// For `LargeShape` (iter 075) the upgrade threshold is seq>=8 —
/// hand-rolled here rather than reusing `blocked_auto`'s seq>=4
/// threshold because the iter 046 microbench showed shared_gate /
/// shared_up only consistently win at seq>=8.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_int4_multi(
    shape: ProjShape,
    packed: &[u8],
    scale_bits: &[u8],
    xs: &[f32],
    n_rows: usize,
    k_cols: usize,
    seq: usize,
    ys: &mut [f32],
) {
    match shape {
        ProjShape::Oproj | ProjShape::SharedDown => {
            // iter 046 dispatcher: routes to blocked variant at seq>=4,
            // iter 042 at seq=2-3, scalar at seq=1.
            dequant_gemm_int4_multi_blocked_auto(packed, scale_bits, xs, n_rows, k_cols, seq, ys);
        }
        ProjShape::LargeShape => {
            // iter 075: shared_gate / shared_up bucket. The iter 046
            // microbench (commit 77bc56f) showed these only
            // consistently win at seq>=8 — at seq=4 the smaller N=2048
            // doesn't pay off the blocked tile's RB=2 register
            // pressure. Keep iter 042 for seq<8 to avoid regressing
            // the chunked-prefill seq=4 case.
            if seq >= 8 {
                dequant_gemm_int4_multi_blocked_auto(
                    packed, scale_bits, xs, n_rows, k_cols, seq, ys,
                );
            } else {
                dequant_gemm_int4_multi_auto(packed, scale_bits, xs, n_rows, k_cols, seq, ys);
            }
        }
        ProjShape::Generic => {
            // iter 042 dispatcher: routes to multi tile at seq>=2,
            // single-token kernel at seq=1.
            dequant_gemm_int4_multi_auto(packed, scale_bits, xs, n_rows, k_cols, seq, ys);
        }
    }
}

/// Quantize a bf16 weight matrix [n_rows, k_cols] (raw bytes, little-endian
/// bf16 = u16) into int4 packed nibbles + per-group bf16 scales.
///
/// Output layout:
///   packed: u8 [n_rows * k_cols / 2], byte i holds nibbles for cols 2i, 2i+1
///   scales: u8 [n_rows * (k_cols / GROUP_SIZE) * 2], bf16 little-endian
pub(crate) fn quantize_int4_group(
    weight_bf16: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(weight_bf16.len(), n_rows * k_cols * 2);
    assert!(k_cols.is_multiple_of(GROUP_SIZE));
    let n_groups = k_cols / GROUP_SIZE;
    let mut packed = vec![0u8; n_rows * k_cols / 2];
    let mut scales = vec![0u8; n_rows * n_groups * 2];

    for r in 0..n_rows {
        for g in 0..n_groups {
            // Find max abs in this group.
            let mut max_abs = 0.0f32;
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let off = (r * k_cols + c) * 2;
                let bits = ((weight_bf16[off + 1] as u32) << 8) | (weight_bf16[off] as u32);
                let w = f32::from_bits(bits << 16);
                let a = w.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            // Symmetric int4 range is [-8, 7]. Use 7 as the scale denominator
            // so the +max maps exactly to 7 (matches NNCF INT4_SYM behavior).
            let scale = if max_abs == 0.0 {
                1.0e-10
            } else {
                max_abs / 7.0
            };
            // Store scale as bf16: round-to-nearest-even of f32 -> bf16.
            let scale_bits = bf16_round(scale);
            let s_off = (r * n_groups + g) * 2;
            scales[s_off] = (scale_bits & 0xFF) as u8;
            scales[s_off + 1] = (scale_bits >> 8) as u8;

            // Quantize each value.
            let scale_q = f32::from_bits((scale_bits as u32) << 16); // re-read after rounding
            let inv = 1.0 / scale_q;
            for k in 0..GROUP_SIZE {
                let c = g * GROUP_SIZE + k;
                let w_off = (r * k_cols + c) * 2;
                let bits = ((weight_bf16[w_off + 1] as u32) << 8) | (weight_bf16[w_off] as u32);
                let w = f32::from_bits(bits << 16);
                let q = (w * inv).round().clamp(-8.0, 7.0) as i32;
                // Map signed [-8, 7] to "byte nibble" — kernel_avx512 expects
                // bytes where low/high nibbles encode columns 2i, 2i+1 with
                // the (unsigned - 8) signed convention. So store
                // (q + 8) as the 4-bit value.
                let nibble = ((q + 8) & 0x0F) as u8;
                let p_off = (r * k_cols + c) / 2;
                if c.is_multiple_of(2) {
                    packed[p_off] = (packed[p_off] & 0xF0) | nibble;
                } else {
                    packed[p_off] = (packed[p_off] & 0x0F) | (nibble << 4);
                }
            }
        }
    }

    (packed, scales)
}

/// Round f32 → bf16 (returns the 16-bit bf16 representation as u16).
#[inline]
fn bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round-to-nearest-even: add (mantissa LSB rounding) bias.
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// All shell weights quantized to int4 + bf16 scales, layer-norm
/// weights kept as bf16, router bias kept as f32.
pub struct Int4Shell {
    pub layer: u32,
    pub input_norm: Vec<u8>,
    pub q_a_proj_packed: Vec<u8>,
    pub q_a_proj_scale: Vec<u8>,
    pub q_a_norm: Vec<u8>,
    pub q_b_proj_packed: Vec<u8>,
    pub q_b_proj_scale: Vec<u8>,
    pub kv_a_proj_packed: Vec<u8>,
    pub kv_a_proj_scale: Vec<u8>,
    pub kv_a_norm: Vec<u8>,
    pub kv_b_proj_packed: Vec<u8>,
    pub kv_b_proj_scale: Vec<u8>,
    pub o_proj_packed: Vec<u8>,
    pub o_proj_scale: Vec<u8>,
    pub post_norm: Vec<u8>,
    pub router_packed: Vec<u8>,
    pub router_scale: Vec<u8>,
    pub router_bias: Vec<u8>,
    pub shared_gate_packed: Vec<u8>,
    pub shared_gate_scale: Vec<u8>,
    pub shared_up_packed: Vec<u8>,
    pub shared_up_scale: Vec<u8>,
    pub shared_down_packed: Vec<u8>,
    pub shared_down_scale: Vec<u8>,
}

impl Int4Shell {
    /// Build from a mmap'd safetensors shell. Quantizes all big matmuls
    /// to int4 + bf16 scales, leaves layer-norm weights bf16. The
    /// resulting buffers are owned (Vec) so they're heap-resident.
    pub fn from_safetensors(shell: &SafetensorsShell) -> Self {
        let (q_a_packed, q_a_scale) = quantize_int4_group(shell.q_a_proj, Q_LORA_RANK, HIDDEN);
        let (q_b_packed, q_b_scale) =
            quantize_int4_group(shell.q_b_proj, NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK);
        let (kv_a_packed, kv_a_scale) =
            quantize_int4_group(shell.kv_a_proj, KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN);
        let (kv_b_packed, kv_b_scale) = quantize_int4_group(
            shell.kv_b_proj,
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            KV_LORA_RANK,
        );
        let (o_packed, o_scale) = quantize_int4_group(shell.o_proj, HIDDEN, NUM_HEADS * V_HEAD_DIM);
        let (router_packed, router_scale) =
            quantize_int4_group(shell.router_weight, N_ROUTED_EXPERTS, HIDDEN);
        let (sg_packed, sg_scale) =
            quantize_int4_group(shell.shared_gate, INTERMEDIATE_SHARED, HIDDEN);
        let (su_packed, su_scale) =
            quantize_int4_group(shell.shared_up, INTERMEDIATE_SHARED, HIDDEN);
        let (sd_packed, sd_scale) =
            quantize_int4_group(shell.shared_down, HIDDEN, INTERMEDIATE_SHARED);
        Self {
            layer: shell.layer,
            input_norm: shell.input_norm.to_vec(),
            q_a_proj_packed: q_a_packed,
            q_a_proj_scale: q_a_scale,
            q_a_norm: shell.q_a_norm.to_vec(),
            q_b_proj_packed: q_b_packed,
            q_b_proj_scale: q_b_scale,
            kv_a_proj_packed: kv_a_packed,
            kv_a_proj_scale: kv_a_scale,
            kv_a_norm: shell.kv_a_norm.to_vec(),
            kv_b_proj_packed: kv_b_packed,
            kv_b_proj_scale: kv_b_scale,
            o_proj_packed: o_packed,
            o_proj_scale: o_scale,
            post_norm: shell.post_norm.to_vec(),
            router_packed,
            router_scale,
            router_bias: shell.router_bias.to_vec(),
            shared_gate_packed: sg_packed,
            shared_gate_scale: sg_scale,
            shared_up_packed: su_packed,
            shared_up_scale: su_scale,
            shared_down_packed: sd_packed,
            shared_down_scale: sd_scale,
        }
    }

    /// Total bytes resident in heap (sum of all the Vec<u8> fields).
    pub fn footprint_bytes(&self) -> usize {
        self.input_norm.len()
            + self.q_a_proj_packed.len()
            + self.q_a_proj_scale.len()
            + self.q_a_norm.len()
            + self.q_b_proj_packed.len()
            + self.q_b_proj_scale.len()
            + self.kv_a_proj_packed.len()
            + self.kv_a_proj_scale.len()
            + self.kv_a_norm.len()
            + self.kv_b_proj_packed.len()
            + self.kv_b_proj_scale.len()
            + self.o_proj_packed.len()
            + self.o_proj_scale.len()
            + self.post_norm.len()
            + self.router_packed.len()
            + self.router_scale.len()
            + self.router_bias.len()
            + self.shared_gate_packed.len()
            + self.shared_gate_scale.len()
            + self.shared_up_packed.len()
            + self.shared_up_scale.len()
            + self.shared_down_packed.len()
            + self.shared_down_scale.len()
    }
}

/// Run one shell forward (decode, seq=1) using int4 weights.
///
/// `past_k`/`past_v` must be sized exactly to `[NUM_HEADS, past_seq_len,
/// HEAD_DIM]`, stored as bf16 bits (`u16`). For callers that pre-allocate
/// to a larger capacity and avoid per-token Vec realloc, use
/// [`shell_forward_decode_int4_with_capacity`].
///
/// **KV cache is bf16-quantized in storage.**
/// The SDPA kernel upconverts to f32 on-the-fly per dot-product element.
pub fn shell_forward_decode_int4(
    shell: &Int4Shell,
    x_f32: &[f32],
    past_k: &[u16],
    past_v: &[u16],
    past_seq_len: usize,
) -> ShellOutputs {
    shell_forward_decode_int4_with_capacity(
        shell,
        x_f32,
        past_k,
        past_v,
        past_seq_len,
        past_seq_len,
    )
}

/// Variant of [`shell_forward_decode_int4`] that accepts a KV cache
/// sized to a larger `capacity` per head (`stride = capacity * HEAD_DIM`),
/// of which only the first `past_seq_len` slots are populated. Lets
/// callers pre-allocate a once-per-session buffer and avoid quadratic
/// alloc/copy traffic across long-context generations.
///
/// Layout of `past_k`: `[NUM_HEADS, capacity, QK_HEAD_DIM]` flat,
/// row-major, **bf16-as-u16**. Head `h`'s
/// populated keys occupy
/// `past_k[h * capacity * QK_HEAD_DIM .. h * capacity * QK_HEAD_DIM + past_seq_len * QK_HEAD_DIM]`.
/// `past_v` is laid out similarly with `V_HEAD_DIM`. KV halves memory
/// vs f32 and halves the per-token bandwidth touched at attention time;
/// the kernel upconverts each bf16 to f32 inline (cheap: `(bits as u32) << 16`).
pub fn shell_forward_decode_int4_with_capacity(
    shell: &Int4Shell,
    x_f32: &[f32],
    past_k: &[u16],
    past_v: &[u16],
    past_seq_len: usize,
    capacity: usize,
) -> ShellOutputs {
    shell_forward_decode_int4_with_capacity_sparse(
        shell,
        x_f32,
        past_k,
        past_v,
        past_seq_len,
        capacity,
        0.0,
    )
}

/// Same as [`shell_forward_decode_int4_with_capacity`] but with an
/// extra `ffn_sparsity_threshold` knob that controls the two-phase
/// Gate-first FFN sparsity in the shared-expert SwiGLU block.
///
/// `ffn_sparsity_threshold == 0.0` (used by the back-compat wrapper
/// `shell_forward_decode_int4_with_capacity`) is bit-identical to the
/// pre-port path.
///
/// Positive values activate the same magnitude-threshold sparsity
/// described in [`cascadia_int4_gemm::ffn_forward_sparse_f32`]. The
/// routed-expert FFNs are dispatched separately by the caller (the
/// runner) — see `Runner::dispatch_expert`. Per-token compute
/// breakdown at K2.6 / top_K=8: routed experts ~87%, shared experts
/// ~11%, layer 0 ~2%. Sparsifying the shared expert here covers the
/// 11%.
pub fn shell_forward_decode_int4_with_capacity_sparse(
    shell: &Int4Shell,
    x_f32: &[f32],
    past_k: &[u16],
    past_v: &[u16],
    past_seq_len: usize,
    capacity: usize,
    ffn_sparsity_threshold: f32,
) -> ShellOutputs {
    // Reuse the shell.rs forward but swap bf16_gemv_auto -> dequant_gemv_int4_auto.
    // Easiest: copy the body and adapt. (Generic functions over a trait would
    // be cleaner but pure functions are fine here.)
    assert_eq!(x_f32.len(), HIDDEN);
    assert!(
        capacity >= past_seq_len,
        "capacity ({capacity}) must be >= past_seq_len ({past_seq_len})"
    );
    // bf16 storage: same number of slots, half the byte footprint.
    assert_eq!(past_k.len(), NUM_HEADS * capacity * QK_HEAD_DIM);
    assert_eq!(past_v.len(), NUM_HEADS * capacity * V_HEAD_DIM);

    // input layernorm (bf16 weight, scalar)
    let h_norm = rmsnorm_apply(x_f32, &shell.input_norm, HIDDEN);

    // q_a_proj (int4)
    let mut q_a = vec![0.0f32; Q_LORA_RANK];
    dequant_gemv_int4_auto(
        &shell.q_a_proj_packed,
        &shell.q_a_proj_scale,
        &h_norm,
        Q_LORA_RANK,
        HIDDEN,
        &mut q_a,
    );
    let q_a_n = rmsnorm_apply(&q_a, &shell.q_a_norm, Q_LORA_RANK);

    // q_b_proj (int4)
    let mut q = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    dequant_gemv_int4_auto(
        &shell.q_b_proj_packed,
        &shell.q_b_proj_scale,
        &q_a_n,
        NUM_HEADS * QK_HEAD_DIM,
        Q_LORA_RANK,
        &mut q,
    );

    // kv_a_proj (int4)
    let mut kv_a_with_rope = vec![0.0f32; KV_LORA_RANK + QK_ROPE_HEAD_DIM];
    dequant_gemv_int4_auto(
        &shell.kv_a_proj_packed,
        &shell.kv_a_proj_scale,
        &h_norm,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        HIDDEN,
        &mut kv_a_with_rope,
    );
    let (kv_a, k_rope_in) = kv_a_with_rope.split_at(KV_LORA_RANK);
    let kv_a_n = rmsnorm_apply(kv_a, &shell.kv_a_norm, KV_LORA_RANK);

    // kv_b_proj (int4)
    let mut kv_b = vec![0.0f32; NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
    dequant_gemv_int4_auto(
        &shell.kv_b_proj_packed,
        &shell.kv_b_proj_scale,
        &kv_a_n,
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        KV_LORA_RANK,
        &mut kv_b,
    );

    // RoPE + assemble Q/K/V (same as bf16 path)
    let (cos, sin) = shell::rope_cos_sin_pub(past_seq_len);
    let mut new_k = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    let mut new_v = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
    let mut k_rope_rot = vec![0.0f32; QK_ROPE_HEAD_DIM];
    shell::apply_rope_kimi_pub(k_rope_in, &cos, &sin, &mut k_rope_rot);

    let mut q_full = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    let mut q_rope_buf = vec![0.0f32; QK_ROPE_HEAD_DIM];
    for h in 0..NUM_HEADS {
        q_full[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM]
            .copy_from_slice(&q[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM]);
        let q_rope_src = &q[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        shell::apply_rope_kimi_pub(q_rope_src, &cos, &sin, &mut q_rope_buf);
        q_full[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM]
            .copy_from_slice(&q_rope_buf);
        let k_nope_src = &kv_b[h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
            ..h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) + QK_NOPE_HEAD_DIM];
        new_k[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM].copy_from_slice(k_nope_src);
        new_k[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM]
            .copy_from_slice(&k_rope_rot);
        let v_src = &kv_b[h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) + QK_NOPE_HEAD_DIM
            ..(h + 1) * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
        new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM].copy_from_slice(v_src);
    }

    // SDPA — parallelize per-head attention.
    // Each head's body is independent (writes to a disjoint V_HEAD_DIM
    // slice of attn_out). Rayon over the 64 heads gives ~core-count
    // speedup on the attention bucket (14.5% of decode per q1).
    //
    // past_k/past_v are bf16-as-u16. The
    // upconvert `f32::from_bits((bits as u32) << 16)` is a single shift
    // per element and stays cheap. The new (this-step) k/v are still
    // f32 — they are written to the bf16 cache by the caller after this
    // function returns.
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut attn_out = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
    attn_out
        .par_chunks_mut(V_HEAD_DIM)
        .enumerate()
        .for_each(|(h, out_h)| {
            let q_h = &q_full[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
            let pk_base = h * capacity * QK_HEAD_DIM;
            let pv_base = h * capacity * V_HEAD_DIM;
            let past_k_h = &past_k[pk_base..pk_base + past_seq_len * QK_HEAD_DIM];
            let past_v_h = &past_v[pv_base..pv_base + past_seq_len * V_HEAD_DIM];
            let new_k_h = &new_k[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
            let new_v_h = &new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];

            let kv_len = past_seq_len + 1;
            let mut scores = vec![0.0f32; kv_len];
            for j in 0..past_seq_len {
                let k_row = &past_k_h[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
                let mut s = 0.0f32;
                for i in 0..QK_HEAD_DIM {
                    let kf = f32::from_bits((k_row[i] as u32) << 16);
                    s += q_h[i] * kf;
                }
                scores[j] = s * scale;
            }
            let mut s = 0.0f32;
            for i in 0..QK_HEAD_DIM {
                s += q_h[i] * new_k_h[i];
            }
            scores[past_seq_len] = s * scale;
            let mut max_s = scores[0];
            for &v in scores.iter().skip(1) {
                if v > max_s {
                    max_s = v;
                }
            }
            let mut sum_e = 0.0f32;
            for v in scores.iter_mut() {
                *v = (*v - max_s).exp();
                sum_e += *v;
            }
            let inv = 1.0 / sum_e;
            for v in scores.iter_mut() {
                *v *= inv;
            }
            out_h.fill(0.0);
            for j in 0..past_seq_len {
                let v_row = &past_v_h[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
                let w = scores[j];
                for i in 0..V_HEAD_DIM {
                    let vf = f32::from_bits((v_row[i] as u32) << 16);
                    out_h[i] += w * vf;
                }
            }
            let w = scores[past_seq_len];
            for i in 0..V_HEAD_DIM {
                out_h[i] += w * new_v_h[i];
            }
        });

    // o_proj (int4)
    let mut o_out = vec![0.0f32; HIDDEN];
    dequant_gemv_int4_auto(
        &shell.o_proj_packed,
        &shell.o_proj_scale,
        &attn_out,
        HIDDEN,
        NUM_HEADS * V_HEAD_DIM,
        &mut o_out,
    );

    let mut residual = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        residual[i] = x_f32[i] + o_out[i];
    }
    let post = rmsnorm_apply(&residual, &shell.post_norm, HIDDEN);

    // Router (int4)
    let mut router_logits = vec![0.0f32; N_ROUTED_EXPERTS];
    dequant_gemv_int4_auto(
        &shell.router_packed,
        &shell.router_scale,
        &post,
        N_ROUTED_EXPERTS,
        HIDDEN,
        &mut router_logits,
    );
    let mut scores_raw = vec![0.0f32; N_ROUTED_EXPERTS];
    for i in 0..N_ROUTED_EXPERTS {
        scores_raw[i] = 1.0f32 / (1.0f32 + (-router_logits[i]).exp());
    }
    let bias: &[f32] = unsafe {
        std::slice::from_raw_parts(shell.router_bias.as_ptr() as *const f32, N_ROUTED_EXPERTS)
    };
    let mut scores_for_choice = vec![0.0f32; N_ROUTED_EXPERTS];
    for i in 0..N_ROUTED_EXPERTS {
        scores_for_choice[i] = scores_raw[i] + bias[i];
    }
    let mut idx_score: Vec<(usize, f32)> = scores_for_choice.iter().copied().enumerate().collect();
    idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut topk_ids = vec![0i64; TOPK];
    let mut topk_w = vec![0.0f32; TOPK];
    for k in 0..TOPK {
        topk_ids[k] = idx_score[k].0 as i64;
        topk_w[k] = scores_raw[idx_score[k].0];
    }
    let s: f32 = topk_w.iter().sum::<f32>() + 1.0e-20;
    for w in topk_w.iter_mut() {
        *w = *w / s * ROUTED_SCALING_FACTOR;
    }

    // Shared expert (int4 ×3) — SwiGLU FFN, runs unconditionally per
    // token. Threaded through `ffn_forward_sparse_f32` so the same
    // `ffn_sparsity_threshold` knob that gates per-routed-expert
    // sparsity also gates the shared-expert path. At `threshold == 0.0`
    // (default) this delegates to a back-to-back dense gate/up/down
    // sequence — bit-identical to the pre-port inline code.
    let mut shared_out = vec![0.0f32; HIDDEN];
    let _shared_active_frac = crate::ffn_sparsity::ffn_forward_sparse_f32(
        &post,
        HIDDEN,
        INTERMEDIATE_SHARED,
        &shell.shared_gate_packed,
        &shell.shared_gate_scale,
        &shell.shared_up_packed,
        &shell.shared_up_scale,
        &shell.shared_down_packed,
        &shell.shared_down_scale,
        &mut shared_out,
        ffn_sparsity_threshold,
    );

    ShellOutputs {
        attn_out_post_norm: post,
        attn_residual: residual,
        shared_expert_out: shared_out,
        routing_ids: topk_ids,
        routing_weights: topk_w,
        present_k: new_k,
        present_v: new_v,
    }
}

// =====================================================================
// Multi-token (seq >= 1) entry points (iter 041 + 042 + 046 + 048)
// =====================================================================
//
// Adds a seq>=1 API on top of the single-token shell forward so future
// chunked-prefill / spec-decode driver loops can batch N tokens through
// one shell forward instead of N sequential seq=1 calls. The KV cache
// here is `[u16]` (bf16 storage) just like
// the seq=1 path; the iter 042 multi-token AVX-512 tile and iter 046
// row-blocked tile are dispatched per-shape via `dispatch_int4_multi`.

/// Per-token outputs from a multi-token shell forward. Flat buffers of
/// length `seq * <per-token dim>` so callers don't have to allocate a
/// `Vec<...>` per token.
pub struct MultiShellOutputs {
    /// Per-token post-attention-layernorm output. Shape `[seq, HIDDEN]`
    /// flat. Caller slices `[t * HIDDEN .. (t + 1) * HIDDEN]` to get
    /// token `t`'s input to expert dispatch.
    pub attn_out_post_norm: Vec<f32>,
    /// Per-token residual (x + attn_out). Shape `[seq, HIDDEN]` flat.
    pub attn_residual: Vec<f32>,
    /// Per-token shared expert output. Shape `[seq, HIDDEN]` flat.
    pub shared_expert_out: Vec<f32>,
    /// Per-token top-K expert ids. Shape `[seq, TOPK]` flat.
    pub routing_ids: Vec<i64>,
    /// Per-token top-K expert weights. Shape `[seq, TOPK]` flat.
    pub routing_weights: Vec<f32>,
}

/// Run a shell forward over `seq` consecutive tokens with the int4
/// kernel. seq>=1 entry point for the spec-decode / chunked-prefill
/// driver loops.
///
/// **Semantics.** Observationally identical to `seq` sequential calls
/// of [`shell_forward_decode_int4_with_capacity`] — the same int4 GEMV
/// kernels run per token, in token order, with the KV cache updated
/// after each step. The only behavioral change for callers is the API:
/// outputs are concatenated across tokens, and `past_k` / `past_v` are
/// written in place rather than returned.
///
/// **Dispatch.** At seq=1, falls through to the per-token kernel via
/// the scalar reference (which calls
/// [`shell_forward_decode_int4_with_capacity`] under the hood). At
/// seq>=2, uses [`shell_forward_decode_int4_multi_batched`] which
/// batches each projection across `seq` tokens through the iter 042 /
/// iter 046 SIMD tiles.
///
/// **Inputs.**
/// - `xs_f32`: layer inputs, shape `[seq, HIDDEN]` flat.
/// - `past_k` / `past_v`: pre-allocated KV cache, shape
///   `[NUM_HEADS, capacity, *_HEAD_DIM]`, **bf16-as-u16** (A8). Only the
///   first `past_seq_len` slots are populated on entry; the kernel
///   writes slots `[past_seq_len, past_seq_len + seq)` on exit.
/// - `past_seq_len`: populated KV length on entry.
/// - `capacity`: total per-head KV slot capacity. Must be
///   `>= past_seq_len + seq`.
/// - `seq`: number of tokens to process. Must be `>= 1`.
pub fn shell_forward_decode_int4_multi_with_capacity(
    shell: &Int4Shell,
    xs_f32: &[f32],
    past_k: &mut [u16],
    past_v: &mut [u16],
    past_seq_len: usize,
    capacity: usize,
    seq: usize,
) -> MultiShellOutputs {
    assert!(seq >= 1, "seq must be >= 1, got {seq}");
    assert_eq!(
        xs_f32.len(),
        seq * HIDDEN,
        "xs_f32.len() = {} != seq * HIDDEN = {} * {} = {}",
        xs_f32.len(),
        seq,
        HIDDEN,
        seq * HIDDEN
    );
    assert!(
        capacity >= past_seq_len + seq,
        "capacity ({capacity}) must be >= past_seq_len ({past_seq_len}) + seq ({seq})",
    );
    assert_eq!(past_k.len(), NUM_HEADS * capacity * QK_HEAD_DIM);
    assert_eq!(past_v.len(), NUM_HEADS * capacity * V_HEAD_DIM);

    // For seq=1 the per-token kernel is faster than the multi-tile —
    // the tile pays a per-row scatter cost that doesn't amortize. Go
    // straight to the scalar reference loop.
    if seq == 1 {
        return shell_forward_decode_int4_multi_scalar(
            shell,
            xs_f32,
            past_k,
            past_v,
            past_seq_len,
            capacity,
            seq,
        );
    }
    shell_forward_decode_int4_multi_batched(
        shell,
        xs_f32,
        past_k,
        past_v,
        past_seq_len,
        capacity,
        seq,
    )
}

/// Reference implementation: loop seq=1 calls. Kept as the bit-identity
/// reference for [`shell_forward_decode_int4_multi_batched`].
pub fn shell_forward_decode_int4_multi_scalar(
    shell: &Int4Shell,
    xs_f32: &[f32],
    past_k: &mut [u16],
    past_v: &mut [u16],
    past_seq_len: usize,
    capacity: usize,
    seq: usize,
) -> MultiShellOutputs {
    let mut attn_out_post_norm = vec![0.0f32; seq * HIDDEN];
    let mut attn_residual = vec![0.0f32; seq * HIDDEN];
    let mut shared_expert_out = vec![0.0f32; seq * HIDDEN];
    let mut routing_ids = vec![0i64; seq * TOPK];
    let mut routing_weights = vec![0.0f32; seq * TOPK];

    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let cur_past = past_seq_len + t;
        let outs =
            shell_forward_decode_int4_with_capacity(shell, x_t, past_k, past_v, cur_past, capacity);
        // Write present_k / present_v into slot `cur_past` for each head.
        // The seq=1 forward returns present_k/present_v as f32 and the
        // engine's `write_present_kv` does the f32->bf16 conversion; here
        // we do the same conversion inline so the cache stays u16.
        write_present_kv_bf16(past_k, &outs.present_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_bf16(past_v, &outs.present_v, cur_past, capacity, V_HEAD_DIM);

        attn_out_post_norm[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.attn_out_post_norm);
        attn_residual[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.attn_residual);
        shared_expert_out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.shared_expert_out);
        routing_ids[t * TOPK..(t + 1) * TOPK].copy_from_slice(&outs.routing_ids);
        routing_weights[t * TOPK..(t + 1) * TOPK].copy_from_slice(&outs.routing_weights);
    }

    MultiShellOutputs {
        attn_out_post_norm,
        attn_residual,
        shared_expert_out,
        routing_ids,
        routing_weights,
    }
}

/// Batched version: structures the forward as three phases so that the
/// big projections (q_a, q_b, kv_a, kv_b, o_proj, router, shared_*)
/// can use the multi-token int4 GEMM kernel (iter 042's
/// `dequant_gemm_int4_multi_auto`). The phases are:
///
/// **Phase A (batched projections, no KV).** Compute h_norm per token,
/// then batch q_a, kv_a across all `seq` tokens. RMSNorm on q_a, kv_a
/// per token, then batch q_b, kv_b.
///
/// **Phase B (per-token, KV-dependent).** RoPE on q + k_rope, assemble
/// q_full / new_k / new_v, SDPA against past KV cache (with inline
/// bf16 -> f32 upconvert per A8), append new K/V into the cache as
/// bf16-as-u16 so the next token sees it.
///
/// **Phase C (batched projections, no KV).** Batch o_proj on the stack
/// of per-token attn_outs (iter 046 row-blocked), per-token residual +
/// post-norm, batch router + sigmoid + topK + shared_gate + shared_up,
/// SwiGLU, shared_down (iter 046 row-blocked).
///
/// All projections in phases A and C are `[seq, K] x [K, N]` int4 GEMMs
/// that amortize one weight load over `seq` tokens.
fn shell_forward_decode_int4_multi_batched(
    shell: &Int4Shell,
    xs_f32: &[f32],
    past_k: &mut [u16],
    past_v: &mut [u16],
    past_seq_len: usize,
    capacity: usize,
    seq: usize,
) -> MultiShellOutputs {
    // --- Allocate outputs and scratch ---
    let mut attn_out_post_norm = vec![0.0f32; seq * HIDDEN];
    let mut attn_residual = vec![0.0f32; seq * HIDDEN];
    let mut shared_expert_out = vec![0.0f32; seq * HIDDEN];
    let mut routing_ids = vec![0i64; seq * TOPK];
    let mut routing_weights = vec![0.0f32; seq * TOPK];

    // ============ PHASE A: pre-attention projections ============
    // Per-token h_norm (cheap RMSNorm).
    let mut h_norms = vec![0.0f32; seq * HIDDEN];
    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let norm = rmsnorm_apply(x_t, &shell.input_norm, HIDDEN);
        h_norms[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&norm);
    }

    // Batched q_a = q_a_proj @ h_norm[t]
    let mut q_a = vec![0.0f32; seq * Q_LORA_RANK];
    dispatch_int4_multi(
        ProjShape::Generic,
        &shell.q_a_proj_packed,
        &shell.q_a_proj_scale,
        &h_norms,
        Q_LORA_RANK,
        HIDDEN,
        seq,
        &mut q_a,
    );

    // Batched kv_a (kv_a_proj output includes the rope shared col).
    let kv_a_out_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
    let mut kv_a_with_rope = vec![0.0f32; seq * kv_a_out_dim];
    dispatch_int4_multi(
        ProjShape::Generic,
        &shell.kv_a_proj_packed,
        &shell.kv_a_proj_scale,
        &h_norms,
        kv_a_out_dim,
        HIDDEN,
        seq,
        &mut kv_a_with_rope,
    );

    // Per-token rmsnorm on q_a and kv_a.
    let mut q_a_n = vec![0.0f32; seq * Q_LORA_RANK];
    let mut kv_a_n = vec![0.0f32; seq * KV_LORA_RANK];
    let mut k_rope_ins = vec![0.0f32; seq * QK_ROPE_HEAD_DIM];
    for t in 0..seq {
        let q_a_t = &q_a[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK];
        let q_a_n_t = rmsnorm_apply(q_a_t, &shell.q_a_norm, Q_LORA_RANK);
        q_a_n[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK].copy_from_slice(&q_a_n_t);

        let kv_a_t = &kv_a_with_rope[t * kv_a_out_dim..t * kv_a_out_dim + KV_LORA_RANK];
        let k_rope_t = &kv_a_with_rope[t * kv_a_out_dim + KV_LORA_RANK..(t + 1) * kv_a_out_dim];
        let kv_a_n_t = rmsnorm_apply(kv_a_t, &shell.kv_a_norm, KV_LORA_RANK);
        kv_a_n[t * KV_LORA_RANK..(t + 1) * KV_LORA_RANK].copy_from_slice(&kv_a_n_t);
        k_rope_ins[t * QK_ROPE_HEAD_DIM..(t + 1) * QK_ROPE_HEAD_DIM].copy_from_slice(k_rope_t);
    }

    // Batched q = q_b_proj @ q_a_n[t]
    let qkv_q_dim = NUM_HEADS * QK_HEAD_DIM;
    let mut qs = vec![0.0f32; seq * qkv_q_dim];
    dispatch_int4_multi(
        ProjShape::Generic,
        &shell.q_b_proj_packed,
        &shell.q_b_proj_scale,
        &q_a_n,
        qkv_q_dim,
        Q_LORA_RANK,
        seq,
        &mut qs,
    );

    // Batched kv_b = kv_b_proj @ kv_a_n[t]
    let kv_b_dim = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM);
    let mut kv_bs = vec![0.0f32; seq * kv_b_dim];
    dispatch_int4_multi(
        ProjShape::Generic,
        &shell.kv_b_proj_packed,
        &shell.kv_b_proj_scale,
        &kv_a_n,
        kv_b_dim,
        KV_LORA_RANK,
        seq,
        &mut kv_bs,
    );

    // ============ PHASE B: per-token RoPE + SDPA + KV append ============
    // A8: past_k/past_v are bf16-as-u16. Each read upconverts via
    // f32::from_bits((bits as u32) << 16). New k/v are f32; we write
    // them into the cache after SDPA via write_present_kv_bf16.
    let mut attn_outs = vec![0.0f32; seq * (NUM_HEADS * V_HEAD_DIM)];
    for t in 0..seq {
        let cur_past = past_seq_len + t;
        let kv_len = cur_past + 1;
        let q = &qs[t * qkv_q_dim..(t + 1) * qkv_q_dim];
        let kv_b = &kv_bs[t * kv_b_dim..(t + 1) * kv_b_dim];
        let k_rope_in = &k_rope_ins[t * QK_ROPE_HEAD_DIM..(t + 1) * QK_ROPE_HEAD_DIM];

        let (cos, sin) = shell::rope_cos_sin_pub(cur_past);
        let mut new_k = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
        let mut new_v = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
        let mut k_rope_rot = vec![0.0f32; QK_ROPE_HEAD_DIM];
        shell::apply_rope_kimi_pub(k_rope_in, &cos, &sin, &mut k_rope_rot);

        let mut q_full = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
        let mut q_rope_buf = vec![0.0f32; QK_ROPE_HEAD_DIM];
        for h in 0..NUM_HEADS {
            q_full[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM]
                .copy_from_slice(&q[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM]);
            let q_rope_src = &q[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
            shell::apply_rope_kimi_pub(q_rope_src, &cos, &sin, &mut q_rope_buf);
            q_full[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM]
                .copy_from_slice(&q_rope_buf);
            let k_nope_src = &kv_b[h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
                ..h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) + QK_NOPE_HEAD_DIM];
            new_k[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM].copy_from_slice(k_nope_src);
            new_k[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM]
                .copy_from_slice(&k_rope_rot);
            let v_src = &kv_b[h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) + QK_NOPE_HEAD_DIM
                ..(h + 1) * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
            new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM].copy_from_slice(v_src);
        }

        // SDPA against past KV in [NUM_HEADS, capacity, *_HEAD_DIM]
        // layout, taking only the first cur_past rows of each head.
        // A8: past_k/past_v are u16; upconvert each element inline.
        let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
        let attn_out_t =
            &mut attn_outs[t * (NUM_HEADS * V_HEAD_DIM)..(t + 1) * (NUM_HEADS * V_HEAD_DIM)];
        for h in 0..NUM_HEADS {
            let q_h = &q_full[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
            let pk_base = h * capacity * QK_HEAD_DIM;
            let pv_base = h * capacity * V_HEAD_DIM;
            let past_k_h = &past_k[pk_base..pk_base + cur_past * QK_HEAD_DIM];
            let past_v_h = &past_v[pv_base..pv_base + cur_past * V_HEAD_DIM];
            let new_k_h = &new_k[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
            let new_v_h = &new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];

            let mut scores = vec![0.0f32; kv_len];
            for j in 0..cur_past {
                let k_row = &past_k_h[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
                let mut s = 0.0f32;
                for i in 0..QK_HEAD_DIM {
                    let kf = f32::from_bits((k_row[i] as u32) << 16);
                    s += q_h[i] * kf;
                }
                scores[j] = s * scale;
            }
            let mut s = 0.0f32;
            for i in 0..QK_HEAD_DIM {
                s += q_h[i] * new_k_h[i];
            }
            scores[cur_past] = s * scale;
            let mut max_s = scores[0];
            for &v in scores.iter().skip(1) {
                if v > max_s {
                    max_s = v;
                }
            }
            let mut sum_e = 0.0f32;
            for v in scores.iter_mut() {
                *v = (*v - max_s).exp();
                sum_e += *v;
            }
            let inv = 1.0 / sum_e;
            for v in scores.iter_mut() {
                *v *= inv;
            }
            let out_h = &mut attn_out_t[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];
            out_h.fill(0.0);
            for j in 0..cur_past {
                let v_row = &past_v_h[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
                let w = scores[j];
                for i in 0..V_HEAD_DIM {
                    let vf = f32::from_bits((v_row[i] as u32) << 16);
                    out_h[i] += w * vf;
                }
            }
            let w = scores[cur_past];
            for i in 0..V_HEAD_DIM {
                out_h[i] += w * new_v_h[i];
            }
        }

        // Append new_k / new_v to past at slot cur_past as bf16 so the
        // next token's SDPA sees them.
        write_present_kv_bf16(past_k, &new_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_bf16(past_v, &new_v, cur_past, capacity, V_HEAD_DIM);
    }

    // ============ PHASE C: post-attention projections ============
    // Batched o_proj on stacked attn_outs.
    //
    // **iter 048 dispatch.** o_proj is the largest single int4 weight in
    // the shell (28 MB) — the row-blocked iter 046 tile halves xs L1
    // traffic and wins +41% at seq>=4. `ProjShape::Oproj` routes through
    // the blocked dispatcher; at seq<4 it falls back to iter 042.
    let mut o_outs = vec![0.0f32; seq * HIDDEN];
    dispatch_int4_multi(
        ProjShape::Oproj,
        &shell.o_proj_packed,
        &shell.o_proj_scale,
        &attn_outs,
        HIDDEN,
        NUM_HEADS * V_HEAD_DIM,
        seq,
        &mut o_outs,
    );

    // Per-token residual + post-norm.
    let mut posts = vec![0.0f32; seq * HIDDEN];
    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let o_t = &o_outs[t * HIDDEN..(t + 1) * HIDDEN];
        let res_t = &mut attn_residual[t * HIDDEN..(t + 1) * HIDDEN];
        for i in 0..HIDDEN {
            res_t[i] = x_t[i] + o_t[i];
        }
        let p = rmsnorm_apply(res_t, &shell.post_norm, HIDDEN);
        posts[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&p);
        attn_out_post_norm[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&p);
    }

    // Batched router.
    let mut router_logits = vec![0.0f32; seq * N_ROUTED_EXPERTS];
    dispatch_int4_multi(
        ProjShape::Generic,
        &shell.router_packed,
        &shell.router_scale,
        &posts,
        N_ROUTED_EXPERTS,
        HIDDEN,
        seq,
        &mut router_logits,
    );

    // Per-token sigmoid + topK + weights.
    let bias: &[f32] = unsafe {
        std::slice::from_raw_parts(shell.router_bias.as_ptr() as *const f32, N_ROUTED_EXPERTS)
    };
    for t in 0..seq {
        let logits_t = &router_logits[t * N_ROUTED_EXPERTS..(t + 1) * N_ROUTED_EXPERTS];
        let mut scores_raw = vec![0.0f32; N_ROUTED_EXPERTS];
        for i in 0..N_ROUTED_EXPERTS {
            scores_raw[i] = 1.0f32 / (1.0f32 + (-logits_t[i]).exp());
        }
        let mut scores_for_choice = vec![0.0f32; N_ROUTED_EXPERTS];
        for i in 0..N_ROUTED_EXPERTS {
            scores_for_choice[i] = scores_raw[i] + bias[i];
        }
        let mut idx_score: Vec<(usize, f32)> =
            scores_for_choice.iter().copied().enumerate().collect();
        idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut tw = vec![0.0f32; TOPK];
        for k in 0..TOPK {
            routing_ids[t * TOPK + k] = idx_score[k].0 as i64;
            tw[k] = scores_raw[idx_score[k].0];
        }
        let s: f32 = tw.iter().sum::<f32>() + 1.0e-20;
        for w in tw.iter_mut() {
            *w = *w / s * ROUTED_SCALING_FACTOR;
        }
        routing_weights[t * TOPK..(t + 1) * TOPK].copy_from_slice(&tw);
    }

    // Batched shared_gate + shared_up.
    //
    // **iter 075 dispatch.** Both shared_gate and shared_up are N=2048,
    // K=7168 → 7 MB packed int4. The iter 046 microbench (commit
    // 77bc56f) showed +28% over iter 042 at seq=16 on this shape, but
    // the win only materializes at seq>=8 — at seq=4 the smaller N
    // doesn't pay off the RB=2 register pressure. `ProjShape::LargeShape`
    // routes through iter 046 blocked at seq>=8 and stays on iter 042
    // for seq<8.
    let mut shared_gate_out = vec![0.0f32; seq * INTERMEDIATE_SHARED];
    let mut shared_up_out = vec![0.0f32; seq * INTERMEDIATE_SHARED];
    dispatch_int4_multi(
        ProjShape::LargeShape,
        &shell.shared_gate_packed,
        &shell.shared_gate_scale,
        &posts,
        INTERMEDIATE_SHARED,
        HIDDEN,
        seq,
        &mut shared_gate_out,
    );
    dispatch_int4_multi(
        ProjShape::LargeShape,
        &shell.shared_up_packed,
        &shell.shared_up_scale,
        &posts,
        INTERMEDIATE_SHARED,
        HIDDEN,
        seq,
        &mut shared_up_out,
    );

    // Per-token SwiGLU.
    let mut shared_inters = vec![0.0f32; seq * INTERMEDIATE_SHARED];
    for t in 0..seq {
        let g_t = &shared_gate_out[t * INTERMEDIATE_SHARED..(t + 1) * INTERMEDIATE_SHARED];
        let u_t = &shared_up_out[t * INTERMEDIATE_SHARED..(t + 1) * INTERMEDIATE_SHARED];
        let i_t = &mut shared_inters[t * INTERMEDIATE_SHARED..(t + 1) * INTERMEDIATE_SHARED];
        shell::swiglu_mul(g_t, u_t, i_t);
    }

    // Batched shared_down.
    //
    // **iter 048 dispatch.** Shared_down's N=HIDDEN=7168 rows match
    // o_proj's row count, with K=2048 instead of 8192. Same RB=2 row-
    // blocking wins (+33% over iter 042 at seq=16).
    dispatch_int4_multi(
        ProjShape::SharedDown,
        &shell.shared_down_packed,
        &shell.shared_down_scale,
        &shared_inters,
        HIDDEN,
        INTERMEDIATE_SHARED,
        seq,
        &mut shared_expert_out,
    );

    MultiShellOutputs {
        attn_out_post_norm,
        attn_residual,
        shared_expert_out,
        routing_ids,
        routing_weights,
    }
}

/// Write `present` (shape `[NUM_HEADS, head_dim]`, f32) into slot `slot`
/// of a `[NUM_HEADS, capacity, head_dim]` **bf16-as-u16** KV buffer.
/// Per-element f32 -> bf16 rounding inlined. Internal helper for the
/// multi-token loop; mirrors the engine's `write_present_kv` but
/// self-contained so the kernel can be unit-tested without pulling in
/// the engine.
fn write_present_kv_bf16(
    buf: &mut [u16],
    present: &[f32],
    slot: usize,
    capacity: usize,
    head_dim: usize,
) {
    debug_assert!(slot < capacity);
    debug_assert_eq!(buf.len(), NUM_HEADS * capacity * head_dim);
    debug_assert_eq!(present.len(), NUM_HEADS * head_dim);
    for h in 0..NUM_HEADS {
        let dst_off = h * capacity * head_dim + slot * head_dim;
        let src_off = h * head_dim;
        let dst = &mut buf[dst_off..dst_off + head_dim];
        let src = &present[src_off..src_off + head_dim];
        for i in 0..head_dim {
            dst[i] = f32_to_bf16_bits_local(src[i]);
        }
    }
}

/// Re-export the bf16-weight RMSNorm (shell.rs's rmsnorm_apply) for use here.
fn rmsnorm_apply(x: &[f32], weight_bf16: &[u8], dim: usize) -> Vec<f32> {
    shell::rmsnorm_apply_pub(x, weight_bf16, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell::{
        INTERMEDIATE_SHARED, QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM,
    };

    /// Build a minimal-fake `Int4Shell` with deterministic non-trivial
    /// bf16 weights so the forward path actually exercises every
    /// dequantization. Layer-norm weights all 0.5 (bf16=0x3F00), packed
    /// nibbles all 0x11 (-7 signed × 2), scales all 1.0 (bf16=0x3F80).
    fn make_test_shell() -> Int4Shell {
        let norm_w = [0x00u8, 0x3F]; // little-endian bf16 = 0.5
        let make_norm = |dim: usize| -> Vec<u8> {
            let mut v = vec![0u8; dim * 2];
            for i in 0..dim {
                v[i * 2] = norm_w[0];
                v[i * 2 + 1] = norm_w[1];
            }
            v
        };
        let make_packed =
            |n_rows: usize, k_cols: usize| -> Vec<u8> { vec![0x11u8; n_rows * k_cols / 2] };
        let make_scale = |n_rows: usize, k_cols: usize| -> Vec<u8> {
            let n_groups = k_cols / GROUP_SIZE;
            let mut v = vec![0u8; n_rows * n_groups * 2];
            for i in 0..n_rows * n_groups {
                v[i * 2] = 0x80;
                v[i * 2 + 1] = 0x3F;
            }
            v
        };
        let router_bias = vec![0u8; N_ROUTED_EXPERTS * 4];

        Int4Shell {
            layer: 0,
            input_norm: make_norm(HIDDEN),
            q_a_proj_packed: make_packed(Q_LORA_RANK, HIDDEN),
            q_a_proj_scale: make_scale(Q_LORA_RANK, HIDDEN),
            q_a_norm: make_norm(Q_LORA_RANK),
            q_b_proj_packed: make_packed(NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK),
            q_b_proj_scale: make_scale(NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK),
            kv_a_proj_packed: make_packed(KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN),
            kv_a_proj_scale: make_scale(KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN),
            kv_a_norm: make_norm(KV_LORA_RANK),
            kv_b_proj_packed: make_packed(
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                KV_LORA_RANK,
            ),
            kv_b_proj_scale: make_scale(NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
            o_proj_packed: make_packed(HIDDEN, NUM_HEADS * V_HEAD_DIM),
            o_proj_scale: make_scale(HIDDEN, NUM_HEADS * V_HEAD_DIM),
            post_norm: make_norm(HIDDEN),
            router_packed: make_packed(N_ROUTED_EXPERTS, HIDDEN),
            router_scale: make_scale(N_ROUTED_EXPERTS, HIDDEN),
            router_bias,
            shared_gate_packed: make_packed(INTERMEDIATE_SHARED, HIDDEN),
            shared_gate_scale: make_scale(INTERMEDIATE_SHARED, HIDDEN),
            shared_up_packed: make_packed(INTERMEDIATE_SHARED, HIDDEN),
            shared_up_scale: make_scale(INTERMEDIATE_SHARED, HIDDEN),
            shared_down_packed: make_packed(HIDDEN, INTERMEDIATE_SHARED),
            shared_down_scale: make_scale(HIDDEN, INTERMEDIATE_SHARED),
        }
    }

    fn make_test_input(seed: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; HIDDEN];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((seed.wrapping_mul(31).wrapping_add(i)) as f32).sin() * 1.0e-3;
        }
        x
    }

    /// Compare two f32 buffers with tolerance. The iter 042/046 tiles
    /// sum in the same nibble order as the scalar reference, so we
    /// expect bit-identity in practice; allow ~1e-3 as a safety net.
    fn assert_outputs_match(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        let mut max_abs: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        for i in 0..actual.len() {
            let a = actual[i];
            let e = expected[i];
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
            let denom = e.abs().max(1.0e-6);
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
        assert!(
            max_abs < 1.0e-3 && max_rel < 1.0e-3,
            "{label}: max_abs={max_abs} max_rel={max_rel}",
        );
    }

    /// iter 048 critical regression guard: at seq=1 every shape must
    /// route to the single-token kernel (bit-identical to
    /// `dequant_gemv_int4_auto`). Every K2.6 inference today hits
    /// seq=1 — any drift here corrupts every token.
    #[test]
    fn dispatch_int4_multi_seq_1_matches_single_token_kernel() {
        let n_rows = 64;
        let k_cols = 128;
        let seq = 1;

        let mut packed = vec![0u8; n_rows * k_cols / 2];
        for r in 0..n_rows {
            for c in 0..(k_cols / 2) {
                packed[r * (k_cols / 2) + c] = ((r.wrapping_mul(31).wrapping_add(c)) & 0xFF) as u8;
            }
        }
        let n_groups = k_cols / GROUP_SIZE;
        let mut scales = vec![0u8; n_rows * n_groups * 2];
        for r in 0..n_rows {
            for g in 0..n_groups {
                let s = 0.5f32 + (((r * 7 + g * 3) % 11) as f32) * 0.1;
                let bits = bf16_round(s);
                let off = (r * n_groups + g) * 2;
                scales[off] = (bits & 0xFF) as u8;
                scales[off + 1] = (bits >> 8) as u8;
            }
        }
        let mut xs = vec![0.0f32; seq * k_cols];
        for t in 0..seq {
            for c in 0..k_cols {
                xs[t * k_cols + c] = ((t * 17 + c * 5) as f32).sin() * 0.5;
            }
        }

        let mut y_single = vec![0.0f32; n_rows];
        dequant_gemv_int4_auto(&packed, &scales, &xs, n_rows, k_cols, &mut y_single);

        for shape in [
            ProjShape::Generic,
            ProjShape::Oproj,
            ProjShape::SharedDown,
            ProjShape::LargeShape,
        ] {
            let mut y_disp = vec![0.0f32; n_rows];
            dispatch_int4_multi(
                shape,
                &packed,
                &scales,
                &xs,
                n_rows,
                k_cols,
                seq,
                &mut y_disp,
            );
            for i in 0..n_rows {
                assert_eq!(
                    y_single[i].to_bits(),
                    y_disp[i].to_bits(),
                    "shape={shape:?}: mismatch at i={i}: single={}, dispatch={}",
                    y_single[i],
                    y_disp[i]
                );
            }
        }
    }

    /// Bit-identity test: seq=1 multi-call (which falls through to the
    /// scalar reference) produces identical KV state + per-token
    /// outputs as a single seq=1 forward.
    #[test]
    fn multi_seq_1_matches_seq_1_reference() {
        let shell = make_test_shell();
        let capacity = 4;
        let past_seq_len = 0;
        let seq = 1;

        let x = make_test_input(0);

        let ref_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let ref_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        let ref_out = shell_forward_decode_int4_with_capacity(
            &shell,
            &x,
            &ref_past_k,
            &ref_past_v,
            past_seq_len,
            capacity,
        );

        let mut multi_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut multi_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        let multi_out = shell_forward_decode_int4_multi_with_capacity(
            &shell,
            &x,
            &mut multi_past_k,
            &mut multi_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        // Per-token outputs match.
        assert_eq!(multi_out.attn_out_post_norm, ref_out.attn_out_post_norm);
        assert_eq!(multi_out.attn_residual, ref_out.attn_residual);
        assert_eq!(multi_out.shared_expert_out, ref_out.shared_expert_out);
        assert_eq!(multi_out.routing_ids, ref_out.routing_ids);
        assert_eq!(multi_out.routing_weights, ref_out.routing_weights);

        // KV state matches: ref didn't write into the cache; we place
        // present_k/present_v at slot 0 of each head (as bf16) and
        // compare against multi_past_k/multi_past_v.
        let mut expected_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut expected_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            let pk_dst = h * capacity * QK_HEAD_DIM;
            let pv_dst = h * capacity * V_HEAD_DIM;
            for i in 0..QK_HEAD_DIM {
                expected_past_k[pk_dst + i] =
                    f32_to_bf16_bits_local(ref_out.present_k[h * QK_HEAD_DIM + i]);
            }
            for i in 0..V_HEAD_DIM {
                expected_past_v[pv_dst + i] =
                    f32_to_bf16_bits_local(ref_out.present_v[h * V_HEAD_DIM + i]);
            }
        }
        assert_eq!(multi_past_k, expected_past_k);
        assert_eq!(multi_past_v, expected_past_v);
    }

    /// iter 048 bit-identity at seq=8 (iter 046 dispatch sweet spot):
    /// batched path must produce per-token outputs and KV state within
    /// f32 tolerance of the scalar reference.
    #[test]
    fn multi_batched_matches_scalar_seq_8_iter046_dispatch() {
        let shell = make_test_shell();
        let capacity = 16;
        let past_seq_len = 4;
        let seq = 8;

        // Seed both caches identically with deterministic non-zero
        // history (as bf16-as-u16 so the bit-pattern compare is exact).
        let mut scalar_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut scalar_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        let mut batched_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut batched_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    let v = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                    let b = f32_to_bf16_bits_local(v);
                    scalar_past_k[off_k + i] = b;
                    batched_past_k[off_k + i] = b;
                }
                for i in 0..V_HEAD_DIM {
                    let v = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                    let b = f32_to_bf16_bits_local(v);
                    scalar_past_v[off_v + i] = b;
                    batched_past_v[off_v + i] = b;
                }
            }
        }

        let mut xs = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = make_test_input(t);
            xs[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&x_t);
        }

        let scalar_out = shell_forward_decode_int4_multi_scalar(
            &shell,
            &xs,
            &mut scalar_past_k,
            &mut scalar_past_v,
            past_seq_len,
            capacity,
            seq,
        );
        let batched_out = shell_forward_decode_int4_multi_batched(
            &shell,
            &xs,
            &mut batched_past_k,
            &mut batched_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        assert_outputs_match(
            &batched_out.attn_out_post_norm,
            &scalar_out.attn_out_post_norm,
            "attn_out_post_norm",
        );
        assert_outputs_match(
            &batched_out.attn_residual,
            &scalar_out.attn_residual,
            "attn_residual",
        );
        assert_outputs_match(
            &batched_out.shared_expert_out,
            &scalar_out.shared_expert_out,
            "shared_expert_out",
        );
        assert_eq!(batched_out.routing_ids, scalar_out.routing_ids);
        assert_outputs_match(
            &batched_out.routing_weights,
            &scalar_out.routing_weights,
            "routing_weights",
        );
        // KV cache (bf16-as-u16) bit-identical.
        assert_eq!(batched_past_k, scalar_past_k, "past_k");
        assert_eq!(batched_past_v, scalar_past_v, "past_v");
    }

    /// iter 075 bit-identity: `LargeShape` at seq=4 must produce
    /// byte-identical output to the per-token scalar loop (the
    /// dispatcher routes seq<8 to iter 042, which is bit-identical
    /// per-cell to the single-token kernel by the existing
    /// `multi_matches_per_token_loop_*` tests).
    #[test]
    fn dispatch_int4_multi_large_shape_seq_4_matches_scalar() {
        use crate::kernel_avx512::dequant_gemv_int4_auto;
        let n_rows = 64;
        let k_cols = 128;
        let seq = 4;
        let (packed, scales, xs) = make_dispatcher_test_data(n_rows, k_cols, seq);

        // Per-token scalar reference.
        let mut y_scalar = vec![0.0f32; seq * n_rows];
        for t in 0..seq {
            dequant_gemv_int4_auto(
                &packed,
                &scales,
                &xs[t * k_cols..(t + 1) * k_cols],
                n_rows,
                k_cols,
                &mut y_scalar[t * n_rows..(t + 1) * n_rows],
            );
        }

        // LargeShape at seq=4 → iter 042 (blocked threshold is seq>=8).
        let mut y_disp = vec![0.0f32; seq * n_rows];
        dispatch_int4_multi(
            ProjShape::LargeShape,
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_disp,
        );
        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_scalar[i].to_bits(),
                y_disp[i].to_bits(),
                "LargeShape seq=4 mismatch at i={i}: scalar={}, dispatch={}",
                y_scalar[i],
                y_disp[i]
            );
        }
    }

    /// iter 075 bit-identity: `LargeShape` at seq=8 must produce
    /// byte-identical output to the per-token scalar loop. At seq=8
    /// the dispatcher routes to iter 046 blocked, which is also
    /// bit-identical per-cell (`blocked_matches_iter042_multi_seq_8`).
    /// This catches any regression in the seq>=8 branch of the
    /// LargeShape dispatcher.
    #[test]
    fn dispatch_int4_multi_large_shape_seq_8_matches_scalar() {
        use crate::kernel_avx512::dequant_gemv_int4_auto;
        let n_rows = 64;
        let k_cols = 128;
        let seq = 8;
        let (packed, scales, xs) = make_dispatcher_test_data(n_rows, k_cols, seq);

        let mut y_scalar = vec![0.0f32; seq * n_rows];
        for t in 0..seq {
            dequant_gemv_int4_auto(
                &packed,
                &scales,
                &xs[t * k_cols..(t + 1) * k_cols],
                n_rows,
                k_cols,
                &mut y_scalar[t * n_rows..(t + 1) * n_rows],
            );
        }

        let mut y_disp = vec![0.0f32; seq * n_rows];
        dispatch_int4_multi(
            ProjShape::LargeShape,
            &packed,
            &scales,
            &xs,
            n_rows,
            k_cols,
            seq,
            &mut y_disp,
        );
        for i in 0..(seq * n_rows) {
            assert_eq!(
                y_scalar[i].to_bits(),
                y_disp[i].to_bits(),
                "LargeShape seq=8 mismatch at i={i}: scalar={}, dispatch={}",
                y_scalar[i],
                y_disp[i]
            );
        }
    }

    /// Factored-out test-data builder (same pattern as the seq=1
    /// dispatcher test). Returns deterministic packed/scales/xs for
    /// the given dimensions and seq count.
    fn make_dispatcher_test_data(
        n_rows: usize,
        k_cols: usize,
        seq: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        let mut packed = vec![0u8; n_rows * k_cols / 2];
        for r in 0..n_rows {
            for c in 0..(k_cols / 2) {
                packed[r * (k_cols / 2) + c] = ((r.wrapping_mul(31).wrapping_add(c)) & 0xFF) as u8;
            }
        }
        let n_groups = k_cols / GROUP_SIZE;
        let mut scales = vec![0u8; n_rows * n_groups * 2];
        for r in 0..n_rows {
            for g in 0..n_groups {
                let s = 0.5f32 + (((r * 7 + g * 3) % 11) as f32) * 0.1;
                let bits = bf16_round(s);
                let off = (r * n_groups + g) * 2;
                scales[off] = (bits & 0xFF) as u8;
                scales[off + 1] = (bits >> 8) as u8;
            }
        }
        let mut xs = vec![0.0f32; seq * k_cols];
        for t in 0..seq {
            for c in 0..k_cols {
                xs[t * k_cols + c] = ((t * 17 + c * 5) as f32).sin() * 0.5;
            }
        }
        (packed, scales, xs)
    }
}
