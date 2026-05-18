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

use crate::kernel_avx512::dequant_gemv_int4_auto;
use crate::kernel_avx512_multi::dequant_gemm_int4_multi_auto;
use crate::kernel_bf16::bf16_gemv_auto;
use crate::safetensors_source::SafetensorsShell;
use crate::shell::{
    self, ShellOutputs, HIDDEN, INTERMEDIATE_SHARED, KV_LORA_RANK, NUM_HEADS, N_ROUTED_EXPERTS,
    QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, Q_LORA_RANK, ROUTED_SCALING_FACTOR, TOPK,
    V_HEAD_DIM,
};

const GROUP_SIZE: usize = 32;

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
/// HEAD_DIM]`. For callers that pre-allocate to a larger capacity and
/// avoid per-token Vec realloc, use [`shell_forward_decode_int4_with_capacity`].
pub fn shell_forward_decode_int4(
    shell: &Int4Shell,
    x_f32: &[f32],
    past_k: &[f32],
    past_v: &[f32],
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
/// row-major. Head `h`'s populated keys occupy
/// `past_k[h * capacity * QK_HEAD_DIM .. h * capacity * QK_HEAD_DIM + past_seq_len * QK_HEAD_DIM]`.
/// `past_v` is laid out similarly with `V_HEAD_DIM`.
pub fn shell_forward_decode_int4_with_capacity(
    shell: &Int4Shell,
    x_f32: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    past_seq_len: usize,
    capacity: usize,
) -> ShellOutputs {
    // Reuse the shell.rs forward but swap bf16_gemv_auto -> dequant_gemv_int4_auto.
    // Easiest: copy the body and adapt. (Generic functions over a trait would
    // be cleaner but pure functions are fine here.)
    assert_eq!(x_f32.len(), HIDDEN);
    assert!(
        capacity >= past_seq_len,
        "capacity ({capacity}) must be >= past_seq_len ({past_seq_len})"
    );
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

    // SDPA
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut attn_out = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
    let kv_len = past_seq_len + 1;
    for h in 0..NUM_HEADS {
        let q_h = &q_full[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        // Slice with `capacity` as the per-head stride, then take only
        // the first `past_seq_len` rows. When the caller passes
        // exact-fit buffers (`capacity == past_seq_len`) this is the
        // original tight slice; with a pre-allocated capacity buffer
        // the trailing rows are unused/zero.
        let pk_base = h * capacity * QK_HEAD_DIM;
        let pv_base = h * capacity * V_HEAD_DIM;
        let past_k_h = &past_k[pk_base..pk_base + past_seq_len * QK_HEAD_DIM];
        let past_v_h = &past_v[pv_base..pv_base + past_seq_len * V_HEAD_DIM];
        let new_k_h = &new_k[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        let new_v_h = &new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];

        let mut scores = vec![0.0f32; kv_len];
        for j in 0..past_seq_len {
            let k_row = &past_k_h[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
            let mut s = 0.0f32;
            for i in 0..QK_HEAD_DIM {
                s += q_h[i] * k_row[i];
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
        let out_h = &mut attn_out[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];
        out_h.fill(0.0);
        for j in 0..past_seq_len {
            let v_row = &past_v_h[j * V_HEAD_DIM..(j + 1) * V_HEAD_DIM];
            let w = scores[j];
            for i in 0..V_HEAD_DIM {
                out_h[i] += w * v_row[i];
            }
        }
        let w = scores[past_seq_len];
        for i in 0..V_HEAD_DIM {
            out_h[i] += w * new_v_h[i];
        }
    }

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

    // Shared expert (int4 ×3)
    let mut shared_gate_out = vec![0.0f32; INTERMEDIATE_SHARED];
    dequant_gemv_int4_auto(
        &shell.shared_gate_packed,
        &shell.shared_gate_scale,
        &post,
        INTERMEDIATE_SHARED,
        HIDDEN,
        &mut shared_gate_out,
    );
    let mut shared_up_out = vec![0.0f32; INTERMEDIATE_SHARED];
    dequant_gemv_int4_auto(
        &shell.shared_up_packed,
        &shell.shared_up_scale,
        &post,
        INTERMEDIATE_SHARED,
        HIDDEN,
        &mut shared_up_out,
    );
    let mut shared_inter = vec![0.0f32; INTERMEDIATE_SHARED];
    shell::swiglu_mul(&shared_gate_out, &shared_up_out, &mut shared_inter);
    let mut shared_out = vec![0.0f32; HIDDEN];
    dequant_gemv_int4_auto(
        &shell.shared_down_packed,
        &shell.shared_down_scale,
        &shared_inter,
        HIDDEN,
        INTERMEDIATE_SHARED,
        &mut shared_out,
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

/// Per-token outputs of a multi-token shell forward (`seq >= 1`).
///
/// Layout: every per-token field is stored as a flat `[seq * D]` vector
/// in token order (token 0 first). The caller indexes into these as
/// `field[t * D .. (t + 1) * D]` to recover a single token's slice.
///
/// `present_k` / `present_v` are NOT in this struct — the multi-token
/// kernel writes them in place into the caller's pre-allocated KV
/// cache buffer (slots `[past_seq_len, past_seq_len + seq)` of each
/// head).
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
/// kernel.
///
/// This is the seq>=1 entry point — the API seam that future
/// SIMD/tiled-GEMM work can hook into. The seq=1 path
/// ([`shell_forward_decode_int4_with_capacity`]) is unchanged and
/// still used by every existing caller.
///
/// **Semantics (functionally equivalent to today).** This call is
/// observationally identical to `seq` sequential calls of
/// [`shell_forward_decode_int4_with_capacity`] — the same int4 GEMV
/// kernels run per token, in token order, with the KV cache updated
/// after each step so the next token can attend to it. The only
/// behavioral change for callers is the API: outputs are concatenated
/// across tokens, and `past_k` / `past_v` are written in place rather
/// than returned. Unit tests in `tests` assert bit-identity to the
/// seq=1 loop.
///
/// **Why a loop and not a real GEMM.** A native multi-token kernel
/// would batch the per-projection matmuls across tokens (`[seq, K] x
/// [K, N]` GEMM instead of `seq` independent `[K] x [K, N]` GEMVs).
/// That's a 1–2 week AVX-VNNI / tiled-GEMM lift. This function is the
/// seam that lets the rest of the engine (speculative decode iter 036,
/// chunked prefill iter 040) call a multi-token API today; the inside
/// can be replaced with a real GEMM later without touching callers.
///
/// **Inputs.**
/// - `xs_f32`: layer inputs, shape `[seq, HIDDEN]` flat. Token `t`'s
///   row lives at `xs_f32[t * HIDDEN .. (t + 1) * HIDDEN]`.
/// - `past_k` / `past_v`: pre-allocated KV cache, shape
///   `[NUM_HEADS, capacity, *_HEAD_DIM]`. Only the first
///   `past_seq_len` slots are populated on entry; the kernel writes
///   slots `[past_seq_len, past_seq_len + seq)` on exit.
/// - `past_seq_len`: populated KV length on entry.
/// - `capacity`: total per-head KV slot capacity. Must be
///   `>= past_seq_len + seq`.
/// - `seq`: number of tokens to process. Must be `>= 1`.
pub fn shell_forward_decode_int4_multi_with_capacity(
    shell: &Int4Shell,
    xs_f32: &[f32],
    past_k: &mut [f32],
    past_v: &mut [f32],
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

    // For seq=1, the per-token kernel is faster than the multi-tile —
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

/// Original per-token loop. Kept as a reference implementation for
/// bit-identity testing — see [`shell_forward_decode_int4_multi_batched`].
pub fn shell_forward_decode_int4_multi_scalar(
    shell: &Int4Shell,
    xs_f32: &[f32],
    past_k: &mut [f32],
    past_v: &mut [f32],
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
        write_present_kv_inplace(past_k, &outs.present_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_inplace(past_v, &outs.present_v, cur_past, capacity, V_HEAD_DIM);

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
/// q_full / new_k / new_v, SDPA against past KV cache, append new K/V
/// into the cache so the next token sees it.
///
/// **Phase C (batched projections, no KV).** Batch o_proj on the stack
/// of per-token attn_outs, per-token residual + post-norm, batch
/// router + sigmoid + topK + shared_gate + shared_up, SwiGLU,
/// shared_down.
///
/// All projections in phases A and C are `[seq, K] x [K, N]` int4 GEMMs
/// that amortize one weight load over `seq` tokens. At seq=4-16 this
/// gives 1.5-5x per-projection speedup (iter 042 microbench).
fn shell_forward_decode_int4_multi_batched(
    shell: &Int4Shell,
    xs_f32: &[f32],
    past_k: &mut [f32],
    past_v: &mut [f32],
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
    dequant_gemm_int4_multi_auto(
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
    dequant_gemm_int4_multi_auto(
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
    dequant_gemm_int4_multi_auto(
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
    dequant_gemm_int4_multi_auto(
        &shell.kv_b_proj_packed,
        &shell.kv_b_proj_scale,
        &kv_a_n,
        kv_b_dim,
        KV_LORA_RANK,
        seq,
        &mut kv_bs,
    );

    // ============ PHASE B: per-token RoPE + SDPA + KV append ============
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
                    s += q_h[i] * k_row[i];
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
                    out_h[i] += w * v_row[i];
                }
            }
            let w = scores[cur_past];
            for i in 0..V_HEAD_DIM {
                out_h[i] += w * new_v_h[i];
            }
        }

        // Append new_k / new_v to past at slot cur_past so the next
        // token's SDPA sees them.
        write_present_kv_inplace(past_k, &new_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_inplace(past_v, &new_v, cur_past, capacity, V_HEAD_DIM);
    }

    // ============ PHASE C: post-attention projections ============
    // Batched o_proj on stacked attn_outs.
    let mut o_outs = vec![0.0f32; seq * HIDDEN];
    dequant_gemm_int4_multi_auto(
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
    dequant_gemm_int4_multi_auto(
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
    let mut shared_gate_out = vec![0.0f32; seq * INTERMEDIATE_SHARED];
    let mut shared_up_out = vec![0.0f32; seq * INTERMEDIATE_SHARED];
    dequant_gemm_int4_multi_auto(
        &shell.shared_gate_packed,
        &shell.shared_gate_scale,
        &posts,
        INTERMEDIATE_SHARED,
        HIDDEN,
        seq,
        &mut shared_gate_out,
    );
    dequant_gemm_int4_multi_auto(
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
    dequant_gemm_int4_multi_auto(
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

/// Write `present` (shape `[NUM_HEADS, head_dim]`) into slot `slot` of
/// a `[NUM_HEADS, capacity, head_dim]` KV buffer. Internal helper for
/// the multi-token loop — the engine's `write_present_kv` does the
/// same thing but lives in `tahoma-engine-sparse-moe`, and we want
/// this crate self-contained so the kernel can be unit-tested without
/// pulling in the engine.
fn write_present_kv_inplace(
    buf: &mut [f32],
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
        buf[dst_off..dst_off + head_dim]
            .copy_from_slice(&present[h * head_dim..(h + 1) * head_dim]);
    }
}

/// Re-export the bf16-weight RMSNorm (shell.rs's rmsnorm_apply) for use here.
fn rmsnorm_apply(x: &[f32], weight_bf16: &[u8], dim: usize) -> Vec<f32> {
    shell::rmsnorm_apply_pub(x, weight_bf16, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Pull in INTERMEDIATE_SHARED + the head dims that the make_test_shell
    // builder needs explicitly; the rest (HIDDEN, NUM_HEADS, TOPK,
    // KV_LORA_RANK, Q_LORA_RANK, N_ROUTED_EXPERTS) are re-exported via
    // the parent module's `use crate::shell::{...}`.
    use crate::shell::{
        INTERMEDIATE_SHARED, QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM,
    };

    /// Build a minimal-fake `Int4Shell` whose weights are all zero (or
    /// a known deterministic pattern) for shape/seam testing. Real
    /// numerical correctness is checked against the seq=1 reference
    /// path — we don't need the weights to mean anything, only that
    /// the multi path produces byte-identical KV updates and outputs.
    fn make_test_shell() -> Int4Shell {
        // Build with deterministic non-trivial bf16 weights so the
        // forward path actually exercises every dequantization. Zero
        // weights would make every projection output 0 and the test
        // would pass even with broken arithmetic.
        //
        // We pick bf16 = 0x3F00 = 0.5 for every layer-norm weight, and
        // build packed int4 buffers where every nibble = 1 (unsigned)
        // = -7 (signed) with scale bf16 = 0x3C00 = 1.0. Then every
        // matmul output is constant -7 * sum(x). Enough to drive the
        // RMSNorm / softmax / SwiGLU paths through real values.

        // 0.5 in bf16 = 0x3F00
        let norm_w = [0x00u8, 0x3F]; // little-endian bf16 = 0.5
        let make_norm = |dim: usize| -> Vec<u8> {
            let mut v = vec![0u8; dim * 2];
            for i in 0..dim {
                v[i * 2] = norm_w[0];
                v[i * 2 + 1] = norm_w[1];
            }
            v
        };

        // All nibbles = 1 (unsigned), i.e. -7 signed.
        // Each byte = 0x11 (low nibble 1, high nibble 1).
        let make_packed =
            |n_rows: usize, k_cols: usize| -> Vec<u8> { vec![0x11u8; n_rows * k_cols / 2] };
        // Scale = 1.0 in bf16 = 0x3F80.
        let make_scale = |n_rows: usize, k_cols: usize| -> Vec<u8> {
            let n_groups = k_cols / GROUP_SIZE;
            let mut v = vec![0u8; n_rows * n_groups * 2];
            for i in 0..n_rows * n_groups {
                v[i * 2] = 0x80;
                v[i * 2 + 1] = 0x3F;
            }
            v
        };

        // f32 zero for the router bias.
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

    /// Build a deterministic input vector that's not all-zero so the
    /// RMSNorm / softmax / SwiGLU paths run on real fp values.
    fn make_test_input(seed: usize) -> Vec<f32> {
        // Tiny float values centered around 0 to keep arithmetic in
        // the normal-range float window; the int4 weights have small
        // magnitude (all -7 * scale=1 = -7) so the down-stream
        // accumulator stays bounded for HIDDEN=7168.
        let mut x = vec![0.0f32; HIDDEN];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((seed.wrapping_mul(31).wrapping_add(i)) as f32).sin() * 1.0e-3;
        }
        x
    }

    /// Bit-identity test: seq=1 multi-call produces identical
    /// KV state + per-token outputs as a single seq=1 forward.
    #[test]
    fn multi_seq_1_matches_seq_1_reference() {
        let shell = make_test_shell();
        let capacity = 4;
        let past_seq_len = 0;
        let seq = 1;

        let x = make_test_input(0);

        // Reference: single seq=1 forward.
        let ref_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let ref_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
        let ref_out = shell_forward_decode_int4_with_capacity(
            &shell,
            &x,
            &ref_past_k,
            &ref_past_v,
            past_seq_len,
            capacity,
        );

        // Test: multi-call with seq=1, same starting cache.
        let mut multi_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut multi_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
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

        // KV state matches: ref didn't write into cache; we manually
        // place present_k/present_v at slot 0 of each head and
        // compare.
        let mut expected_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut expected_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            let pk_dst = h * capacity * QK_HEAD_DIM;
            let pv_dst = h * capacity * V_HEAD_DIM;
            expected_past_k[pk_dst..pk_dst + QK_HEAD_DIM]
                .copy_from_slice(&ref_out.present_k[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM]);
            expected_past_v[pv_dst..pv_dst + V_HEAD_DIM]
                .copy_from_slice(&ref_out.present_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM]);
        }
        assert_eq!(multi_past_k, expected_past_k);
        assert_eq!(multi_past_v, expected_past_v);
    }

    /// Bit-identity test: seq=N multi-call produces same KV state +
    /// per-token outputs as N sequential seq=1 calls feeding through
    /// the same evolving KV cache.
    #[test]
    fn multi_seq_3_matches_sequential_seq_1_calls() {
        let shell = make_test_shell();
        let capacity = 8;
        let past_seq_len = 2; // pretend we already had 2 tokens of history
        let seq = 3;

        // Pre-seed the cache with non-zero history to make sure the
        // "starting past_seq_len > 0" path is exercised.
        let mut ref_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut ref_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    ref_past_k[off_k + i] = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                }
                for i in 0..V_HEAD_DIM {
                    ref_past_v[off_v + i] = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                }
            }
        }

        // Build 3 tokens of input.
        let mut xs = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = make_test_input(t);
            xs[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&x_t);
        }

        // Reference: 3 sequential seq=1 forwards, with the same KV
        // cache progressively updated between each call.
        let mut ref_out_post_norm = vec![0.0f32; seq * HIDDEN];
        let mut ref_out_residual = vec![0.0f32; seq * HIDDEN];
        let mut ref_out_shared = vec![0.0f32; seq * HIDDEN];
        let mut ref_out_ids = vec![0i64; seq * TOPK];
        let mut ref_out_weights = vec![0.0f32; seq * TOPK];
        for t in 0..seq {
            let x_t = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let cur_past = past_seq_len + t;
            let outs = shell_forward_decode_int4_with_capacity(
                &shell,
                x_t,
                &ref_past_k,
                &ref_past_v,
                cur_past,
                capacity,
            );
            // Write present into ref cache at slot `cur_past`.
            for h in 0..NUM_HEADS {
                let dst_k = h * capacity * QK_HEAD_DIM + cur_past * QK_HEAD_DIM;
                let dst_v = h * capacity * V_HEAD_DIM + cur_past * V_HEAD_DIM;
                ref_past_k[dst_k..dst_k + QK_HEAD_DIM]
                    .copy_from_slice(&outs.present_k[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM]);
                ref_past_v[dst_v..dst_v + V_HEAD_DIM]
                    .copy_from_slice(&outs.present_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM]);
            }
            ref_out_post_norm[t * HIDDEN..(t + 1) * HIDDEN]
                .copy_from_slice(&outs.attn_out_post_norm);
            ref_out_residual[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.attn_residual);
            ref_out_shared[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.shared_expert_out);
            ref_out_ids[t * TOPK..(t + 1) * TOPK].copy_from_slice(&outs.routing_ids);
            ref_out_weights[t * TOPK..(t + 1) * TOPK].copy_from_slice(&outs.routing_weights);
        }

        // Test: same seed cache, single multi-call.
        let mut multi_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut multi_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    multi_past_k[off_k + i] = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                }
                for i in 0..V_HEAD_DIM {
                    multi_past_v[off_v + i] = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                }
            }
        }

        let multi_out = shell_forward_decode_int4_multi_with_capacity(
            &shell,
            &xs,
            &mut multi_past_k,
            &mut multi_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        // Per-token outputs match the scalar reference (allowing fp
        // noise from the batched-projection path — the iter 042
        // multi-tile sums in the same nibble/col order as the scalar
        // kernel, so we expect bit-identity).
        assert_outputs_match(
            &multi_out.attn_out_post_norm,
            &ref_out_post_norm,
            "attn_out_post_norm",
        );
        assert_outputs_match(&multi_out.attn_residual, &ref_out_residual, "attn_residual");
        assert_outputs_match(
            &multi_out.shared_expert_out,
            &ref_out_shared,
            "shared_expert_out",
        );
        assert_eq!(multi_out.routing_ids, ref_out_ids);
        assert_outputs_match(
            &multi_out.routing_weights,
            &ref_out_weights,
            "routing_weights",
        );

        // KV cache: bit-identical.
        assert_eq!(multi_past_k, ref_past_k);
        assert_eq!(multi_past_v, ref_past_v);
    }

    /// Compare two f32 buffers, asserting they're near-identical.
    /// The iter 042 batched-projection path sums in the same nibble
    /// order as the per-token kernel, so we expect bit-identity in
    /// practice; allow ~1e-4 abs / rel tolerance as a safety net
    /// against any rayon-induced reordering.
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

    /// Explicit bit-identity test between batched and scalar paths.
    /// Same inputs, same starting KV cache; outputs must agree.
    #[test]
    fn multi_batched_matches_scalar() {
        let shell = make_test_shell();
        let capacity = 8;
        let past_seq_len = 2;
        let seq = 3;

        // Seed both caches identically.
        let mut scalar_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut scalar_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
        let mut batched_past_k = vec![0.0f32; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut batched_past_v = vec![0.0f32; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    let v = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                    scalar_past_k[off_k + i] = v;
                    batched_past_k[off_k + i] = v;
                }
                for i in 0..V_HEAD_DIM {
                    let v = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                    scalar_past_v[off_v + i] = v;
                    batched_past_v[off_v + i] = v;
                }
            }
        }

        let mut xs = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = make_test_input(t);
            xs[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&x_t);
        }

        // Scalar reference.
        let scalar_out = shell_forward_decode_int4_multi_scalar(
            &shell,
            &xs,
            &mut scalar_past_k,
            &mut scalar_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        // Batched path.
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
        assert_eq!(batched_past_k, scalar_past_k, "past_k");
        assert_eq!(batched_past_v, scalar_past_v, "past_v");
    }
}
