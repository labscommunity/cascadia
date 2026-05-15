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
use crate::kernel_bf16::bf16_gemv_auto;
use crate::safetensors_source::SafetensorsShell;
use crate::shell::{
    self, ShellOutputs, HIDDEN, INTERMEDIATE_SHARED, KV_LORA_RANK, NUM_HEADS,
    N_ROUTED_EXPERTS, QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, Q_LORA_RANK,
    ROUTED_SCALING_FACTOR, TOPK, V_HEAD_DIM,
};

const GROUP_SIZE: usize = 32;

/// Quantize a bf16 weight matrix [n_rows, k_cols] (raw bytes, little-endian
/// bf16 = u16) into int4 packed nibbles + per-group bf16 scales.
///
/// Output layout:
///   packed: u8 [n_rows * k_cols / 2], byte i holds nibbles for cols 2i, 2i+1
///   scales: u8 [n_rows * (k_cols / GROUP_SIZE) * 2], bf16 little-endian
fn quantize_int4_group(
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
            let scale = if max_abs == 0.0 { 1.0e-10 } else { max_abs / 7.0 };
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
        let (q_a_packed, q_a_scale) =
            quantize_int4_group(shell.q_a_proj, Q_LORA_RANK, HIDDEN);
        let (q_b_packed, q_b_scale) =
            quantize_int4_group(shell.q_b_proj, NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK);
        let (kv_a_packed, kv_a_scale) =
            quantize_int4_group(shell.kv_a_proj, KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN);
        let (kv_b_packed, kv_b_scale) =
            quantize_int4_group(shell.kv_b_proj, NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK);
        let (o_packed, o_scale) =
            quantize_int4_group(shell.o_proj, HIDDEN, NUM_HEADS * V_HEAD_DIM);
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
            + self.q_a_proj_packed.len() + self.q_a_proj_scale.len() + self.q_a_norm.len()
            + self.q_b_proj_packed.len() + self.q_b_proj_scale.len()
            + self.kv_a_proj_packed.len() + self.kv_a_proj_scale.len() + self.kv_a_norm.len()
            + self.kv_b_proj_packed.len() + self.kv_b_proj_scale.len()
            + self.o_proj_packed.len() + self.o_proj_scale.len()
            + self.post_norm.len()
            + self.router_packed.len() + self.router_scale.len() + self.router_bias.len()
            + self.shared_gate_packed.len() + self.shared_gate_scale.len()
            + self.shared_up_packed.len() + self.shared_up_scale.len()
            + self.shared_down_packed.len() + self.shared_down_scale.len()
    }
}

/// Run one shell forward (decode, seq=1) using int4 weights.
pub fn shell_forward_decode_int4(
    shell: &Int4Shell,
    x_f32: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    past_seq_len: usize,
) -> ShellOutputs {
    // Reuse the shell.rs forward but swap bf16_gemv_auto -> dequant_gemv_int4_auto.
    // Easiest: copy the body and adapt. (Generic functions over a trait would
    // be cleaner but pure functions are fine here.)
    assert_eq!(x_f32.len(), HIDDEN);
    assert_eq!(past_k.len(), NUM_HEADS * past_seq_len * QK_HEAD_DIM);
    assert_eq!(past_v.len(), NUM_HEADS * past_seq_len * V_HEAD_DIM);

    // input layernorm (bf16 weight, scalar)
    let h_norm = rmsnorm_apply(x_f32, &shell.input_norm, HIDDEN);

    // q_a_proj (int4)
    let mut q_a = vec![0.0f32; Q_LORA_RANK];
    dequant_gemv_int4_auto(&shell.q_a_proj_packed, &shell.q_a_proj_scale, &h_norm, Q_LORA_RANK, HIDDEN, &mut q_a);
    let q_a_n = rmsnorm_apply(&q_a, &shell.q_a_norm, Q_LORA_RANK);

    // q_b_proj (int4)
    let mut q = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    dequant_gemv_int4_auto(&shell.q_b_proj_packed, &shell.q_b_proj_scale, &q_a_n, NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK, &mut q);

    // kv_a_proj (int4)
    let mut kv_a_with_rope = vec![0.0f32; KV_LORA_RANK + QK_ROPE_HEAD_DIM];
    dequant_gemv_int4_auto(&shell.kv_a_proj_packed, &shell.kv_a_proj_scale, &h_norm, KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN, &mut kv_a_with_rope);
    let (kv_a, k_rope_in) = kv_a_with_rope.split_at(KV_LORA_RANK);
    let kv_a_n = rmsnorm_apply(kv_a, &shell.kv_a_norm, KV_LORA_RANK);

    // kv_b_proj (int4)
    let mut kv_b = vec![0.0f32; NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
    dequant_gemv_int4_auto(&shell.kv_b_proj_packed, &shell.kv_b_proj_scale, &kv_a_n, NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK, &mut kv_b);

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
        q_full[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM].copy_from_slice(&q_rope_buf);
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
        let past_k_h = &past_k[h * past_seq_len * QK_HEAD_DIM..(h + 1) * past_seq_len * QK_HEAD_DIM];
        let past_v_h = &past_v[h * past_seq_len * V_HEAD_DIM..(h + 1) * past_seq_len * V_HEAD_DIM];
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
            if v > max_s { max_s = v; }
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
    dequant_gemv_int4_auto(&shell.o_proj_packed, &shell.o_proj_scale, &attn_out, HIDDEN, NUM_HEADS * V_HEAD_DIM, &mut o_out);

    let mut residual = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN { residual[i] = x_f32[i] + o_out[i]; }
    let post = rmsnorm_apply(&residual, &shell.post_norm, HIDDEN);

    // Router (int4)
    let mut router_logits = vec![0.0f32; N_ROUTED_EXPERTS];
    dequant_gemv_int4_auto(&shell.router_packed, &shell.router_scale, &post, N_ROUTED_EXPERTS, HIDDEN, &mut router_logits);
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
    dequant_gemv_int4_auto(&shell.shared_gate_packed, &shell.shared_gate_scale, &post, INTERMEDIATE_SHARED, HIDDEN, &mut shared_gate_out);
    let mut shared_up_out = vec![0.0f32; INTERMEDIATE_SHARED];
    dequant_gemv_int4_auto(&shell.shared_up_packed, &shell.shared_up_scale, &post, INTERMEDIATE_SHARED, HIDDEN, &mut shared_up_out);
    let mut shared_inter = vec![0.0f32; INTERMEDIATE_SHARED];
    for i in 0..INTERMEDIATE_SHARED {
        let g = shared_gate_out[i];
        let silu = g / (1.0f32 + (-g).exp());
        shared_inter[i] = silu * shared_up_out[i];
    }
    let mut shared_out = vec![0.0f32; HIDDEN];
    dequant_gemv_int4_auto(&shell.shared_down_packed, &shell.shared_down_scale, &shared_inter, HIDDEN, INTERMEDIATE_SHARED, &mut shared_out);

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

/// Re-export the bf16-weight RMSNorm (shell.rs's rmsnorm_apply) for use here.
fn rmsnorm_apply(x: &[f32], weight_bf16: &[u8], dim: usize) -> Vec<f32> {
    shell::rmsnorm_apply_pub(x, weight_bf16, dim)
}
