//! Rust int4 implementation of K2.6's dense layer 0.
//!
//! Layer 0 differs from a shell in just the MLP: the shell has a
//! router + 384 routed experts + a shared expert; layer 0 has a
//! plain SwiGLU dense MLP with `intermediate_size = 18432`.
//! Attention (MLA, YARN-RoPE) is identical to the shell.
//!
//! Why this exists: the original OV layer-0 IR was stateless — every
//! decode step ran the full prefix through the attention block,
//! making prefill + generation O(N²) in attention. This module
//! gives layer 0 its own pre-allocated KV cache so it joins the
//! shells on the O(N) per-token path.

use crate::kernel_avx512::dequant_gemv_int4_auto;
use crate::safetensors_source::SafetensorsLayer0;
use crate::shell::{
    apply_rope_kimi_pub, rmsnorm_apply_pub, rope_cos_sin_pub, HIDDEN, INTERMEDIATE_DENSE,
    KV_LORA_RANK, NUM_HEADS, QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, Q_LORA_RANK,
    V_HEAD_DIM,
};
use crate::shell_int4::quantize_int4_group;

/// All layer-0 weights in int4 + bf16 scales, layer-norms kept as bf16.
pub struct Int4Layer0 {
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
    pub gate_proj_packed: Vec<u8>,
    pub gate_proj_scale: Vec<u8>,
    pub up_proj_packed: Vec<u8>,
    pub up_proj_scale: Vec<u8>,
    pub down_proj_packed: Vec<u8>,
    pub down_proj_scale: Vec<u8>,
}

impl Int4Layer0 {
    /// Quantize the bf16 safetensors weights into int4 + bf16 scales.
    pub fn from_safetensors(layer: &SafetensorsLayer0) -> Self {
        let (q_a_p, q_a_s) = quantize_int4_group(layer.q_a_proj, Q_LORA_RANK, HIDDEN);
        let (q_b_p, q_b_s) =
            quantize_int4_group(layer.q_b_proj, NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK);
        let (kv_a_p, kv_a_s) =
            quantize_int4_group(layer.kv_a_proj, KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN);
        let (kv_b_p, kv_b_s) = quantize_int4_group(
            layer.kv_b_proj,
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            KV_LORA_RANK,
        );
        let (o_p, o_s) = quantize_int4_group(layer.o_proj, HIDDEN, NUM_HEADS * V_HEAD_DIM);
        let (g_p, g_s) = quantize_int4_group(layer.gate_proj, INTERMEDIATE_DENSE, HIDDEN);
        let (u_p, u_s) = quantize_int4_group(layer.up_proj, INTERMEDIATE_DENSE, HIDDEN);
        let (d_p, d_s) = quantize_int4_group(layer.down_proj, HIDDEN, INTERMEDIATE_DENSE);
        Self {
            input_norm: layer.input_norm.to_vec(),
            q_a_proj_packed: q_a_p,
            q_a_proj_scale: q_a_s,
            q_a_norm: layer.q_a_norm.to_vec(),
            q_b_proj_packed: q_b_p,
            q_b_proj_scale: q_b_s,
            kv_a_proj_packed: kv_a_p,
            kv_a_proj_scale: kv_a_s,
            kv_a_norm: layer.kv_a_norm.to_vec(),
            kv_b_proj_packed: kv_b_p,
            kv_b_proj_scale: kv_b_s,
            o_proj_packed: o_p,
            o_proj_scale: o_s,
            post_norm: layer.post_norm.to_vec(),
            gate_proj_packed: g_p,
            gate_proj_scale: g_s,
            up_proj_packed: u_p,
            up_proj_scale: u_s,
            down_proj_packed: d_p,
            down_proj_scale: d_s,
        }
    }

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
            + self.gate_proj_packed.len()
            + self.gate_proj_scale.len()
            + self.up_proj_packed.len()
            + self.up_proj_scale.len()
            + self.down_proj_packed.len()
            + self.down_proj_scale.len()
    }
}

/// Outputs of one layer-0 decode step.
pub struct Layer0Outputs {
    /// Hidden state after attention + MLP residual — the input to the
    /// first MoE shell.
    pub hidden_out: Vec<f32>,
    /// This step's new K row, shape `[NUM_HEADS, 1, QK_HEAD_DIM]`.
    pub present_k: Vec<f32>,
    /// This step's new V row, shape `[NUM_HEADS, 1, V_HEAD_DIM]`.
    pub present_v: Vec<f32>,
}

/// Look up one token's embedding row from a bf16 `[vocab, hidden]`
/// flat slice and return it as f32. The caller owns the result Vec.
///
/// The embed_tokens table is huge (~2.3 GB for K2.6) — the standard
/// usage is to mmap the safetensors shard and pass the slice in.
pub fn embed_token_bf16(embed_table_bf16: &[u8], token_id: i64) -> Vec<f32> {
    assert!(token_id >= 0, "token_id < 0: {token_id}");
    let id = token_id as usize;
    let row_bytes = HIDDEN * 2;
    let start = id * row_bytes;
    debug_assert!(
        start + row_bytes <= embed_table_bf16.len(),
        "embed lookup out of range: token {token_id} row start {start} table len {}",
        embed_table_bf16.len()
    );
    let mut out = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        let lo = embed_table_bf16[start + i * 2] as u32;
        let hi = embed_table_bf16[start + i * 2 + 1] as u32;
        let bits16: u32 = (hi << 8) | lo;
        out[i] = f32::from_bits(bits16 << 16);
    }
    out
}

/// One layer-0 decode step. Pre-allocated KV cache variant — same
/// contract as `shell_forward_decode_int4_with_capacity` but with a
/// dense SwiGLU MLP and no router / shared expert / routing outputs.
///
/// `x_f32` is `[HIDDEN]` — the layer input (typically embed_token_bf16
/// of the new token id).
///
/// `past_k` shape `[NUM_HEADS, capacity, QK_HEAD_DIM]`, `past_v`
/// shape `[NUM_HEADS, capacity, V_HEAD_DIM]`; only the first
/// `past_seq_len` slots per head are populated.
pub fn layer0_forward_decode_int4_with_capacity(
    layer: &Int4Layer0,
    x_f32: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    past_seq_len: usize,
    capacity: usize,
) -> Layer0Outputs {
    assert_eq!(x_f32.len(), HIDDEN);
    assert!(
        capacity >= past_seq_len,
        "capacity ({capacity}) must be >= past_seq_len ({past_seq_len})"
    );
    assert_eq!(past_k.len(), NUM_HEADS * capacity * QK_HEAD_DIM);
    assert_eq!(past_v.len(), NUM_HEADS * capacity * V_HEAD_DIM);

    // ----- Attention (identical to shell_forward_decode_int4_with_capacity) -----
    let h_norm = rmsnorm_apply_pub(x_f32, &layer.input_norm, HIDDEN);

    let mut q_a = vec![0.0f32; Q_LORA_RANK];
    dequant_gemv_int4_auto(
        &layer.q_a_proj_packed,
        &layer.q_a_proj_scale,
        &h_norm,
        Q_LORA_RANK,
        HIDDEN,
        &mut q_a,
    );
    let q_a_n = rmsnorm_apply_pub(&q_a, &layer.q_a_norm, Q_LORA_RANK);

    let mut q = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    dequant_gemv_int4_auto(
        &layer.q_b_proj_packed,
        &layer.q_b_proj_scale,
        &q_a_n,
        NUM_HEADS * QK_HEAD_DIM,
        Q_LORA_RANK,
        &mut q,
    );

    let mut kv_a_with_rope = vec![0.0f32; KV_LORA_RANK + QK_ROPE_HEAD_DIM];
    dequant_gemv_int4_auto(
        &layer.kv_a_proj_packed,
        &layer.kv_a_proj_scale,
        &h_norm,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        HIDDEN,
        &mut kv_a_with_rope,
    );
    let (kv_a, k_rope_in) = kv_a_with_rope.split_at(KV_LORA_RANK);
    let kv_a_n = rmsnorm_apply_pub(kv_a, &layer.kv_a_norm, KV_LORA_RANK);

    let mut kv_b = vec![0.0f32; NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
    dequant_gemv_int4_auto(
        &layer.kv_b_proj_packed,
        &layer.kv_b_proj_scale,
        &kv_a_n,
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        KV_LORA_RANK,
        &mut kv_b,
    );

    let (cos, sin) = rope_cos_sin_pub(past_seq_len);
    let mut new_k = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    let mut new_v = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
    let mut k_rope_rot = vec![0.0f32; QK_ROPE_HEAD_DIM];
    apply_rope_kimi_pub(k_rope_in, &cos, &sin, &mut k_rope_rot);

    let mut q_full = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    let mut q_rope_buf = vec![0.0f32; QK_ROPE_HEAD_DIM];
    for h in 0..NUM_HEADS {
        q_full[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM]
            .copy_from_slice(&q[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM]);
        let q_rope_src = &q[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        apply_rope_kimi_pub(q_rope_src, &cos, &sin, &mut q_rope_buf);
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

    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut attn_out = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
    let kv_len = past_seq_len + 1;
    for h in 0..NUM_HEADS {
        let q_h = &q_full[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
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

    let mut o_out = vec![0.0f32; HIDDEN];
    dequant_gemv_int4_auto(
        &layer.o_proj_packed,
        &layer.o_proj_scale,
        &attn_out,
        HIDDEN,
        NUM_HEADS * V_HEAD_DIM,
        &mut o_out,
    );

    let mut residual = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        residual[i] = x_f32[i] + o_out[i];
    }
    let post = rmsnorm_apply_pub(&residual, &layer.post_norm, HIDDEN);

    // ----- Dense SwiGLU MLP (the only place layer 0 differs from a shell) -----
    let mut gate_out = vec![0.0f32; INTERMEDIATE_DENSE];
    dequant_gemv_int4_auto(
        &layer.gate_proj_packed,
        &layer.gate_proj_scale,
        &post,
        INTERMEDIATE_DENSE,
        HIDDEN,
        &mut gate_out,
    );
    let mut up_out = vec![0.0f32; INTERMEDIATE_DENSE];
    dequant_gemv_int4_auto(
        &layer.up_proj_packed,
        &layer.up_proj_scale,
        &post,
        INTERMEDIATE_DENSE,
        HIDDEN,
        &mut up_out,
    );
    let mut inter = vec![0.0f32; INTERMEDIATE_DENSE];
    for i in 0..INTERMEDIATE_DENSE {
        let g = gate_out[i];
        let silu = g / (1.0f32 + (-g).exp());
        inter[i] = silu * up_out[i];
    }
    let mut mlp_out = vec![0.0f32; HIDDEN];
    dequant_gemv_int4_auto(
        &layer.down_proj_packed,
        &layer.down_proj_scale,
        &inter,
        HIDDEN,
        INTERMEDIATE_DENSE,
        &mut mlp_out,
    );

    let mut hidden_out = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        hidden_out[i] = residual[i] + mlp_out[i];
    }

    Layer0Outputs {
        hidden_out,
        present_k: new_k,
        present_v: new_v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_token_bf16_decodes_one_row() {
        // Build a fake [vocab=3, hidden=7168] bf16 table where row k
        // is filled with bf16 value (k+1).0. Look up id=1 → all 2.0.
        let row_bytes = HIDDEN * 2;
        let mut table = vec![0u8; 3 * row_bytes];
        for k in 0..3 {
            // bf16 encoding of (k+1) — just take high 16 bits of f32.
            let v = (k as f32 + 1.0).to_bits();
            let bf16 = (v >> 16) as u16;
            let lo = (bf16 & 0xFF) as u8;
            let hi = (bf16 >> 8) as u8;
            for i in 0..HIDDEN {
                let off = k * row_bytes + i * 2;
                table[off] = lo;
                table[off + 1] = hi;
            }
        }
        let row = embed_token_bf16(&table, 1);
        assert_eq!(row.len(), HIDDEN);
        for v in row {
            assert_eq!(v, 2.0);
        }
    }
}
