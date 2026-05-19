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
    apply_rope_kimi_pub, rmsnorm_apply_pub, rope_cos_sin_pub, swiglu_mul, HIDDEN,
    INTERMEDIATE_DENSE, KV_LORA_RANK, NUM_HEADS, QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM,
    Q_LORA_RANK, V_HEAD_DIM,
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
    // Bounds-check in release too. A corrupted vocab id or an
    // off-by-one in the sampler would otherwise read past the mmap
    // byte-by-byte in the loop and panic with the much less actionable
    // `index out of bounds`. Use `checked_*` to defend against the
    // (impossible-but-cheap) usize overflow.
    let start = id.checked_mul(row_bytes).expect("embed offset overflow");
    let end = start.checked_add(row_bytes).expect("embed offset overflow");
    assert!(
        end <= embed_table_bf16.len(),
        "embed lookup out of range: token {token_id} row [{start},{end}) table len {}",
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
/// shape `[NUM_HEADS, capacity, V_HEAD_DIM]`, **bf16-as-u16** storage
/// (autolab campaign 029 / A8). Only the first `past_seq_len` slots
/// per head are populated.
pub fn layer0_forward_decode_int4_with_capacity(
    layer: &Int4Layer0,
    x_f32: &[f32],
    past_k: &[u16],
    past_v: &[u16],
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

    // SDPA — autolab campaign 029 (A8): past_k/past_v are bf16-as-u16,
    // upconverted to f32 inline at each dot-product element. The new
    // (this-step) k/v stay f32 — the caller writes them into the bf16
    // cache after we return.
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
        let out_h = &mut attn_out[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];
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
    swiglu_mul(&gate_out, &up_out, &mut inter);
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

/// Per-token outputs of a multi-token layer-0 forward (`seq >= 1`).
///
/// Layout: `hidden_out` is flat `[seq, HIDDEN]` in token order.
/// `present_k` / `present_v` are NOT in this struct — the multi-token
/// kernel writes them in place into the caller's pre-allocated KV
/// cache buffer (slots `[past_seq_len, past_seq_len + seq)` of each
/// head).
pub struct MultiLayer0Outputs {
    /// Per-token hidden-state output (after attention + MLP + residual).
    /// Shape `[seq, HIDDEN]` flat — caller slices
    /// `[t * HIDDEN .. (t + 1) * HIDDEN]` for token `t`.
    pub hidden_out: Vec<f32>,
}

/// Multi-token layer-0 forward — the seq>=1 entry point. The seq=1
/// path ([`layer0_forward_decode_int4_with_capacity`]) is unchanged.
///
/// Like [`crate::shell_int4::shell_forward_decode_int4_multi_with_capacity`],
/// this is currently an internal scalar loop over `seq` sequential
/// seq=1 calls — the API seam that future tiled-GEMM work plugs into.
///
/// **Inputs.**
/// - `xs_f32`: `[seq, HIDDEN]` flat, the per-token layer inputs
///   (typically `embed_token_bf16` of each new token id, concatenated).
/// - `past_k` / `past_v`: pre-allocated KV cache,
///   `[NUM_HEADS, capacity, *_HEAD_DIM]`. Only the first
///   `past_seq_len` slots are populated on entry; the kernel writes
///   slots `[past_seq_len, past_seq_len + seq)` on exit.
/// - `past_seq_len`: populated KV length on entry.
/// - `capacity`: total per-head KV slot capacity. Must be
///   `>= past_seq_len + seq`.
/// - `seq`: number of tokens. Must be `>= 1`.
pub fn layer0_forward_decode_int4_multi_with_capacity(
    layer: &Int4Layer0,
    xs_f32: &[f32],
    past_k: &mut [u16],
    past_v: &mut [u16],
    past_seq_len: usize,
    capacity: usize,
    seq: usize,
) -> MultiLayer0Outputs {
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

    let mut hidden_out = vec![0.0f32; seq * HIDDEN];

    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let cur_past = past_seq_len + t;
        let outs = layer0_forward_decode_int4_with_capacity(
            layer, x_t, past_k, past_v, cur_past, capacity,
        );
        // Write present_k / present_v into slot `cur_past` for each head.
        write_present_kv_inplace(past_k, &outs.present_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_inplace(past_v, &outs.present_v, cur_past, capacity, V_HEAD_DIM);
        hidden_out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.hidden_out);
    }

    MultiLayer0Outputs { hidden_out }
}

/// Same in-place KV write as `shell_int4`'s helper. Lives here to keep
/// `layer0_int4` self-contained for unit testing.
///
/// autolab campaign 029 (A8): the cache is bf16-as-u16; f32→bf16
/// round-to-nearest-even per element matches the engine-side
/// `write_present_kv`.
fn write_present_kv_inplace(
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
        let dst = &mut buf[dst_off..dst_off + head_dim];
        let src = &present[h * head_dim..(h + 1) * head_dim];
        for i in 0..head_dim {
            dst[i] = crate::format::f32_to_bf16_bits(src[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GROUP_SIZE;

    /// Build a deterministic Int4Layer0 with non-trivial weights — same
    /// fake-shell pattern as `shell_int4::tests::make_test_shell`.
    fn make_test_layer0() -> Int4Layer0 {
        let norm_w = [0x00u8, 0x3F]; // 0.5 in bf16
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
        Int4Layer0 {
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
            gate_proj_packed: make_packed(INTERMEDIATE_DENSE, HIDDEN),
            gate_proj_scale: make_scale(INTERMEDIATE_DENSE, HIDDEN),
            up_proj_packed: make_packed(INTERMEDIATE_DENSE, HIDDEN),
            up_proj_scale: make_scale(INTERMEDIATE_DENSE, HIDDEN),
            down_proj_packed: make_packed(HIDDEN, INTERMEDIATE_DENSE),
            down_proj_scale: make_scale(HIDDEN, INTERMEDIATE_DENSE),
        }
    }

    fn make_test_input(seed: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; HIDDEN];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((seed.wrapping_mul(31).wrapping_add(i)) as f32).sin() * 1.0e-3;
        }
        x
    }

    /// Bit-identity test: seq=1 multi-call produces identical KV +
    /// hidden_out as a single seq=1 forward.
    #[test]
    fn multi_layer0_seq_1_matches_seq_1_reference() {
        let layer = make_test_layer0();
        let capacity = 4;
        let past_seq_len = 0;
        let seq = 1;
        let x = make_test_input(0);

        // KV is bf16-as-u16 (autolab campaign 029 / A8).
        let ref_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let ref_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        let ref_out = layer0_forward_decode_int4_with_capacity(
            &layer,
            &x,
            &ref_past_k,
            &ref_past_v,
            past_seq_len,
            capacity,
        );

        let mut multi_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut multi_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        let multi_out = layer0_forward_decode_int4_multi_with_capacity(
            &layer,
            &x,
            &mut multi_past_k,
            &mut multi_past_v,
            past_seq_len,
            capacity,
            seq,
        );
        assert_eq!(multi_out.hidden_out, ref_out.hidden_out);

        // KV state: ref is unchanged (seq=1 API returns present_k/v),
        // so we manually splat present_k/v at slot 0, encoding f32 →
        // bf16-as-u16 to mirror the engine seam.
        let mut expected_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut expected_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            let pk_dst = h * capacity * QK_HEAD_DIM;
            let pv_dst = h * capacity * V_HEAD_DIM;
            for i in 0..QK_HEAD_DIM {
                expected_past_k[pk_dst + i] =
                    crate::format::f32_to_bf16_bits(ref_out.present_k[h * QK_HEAD_DIM + i]);
            }
            for i in 0..V_HEAD_DIM {
                expected_past_v[pv_dst + i] =
                    crate::format::f32_to_bf16_bits(ref_out.present_v[h * V_HEAD_DIM + i]);
            }
        }
        assert_eq!(multi_past_k, expected_past_k);
        assert_eq!(multi_past_v, expected_past_v);
    }

    /// Bit-identity test: seq=3 multi-call matches 3 sequential seq=1
    /// calls feeding through the same evolving KV cache (starting at
    /// past_seq_len=2 with pre-seeded history).
    #[test]
    fn multi_layer0_seq_3_matches_sequential_seq_1_calls() {
        let layer = make_test_layer0();
        let capacity = 8;
        let past_seq_len = 2;
        let seq = 3;

        // Pre-seed cache with deterministic non-zero history. KV is
        // bf16-as-u16 so we encode each seed through f32_to_bf16_bits.
        let mut ref_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut ref_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    let v = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                    ref_past_k[off_k + i] = crate::format::f32_to_bf16_bits(v);
                }
                for i in 0..V_HEAD_DIM {
                    let v = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                    ref_past_v[off_v + i] = crate::format::f32_to_bf16_bits(v);
                }
            }
        }

        let mut xs = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = make_test_input(t);
            xs[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&x_t);
        }

        // Reference: 3 sequential seq=1 forwards, KV updated between.
        let mut ref_hidden = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let cur_past = past_seq_len + t;
            let outs = layer0_forward_decode_int4_with_capacity(
                &layer,
                x_t,
                &ref_past_k,
                &ref_past_v,
                cur_past,
                capacity,
            );
            // Encode f32 present → bf16 cache slot, matching the
            // engine-side write_present_kv conversion.
            for h in 0..NUM_HEADS {
                let dst_k = h * capacity * QK_HEAD_DIM + cur_past * QK_HEAD_DIM;
                let dst_v = h * capacity * V_HEAD_DIM + cur_past * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    ref_past_k[dst_k + i] =
                        crate::format::f32_to_bf16_bits(outs.present_k[h * QK_HEAD_DIM + i]);
                }
                for i in 0..V_HEAD_DIM {
                    ref_past_v[dst_v + i] =
                        crate::format::f32_to_bf16_bits(outs.present_v[h * V_HEAD_DIM + i]);
                }
            }
            ref_hidden[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.hidden_out);
        }

        // Test: same seed cache (bf16-as-u16), single multi-call.
        let mut multi_past_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut multi_past_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    let v = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                    multi_past_k[off_k + i] = crate::format::f32_to_bf16_bits(v);
                }
                for i in 0..V_HEAD_DIM {
                    let v = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                    multi_past_v[off_v + i] = crate::format::f32_to_bf16_bits(v);
                }
            }
        }
        let multi_out = layer0_forward_decode_int4_multi_with_capacity(
            &layer,
            &xs,
            &mut multi_past_k,
            &mut multi_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        assert_eq!(multi_out.hidden_out, ref_hidden);
        assert_eq!(multi_past_k, ref_past_k);
        assert_eq!(multi_past_v, ref_past_v);
    }

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
