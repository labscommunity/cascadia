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

use crate::format::f32_to_bf16_bits as f32_to_bf16_bits_local;
use crate::kernel_avx512::dequant_gemv_int4_auto;
use crate::safetensors_source::SafetensorsLayer0;
use crate::shell::{
    apply_rope_kimi_pub, rmsnorm_apply_pub, rope_cos_sin_pub, swiglu_mul, HIDDEN,
    INTERMEDIATE_DENSE, KV_LORA_RANK, NUM_HEADS, QK_HEAD_DIM, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM,
    Q_LORA_RANK, V_HEAD_DIM,
};
use crate::shell_int4::{dispatch_int4_multi, quantize_int4_group, ProjShape};

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
/// shape `[NUM_HEADS, capacity, V_HEAD_DIM]`, **bf16-as-u16** storage.
/// Only the first `past_seq_len` slots per head are populated.
pub fn layer0_forward_decode_int4_with_capacity(
    layer: &Int4Layer0,
    x_f32: &[f32],
    past_k: &[u16],
    past_v: &[u16],
    past_seq_len: usize,
    capacity: usize,
) -> Layer0Outputs {
    layer0_forward_decode_int4_with_capacity_sparse(
        layer,
        x_f32,
        past_k,
        past_v,
        past_seq_len,
        capacity,
        0.0,
    )
}

/// Same as [`layer0_forward_decode_int4_with_capacity`] but with an
/// extra `ffn_sparsity_threshold` knob that controls the two-phase
/// Gate-first FFN sparsity in the dense MLP block of layer 0.
///
/// `ffn_sparsity_threshold == 0.0` (used by the back-compat wrapper
/// `layer0_forward_decode_int4_with_capacity`) is bit-identical to the
/// pre-port path.
///
/// Positive values activate the same magnitude-threshold sparsity
/// described in [`cascadia_int4_gemm::ffn_forward_sparse_f32`]. K2.6's
/// layer 0 is dense (no MoE) so this is the only place the layer-0
/// FFN sparsity applies; the FFN inside layer 0 follows the same
/// SwiGLU pattern as the routed experts, so the same threshold value
/// is meaningful for both.
pub fn layer0_forward_decode_int4_with_capacity_sparse(
    layer: &Int4Layer0,
    x_f32: &[f32],
    past_k: &[u16],
    past_v: &[u16],
    past_seq_len: usize,
    capacity: usize,
    ffn_sparsity_threshold: f32,
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

    // SDPA — past_k/past_v are bf16-as-u16,
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
    //
    // The two-phase Gate-first sparse path lives in
    // `cascadia_int4_gemm::ffn_forward_sparse_f32`. At
    // `ffn_sparsity_threshold == 0.0` it runs a back-to-back
    // gate / SwiGLU / up / down sequence identical to the pre-port
    // inline code — bit-identical, no overhead.
    let mut mlp_out = vec![0.0f32; HIDDEN];
    let _active_frac = crate::ffn_sparsity::ffn_forward_sparse_f32(
        &post,
        HIDDEN,
        INTERMEDIATE_DENSE,
        &layer.gate_proj_packed,
        &layer.gate_proj_scale,
        &layer.up_proj_packed,
        &layer.up_proj_scale,
        &layer.down_proj_packed,
        &layer.down_proj_scale,
        &mut mlp_out,
        ffn_sparsity_threshold,
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

// =====================================================================
// Multi-token (seq >= 1) entry point (iter 041)
// =====================================================================
//
// Per the iter 048 commit body: "Layer-0 multi still uses the scalar
// loop. Layer 0 is one call per token (not per layer × per token), so
// the wiring effort isn't justified yet." Future iter can swap the body
// for a tile if profiles ever flag layer 0.
//
// KV cache here is `[u16]` (bf16 storage, A8) just like the seq=1 path.
// `write_present_kv_bf16` does the inline f32 -> bf16 round on each
// per-token KV append.

/// Per-token outputs of a multi-token layer-0 forward (`seq >= 1`).
///
/// Layout: `hidden_out` is flat `[seq, HIDDEN]` in token order.
/// `present_k` / `present_v` are NOT in this struct — the multi-token
/// kernel writes them in place into the caller's pre-allocated KV
/// cache buffer (slots `[past_seq_len, past_seq_len + seq)` of each
/// head), as bf16-as-u16.
pub struct MultiLayer0Outputs {
    /// Per-token hidden-state output (after attention + MLP + residual).
    /// Shape `[seq, HIDDEN]` flat — caller slices
    /// `[t * HIDDEN .. (t + 1) * HIDDEN]` for token `t`.
    pub hidden_out: Vec<f32>,
}

/// Multi-token layer-0 forward — the seq>=1 entry point. The seq=1
/// path ([`layer0_forward_decode_int4_with_capacity`]) is unchanged.
///
/// **Semantics.** Observationally identical to `seq` sequential calls
/// of [`layer0_forward_decode_int4_with_capacity`] — same int4 GEMV
/// math runs per token, in token order, with the KV cache updated
/// after each step so the next token's attention sees it. The
/// behavioral change is purely the API: outputs are concatenated
/// across tokens, and `past_k` / `past_v` are written in place rather
/// than returned.
///
/// **Bit-identity.** At seq=1 this delegates straight to the seq=1
/// scalar reference path. At seq>=2 the per-projection GEMVs are
/// batched into iter 042 / iter 046 SIMD multi-token tiles via
/// [`crate::shell_int4::dispatch_int4_multi`]. Those tiles are
/// bit-identical per cell to the scalar kernel — same FMA accumulation
/// order, same dequant grouping — so the seq>=2 path produces
/// byte-identical KV state and per-token hidden outputs as the scalar
/// reference (proved by the `multi_layer0_batched_matches_scalar_seq_*_iter048_dispatch`
/// tests below).
///
/// **Inputs.**
/// - `xs_f32`: `[seq, HIDDEN]` flat, the per-token layer inputs.
/// - `past_k` / `past_v`: pre-allocated **bf16-as-u16** KV cache,
///   `[NUM_HEADS, capacity, *_HEAD_DIM]`. Only the first `past_seq_len`
///   slots are populated on entry; the kernel writes slots
///   `[past_seq_len, past_seq_len + seq)` on exit.
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

    // seq=1 hot path: delegate to the per-token kernel exactly once.
    // Matches the shell wrapper's seq=1 shortcut — the multi-token tile
    // pays a per-row scatter cost that doesn't amortize across a single
    // token, and seq=1 is the dominant K2.6 decode path.
    if seq == 1 {
        return layer0_forward_decode_int4_multi_scalar(
            layer,
            xs_f32,
            past_k,
            past_v,
            past_seq_len,
            capacity,
            seq,
        );
    }
    layer0_forward_decode_int4_multi_batched(
        layer,
        xs_f32,
        past_k,
        past_v,
        past_seq_len,
        capacity,
        seq,
    )
}

/// Reference per-token scalar loop. Kept as the seq=1 fast path of
/// [`layer0_forward_decode_int4_multi_with_capacity`] AND as the
/// reference implementation for the bit-identity tests against
/// [`layer0_forward_decode_int4_multi_batched`].
///
/// KV cache here is **bf16-as-u16** just like the seq=1 path.
/// The per-token KV append uses [`write_present_kv_bf16`]
/// for the inline f32 → bf16 round at each slot.
pub fn layer0_forward_decode_int4_multi_scalar(
    layer: &Int4Layer0,
    xs_f32: &[f32],
    past_k: &mut [u16],
    past_v: &mut [u16],
    past_seq_len: usize,
    capacity: usize,
    seq: usize,
) -> MultiLayer0Outputs {
    let mut hidden_out = vec![0.0f32; seq * HIDDEN];

    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let cur_past = past_seq_len + t;
        let outs = layer0_forward_decode_int4_with_capacity(
            layer, x_t, past_k, past_v, cur_past, capacity,
        );
        // Write present_k / present_v into slot `cur_past` for each head
        // (f32 -> bf16 inline).
        write_present_kv_bf16(past_k, &outs.present_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_bf16(past_v, &outs.present_v, cur_past, capacity, V_HEAD_DIM);
        hidden_out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&outs.hidden_out);
    }

    MultiLayer0Outputs { hidden_out }
}

/// Batched seq>=2 layer-0 forward. Structures the per-token work as
/// three phases so the big dense projections can amortize their weight
/// load across `seq` tokens via [`crate::shell_int4::dispatch_int4_multi`]:
///
/// **Phase A (batched projections, no KV).** Per-token `h_norm` →
/// batched `q_a`, `kv_a` → per-token rmsnorm on the LoRA outputs →
/// batched `q_b`, `kv_b`.
///
/// **Phase B (per-token, KV-dependent).** Per-token RoPE on q + k_rope,
/// assemble `q_full` / `new_k` / `new_v`, SDPA against the running KV
/// cache, append the new K/V into the cache so the next token sees them.
///
/// **Phase C (batched projections, no KV).** Batched `o_proj` on the
/// stack of per-token `attn_outs` → per-token residual + post-norm →
/// batched `gate_proj`, `up_proj` → per-token SwiGLU → batched
/// `down_proj` → per-token residual.
///
/// **Why batching helps.** Every batched projection is a
/// `[seq, K] x [K, N]` int4 GEMM that loads each packed weight byte
/// once and FMAs it against `seq` token rows. At seq=4-16 the iter 042
/// tile gives 1.4-4.75x per-projection speedup (iter 042 microbench),
/// and iter 046's row-blocking adds another +40% on the two largest
/// shapes (`o_proj`: N=7168, K=8192 — identical to the shell's o_proj;
/// `down_proj`: N=7168, K=18432 — the biggest single int4 matrix in
/// layer 0). All other layer-0 projections fall under `ProjShape::Generic`.
///
/// **Why this matters for spec-decode.** Layer 0 is one call per token,
/// so seq=K spec-decode-verify (iter 044, K=4) serializes it K times.
/// Routing each GEMV through the iter 042 tile recovers the bulk of
/// that serialized cost — the SIMD wins iter 048 wired for the 60
/// shells now also apply to the single layer-0 call.
///
/// **KV cache is bf16-as-u16.** SDPA
/// reads past_k/past_v with inline `f32::from_bits((bits as u32) << 16)`
/// upconvert per element, and the per-token KV append uses
/// [`write_present_kv_bf16`] for the inline f32 → bf16 round.
#[allow(clippy::needless_range_loop)]
fn layer0_forward_decode_int4_multi_batched(
    layer: &Int4Layer0,
    xs_f32: &[f32],
    past_k: &mut [u16],
    past_v: &mut [u16],
    past_seq_len: usize,
    capacity: usize,
    seq: usize,
) -> MultiLayer0Outputs {
    // --- Outputs ---
    let mut hidden_out = vec![0.0f32; seq * HIDDEN];

    // ============ PHASE A: pre-attention projections ============
    // Per-token h_norm (cheap RMSNorm).
    let mut h_norms = vec![0.0f32; seq * HIDDEN];
    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let norm = rmsnorm_apply_pub(x_t, &layer.input_norm, HIDDEN);
        h_norms[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&norm);
    }

    // Batched q_a = q_a_proj @ h_norm[t]
    // Shape N=1536, K=7168 — Generic.
    let mut q_a = vec![0.0f32; seq * Q_LORA_RANK];
    dispatch_int4_multi(
        ProjShape::Generic,
        &layer.q_a_proj_packed,
        &layer.q_a_proj_scale,
        &h_norms,
        Q_LORA_RANK,
        HIDDEN,
        seq,
        &mut q_a,
    );

    // Batched kv_a (output includes the rope shared col, dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM = 576).
    // Shape N=576, K=7168 — Generic.
    let kv_a_out_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
    let mut kv_a_with_rope = vec![0.0f32; seq * kv_a_out_dim];
    dispatch_int4_multi(
        ProjShape::Generic,
        &layer.kv_a_proj_packed,
        &layer.kv_a_proj_scale,
        &h_norms,
        kv_a_out_dim,
        HIDDEN,
        seq,
        &mut kv_a_with_rope,
    );

    // Per-token rmsnorm on q_a and kv_a, plus split out the k_rope_in
    // column the rope step needs.
    let mut q_a_n = vec![0.0f32; seq * Q_LORA_RANK];
    let mut kv_a_n = vec![0.0f32; seq * KV_LORA_RANK];
    let mut k_rope_ins = vec![0.0f32; seq * QK_ROPE_HEAD_DIM];
    for t in 0..seq {
        let q_a_t = &q_a[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK];
        let q_a_n_t = rmsnorm_apply_pub(q_a_t, &layer.q_a_norm, Q_LORA_RANK);
        q_a_n[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK].copy_from_slice(&q_a_n_t);

        let kv_a_t = &kv_a_with_rope[t * kv_a_out_dim..t * kv_a_out_dim + KV_LORA_RANK];
        let k_rope_t = &kv_a_with_rope[t * kv_a_out_dim + KV_LORA_RANK..(t + 1) * kv_a_out_dim];
        let kv_a_n_t = rmsnorm_apply_pub(kv_a_t, &layer.kv_a_norm, KV_LORA_RANK);
        kv_a_n[t * KV_LORA_RANK..(t + 1) * KV_LORA_RANK].copy_from_slice(&kv_a_n_t);
        k_rope_ins[t * QK_ROPE_HEAD_DIM..(t + 1) * QK_ROPE_HEAD_DIM].copy_from_slice(k_rope_t);
    }

    // Batched q = q_b_proj @ q_a_n[t]
    // Shape N=12288, K=1536 — Generic.
    let qkv_q_dim = NUM_HEADS * QK_HEAD_DIM;
    let mut qs = vec![0.0f32; seq * qkv_q_dim];
    dispatch_int4_multi(
        ProjShape::Generic,
        &layer.q_b_proj_packed,
        &layer.q_b_proj_scale,
        &q_a_n,
        qkv_q_dim,
        Q_LORA_RANK,
        seq,
        &mut qs,
    );

    // Batched kv_b = kv_b_proj @ kv_a_n[t]
    // Shape N=16384, K=512 — Generic.
    let kv_b_dim = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM);
    let mut kv_bs = vec![0.0f32; seq * kv_b_dim];
    dispatch_int4_multi(
        ProjShape::Generic,
        &layer.kv_b_proj_packed,
        &layer.kv_b_proj_scale,
        &kv_a_n,
        kv_b_dim,
        KV_LORA_RANK,
        seq,
        &mut kv_bs,
    );

    // ============ PHASE B: per-token RoPE + SDPA + KV append ============
    // SDPA reads past_k/past_v as bf16-as-u16 with inline upconvert.
    // new_k/new_v stay f32 for the
    // current-step dot product, then are written into the cache as
    // bf16 by `write_present_kv_bf16` so the next token sees them.
    let mut attn_outs = vec![0.0f32; seq * (NUM_HEADS * V_HEAD_DIM)];
    for t in 0..seq {
        let cur_past = past_seq_len + t;
        let kv_len = cur_past + 1;
        let q = &qs[t * qkv_q_dim..(t + 1) * qkv_q_dim];
        let kv_b = &kv_bs[t * kv_b_dim..(t + 1) * kv_b_dim];
        let k_rope_in = &k_rope_ins[t * QK_ROPE_HEAD_DIM..(t + 1) * QK_ROPE_HEAD_DIM];

        let (cos, sin) = rope_cos_sin_pub(cur_past);
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

        // SDPA — past_k/past_v are bf16-as-u16, upconverted to f32
        // inline at each dot-product element (matches the seq=1 path).
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

        // Append new_k / new_v to past at slot cur_past so the next
        // token's SDPA sees them. Inline f32 → bf16 round.
        write_present_kv_bf16(past_k, &new_k, cur_past, capacity, QK_HEAD_DIM);
        write_present_kv_bf16(past_v, &new_v, cur_past, capacity, V_HEAD_DIM);
    }

    // ============ PHASE C: post-attention projections + dense MLP ============
    // Batched o_proj on stacked attn_outs.
    //
    // Shape N=7168, K=8192 — identical to the shell's o_proj, so
    // `ProjShape::Oproj` routes through the iter 046 row-blocked tile
    // at seq>=4 (+41% over iter 042 at seq=4-16, verified by the iter
    // 046 Xeon microbench). At seq=2-3 it auto-falls-back to iter 042.
    let mut o_outs = vec![0.0f32; seq * HIDDEN];
    dispatch_int4_multi(
        ProjShape::Oproj,
        &layer.o_proj_packed,
        &layer.o_proj_scale,
        &attn_outs,
        HIDDEN,
        NUM_HEADS * V_HEAD_DIM,
        seq,
        &mut o_outs,
    );

    // Per-token residual + post-norm.
    let mut residuals = vec![0.0f32; seq * HIDDEN];
    let mut posts = vec![0.0f32; seq * HIDDEN];
    for t in 0..seq {
        let x_t = &xs_f32[t * HIDDEN..(t + 1) * HIDDEN];
        let o_t = &o_outs[t * HIDDEN..(t + 1) * HIDDEN];
        let res_t = &mut residuals[t * HIDDEN..(t + 1) * HIDDEN];
        for i in 0..HIDDEN {
            res_t[i] = x_t[i] + o_t[i];
        }
        let p = rmsnorm_apply_pub(res_t, &layer.post_norm, HIDDEN);
        posts[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&p);
    }

    // ----- Dense SwiGLU MLP (layer 0's only structural difference from shells) -----
    // Batched gate_proj + up_proj.
    // Shape N=18432, K=7168 — Generic (taller than oproj's N=7168 but
    // K is smaller than oproj's K=8192; iter 042's per-row sweep is the
    // right tile here. Iter 046's row-blocking would help on a wider
    // (N,K) shape but adds register pressure with no clear win at this
    // N — leaving as Generic and letting the iter 042 microbench
    // sweep flag it as a future opt if profiles say otherwise).
    let mut gate_out = vec![0.0f32; seq * INTERMEDIATE_DENSE];
    dispatch_int4_multi(
        ProjShape::Generic,
        &layer.gate_proj_packed,
        &layer.gate_proj_scale,
        &posts,
        INTERMEDIATE_DENSE,
        HIDDEN,
        seq,
        &mut gate_out,
    );
    let mut up_out = vec![0.0f32; seq * INTERMEDIATE_DENSE];
    dispatch_int4_multi(
        ProjShape::Generic,
        &layer.up_proj_packed,
        &layer.up_proj_scale,
        &posts,
        INTERMEDIATE_DENSE,
        HIDDEN,
        seq,
        &mut up_out,
    );

    // Per-token SwiGLU.
    let mut inters = vec![0.0f32; seq * INTERMEDIATE_DENSE];
    for t in 0..seq {
        let g_t = &gate_out[t * INTERMEDIATE_DENSE..(t + 1) * INTERMEDIATE_DENSE];
        let u_t = &up_out[t * INTERMEDIATE_DENSE..(t + 1) * INTERMEDIATE_DENSE];
        let i_t = &mut inters[t * INTERMEDIATE_DENSE..(t + 1) * INTERMEDIATE_DENSE];
        swiglu_mul(g_t, u_t, i_t);
    }

    // Batched down_proj.
    // Shape N=7168, K=18432 — N matches oproj, K is 2.25x larger. This
    // is the largest single int4 matrix in layer 0 (66 MB). Tall-and-thin
    // GEMM where the row-blocked iter 046 tile is the natural fit
    // (same N=7168 row-blocking sweet spot as the shell's oproj +
    // shared_down). At seq=2-3 the dispatcher's seq>=4 threshold makes
    // it fall back to iter 042 — still wins over scalar.
    let mut mlp_out = vec![0.0f32; seq * HIDDEN];
    dispatch_int4_multi(
        ProjShape::Oproj,
        &layer.down_proj_packed,
        &layer.down_proj_scale,
        &inters,
        HIDDEN,
        INTERMEDIATE_DENSE,
        seq,
        &mut mlp_out,
    );

    // Final per-token residual: hidden_out = residual + mlp_out.
    for t in 0..seq {
        let res_t = &residuals[t * HIDDEN..(t + 1) * HIDDEN];
        let m_t = &mlp_out[t * HIDDEN..(t + 1) * HIDDEN];
        let h_t = &mut hidden_out[t * HIDDEN..(t + 1) * HIDDEN];
        for i in 0..HIDDEN {
            h_t[i] = res_t[i] + m_t[i];
        }
    }

    MultiLayer0Outputs { hidden_out }
}

/// f32 -> bf16-as-u16 KV-slot writer; same routine as `shell_int4`'s
/// `write_present_kv_bf16` but kept here so layer0 stays self-contained.
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

    /// Seed two bf16-as-u16 KV pairs with the same deterministic non-zero
    /// history (one for the scalar reference, one for the batched path).
    /// Factored out so the seq=4 and seq=8 iter 048-dispatch tests can
    /// share it. Pre-existing slots are encoded through
    /// `f32_to_bf16_bits_local` so the bit-pattern compare is exact.
    #[allow(clippy::type_complexity)]
    fn seed_layer0_kv_pair(
        capacity: usize,
        past_seq_len: usize,
    ) -> ((Vec<u16>, Vec<u16>), (Vec<u16>, Vec<u16>)) {
        let mut a_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut a_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        let mut b_k = vec![0u16; NUM_HEADS * capacity * QK_HEAD_DIM];
        let mut b_v = vec![0u16; NUM_HEADS * capacity * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past_seq_len {
                let off_k = h * capacity * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let off_v = h * capacity * V_HEAD_DIM + s * V_HEAD_DIM;
                for i in 0..QK_HEAD_DIM {
                    let v = (((h * 7 + s * 13 + i) as f32).sin()) * 1.0e-3;
                    let b = f32_to_bf16_bits_local(v);
                    a_k[off_k + i] = b;
                    b_k[off_k + i] = b;
                }
                for i in 0..V_HEAD_DIM {
                    let v = (((h * 11 + s * 17 + i) as f32).cos()) * 1.0e-3;
                    let b = f32_to_bf16_bits_local(v);
                    a_v[off_v + i] = b;
                    b_v[off_v + i] = b;
                }
            }
        }
        ((a_k, a_v), (b_k, b_v))
    }

    /// iter 052 bit-identity: at seq=4 the layer-0 multi-token path
    /// routes oproj + down_proj through the iter 046 row-blocked tile
    /// (`ProjShape::Oproj` → seq>=4 threshold) and q_a / kv_a / q_b /
    /// kv_b / gate / up through the iter 042 multi tile
    /// (`ProjShape::Generic`). The batched path must produce
    /// byte-identical KV state and per-token hidden outputs as 4
    /// sequential seq=1 forwards driving the same evolving KV cache.
    ///
    /// This is the layer-0 analogue of shell_int4's
    /// `multi_batched_matches_scalar_seq_4_iter046_dispatch` — if this
    /// test fails, the iter 052 dispatch wiring has regressed.
    #[test]
    fn multi_layer0_batched_matches_scalar_seq_4_iter048_dispatch() {
        let layer = make_test_layer0();
        let capacity = 16;
        let past_seq_len = 4;
        let seq = 4;
        let ((mut scalar_past_k, mut scalar_past_v), (mut batched_past_k, mut batched_past_v)) =
            seed_layer0_kv_pair(capacity, past_seq_len);

        let mut xs = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = make_test_input(t);
            xs[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&x_t);
        }

        let scalar_out = layer0_forward_decode_int4_multi_scalar(
            &layer,
            &xs,
            &mut scalar_past_k,
            &mut scalar_past_v,
            past_seq_len,
            capacity,
            seq,
        );
        let batched_out = layer0_forward_decode_int4_multi_batched(
            &layer,
            &xs,
            &mut batched_past_k,
            &mut batched_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        // Per-token hidden outputs: bit-identical (iter 042 and iter 046
        // tiles preserve the per-cell FMA order — see
        // `multi_matches_per_token_loop_seq_4` in
        // `kernel_avx512_multi.rs` and `blocked_matches_iter042_multi_seq_8`
        // in `kernel_avx512_multi_blocked.rs`).
        assert_eq!(
            batched_out.hidden_out, scalar_out.hidden_out,
            "hidden_out mismatch"
        );
        // KV state (bf16-as-u16): bit-identical (every K/V write goes
        // through the same per-token RoPE+assemble+bf16-round code path
        // in both branches).
        assert_eq!(batched_past_k, scalar_past_k, "past_k mismatch");
        assert_eq!(batched_past_v, scalar_past_v, "past_v mismatch");
    }

    /// Same as `multi_layer0_batched_matches_scalar_seq_4_iter048_dispatch`
    /// but at seq=8 — exercises iter 046's sweet spot (consistent +40%
    /// over iter 042 per the iter 046 Xeon microbench at seq=8).
    /// Layer 0's down_proj (66 MB, biggest single int4 matrix in the
    /// model) is the projection most sensitive to row-blocking; if its
    /// dispatch breaks, this test catches it.
    #[test]
    fn multi_layer0_batched_matches_scalar_seq_8_iter048_dispatch() {
        let layer = make_test_layer0();
        let capacity = 16;
        let past_seq_len = 4;
        let seq = 8;
        let ((mut scalar_past_k, mut scalar_past_v), (mut batched_past_k, mut batched_past_v)) =
            seed_layer0_kv_pair(capacity, past_seq_len);

        let mut xs = vec![0.0f32; seq * HIDDEN];
        for t in 0..seq {
            let x_t = make_test_input(t);
            xs[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&x_t);
        }

        let scalar_out = layer0_forward_decode_int4_multi_scalar(
            &layer,
            &xs,
            &mut scalar_past_k,
            &mut scalar_past_v,
            past_seq_len,
            capacity,
            seq,
        );
        let batched_out = layer0_forward_decode_int4_multi_batched(
            &layer,
            &xs,
            &mut batched_past_k,
            &mut batched_past_v,
            past_seq_len,
            capacity,
            seq,
        );

        assert_eq!(
            batched_out.hidden_out, scalar_out.hidden_out,
            "hidden_out mismatch"
        );
        assert_eq!(batched_past_k, scalar_past_k, "past_k mismatch");
        assert_eq!(batched_past_v, scalar_past_v, "past_v mismatch");
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
