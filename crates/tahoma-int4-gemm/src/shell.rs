//! Rust implementation of one K2.6 transformer "shell" — everything in
//! a transformer layer except the routed experts (which are dispatched
//! separately by the engine).
//!
//! Mirrors the contract that `kimi-k26-shells-kv/layer_NN/openvino_model.xml`
//! exposes:
//!   inputs:  x.1, past_k, past_v, attn_mask_ext, past_seq_len
//!   outputs: attn_out_post_norm, attn_residual, shared_expert_out,
//!            routing_ids, routing_weights, present_k, present_v
//!
//! All matmuls use bf16 weights (read straight from the safetensors
//! shards via `SafetensorsShell`) and the AVX-512 GEMV from
//! `kernel_bf16`. The attention is MLA (Multi-Latent Attention) as in
//! DeepseekV3 — q is downprojected through q_a/q_b, kv is downprojected
//! through kv_a/kv_b with a separate rope-only column for k, and
//! position-rotated head halves get spliced before SDPA.
//!
//! Constants below are wired for K2.6 specifically. Hard-coded for
//! perf (let the compiler unroll).

use crate::kernel_bf16::bf16_gemv_auto;
use crate::safetensors_source::SafetensorsShell;

pub const HIDDEN: usize = 7168;
pub const Q_LORA_RANK: usize = 1536;
pub const KV_LORA_RANK: usize = 512;
pub const NUM_HEADS: usize = 64;
pub const QK_NOPE_HEAD_DIM: usize = 128;
pub const QK_ROPE_HEAD_DIM: usize = 64;
pub const QK_HEAD_DIM: usize = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM; // 192
pub const V_HEAD_DIM: usize = 128;
pub const INTERMEDIATE_SHARED: usize = 2048;
pub const N_ROUTED_EXPERTS: usize = 384;
pub const N_GROUPS: usize = 1;
pub const TOPK_GROUP: usize = 1;
pub const TOPK: usize = 8;
pub const ROUTED_SCALING_FACTOR: f32 = 2.827;
pub const RMS_NORM_EPS: f32 = 1.0e-6;

// K2.6 RoPE configuration. The model uses YARN scaling — without it
// our cos/sin tables don't match what OV's traced graph computed,
// which manifests as present_k cosine 0.07 against the OV reference
// despite Q, V, residual, and routing all matching at cosine 1.0.
pub const ROPE_BASE: f64 = 50000.0;
pub const YARN_FACTOR: f64 = 64.0;
pub const YARN_BETA_FAST: f64 = 32.0;
pub const YARN_BETA_SLOW: f64 = 1.0;
pub const YARN_ORIGINAL_MAX_POSITION: f64 = 4096.0;
pub const YARN_MSCALE: f64 = 1.0;
pub const YARN_MSCALE_ALL_DIM: f64 = 1.0;

/// Output struct: exactly the seven tensors the engine consumes.
pub struct ShellOutputs {
    /// post_attention_layernorm output — fed to experts + shared.
    pub attn_out_post_norm: Vec<f32>,
    /// hidden_states + attn_output (i.e. the residual fed back at the
    /// end alongside MoE / shared_expert outputs).
    pub attn_residual: Vec<f32>,
    /// shared_experts(attn_out_post_norm).
    pub shared_expert_out: Vec<f32>,
    /// top-8 expert ids per token.
    pub routing_ids: Vec<i64>,
    /// top-8 expert weights per token (post norm + scaling).
    pub routing_weights: Vec<f32>,
    /// NEW K values for this step only — shape [NUM_HEADS, seq, QK_HEAD_DIM]. The
    /// engine concatenates them onto the running cache.
    pub present_k: Vec<f32>,
    /// NEW V values for this step only — shape [NUM_HEADS, seq, V_HEAD_DIM].
    pub present_v: Vec<f32>,
}

/// Run one shell forward for a single token (seq=1).
///
/// `x_f32` is [HIDDEN] — the layer input.
/// `past_k`/`past_v` are the running KV cache (shape
/// [NUM_HEADS, past_seq_len, QK_HEAD_DIM] / [NUM_HEADS, past_seq_len,
/// V_HEAD_DIM]) flattened to f32. `past_seq_len` is the number of past
/// positions already in the cache; the new token's position is
/// `past_seq_len`.
pub fn shell_forward_decode(
    shell: &SafetensorsShell,
    x_f32: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    past_seq_len: usize,
) -> ShellOutputs {
    assert_eq!(x_f32.len(), HIDDEN);
    assert_eq!(past_k.len(), NUM_HEADS * past_seq_len * QK_HEAD_DIM);
    assert_eq!(past_v.len(), NUM_HEADS * past_seq_len * V_HEAD_DIM);

    let kv_len = past_seq_len + 1;

    // ---- input layernorm ----
    let h_norm = rmsnorm_apply(x_f32, shell.input_norm, HIDDEN);

    // ---- Q projection ----
    let mut q_a = vec![0.0f32; Q_LORA_RANK];
    bf16_gemv_auto(shell.q_a_proj, &h_norm, Q_LORA_RANK, HIDDEN, &mut q_a);
    let q_a_n = rmsnorm_apply(&q_a, shell.q_a_norm, Q_LORA_RANK);
    let mut q = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    bf16_gemv_auto(shell.q_b_proj, &q_a_n, NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK, &mut q);
    // q is laid out [NUM_HEADS, QK_HEAD_DIM] = [NUM_HEADS, QK_NOPE + QK_ROPE].

    // ---- KV projection ----
    let mut kv_a_with_rope = vec![0.0f32; KV_LORA_RANK + QK_ROPE_HEAD_DIM];
    bf16_gemv_auto(
        shell.kv_a_proj,
        &h_norm,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        HIDDEN,
        &mut kv_a_with_rope,
    );
    let (kv_a, k_rope_in) = kv_a_with_rope.split_at(KV_LORA_RANK);
    let kv_a_n = rmsnorm_apply(kv_a, shell.kv_a_norm, KV_LORA_RANK);
    let mut kv_b = vec![0.0f32; NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
    bf16_gemv_auto(
        shell.kv_b_proj,
        &kv_a_n,
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        KV_LORA_RANK,
        &mut kv_b,
    );
    // kv_b is [NUM_HEADS, QK_NOPE + V_HEAD] per head.

    // ---- RoPE: apply rotation at position `past_seq_len` to q_rope and k_rope ----
    let (cos, sin) = rope_cos_sin(past_seq_len);

    // Build the per-head Q (nope + rope) and the new K (nope + rope_shared) and V.
    let mut new_k = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    let mut new_v = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];
    // k_rope is broadcast across heads (MLA convention).
    let mut k_rope_rot = vec![0.0f32; QK_ROPE_HEAD_DIM];
    apply_rope_kimi(k_rope_in, &cos, &sin, &mut k_rope_rot);

    // q_rope per head: it's already split inside q[..h_dim_qk] as
    // [nope(128), rope(64)] per head.
    let mut q_full = vec![0.0f32; NUM_HEADS * QK_HEAD_DIM];
    let mut q_rope_buf = vec![0.0f32; QK_ROPE_HEAD_DIM];
    for h in 0..NUM_HEADS {
        // Copy q nope part
        let dst = &mut q_full[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM];
        let src = &q[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM];
        dst.copy_from_slice(src);
        // Rotate q_rope part per head
        let q_rope_src = &q[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        apply_rope_kimi(q_rope_src, &cos, &sin, &mut q_rope_buf);
        q_full[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM]
            .copy_from_slice(&q_rope_buf);

        // K nope per head
        let k_nope_src = &kv_b[h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
            ..h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) + QK_NOPE_HEAD_DIM];
        let k_dst = &mut new_k[h * QK_HEAD_DIM..h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM];
        k_dst.copy_from_slice(k_nope_src);
        // K rope (shared across heads)
        new_k[h * QK_HEAD_DIM + QK_NOPE_HEAD_DIM..(h + 1) * QK_HEAD_DIM]
            .copy_from_slice(&k_rope_rot);
        // V per head
        let v_src = &kv_b[h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) + QK_NOPE_HEAD_DIM
            ..(h + 1) * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)];
        new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM].copy_from_slice(v_src);
    }

    // ---- SDPA: per head: scores = Q[h, qk_head_dim] @ K[h, :, qk_head_dim].T
    //                       softmax(scores / sqrt(qk_head_dim))
    //                       out[h, :] = scores @ V[h, :, v_head_dim]
    // K cache is [past + new] per head.
    let scale = 1.0f32 / (QK_HEAD_DIM as f32).sqrt();
    let mut attn_out = vec![0.0f32; NUM_HEADS * V_HEAD_DIM];

    // Pre-build per-head full K/V views by walking past_k/past_v + new_k/new_v.
    // Layout of past_k: [num_heads, past_seq_len, qk_head_dim] (row-major).
    for h in 0..NUM_HEADS {
        let q_h = &q_full[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        let past_k_h = &past_k[h * past_seq_len * QK_HEAD_DIM
            ..(h + 1) * past_seq_len * QK_HEAD_DIM];
        let past_v_h = &past_v[h * past_seq_len * V_HEAD_DIM
            ..(h + 1) * past_seq_len * V_HEAD_DIM];
        let new_k_h = &new_k[h * QK_HEAD_DIM..(h + 1) * QK_HEAD_DIM];
        let new_v_h = &new_v[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM];

        // scores: kv_len scalars per head.
        let mut scores = vec![0.0f32; kv_len];
        for j in 0..past_seq_len {
            let k_row = &past_k_h[j * QK_HEAD_DIM..(j + 1) * QK_HEAD_DIM];
            let mut s = 0.0f32;
            for i in 0..QK_HEAD_DIM {
                s += q_h[i] * k_row[i];
            }
            scores[j] = s * scale;
        }
        // last row = new K
        {
            let mut s = 0.0f32;
            for i in 0..QK_HEAD_DIM {
                s += q_h[i] * new_k_h[i];
            }
            scores[past_seq_len] = s * scale;
        }
        // Causal mask: the new token at past_seq_len attends to all
        // 0..=past_seq_len; everything before fits within mask=0. No mask
        // operation needed for seq=1 decode.
        // Softmax in fp32:
        let mut max_s = scores[0];
        for &s in scores.iter().skip(1) {
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_e = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum_e += *s;
        }
        let inv = 1.0f32 / sum_e;
        for s in scores.iter_mut() {
            *s *= inv;
        }

        // out[h, :] = sum_j scores[j] * V_full[h, j, :]
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

    // ---- o_proj ----
    let mut o_out = vec![0.0f32; HIDDEN];
    bf16_gemv_auto(shell.o_proj, &attn_out, HIDDEN, NUM_HEADS * V_HEAD_DIM, &mut o_out);

    // ---- residual = x + o_out ----
    let mut residual = vec![0.0f32; HIDDEN];
    for i in 0..HIDDEN {
        residual[i] = x_f32[i] + o_out[i];
    }

    // ---- post_attention_layernorm ----
    let post = rmsnorm_apply(&residual, shell.post_norm, HIDDEN);

    // ---- Router ----
    let mut router_logits = vec![0.0f32; N_ROUTED_EXPERTS];
    bf16_gemv_auto(shell.router_weight, &post, N_ROUTED_EXPERTS, HIDDEN, &mut router_logits);
    // sigmoid scores
    let mut scores = vec![0.0f32; N_ROUTED_EXPERTS];
    for i in 0..N_ROUTED_EXPERTS {
        scores[i] = 1.0f32 / (1.0f32 + (-router_logits[i]).exp());
    }
    // noaux_tc: scores_for_choice = scores + bias
    // bias is f32 [N_ROUTED_EXPERTS]
    let bias = unsafe {
        std::slice::from_raw_parts(
            shell.router_bias.as_ptr() as *const f32,
            N_ROUTED_EXPERTS,
        )
    };
    let mut scores_for_choice = vec![0.0f32; N_ROUTED_EXPERTS];
    for i in 0..N_ROUTED_EXPERTS {
        scores_for_choice[i] = scores[i] + bias[i];
    }
    // For K2.6 N_GROUPS=1, so no group masking needed — top-k directly.
    // top-8 by scores_for_choice. Get indices.
    let mut idx_score: Vec<(usize, f32)> = scores_for_choice
        .iter()
        .copied()
        .enumerate()
        .collect();
    idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut topk_ids = vec![0i64; TOPK];
    let mut topk_w = vec![0.0f32; TOPK];
    for k in 0..TOPK {
        topk_ids[k] = idx_score[k].0 as i64;
        topk_w[k] = scores[idx_score[k].0]; // use ORIGINAL sigmoid score (not + bias)
    }
    // Normalize topk weights to sum 1, then scale.
    let s: f32 = topk_w.iter().sum::<f32>() + 1.0e-20;
    for w in topk_w.iter_mut() {
        *w = *w / s * ROUTED_SCALING_FACTOR;
    }

    // ---- Shared expert ----
    let mut shared_gate_out = vec![0.0f32; INTERMEDIATE_SHARED];
    bf16_gemv_auto(shell.shared_gate, &post, INTERMEDIATE_SHARED, HIDDEN, &mut shared_gate_out);
    let mut shared_up_out = vec![0.0f32; INTERMEDIATE_SHARED];
    bf16_gemv_auto(shell.shared_up, &post, INTERMEDIATE_SHARED, HIDDEN, &mut shared_up_out);
    let mut shared_inter = vec![0.0f32; INTERMEDIATE_SHARED];
    for i in 0..INTERMEDIATE_SHARED {
        let g = shared_gate_out[i];
        let silu = g / (1.0f32 + (-g).exp());
        shared_inter[i] = silu * shared_up_out[i];
    }
    let mut shared_out = vec![0.0f32; HIDDEN];
    bf16_gemv_auto(shell.shared_down, &shared_inter, HIDDEN, INTERMEDIATE_SHARED, &mut shared_out);

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

/// Apply RMSNorm with bf16 weights to an f32 vector, return f32.
fn rmsnorm_apply(x: &[f32], weight_bf16: &[u8], dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), dim);
    assert_eq!(weight_bf16.len(), dim * 2);
    let mut var: f64 = 0.0;
    for v in x.iter() {
        var += (*v as f64) * (*v as f64);
    }
    let mean_sq = (var / dim as f64) as f32;
    let inv = (mean_sq + RMS_NORM_EPS).sqrt().recip();
    let mut out = vec![0.0f32; dim];
    for i in 0..dim {
        let lo = weight_bf16[i * 2];
        let hi = weight_bf16[i * 2 + 1];
        let bits = ((hi as u32) << 8) | (lo as u32);
        let w = f32::from_bits(bits << 16);
        out[i] = x[i] * inv * w;
    }
    out
}

/// YARN-blended inv_freq for K2.6's RoPE. Returns an array of length
/// QK_ROPE_HEAD_DIM/2 holding the inverse frequency for each rotation
/// pair, blended between the standard (high-freq) extrapolation table
/// and the YARN-interpolated (low-freq) table.
fn yarn_inv_freq() -> [f64; QK_ROPE_HEAD_DIM / 2] {
    let dim = QK_ROPE_HEAD_DIM as f64;
    let mut freq_extra = [0.0f64; QK_ROPE_HEAD_DIM / 2];
    let mut freq_inter = [0.0f64; QK_ROPE_HEAD_DIM / 2];
    for i in 0..(QK_ROPE_HEAD_DIM / 2) {
        let exp = (2.0 * i as f64) / dim;
        let f = 1.0 / ROPE_BASE.powf(exp);
        freq_extra[i] = f;
        freq_inter[i] = f / YARN_FACTOR;
    }

    // yarn_find_correction_dim(num_rotations, dim, base, max_pos):
    //   (dim * log(max_pos / (num_rotations * 2 * pi))) / (2 * log(base))
    let correction_dim = |num_rot: f64| -> f64 {
        (dim * (YARN_ORIGINAL_MAX_POSITION / (num_rot * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * ROPE_BASE.ln())
    };
    let low_raw = correction_dim(YARN_BETA_FAST).floor();
    let high_raw = correction_dim(YARN_BETA_SLOW).ceil();
    let low = low_raw.max(0.0);
    let high = high_raw.min(dim - 1.0);
    let mut inv_freq = [0.0f64; QK_ROPE_HEAD_DIM / 2];
    let denom = if (high - low).abs() < 1.0e-6 {
        0.001
    } else {
        high - low
    };
    for i in 0..(QK_ROPE_HEAD_DIM / 2) {
        let lin = ((i as f64) - low) / denom;
        let ramp = lin.clamp(0.0, 1.0);
        let inv_mask = 1.0 - ramp;
        inv_freq[i] = freq_inter[i] * (1.0 - inv_mask) + freq_extra[i] * inv_mask;
    }
    inv_freq
}

fn yarn_mscale() -> f64 {
    // mscale = mscale_all_dim → ratio is 1.0 in K2.6's config; we keep
    // the formula here for completeness.
    fn get_mscale(scale: f64, m: f64) -> f64 {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * m * scale.ln() + 1.0
        }
    }
    get_mscale(YARN_FACTOR, YARN_MSCALE) / get_mscale(YARN_FACTOR, YARN_MSCALE_ALL_DIM)
}

/// RoPE for K2.6 — YARN-blended frequencies, paired (i, i+half) rotation.
fn rope_cos_sin(pos: usize) -> (Vec<f32>, Vec<f32>) {
    let dim = QK_ROPE_HEAD_DIM;
    let half = dim / 2;
    let inv_freq = yarn_inv_freq();
    let mscale = yarn_mscale() as f32;
    let mut cos = vec![0.0f32; dim];
    let mut sin = vec![0.0f32; dim];
    let p = pos as f64;
    for i in 0..half {
        let theta = p * inv_freq[i];
        let c = theta.cos() as f32 * mscale;
        let s = theta.sin() as f32 * mscale;
        cos[i] = c;
        cos[i + half] = c;
        sin[i] = s;
        sin[i + half] = s;
    }
    (cos, sin)
}

/// K2.6's apply_rotary_pos_emb pre-interleaves dims via
/// `x.view(..., d//2, 2).transpose(-1, -2).reshape(..., d)` and THEN
/// runs standard HF rotate_half rotation. Net:
///
///   even_i = x[2i]    (i in 0..half)
///   odd_i  = x[2i+1]
///   out[i]      = even_i * cos[i] - odd_i  * sin[i]
///   out[i+half] = odd_i  * cos[i+half] + even_i * sin[i+half]
///
/// This uses K2.6's own convention. OV's traced shell apparently bakes
/// in an extra layout transform so `present_k` doesn't byte-match this,
/// but within a pure-Rust pipeline this convention is self-consistent
/// (Q and K both rotate the same way, so q·k is invariant). The final
/// generated text matches.
fn apply_rope_kimi(x: &[f32], cos: &[f32], sin: &[f32], out: &mut [f32]) {
    let half = QK_ROPE_HEAD_DIM / 2;
    for i in 0..half {
        let even_i = x[2 * i];
        let odd_i = x[2 * i + 1];
        out[i] = even_i * cos[i] - odd_i * sin[i];
        out[i + half] = odd_i * cos[i + half] + even_i * sin[i + half];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_basic() {
        // weight = 1.0 (bf16 0x3f80), x = [1, 1, 1, 1]: rms=1, out=1*1=1 each
        let mut w = vec![0u8; 4 * 2];
        for i in 0..4 {
            w[i * 2] = 0x80;
            w[i * 2 + 1] = 0x3f;
        }
        let x = vec![1.0f32; 4];
        let out = rmsnorm_apply(&x, &w, 4);
        for v in out.iter() {
            assert!((v - 1.0).abs() < 1e-3, "got {}", v);
        }
    }

    #[test]
    fn rope_pos_zero_reinterleaves() {
        // At pos=0, cos=1 sin=0 so apply_rope_kimi degenerates to the
        // pre-rotation interleave: y[i]=x[2i] for i<half, y[i+half]=x[2i+1].
        let (cos, sin) = rope_cos_sin(0);
        let x: Vec<f32> = (0..QK_ROPE_HEAD_DIM).map(|i| i as f32).collect();
        let mut y = vec![0.0f32; QK_ROPE_HEAD_DIM];
        apply_rope_kimi(&x, &cos, &sin, &mut y);
        let half = QK_ROPE_HEAD_DIM / 2;
        for i in 0..half {
            assert!((y[i] - (2 * i) as f32).abs() < 1e-4, "y[{i}]={} expected {}", y[i], 2 * i);
            assert!(
                (y[i + half] - (2 * i + 1) as f32).abs() < 1e-4,
                "y[{}]={} expected {}", i + half, y[i + half], 2 * i + 1,
            );
        }
    }
}
