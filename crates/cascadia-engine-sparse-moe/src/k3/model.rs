//! K3 layer stack — the AttnRes state machine plus the inter-stage wire format.
//!
//! # Inter-stage wire
//!
//! Unlike a plain residual stream, a K3 layer consumes and produces a PAIR:
//! the running `prefix_sum` and the stack of per-block residuals. Because
//! [`crate::k3::attn_res`] mixes over every prior block, a pipeline rank
//! boundary cannot drop the stack — it has to travel.
//!
//! The wire is therefore widened, as dsv4 does for Hyper-Connections:
//!
//! ```text
//! [ prefix_sum (H) | block_0 (H) | block_1 (H) | ... | block_{maxb-1} (H) ]
//! ```
//!
//! with `maxb = ceil(num_layers / attn_res_block_size)`. Slots beyond the live
//! count are zero. The live count is not transmitted because it is derivable:
//! at the entry to layer `i` exactly `ceil(i / block_size)` boundaries have
//! been passed, and every rank knows its own first layer index.

use crate::dsv4::math::{linear_bf16_w, rmsnorm, to_bf16};
use crate::k3::attn::{mla_step, MlaDims, MlaKv, MlaWeights};
use crate::k3::attn_res::apply_attn_res;
use crate::k3::kda::{kda_gate, kda_step, l2norm_heads, short_conv};
use crate::k3::moe::{moe_forward, moe_forward_batch, ExpertSource, MoeDims, MoeWeights};
use crate::k3::prof;
use crate::k3::situ::situ;

/// True when `layer` (0-indexed) is a KDA layer.
///
/// The manifest's `kda_layers` is already 0-indexed — the exporter shifts it,
/// because `linear_attn_config` in the HF config lists layers 1-INDEXED.
/// Shifted, `kda_layers` and `full_attn_layers` partition 0..n-1 exactly
/// (69 KDA + 24 MLA = 93), which the tensor index confirms: layer 0 carries
/// KDA weights and layer 3 carries MLA ones.
///
/// Number of block-residual slots live at the entry to layer `i`.
#[inline]
pub fn blocks_at(layer: usize, block_size: usize) -> usize {
    layer.div_ceil(block_size)
}

/// Total slots the wire must carry for a model of `n` layers.
#[inline]
pub fn max_blocks(n: usize, block_size: usize) -> usize {
    n.div_ceil(block_size)
}

/// Per-layer KDA weights.
pub struct KdaWeights {
    pub q_proj: Vec<u16>,
    pub k_proj: Vec<u16>,
    pub v_proj: Vec<u16>,
    pub q_conv1d: Vec<f32>,
    pub k_conv1d: Vec<f32>,
    pub v_conv1d: Vec<f32>,
    pub f_a_proj: Vec<u16>,
    pub f_b_proj: Vec<u16>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub b_proj: Vec<u16>,
    pub g_proj: Vec<u16>,
    pub o_norm: Vec<f32>,
    pub o_proj: Vec<u16>,
}

/// KDA shape contract.
#[derive(Clone, Copy, Debug)]
pub struct KdaDims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_size: usize,
    pub gate_lower_bound: Option<f32>,
    pub eps: f32,
}

/// Carried KDA state: the `[heads, K, V]` recurrence plus three conv windows.
///
/// `Clone` is what makes prefix caching and speculative rollback possible: the
/// recurrence is destructive (`kda_step` overwrites `recurrent` in place), so the
/// only way back to an earlier position is to have kept a copy. It is fixed-size
/// regardless of sequence length, which is what makes keeping one cheap.
#[derive(Clone)]
pub struct KdaState {
    pub recurrent: Vec<f32>,
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
}

impl KdaState {
    pub fn new(d: KdaDims) -> Self {
        let proj = d.heads * d.head_dim;
        let w = proj * (d.conv_size - 1);
        Self {
            recurrent: vec![0.0; d.heads * d.head_dim * d.head_dim],
            conv_q: vec![0.0; w],
            conv_k: vec![0.0; w],
            conv_v: vec![0.0; w],
        }
    }
    pub fn clear(&mut self) {
        self.recurrent.fill(0.0);
        self.conv_q.fill(0.0);
        self.conv_k.fill(0.0);
        self.conv_v.fill(0.0);
    }
}

/// One KDA layer step. `x`, `out`: `[hidden]`.
pub fn kda_layer_step(x: &[f32], w: &KdaWeights, d: KdaDims, st: &mut KdaState, out: &mut [f32]) {
    let proj = d.heads * d.head_dim;
    let (mut q, mut k, mut v) = (vec![0.0; proj], vec![0.0; proj], vec![0.0; proj]);
    linear_bf16_w(x, &w.q_proj, proj, d.hidden, &mut q);
    linear_bf16_w(x, &w.k_proj, proj, d.hidden, &mut k);
    linear_bf16_w(x, &w.v_proj, proj, d.hidden, &mut v);

    let (mut qc, mut kc, mut vc) = (vec![0.0; proj], vec![0.0; proj], vec![0.0; proj]);
    short_conv(&q, &w.q_conv1d, &mut st.conv_q, proj, d.conv_size, &mut qc);
    short_conv(&k, &w.k_conv1d, &mut st.conv_k, proj, d.conv_size, &mut kc);
    short_conv(&v, &w.v_conv1d, &mut st.conv_v, proj, d.conv_size, &mut vc);

    // low-rank decay gate
    let mut fa = vec![0.0; d.head_dim];
    linear_bf16_w(x, &w.f_a_proj, d.head_dim, d.hidden, &mut fa);
    let mut g_raw = vec![0.0; proj];
    linear_bf16_w(&fa, &w.f_b_proj, proj, d.head_dim, &mut g_raw);
    let mut g = vec![0.0; proj];
    kda_gate(
        &g_raw,
        &w.a_log,
        &w.dt_bias,
        d.gate_lower_bound,
        d.heads,
        d.head_dim,
        &mut g,
    );

    let mut beta = vec![0.0; d.heads];
    linear_bf16_w(x, &w.b_proj, d.heads, d.hidden, &mut beta);
    for b in beta.iter_mut() {
        *b = 1.0 / (1.0 + (-*b).exp());
    }

    // q/k are L2-normalised per head; q additionally carries the K^-0.5 scale
    l2norm_heads(&mut qc, d.heads, d.head_dim, (d.head_dim as f32).powf(-0.5));
    l2norm_heads(&mut kc, d.heads, d.head_dim, 1.0);

    let mut o = vec![0.0; proj];
    kda_step(
        &qc,
        &kc,
        &vc,
        &g,
        &beta,
        &mut st.recurrent,
        d.heads,
        d.head_dim,
        d.head_dim,
        &mut o,
    );

    // FusedRMSNormGated(head_dim, activation="sigmoid")
    let mut go = vec![0.0; proj];
    linear_bf16_w(x, &w.g_proj, proj, d.hidden, &mut go);
    for h in 0..d.heads {
        let s = h * d.head_dim;
        let e = s + d.head_dim;
        rmsnorm(&mut o[s..e], &w.o_norm, d.eps);
        for i in s..e {
            o[i] = to_bf16(o[i] * (1.0 / (1.0 + (-go[i]).exp())));
        }
    }
    linear_bf16_w(&o, &w.o_proj, d.hidden, proj, out);
}

/// Attention flavour of one layer.
pub enum LayerAttn {
    Kda(Box<KdaWeights>, KdaDims),
    Mla(Box<MlaWeights>, MlaDims),
}

/// Feed-forward flavour: dense (layer 0) or LatentMoE.
pub enum LayerFfn<E: ExpertSource> {
    Dense {
        w1: Vec<u16>,
        w3: Vec<u16>,
        w2: Vec<u16>,
        inter: usize,
    },
    Moe(Box<MoeWeights>, MoeDims, E),
}

/// One decoder layer: norms, AttnRes projections, attention and FFN.
pub struct K3Layer<E: ExpertSource> {
    pub idx: usize,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub attn_res_proj: Vec<f32>,
    pub attn_res_norm: Vec<f32>,
    pub mlp_res_proj: Vec<f32>,
    pub mlp_res_norm: Vec<f32>,
    pub attn: LayerAttn,
    pub ffn: LayerFfn<E>,
}

/// Mutable per-layer state (KDA recurrence or MLA KV).
#[derive(Clone)]
pub enum LayerState {
    Kda(Box<KdaState>),
    Mla(MlaKv),
}

impl LayerState {
    pub fn clear(&mut self) {
        match self {
            LayerState::Kda(s) => s.clear(),
            LayerState::Mla(kv) => kv.clear(),
        }
    }

    /// Bytes this state currently occupies — the cost of snapshotting it.
    ///
    /// KDA is fixed-size; only the MLA latent cache grows with position, so a
    /// long-prefix snapshot is dominated by the 24 MLA layers, not the 69 KDA
    /// ones.
    pub fn approx_bytes(&self) -> usize {
        const F: usize = std::mem::size_of::<f32>();
        match self {
            LayerState::Kda(s) => {
                (s.recurrent.len() + s.conv_q.len() + s.conv_k.len() + s.conv_v.len()) * F
            }
            LayerState::Mla(kv) => kv.latent.len() * F,
        }
    }
}

/// Model-wide shape knobs the layer loop needs.
#[derive(Clone, Copy, Debug)]
pub struct K3Dims {
    pub hidden: usize,
    pub num_layers: usize,
    pub block_size: usize,
    pub eps: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: Option<f32>,
}

/// Run one token through a contiguous slice of layers, threading the AttnRes
/// pair. `prefix_sum` is `[H]`; `blocks` holds `max_blocks * H` with the first
/// `nb` slots live and is grown in place as boundaries are crossed.
///
/// Returns the number of live block slots after the slice.
pub fn forward_slice<E: ExpertSource>(
    layers: &mut [K3Layer<E>],
    states: &mut [LayerState],
    d: K3Dims,
    prefix_sum: &mut [f32],
    blocks: &mut [f32],
    mut nb: usize,
) -> usize {
    let h = d.hidden;
    let mut buf = vec![0.0f32; h];
    let mut attn_out = vec![0.0f32; h];
    let mut ffn_out = vec![0.0f32; h];

    for (layer, state) in layers.iter_mut().zip(states.iter_mut()) {
        let t_ar = std::time::Instant::now();
        // pre-attention mixture (skipped while the stack is empty)
        if nb > 0 {
            apply_attn_res(
                prefix_sum,
                &blocks[..nb * h],
                &layer.attn_res_proj,
                &layer.attn_res_norm,
                d.eps,
                &mut buf,
            );
        } else {
            buf.copy_from_slice(prefix_sum);
        }

        // crossing a block boundary pushes the prefix sum and resets the carry
        let mut carry = true;
        if layer.idx % d.block_size == 0 {
            blocks[nb * h..(nb + 1) * h].copy_from_slice(prefix_sum);
            nb += 1;
            carry = false;
        }

        prof::add(prof::ATTNRES, t_ar);
        rmsnorm(&mut buf, &layer.input_layernorm, d.eps);
        let t_at = std::time::Instant::now();
        match (&mut layer.attn, &mut *state) {
            (LayerAttn::Kda(w, kd), LayerState::Kda(st)) => {
                kda_layer_step(&buf, w, *kd, st, &mut attn_out);
                prof::add(prof::KDA, t_at);
            }
            (LayerAttn::Mla(w, md), LayerState::Mla(kv)) => {
                mla_step(&buf, w, *md, kv, &mut attn_out);
                prof::add(prof::MLA, t_at);
            }
            _ => panic!("k3: layer {} attention/state kind mismatch", layer.idx),
        }

        if carry {
            for (p, &a) in prefix_sum.iter_mut().zip(attn_out.iter()) {
                *p = to_bf16(*p + a);
            }
        } else {
            prefix_sum.copy_from_slice(&attn_out);
        }

        // pre-FFN mixture (the stack is never empty here: layer 0 pushed)
        let t_ar2 = std::time::Instant::now();
        apply_attn_res(
            prefix_sum,
            &blocks[..nb * h],
            &layer.mlp_res_proj,
            &layer.mlp_res_norm,
            d.eps,
            &mut buf,
        );
        prof::add(prof::ATTNRES, t_ar2);
        rmsnorm(&mut buf, &layer.post_attention_layernorm, d.eps);

        match &layer.ffn {
            LayerFfn::Dense { w1, w3, w2, inter } => {
                let (mut g, mut u) = (vec![0.0; *inter], vec![0.0; *inter]);
                linear_bf16_w(&buf, w1, *inter, h, &mut g);
                linear_bf16_w(&buf, w3, *inter, h, &mut u);
                let mut hid = vec![0.0; *inter];
                situ(&g, &u, &mut hid, d.situ_beta, d.situ_linear_beta);
                for v in hid.iter_mut() {
                    *v = to_bf16(*v);
                }
                linear_bf16_w(&hid, w2, h, *inter, &mut ffn_out);
            }
            LayerFfn::Moe(w, md, ex) => moe_forward(&buf, w, *md, ex, &mut ffn_out),
        }

        for (p, &f) in prefix_sum.iter_mut().zip(ffn_out.iter()) {
            *p = to_bf16(*p + f);
        }
    }
    nb
}

/// Batched prefill: `rows` contiguous positions through a layer slice.
///
/// Attention stays per position (KDA's recurrence and the MLA cache are
/// order-dependent); the MoE is batch-unioned, so each distinct expert is
/// fetched once per layer rather than once per token.
///
/// Bit-exact against looping [`forward_slice`] over the same rows.
///
/// `prefix`: `[rows * H]`; `blocks`: `[rows * max_blocks * H]`.
/// Returns the live block count after the slice.
#[allow(clippy::too_many_arguments)]
pub fn forward_slice_batch<E: ExpertSource>(
    layers: &mut [K3Layer<E>],
    states: &mut [LayerState],
    d: K3Dims,
    prefix: &mut [f32],
    blocks: &mut [f32],
    rows: usize,
    maxb: usize,
    mut nb: usize,
) -> usize {
    let h = d.hidden;
    debug_assert_eq!(prefix.len(), rows * h);
    debug_assert_eq!(blocks.len(), rows * maxb * h);

    let mut buf = vec![0.0f32; rows * h];
    let mut attn_out = vec![0.0f32; rows * h];
    let mut ffn_out = vec![0.0f32; rows * h];
    let mut one = vec![0.0f32; h];

    for (layer, state) in layers.iter_mut().zip(states.iter_mut()) {
        // pre-attention mixture, per row
        for r in 0..rows {
            let ps = &prefix[r * h..(r + 1) * h];
            if nb > 0 {
                let br = &blocks[r * maxb * h..r * maxb * h + nb * h];
                apply_attn_res(
                    ps,
                    br,
                    &layer.attn_res_proj,
                    &layer.attn_res_norm,
                    d.eps,
                    &mut one,
                );
                buf[r * h..(r + 1) * h].copy_from_slice(&one);
            } else {
                buf[r * h..(r + 1) * h].copy_from_slice(ps);
            }
        }

        let mut carry = true;
        if layer.idx % d.block_size == 0 {
            for r in 0..rows {
                let dst = r * maxb * h + nb * h;
                blocks[dst..dst + h].copy_from_slice(&prefix[r * h..(r + 1) * h]);
            }
            nb += 1;
            carry = false;
        }

        // attention: strictly per position, in order
        for r in 0..rows {
            one.copy_from_slice(&buf[r * h..(r + 1) * h]);
            rmsnorm(&mut one, &layer.input_layernorm, d.eps);
            let dst = &mut attn_out[r * h..(r + 1) * h];
            match (&mut layer.attn, &mut *state) {
                (LayerAttn::Kda(w, kd), LayerState::Kda(st)) => {
                    kda_layer_step(&one, w, *kd, st, dst)
                }
                (LayerAttn::Mla(w, md), LayerState::Mla(kv)) => mla_step(&one, w, *md, kv, dst),
                _ => panic!("k3: layer {} attention/state kind mismatch", layer.idx),
            }
        }

        for r in 0..rows {
            let (ps, ao) = (r * h, r * h);
            if carry {
                for i in 0..h {
                    prefix[ps + i] = to_bf16(prefix[ps + i] + attn_out[ao + i]);
                }
            } else {
                prefix[ps..ps + h].copy_from_slice(&attn_out[ao..ao + h]);
            }
        }

        // pre-FFN mixture, per row
        for r in 0..rows {
            let ps = &prefix[r * h..(r + 1) * h];
            let br = &blocks[r * maxb * h..r * maxb * h + nb * h];
            apply_attn_res(
                ps,
                br,
                &layer.mlp_res_proj,
                &layer.mlp_res_norm,
                d.eps,
                &mut one,
            );
            rmsnorm(&mut one, &layer.post_attention_layernorm, d.eps);
            buf[r * h..(r + 1) * h].copy_from_slice(&one);
        }

        match &layer.ffn {
            LayerFfn::Dense { w1, w3, w2, inter } => {
                let (mut g, mut u) = (vec![0.0; *inter], vec![0.0; *inter]);
                let mut hid = vec![0.0; *inter];
                for r in 0..rows {
                    let x = &buf[r * h..(r + 1) * h];
                    linear_bf16_w(x, w1, *inter, h, &mut g);
                    linear_bf16_w(x, w3, *inter, h, &mut u);
                    situ(&g, &u, &mut hid, d.situ_beta, d.situ_linear_beta);
                    for v in hid.iter_mut() {
                        *v = to_bf16(*v);
                    }
                    linear_bf16_w(&hid, w2, h, *inter, &mut ffn_out[r * h..(r + 1) * h]);
                }
            }
            // the win: each distinct expert loaded once for the whole batch
            LayerFfn::Moe(w, md, ex) => moe_forward_batch(&buf, w, *md, ex, rows, &mut ffn_out),
        }

        for i in 0..rows * h {
            prefix[i] = to_bf16(prefix[i] + ffn_out[i]);
        }
    }
    nb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_counts_match_the_reference_schedule() {
        // boundaries at 0, 12, 24, ... -> ceil(i / 12) live slots at layer i
        assert_eq!(blocks_at(0, 12), 0);
        assert_eq!(blocks_at(1, 12), 1);
        assert_eq!(blocks_at(12, 12), 1);
        assert_eq!(blocks_at(13, 12), 2);
        assert_eq!(blocks_at(93, 12), 8);
        // the real model: 93 layers, block 12 -> 8 slots on the wire
        assert_eq!(max_blocks(93, 12), 8);
        // the tiny model: 6 layers, block 2 -> 3 slots
        assert_eq!(max_blocks(6, 2), 3);
    }

    #[test]
    fn wire_width_is_one_plus_max_blocks() {
        // 9 * 7168 f32 = 258 KB/token for the real model
        let (h, n, bs) = (7168usize, 93usize, 12usize);
        assert_eq!((1 + max_blocks(n, bs)) * h, 9 * 7168);
    }
}
