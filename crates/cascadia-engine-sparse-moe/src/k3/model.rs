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
use crate::k3::moe::{moe_forward, ExpertSource, MoeDims, MoeWeights};
use crate::k3::situ::situ;

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

        rmsnorm(&mut buf, &layer.input_layernorm, d.eps);
        match (&mut layer.attn, &mut *state) {
            (LayerAttn::Kda(w, kd), LayerState::Kda(st)) => {
                kda_layer_step(&buf, w, *kd, st, &mut attn_out)
            }
            (LayerAttn::Mla(w, md), LayerState::Mla(kv)) => {
                mla_step(&buf, w, *md, kv, &mut attn_out)
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
        apply_attn_res(
            prefix_sum,
            &blocks[..nb * h],
            &layer.mlp_res_proj,
            &layer.mlp_res_norm,
            d.eps,
            &mut buf,
        );
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
