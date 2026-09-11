//! Inkling decoder layer (`InklingDecoderLayer`) and full text model
//! (`InklingForCausalLM`).
//!
//! Layer (pre-norm, convs INSIDE the residual branches — `PORT_SPEC.md` §3):
//!
//! ```text
//!   x  = x + sconv(attn_sconv, attention(rmsnorm(x, attn_norm)))
//!   x  = x + sconv(mlp_sconv,  mlp(rmsnorm(x, mlp_norm)))
//! ```
//!
//! Model: `x = rmsnorm(embed[t], embed_norm)` → layers →
//! `logits = unembed · (rmsnorm(x, norm) / mup)[..unpadded_vocab]`.
//!
//! Single-stream incremental decode; every `forward_token` advances each
//! layer's KV cache + conv histories. Pipeline-parallel sharding is layered
//! on at the stage level over [`Layer`]s.

use super::attn::{AttentionLayer, AttnKv};
use super::conv::{ConvState, ShortConv};
use super::moe::{DenseMlp, MoeLayer};
use super::rmsnorm_f32;
use crate::dsv4::math::{dot, dot_bf16w};

/// The per-layer feed-forward: dense SwiGLU (first `dense_mlp_idx` layers) or
/// the routed + shared MoE.
pub enum LayerMlp {
    Dense(DenseMlp),
    Moe(MoeLayer),
}

pub struct Layer {
    pub hidden: usize,
    pub eps: f32,
    attn_norm: Vec<f32>, // `attn_norm.weight` [hidden]
    attn: AttentionLayer,
    attn_sconv: ShortConv, // `attn_sconv.weight` [hidden, K]
    mlp_norm: Vec<f32>,    // `mlp_norm.weight` [hidden]
    mlp: LayerMlp,
    mlp_sconv: ShortConv, // `mlp_sconv.weight` [hidden, K]
}

/// One layer's complete sequence state (attention KV + k/v convs, plus the
/// attn-out and mlp-out conv histories) for prefix caching.
#[derive(Clone)]
pub struct LayerState {
    pub attn: AttnKv,
    pub attn_conv: ConvState,
    pub mlp_conv: ConvState,
}

impl LayerState {
    pub fn len(&self) -> usize {
        self.attn.len()
    }

    pub fn is_empty(&self) -> bool {
        self.attn.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.attn.bytes() + self.attn_conv.bytes() + self.mlp_conv.bytes()
    }
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        eps: f32,
        attn_norm: Vec<f32>,
        attn: AttentionLayer,
        attn_sconv: ShortConv,
        mlp_norm: Vec<f32>,
        mlp: LayerMlp,
        mlp_sconv: ShortConv,
    ) -> Self {
        assert_eq!(attn_norm.len(), hidden, "layer: attn_norm len");
        assert_eq!(mlp_norm.len(), hidden, "layer: mlp_norm len");
        assert_eq!(attn.dims.hidden, hidden, "layer: attention hidden");
        assert_eq!(attn_sconv.c, hidden, "layer: attn_sconv channels");
        assert_eq!(mlp_sconv.c, hidden, "layer: mlp_sconv channels");
        match &mlp {
            LayerMlp::Moe(m) => assert_eq!(m.hidden, hidden, "layer: moe hidden"),
            LayerMlp::Dense(_) => {}
        }
        Self {
            hidden,
            eps,
            attn_norm,
            attn,
            attn_sconv,
            mlp_norm,
            mlp,
            mlp_sconv,
        }
    }

    /// This layer's MoE block, if sparse.
    pub fn moe(&self) -> Option<&MoeLayer> {
        match &self.mlp {
            LayerMlp::Moe(m) => Some(m),
            LayerMlp::Dense(_) => None,
        }
    }

    /// Cached positions (attention and convs agree).
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.attn.len(), self.attn_sconv.len());
        debug_assert_eq!(self.attn.len(), self.mlp_sconv.len());
        self.attn.len()
    }

    pub fn is_empty(&self) -> bool {
        self.attn.is_empty()
    }

    /// Bytes of sequence state this layer allocates (KV cache + 4 conv rings).
    pub fn cache_bytes(&self) -> usize {
        self.attn.cache_bytes() + self.attn_sconv.cache_bytes() + self.mlp_sconv.cache_bytes()
    }

    /// Clear all sequence state (new sequence). O(1) — see
    /// [`AttentionLayer::reset`] / [`ShortConv::reset`].
    pub fn reset(&mut self) {
        self.attn.reset();
        self.attn_sconv.reset();
        self.mlp_sconv.reset();
    }

    /// Roll all sequence state back to `len` positions (spec-decode reject).
    pub fn truncate(&mut self, len: usize) {
        self.attn.truncate(len);
        self.attn_sconv.truncate(len);
        self.mlp_sconv.truncate(len);
    }

    /// Snapshot this layer's sequence state for prefix caching.
    pub fn snapshot(&self) -> LayerState {
        LayerState {
            attn: self.attn.snapshot(),
            attn_conv: self.attn_sconv.snapshot(),
            mlp_conv: self.mlp_sconv.snapshot(),
        }
    }

    /// Restore a snapshot (call after [`Self::reset`]).
    pub fn restore(&mut self, s: &LayerState) {
        self.attn.restore(&s.attn);
        self.attn_sconv.restore(&s.attn_conv);
        self.mlp_sconv.restore(&s.mlp_conv);
    }

    /// One token at the next cached position: `x` is the residual-stream hidden
    /// `[hidden]`; returns the updated hidden.
    pub fn forward_token(&mut self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden, "layer forward_token: x len");
        // x1 = x + attn_sconv(attention(rmsnorm(x, attn_norm)))
        let mut h = x.to_vec();
        rmsnorm_f32(&mut h, &self.attn_norm, self.eps);
        let a = self.attn.forward_token(&h);
        let a = self.attn_sconv.decode(&a);
        let mut x1: Vec<f32> = x.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        // x2 = x1 + mlp_sconv(mlp(rmsnorm(x1, mlp_norm)))
        let mut h2 = x1.clone();
        rmsnorm_f32(&mut h2, &self.mlp_norm, self.eps);
        let m = match &self.mlp {
            LayerMlp::Moe(m) => m.forward(&h2),
            LayerMlp::Dense(d) => d.forward(&h2, self.hidden),
        };
        let m = self.mlp_sconv.decode(&m);
        for (xi, &mi) in x1.iter_mut().zip(&m) {
            *xi += mi;
        }
        x1
    }

    /// Batched prefill of `rows` tokens (`xs` = `[rows, hidden]`), returning
    /// `[rows, hidden]`. Attention runs per position, the convs as batched
    /// prefills, the MoE as one batch-union (each expert loaded once per
    /// block). Bit-identical to [`Self::forward_token`] per row.
    pub fn forward_prefill(&mut self, xs: &[f32], rows: usize) -> Vec<f32> {
        let hd = self.hidden;
        assert_eq!(xs.len(), rows * hd, "layer forward_prefill: xs len");
        // attention branch
        let mut h = xs.to_vec();
        rmsnorm_f32(&mut h, &self.attn_norm, self.eps);
        let a = self.attn.forward_prefill(&h, rows);
        let a = self.attn_sconv.prefill(&a, rows);
        let mut x1: Vec<f32> = xs.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        // mlp branch
        let mut h2 = x1.clone();
        rmsnorm_f32(&mut h2, &self.mlp_norm, self.eps);
        let m = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_batch(&h2, rows),
            LayerMlp::Dense(d) => {
                let mut m = vec![0.0f32; rows * hd];
                for (r, row) in h2.chunks_exact(hd).enumerate() {
                    m[r * hd..(r + 1) * hd].copy_from_slice(&d.forward(row, hd));
                }
                m
            }
        };
        let m = self.mlp_sconv.prefill(&m, rows);
        for (xi, &mi) in x1.iter_mut().zip(&m) {
            *xi += mi;
        }
        x1
    }
}

/// A `vocab × hidden` edge table (embedding or unembed) held as exact f32 or
/// bf16 bits (the checkpoint dtype — lossless for bf16 weights and half the
/// RAM). Only the edge ranks hold one.
pub enum WideTable {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl WideTable {
    pub fn len(&self) -> usize {
        match self {
            WideTable::F32(v) => v.len(),
            WideTable::Bf16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `t` (`hidden` floats) — the embedding lookup.
    pub fn row(&self, t: usize, hidden: usize) -> Vec<f32> {
        let r = t * hidden..(t + 1) * hidden;
        match self {
            WideTable::F32(v) => v[r].to_vec(),
            WideTable::Bf16(v) => v[r]
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect(),
        }
    }

    /// `out[i] = row_i · x` for the first `out.len()` rows — f32 output with NO
    /// bf16 rounding (logits feed an argmax; keeping them f32 avoids bf16
    /// ties flipping greedy parity against the f32 reference).
    pub fn matvec_f32(&self, x: &[f32], hidden: usize, out: &mut [f32]) {
        use rayon::prelude::*;
        assert_eq!(x.len(), hidden);
        assert!(
            out.len() * hidden <= self.len(),
            "matvec_f32: rows exceed table"
        );
        match self {
            WideTable::F32(w) => out.par_iter_mut().enumerate().for_each(|(o, y)| {
                *y = dot(&w[o * hidden..(o + 1) * hidden], x);
            }),
            WideTable::Bf16(w) => out.par_iter_mut().enumerate().for_each(|(o, y)| {
                *y = dot_bf16w(&w[o * hidden..(o + 1) * hidden], x);
            }),
        }
    }
}

/// The output head: `logits = unembed · (rmsnorm(x, norm) / mup)` sliced to
/// `unpadded_vocab` (f32 logits, no bf16 rounding — see
/// [`WideTable::matvec_f32`]). The ONE implementation [`Model`], the staged
/// runner and the layer-dump example share, so their logits cannot drift: the
/// division is a true `x / mup`, not `x * (1 / mup)` — at the real
/// `logits_mup_width_multiplier` of 24 the reciprocal form is 1 ULP off on
/// most elements, enough to flip argmax ties.
pub struct Head {
    /// `norm.weight` `[hidden]`.
    pub norm: Vec<f32>,
    /// `unembed.weight` `[vocab, hidden]`.
    pub unembed: WideTable,
    pub eps: f32,
    /// `logits_mup_width_multiplier` (> 0).
    pub mup: f32,
    /// Logits are sliced to this many entries (`unpadded_vocab_size`).
    pub unpadded_vocab: usize,
}

impl Head {
    pub fn new(
        norm: Vec<f32>,
        unembed: WideTable,
        eps: f32,
        mup: f32,
        unpadded_vocab: usize,
    ) -> Self {
        let hidden = norm.len();
        assert!(hidden > 0, "head: empty norm");
        assert!(mup > 0.0, "head: mup must be positive");
        assert!(
            unpadded_vocab >= 1 && unpadded_vocab * hidden <= unembed.len(),
            "head: unpadded_vocab {unpadded_vocab} vs unembed rows {}",
            unembed.len() / hidden
        );
        Self {
            norm,
            unembed,
            eps,
            mup,
            unpadded_vocab,
        }
    }

    pub fn hidden(&self) -> usize {
        self.norm.len()
    }

    /// `unembed · (rmsnorm(x, norm) / mup)[..unpadded_vocab]` for one hidden
    /// `x` (`[hidden]`).
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        let hidden = self.hidden();
        assert_eq!(x.len(), hidden, "head logits: x len");
        let mut y = x.to_vec();
        rmsnorm_f32(&mut y, &self.norm, self.eps);
        for v in y.iter_mut() {
            *v /= self.mup;
        }
        let mut logits = vec![0.0f32; self.unpadded_vocab];
        self.unembed.matvec_f32(&y, hidden, &mut logits);
        logits
    }
}

/// Full Inkling text model: embed + embed_norm → layers → [`Head`].
pub struct Model {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    embed: WideTable,     // [vocab, hidden]
    embed_norm: Vec<f32>, // [hidden]
    layers: Vec<Layer>,
    head: Head,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        vocab: usize,
        unpadded_vocab: usize,
        eps: f32,
        mup: f32,
        embed: WideTable,
        embed_norm: Vec<f32>,
        layers: Vec<Layer>,
        norm: Vec<f32>,
        unembed: WideTable,
    ) -> Self {
        assert_eq!(embed.len(), vocab * hidden, "model: embed shape");
        assert_eq!(unembed.len(), vocab * hidden, "model: unembed shape");
        assert_eq!(embed_norm.len(), hidden, "model: embed_norm len");
        assert_eq!(norm.len(), hidden, "model: norm len");
        assert!(
            unpadded_vocab >= 1 && unpadded_vocab <= vocab,
            "model: unpadded_vocab {unpadded_vocab} vs vocab {vocab}"
        );
        for (i, l) in layers.iter().enumerate() {
            assert_eq!(l.hidden, hidden, "model: layer {i} hidden");
        }
        Self {
            hidden,
            vocab,
            eps,
            embed,
            embed_norm,
            layers,
            head: Head::new(norm, unembed, eps, mup, unpadded_vocab),
        }
    }

    /// Logits are sliced to this many entries (`unpadded_vocab_size`).
    pub fn unpadded_vocab(&self) -> usize {
        self.head.unpadded_vocab
    }

    /// `logits_mup_width_multiplier` — the final hidden is divided by it.
    pub fn mup(&self) -> f32 {
        self.head.mup
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    /// Cached positions (every layer agrees).
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, Layer::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear every layer's sequence state (new sequence).
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }

    /// Roll every layer back to `len` positions (spec-decode reject).
    pub fn truncate(&mut self, len: usize) {
        for l in &mut self.layers {
            l.truncate(len);
        }
    }

    /// Snapshot every layer's state — a reusable prompt prefix.
    pub fn snapshot_prefix(&self) -> Vec<LayerState> {
        self.layers.iter().map(Layer::snapshot).collect()
    }

    /// Restore a prefix snapshot into every layer (after [`Self::reset`]).
    pub fn restore_prefix(&mut self, snap: &[LayerState]) {
        assert_eq!(
            snap.len(),
            self.layers.len(),
            "prefix snapshot layer count {} != {}",
            snap.len(),
            self.layers.len()
        );
        for (l, s) in self.layers.iter_mut().zip(snap) {
            l.restore(s);
        }
    }

    /// `rmsnorm(embed[id], embed_norm)` — the residual stream entering layer 0.
    pub fn embed_token(&self, id: u32) -> Vec<f32> {
        let t = id as usize;
        assert!(t < self.vocab, "token {t} >= vocab {}", self.vocab);
        let mut x = self.embed.row(t, self.hidden);
        rmsnorm_f32(&mut x, &self.embed_norm, self.eps);
        x
    }

    /// [`Head::logits`]: `unembed · (rmsnorm(x, norm) / mup)`, sliced to
    /// `unpadded_vocab` (f32 logits).
    pub fn head_logits(&self, x: &[f32]) -> Vec<f32> {
        self.head.logits(x)
    }

    /// Embed `token`, run every layer and the head; returns logits
    /// `[unpadded_vocab]` at this position and advances the caches.
    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let mut x = self.embed_token(token);
        for l in &mut self.layers {
            x = l.forward_token(&x);
        }
        self.head_logits(&x)
    }

    /// Batched prefill of `prompt`; returns the LAST position's logits and
    /// advances the caches (decode continues with [`Self::forward_token`]).
    /// Bit-identical to looping `forward_token` and keeping the last logits.
    pub fn prefill(&mut self, prompt: &[u32]) -> Vec<f32> {
        let rows = prompt.len();
        assert!(rows > 0, "prefill needs a non-empty prompt");
        let hd = self.hidden;
        let mut xs = vec![0.0f32; rows * hd];
        for (r, &t) in prompt.iter().enumerate() {
            xs[r * hd..(r + 1) * hd].copy_from_slice(&self.embed_token(t));
        }
        for l in &mut self.layers {
            xs = l.forward_prefill(&xs, rows);
        }
        self.head_logits(&xs[(rows - 1) * hd..])
    }

    /// Greedy generation: reset, batched prefill of `prompt`, then `n_gen`
    /// argmax tokens.
    pub fn greedy(&mut self, prompt: &[u32], n_gen: usize) -> Vec<u32> {
        self.reset();
        let mut logits = self.prefill(prompt);
        let mut out = Vec::with_capacity(n_gen);
        for _ in 0..n_gen {
            let nxt = argmax(&logits) as u32;
            out.push(nxt);
            if out.len() == n_gen {
                break;
            }
            logits = self.forward_token(nxt);
        }
        out
    }
}

/// Index of the first maximum (ties -> lowest index, matching `torch.argmax`).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}
