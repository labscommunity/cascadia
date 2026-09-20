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

use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Optional diagnostic measurements. Branch durations include norms, residuals
/// and short convolutions; observer overhead is outside `total`.
#[derive(Debug, Clone, Copy)]
pub struct LayerTiming {
    pub rows: usize,
    pub prefill: bool,
    pub attention: Duration,
    pub mlp: Duration,
    pub total: Duration,
}

pub type LayerTimingObserver = Arc<dyn Fn(LayerTiming) + Send + Sync>;

/// Target layer index and its predicted gate, evaluated before its predecessor.
pub type PreviousLayerRouteObserver = Arc<dyn Fn(usize, &super::gate::GateOut) + Send + Sync>;

pub struct Layer {
    pub hidden: usize,
    pub eps: f32,
    attn_norm: Vec<f32>, // `attn_norm.weight` [hidden]
    attn: AttentionLayer,
    attn_sconv: ShortConv, // `attn_sconv.weight` [hidden, K]
    mlp_norm: Vec<f32>,    // `mlp_norm.weight` [hidden]
    mlp: LayerMlp,
    mlp_sconv: ShortConv, // `mlp_sconv.weight` [hidden, K]
    timing_observer: Option<LayerTimingObserver>,
    pre_attention_route_observer: Option<super::moe::RouteObserver>,
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
        assert_eq!(attn.dims().hidden, hidden, "layer: attention hidden");
        assert_eq!(attn_sconv.c(), hidden, "layer: attn_sconv channels");
        assert_eq!(mlp_sconv.c(), hidden, "layer: mlp_sconv channels");
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
            timing_observer: None,
            pre_attention_route_observer: None,
        }
    }

    /// Install diagnostics without changing model arithmetic. Defaults off;
    /// callbacks must not change the floating-point environment or block on I/O.
    pub fn set_timing_observer(&mut self, observer: Option<LayerTimingObserver>) {
        self.timing_observer = observer;
    }

    /// Observe a causal decode-only route prediction from this layer's input,
    /// before attention. Uses the existing MLP norm/router without modifying
    /// actual routing, cache history, weights, or sequence state. Defaults off;
    /// the callback must not block or alter the floating-point environment.
    pub fn set_pre_attention_route_observer(
        &mut self,
        observer: Option<super::moe::RouteObserver>,
    ) {
        self.pre_attention_route_observer = observer;
    }

    fn observe_timing(
        &self,
        start: Option<Instant>,
        mlp_start: Option<Instant>,
        rows: usize,
        prefill: bool,
    ) {
        if let (Some(observer), Some(start), Some(mlp_start)) =
            (&self.timing_observer, start, mlp_start)
        {
            let end = Instant::now();
            observer(LayerTiming {
                rows,
                prefill,
                attention: mlp_start.duration_since(start),
                mlp: end.duration_since(mlp_start),
                total: end.duration_since(start),
            });
        }
    }

    /// This layer's MoE block, if sparse.
    pub fn moe(&self) -> Option<&MoeLayer> {
        match &self.mlp {
            LayerMlp::Moe(m) => Some(m),
            LayerMlp::Dense(_) => None,
        }
    }

    /// Mutable access to the MoE block (to attach an expert-parallel client —
    /// [`MoeLayer::attach_remote`]).
    pub fn moe_mut(&mut self) -> Option<&mut MoeLayer> {
        match &mut self.mlp {
            LayerMlp::Moe(m) => Some(m),
            LayerMlp::Dense(_) => None,
        }
    }

    /// Route this layer's MLP (MoE experts or the dense FFN) through an
    /// OpenVINO backend; `layer` is the global layer index.
    pub fn attach_ov(&mut self, layer: u32, ov: Arc<super::ov_expert::OvExperts>) {
        match &mut self.mlp {
            LayerMlp::Moe(m) => m.attach_ov(layer, ov),
            LayerMlp::Dense(d) => d.attach_ov(layer, ov),
        }
    }

    /// Route this layer's MoE through a fused-MoE backend (dense layers keep
    /// their path); `layer` is the global layer index.
    pub fn attach_ov_moe(&mut self, layer: u32, ov: Arc<super::ov_moe::OvMoe>) {
        if let LayerMlp::Moe(m) = &mut self.mlp {
            m.attach_ov_moe(layer, ov);
        }
    }

    /// Run a dense layer's MLP on the all-rows device backend; no-op on a MoE layer.
    pub fn attach_ov_dense(&mut self, layer: u32, ov: Arc<super::ov_dense::OvDense>) {
        if let LayerMlp::Dense(d) = &mut self.mlp {
            d.attach_ov_dense(layer, ov);
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self.mlp, LayerMlp::Dense(_))
    }

    /// Route this layer's attention projections through an OpenVINO backend.
    pub fn attach_ov_attn(&mut self, layer: u32, ov: Arc<super::ov_attn::OvAttn>) {
        self.attn.attach_ov(layer, ov);
    }

    pub fn ov_attn(&self) -> Option<&Arc<super::ov_attn::OvAttn>> {
        self.attn.ov().map(|(_, o)| o)
    }

    pub fn warm_ov_attn(&self) -> Option<bool> {
        self.attn.warm_ov()
    }

    /// Free this layer's Rust projection tables once its OpenVINO attention
    /// backend compiled (`None` without a backend, `Some(0)` if it failed to
    /// warm and the Rust tables must stay).
    pub fn release_rust_attention_weights(&mut self) -> Option<usize> {
        let ok = self.attn.warm_ov()?;
        Some(if ok {
            self.attn.release_rust_projections()
        } else {
            0
        })
    }

    /// The attached fused-MoE backend, if any.
    pub fn ov_moe(&self) -> Option<&Arc<super::ov_moe::OvMoe>> {
        match &self.mlp {
            LayerMlp::Moe(m) => m.ov_moe().map(|(_, o)| o),
            LayerMlp::Dense(_) => None,
        }
    }

    /// Compile this layer's fused IR ahead of time; `None` without a backend.
    pub fn warm_ov_moe(&self) -> Option<bool> {
        match &self.mlp {
            LayerMlp::Moe(m) => m.warm_ov_moe(),
            LayerMlp::Dense(_) => None,
        }
    }

    /// The attached OV backend, if any.
    pub fn ov(&self) -> Option<&Arc<super::ov_expert::OvExperts>> {
        match &self.mlp {
            LayerMlp::Moe(m) => m.ov().map(|(_, o)| o),
            LayerMlp::Dense(d) => d.ov().map(|(_, o)| o),
        }
    }

    /// Compile this layer's experts / MLP on the attached OV backend;
    /// `(compiled, failed keys)`, or `None` without a backend.
    pub fn warm_ov(&self) -> Option<(usize, Vec<(u32, u32)>)> {
        match &self.mlp {
            LayerMlp::Moe(m) => m.warm_ov(),
            LayerMlp::Dense(d) => d.warm_ov(),
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

    /// Size this layer's multi-stream slot pool (attention KV + the four conv
    /// histories per sequence). See [`Self::select_slot`].
    pub fn ensure_slots(&mut self, n: usize) {
        self.attn.ensure_slots(n);
        self.attn_sconv.ensure_slots(n);
        self.mlp_sconv.ensure_slots(n);
    }

    pub fn slot_count(&self) -> usize {
        self.attn.slot_count()
    }

    /// Bytes one additional sequence slot costs on this layer.
    pub fn slot_bytes(&self) -> usize {
        self.cache_bytes()
    }

    /// Make sequence `slot` the live one for every stateful part of this
    /// layer (O(1) swaps); `reset` / `truncate` / `forward_token` /
    /// `forward_prefill` then act on that sequence.
    pub fn select_slot(&mut self, slot: usize) {
        self.attn.select(slot);
        self.attn_sconv.select(slot);
        self.mlp_sconv.select(slot);
    }

    /// Clear all sequence state (new sequence). O(1) — see
    /// [`AttentionLayer::reset`] / [`ShortConv::reset`].
    pub fn reset(&mut self) {
        self.attn.reset();
        self.attn_sconv.reset();
        self.mlp_sconv.reset();
        if let Some(moe) = self.moe() {
            moe.reset_expert_cache_history();
        }
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
        self.forward_token_with_prediction(x, None, false)
    }

    fn forward_token_with_prediction(
        &mut self,
        x: &[f32],
        supplied: Option<super::predicted_read::PendingReadGroup>,
        skip_current_prediction: bool,
    ) -> Vec<f32> {
        let start = self.timing_observer.as_ref().map(|_| Instant::now());
        assert_eq!(x.len(), self.hidden, "layer forward_token: x len");
        let pending_read = self
            .moe()
            .and_then(|moe| {
                if self.pre_attention_route_observer.is_none()
                    && (skip_current_prediction || !moe.prediction_reads_enabled())
                {
                    return None;
                }
                let mut predicted_input = x.to_vec();
                rmsnorm_f32(&mut predicted_input, &self.mlp_norm, self.eps);
                let prediction = moe.route_unobserved(&predicted_input);
                if let Some(observer) = &self.pre_attention_route_observer {
                    observer(&prediction);
                }
                if skip_current_prediction {
                    None
                } else {
                    moe.start_predicted_read(&prediction)
                }
            })
            .or(supplied);
        // x1 = x + attn_sconv(attention(rmsnorm(x, attn_norm)))
        let mut h = x.to_vec();
        rmsnorm_f32(&mut h, &self.attn_norm, self.eps);
        let a = self.attn.forward_token(&h);
        let a = self.attn_sconv.decode(&a);
        let mut x1: Vec<f32> = x.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        let mlp_start = start.map(|_| Instant::now());
        // x2 = x1 + mlp_sconv(mlp(rmsnorm(x1, mlp_norm)))
        let mut h2 = x1.clone();
        rmsnorm_f32(&mut h2, &self.mlp_norm, self.eps);
        let m = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_with_prediction(&h2, pending_read),
            LayerMlp::Dense(d) => d.forward(&h2, self.hidden),
        };
        let m = self.mlp_sconv.decode(&m);
        for (xi, &mi) in x1.iter_mut().zip(&m) {
            *xi += mi;
        }
        self.observe_timing(start, mlp_start, 1, false);
        x1
    }

    /// Batched prefill of `rows` tokens (`xs` = `[rows, hidden]`), returning
    /// `[rows, hidden]`. Attention runs per position, the convs as batched
    /// prefills, the MoE as one batch-union (each expert loaded once per
    /// block). Bit-identical to [`Self::forward_token`] per row.
    pub fn forward_prefill(&mut self, xs: &[f32], rows: usize) -> Vec<f32> {
        let start = self.timing_observer.as_ref().map(|_| Instant::now());
        let hd = self.hidden;
        assert_eq!(xs.len(), rows * hd, "layer forward_prefill: xs len");
        // attention branch
        let mut h = xs.to_vec();
        rmsnorm_f32(&mut h, &self.attn_norm, self.eps);
        let a = self.attn.forward_prefill(&h, rows);
        let a = self.attn_sconv.prefill(&a, rows);
        let mut x1: Vec<f32> = xs.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        let mlp_start = start.map(|_| Instant::now());
        // mlp branch
        let mut h2 = x1.clone();
        rmsnorm_f32(&mut h2, &self.mlp_norm, self.eps);
        let m = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_batch(&h2, rows),
            LayerMlp::Dense(d) => d.forward_rows(&h2, rows, hd),
        };
        let m = self.mlp_sconv.prefill(&m, rows);
        for (xi, &mi) in x1.iter_mut().zip(&m) {
            *xi += mi;
        }
        self.observe_timing(start, mlp_start, rows, true);
        x1
    }
}

impl Layer {
    /// Prefill several sequences at once: `segs[i] = (slot, rows)`, their rows
    /// laid end to end in `xs`. Attention and the convs run per sequence on its
    /// own slot (each appending at that slot's position); the MoE runs ALL rows
    /// as one batch-union, so an expert several prompts touch is read once.
    /// A 24-token prompt touches about 135 of a layer's 256 experts and ten of
    /// them about 250, not 1350: a burst of requests costs the expert reads of
    /// roughly two. Per row the ops are [`Self::forward_prefill`]'s.
    pub fn forward_prefill_slots(&mut self, xs: &[f32], segs: &[(usize, usize)]) -> Vec<f32> {
        let start = self.timing_observer.as_ref().map(|_| Instant::now());
        let hd = self.hidden;
        let rows: usize = segs.iter().map(|&(_, r)| r).sum();
        assert_eq!(xs.len(), rows * hd, "layer forward_prefill_slots: xs len");
        let mut h = xs.to_vec();
        rmsnorm_f32(&mut h, &self.attn_norm, self.eps);
        let mut a: Vec<f32> = Vec::with_capacity(rows * hd);
        let mut at = 0usize;
        for &(slot, r) in segs {
            self.select_slot(slot);
            let seg = self.attn.forward_prefill(&h[at * hd..(at + r) * hd], r);
            a.extend(self.attn_sconv.prefill(&seg, r));
            at += r;
        }
        let mut x1: Vec<f32> = xs.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        let mlp_start = start.map(|_| Instant::now());
        let mut h2 = x1.clone();
        rmsnorm_f32(&mut h2, &self.mlp_norm, self.eps);
        let m = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_batch(&h2, rows),
            LayerMlp::Dense(d) => d.forward_rows(&h2, rows, hd),
        };
        let mut at = 0usize;
        for &(slot, r) in segs {
            self.mlp_sconv.select(slot);
            let ms = self.mlp_sconv.prefill(&m[at * hd..(at + r) * hd], r);
            for (xi, &mi) in x1[at * hd..(at + r) * hd].iter_mut().zip(&ms) {
                *xi += mi;
            }
            at += r;
        }
        self.observe_timing(start, mlp_start, rows, true);
        x1
    }

    /// Multi-stream decode step: `rows` residual-stream rows (`xs` =
    /// `[rows, hidden]`), row `i` the next token of sequence `slots[i]`.
    /// Attention, the convs and the residuals run per row on that row's slot;
    /// the MoE runs all rows as one batch-union (each expert loaded once for
    /// every stream that chose it — the aggregate-throughput lever). Per row
    /// the op sequence is [`Self::forward_token`]'s, so a stream decoded in a
    /// batch equals the same stream decoded alone (bit-identical on the CPU
    /// kernels). Returns `[rows, hidden]`.
    pub fn forward_rows(&mut self, xs: &[f32], rows: usize, slots: &[usize]) -> Vec<f32> {
        let start = self.timing_observer.as_ref().map(|_| Instant::now());
        let hd = self.hidden;
        assert_eq!(xs.len(), rows * hd, "layer forward_rows: xs len");
        assert_eq!(slots.len(), rows, "layer forward_rows: one slot per row");
        // attention branch: batched projections, per-slot attention + conv
        let mut h = xs.to_vec();
        rmsnorm_f32(&mut h, &self.attn_norm, self.eps);
        let a = self.attn.forward_rows(&h, rows, slots);
        let mut x1: Vec<f32> = Vec::with_capacity(rows * hd);
        for (r, &slot) in slots.iter().enumerate() {
            self.attn_sconv.select(slot);
            let ar = self.attn_sconv.decode(&a[r * hd..(r + 1) * hd]);
            x1.extend(
                xs[r * hd..(r + 1) * hd]
                    .iter()
                    .zip(&ar)
                    .map(|(&xi, &ai)| xi + ai),
            );
        }
        let mlp_start = start.map(|_| Instant::now());
        // mlp branch: one batch across streams
        let mut h2 = x1.clone();
        rmsnorm_f32(&mut h2, &self.mlp_norm, self.eps);
        let m = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_batch(&h2, rows),
            LayerMlp::Dense(d) => d.forward_rows(&h2, rows, hd),
        };
        for (r, &slot) in slots.iter().enumerate() {
            self.mlp_sconv.select(slot);
            let mr = self.mlp_sconv.decode(&m[r * hd..(r + 1) * hd]);
            for (xi, &mi) in x1[r * hd..(r + 1) * hd].iter_mut().zip(&mr) {
                *xi += mi;
            }
        }
        self.observe_timing(start, mlp_start, rows, false);
        x1
    }
}

/// A `vocab × hidden` edge table (embedding or unembed) held as exact f32 or
/// bf16 bits (the checkpoint dtype — lossless for bf16 weights and half the
/// RAM), or a read-only mapped BF16 table for sparse embedding lookups.
/// Only the edge ranks hold one.
pub enum WideTable {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    MappedBf16(crate::dsv4::st::MappedBf16),
}

impl WideTable {
    pub fn len(&self) -> usize {
        match self {
            WideTable::F32(v) => v.len(),
            WideTable::Bf16(v) => v.len(),
            WideTable::MappedBf16(v) => v.as_slice().len(),
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
            WideTable::MappedBf16(v) => v.as_slice()[r]
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
            WideTable::MappedBf16(w) => {
                let weights = w.as_slice();
                out.par_iter_mut().enumerate().for_each(|(o, y)| {
                    *y = dot_bf16w(&weights[o * hidden..(o + 1) * hidden], x);
                });
                w.trim_working_set();
            }
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
    /// Optional OpenVINO backend for the unembed GEMV; see [`super::ov_head`].
    ov: Option<Arc<super::ov_head::OvHead>>,
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
            ov: None,
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

    /// Route the unembed GEMV through an OpenVINO backend (see
    /// [`super::ov_head`]).
    pub fn attach_ov(&mut self, ov: Arc<super::ov_head::OvHead>) {
        self.ov = Some(ov);
    }

    pub fn ov(&self) -> Option<&Arc<super::ov_head::OvHead>> {
        self.ov.as_ref()
    }

    /// Compile the head's IR ahead of time; `None` without a backend.
    pub fn warm_ov(&self) -> Option<bool> {
        Some(self.ov.as_ref()?.warm())
    }

    /// [`Self::logits`] for `rows` hidden states (`xs` = `[rows, hidden]`),
    /// returning `[rows, unpadded_vocab]`. The OpenVINO head takes all rows in
    /// one call (the 1.2 GB table read once per step); the Rust head loops.
    pub fn logits_rows(&self, xs: &[f32], rows: usize) -> Vec<f32> {
        let hidden = self.hidden();
        assert_eq!(xs.len(), rows * hidden, "head logits_rows: xs len");
        if rows == 1 {
            return self.logits(xs);
        }
        if let Some(ov) = &self.ov {
            let mut ys = xs.to_vec();
            for y in ys.chunks_exact_mut(hidden) {
                rmsnorm_f32(y, &self.norm, self.eps);
                for v in y.iter_mut() {
                    *v /= self.mup;
                }
            }
            if let Some(l) = ov.logits_rows(&ys, rows) {
                return l;
            }
        }
        let mut out = Vec::with_capacity(rows * self.unpadded_vocab);
        for x in xs.chunks_exact(hidden) {
            out.extend(self.logits(x));
        }
        out
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
        if let Some(ov) = &self.ov {
            if let Some(l) = ov.logits(&y) {
                return l;
            }
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
    previous_layer_route_observer: Option<PreviousLayerRouteObserver>,
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
            previous_layer_route_observer: None,
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

    /// Decode-only diagnostics for predicting layer i from the residual entering
    /// layer i-1, using i's existing MLP norm/router. Layer 0 has no predecessor
    /// and is omitted. No expert read or actual route observation is performed.
    /// Defaults off; callbacks must not block or alter floating-point state.
    /// Observer overhead is included in model time, outside layer timing spans.
    pub fn set_previous_layer_route_observer(
        &mut self,
        observer: Option<PreviousLayerRouteObserver>,
    ) {
        self.previous_layer_route_observer = observer;
    }

    /// Opt-in whole-model decode scheduling; local pipelined expert caching and
    /// predicted reads must also be enabled. At most current + next are pending
    /// in this model. Layer-only/staged execution retains current-layer reads.
    pub fn early_prediction_reads_enabled(&self) -> bool {
        super::predicted_read::early_reads_requested()
            && self
                .layers
                .iter()
                .filter_map(Layer::moe)
                .any(MoeLayer::prediction_reads_enabled)
    }

    /// Whether sparse embedding lookups use file-backed BF16 rows.
    pub fn embedding_is_mapped(&self) -> bool {
        matches!(self.embed, WideTable::MappedBf16(_))
    }

    pub fn owned_shared_bytes(&self) -> usize {
        self.layers
            .iter()
            .filter_map(Layer::moe)
            .map(MoeLayer::owned_shared_bytes)
            .sum()
    }

    /// Aggregate actual routed-cache storage and decode hits for this model.
    pub fn expert_cache_stats(&self) -> super::ExpertCacheStats {
        let mut total = super::ExpertCacheStats::default();
        for moe in self.layers.iter().filter_map(Layer::moe) {
            total.add(moe.expert_cache_stats());
        }
        total
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
    /// Route every layer's experts / dense MLP through an OpenVINO backend
    /// (layer index = position; the model holds all layers).
    ///
    /// Precondition: this `Model` holds the full, 0-based layer set, so a
    /// layer's vec position IS its absolute index in the IR-file namespace.
    /// A partial stage must instead attach through the loader's per-layer
    /// global index (`lo + i`, gated by `has_layer`) — never this by-position
    /// helper, or a layer's hidden state would run through another layer's IR.
    pub fn attach_ov(&mut self, ov: Arc<super::ov_expert::OvExperts>) {
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.attach_ov(i as u32, Arc::clone(&ov));
        }
    }

    /// Route the head's unembed GEMV through an OpenVINO backend (last rank
    /// only; a stage without a head ignores it).
    pub fn attach_ov_head(&mut self, ov: Arc<super::ov_head::OvHead>) {
        self.head.attach_ov(ov);
    }

    pub fn ov_head(&self) -> Option<&Arc<super::ov_head::OvHead>> {
        self.head.ov()
    }

    /// Compile every attached OpenVINO backend (attention, fused MoE,
    /// per-expert, head) ahead of the first forward so their compile and
    /// first-shape costs land outside any timed region. Returns
    /// `(backends warmed, backends that failed)`.
    pub fn warm_ov_backends(&self) -> (usize, usize) {
        let (mut ok, mut bad) = (0usize, 0usize);
        let mut tally = |r: Option<bool>| match r {
            Some(true) => ok += 1,
            Some(false) => bad += 1,
            None => {}
        };
        for l in &self.layers {
            tally(l.warm_ov_attn());
            tally(l.warm_ov_moe());
            tally(l.warm_ov().map(|(_, failed)| failed.is_empty()));
        }
        tally(self.head.warm_ov());
        (ok, bad)
    }

    /// Free the Rust attention projections of every layer whose OpenVINO
    /// attention backend compiles (see `Layer::release_rust_attention_weights`);
    /// one entry per layer.
    pub fn release_rust_attention_weights(&mut self) -> Vec<Option<usize>> {
        self.layers
            .iter_mut()
            .map(|l| l.release_rust_attention_weights())
            .collect()
    }

    /// Route every MoE layer through a fused-MoE backend (layer index =
    /// position) — layers whose IR the backend lacks keep their path.
    /// Same full-0-based-model precondition as [`Self::attach_ov`].
    pub fn attach_ov_moe(&mut self, ov: Arc<super::ov_moe::OvMoe>) {
        for (i, l) in self.layers.iter_mut().enumerate() {
            if ov.has_layer(i as u32) {
                l.attach_ov_moe(i as u32, Arc::clone(&ov));
            }
        }
    }

    /// Route every layer's attention projections that have IRs through an
    /// OpenVINO backend (layer index = position).
    /// Same full-0-based-model precondition as [`Self::attach_ov`].
    pub fn attach_ov_attn(&mut self, ov: Arc<super::ov_attn::OvAttn>) {
        for (i, l) in self.layers.iter_mut().enumerate() {
            if ov.has_layer(i as u32) {
                l.attach_ov_attn(i as u32, Arc::clone(&ov));
            }
        }
    }

    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let mut x = self.embed_token(token);
        let early = self.early_prediction_reads_enabled();
        let mut pending_read = None;
        for index in 0..self.layers.len() {
            let next_read = self.layers.get(index + 1).and_then(|target| {
                if let Some(moe) = target.moe() {
                    if !early && self.previous_layer_route_observer.is_none() {
                        return None;
                    }
                    let mut predicted_input = x.clone();
                    rmsnorm_f32(&mut predicted_input, &target.mlp_norm, target.eps);
                    let prediction = moe.route_unobserved(&predicted_input);
                    if let Some(observer) = &self.previous_layer_route_observer {
                        observer(index + 1, &prediction);
                    }
                    if early {
                        return moe.start_predicted_read(&prediction);
                    }
                }
                None
            });
            // Suppress a second prediction even when the earlier lookup found
            // no uncached expert. Both current and future leases drain on unwind.
            x = self.layers[index].forward_token_with_prediction(
                &x,
                pending_read.take(),
                early && index > 0,
            );
            pending_read = next_read;
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
