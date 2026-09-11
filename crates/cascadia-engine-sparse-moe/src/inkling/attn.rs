//! Inkling attention — `InklingAttention`: GQA with per-head q/k RMSNorm,
//! causal short convs on k and v, a learned relative-position bias, `1/D`
//! scaling (q and k are RMS-normalised, hence not `1/sqrt(D)`), sliding
//! (`window`) or global keys, and — on global layers — log scaling.
//!
//! Per token at position `p` (`h` = the input-normed hidden, `[H]`):
//!
//! ```text
//!   q  = Wq h -> [Hq, D];   q_i = rmsnorm(q_i, q_norm)            per head
//!   kr = Wk h;  kc = sconv(k_sconv, kr) -> [Hkv, D];  k_i = rmsnorm(kc_i, k_norm)
//!   vr = Wv h;  v  = sconv(v_sconv, vr) -> [Hkv, D]
//!   r  = Wr h -> [Hq, d_rel]
//!   cache k, v at p (per kv head)
//!   tau = 1 + alpha · ln(max((p+1)/n_floor, 1))   global layers with log scaling, else 1
//!   q *= tau
//!   keys j: global 0..=p; sliding p-window < j <= p
//!   score[i, j] = (q_i · k_{i / (Hq/Hkv)}[j]) / D + tau · bias_i(p - j)
//!   a_i = softmax_j(score[i, :]) · v_{kv(i)}
//!   out = Wo concat_i(a_i) -> [H]                (the layer applies attn_sconv)
//! ```
//!
//! Numerics: bf16 round after each linear (`linear_bf16_w`); the head norms,
//! convs, bias, softmax and the p·v accumulate are f32; the KV cache is f32.
//!
//! Cache: sliding layers keep a `window + rewind` ring per kv head (slot =
//! `pos % rows`), global layers `max_seq` rows (slot = `pos`). The two conv
//! histories live inside the layer, so `reset` / `truncate` / `snapshot` /
//! `restore` cover the whole attention state.

use super::conv::{ConvState, ShortConv};
use super::relpos::RelPos;
use super::{rmsnorm_f32, DEFAULT_REWIND};
use crate::dsv4::math::{dot, linear_bf16_w};

/// Shape + behaviour of one attention layer.
#[derive(Clone, Debug)]
pub struct AttnDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    /// `Some(window)` for a sliding layer (keys `p - window < j <= p`), `None`
    /// for a global layer (all keys).
    pub window: Option<usize>,
    /// KV rows for a global layer (the context cap); ignored for sliding.
    pub max_seq: usize,
    /// RMSNorm eps for the q/k head norms.
    pub eps: f32,
    /// `log_scaling_n_floor` — log scaling applies only on global layers, so a
    /// sliding layer may carry the manifest value harmlessly.
    pub n_floor: Option<f32>,
    /// `log_scaling_alpha`.
    pub alpha: f32,
    /// Extra ring rows for [`AttentionLayer::truncate`] on a sliding layer
    /// ([`DEFAULT_REWIND`]); must match the convs' rewind for a uniform bound.
    pub rewind: usize,
}

impl AttnDims {
    /// A global layer with the default rewind slack.
    pub fn global(
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_rel: usize,
        max_seq: usize,
        eps: f32,
    ) -> Self {
        Self {
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            d_rel,
            window: None,
            max_seq,
            eps,
            n_floor: None,
            alpha: 0.0,
            rewind: DEFAULT_REWIND,
        }
    }

    /// A sliding layer (`window` keys) with the default rewind slack.
    pub fn sliding(
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        d_rel: usize,
        window: usize,
        eps: f32,
    ) -> Self {
        Self {
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            d_rel,
            window: Some(window),
            max_seq: 0,
            eps,
            n_floor: None,
            alpha: 0.0,
            rewind: DEFAULT_REWIND,
        }
    }

    /// Attach log scaling (`n_floor`, `alpha`); a no-op on sliding layers.
    pub fn with_log_scaling(mut self, n_floor: f32, alpha: f32) -> Self {
        self.n_floor = Some(n_floor);
        self.alpha = alpha;
        self
    }

    pub fn with_rewind(mut self, rewind: usize) -> Self {
        self.rewind = rewind;
        self
    }
}

/// Projection weights as bf16 bits (batch-1 GEMVs are bandwidth-bound and the
/// checkpoint is bf16-native) plus the f32 head-norm weights.
pub struct AttnWeights {
    pub wq: Vec<u16>,     // [Hq·D, H]      `attn.wq_du.weight`
    pub wk: Vec<u16>,     // [Hkv·D, H]     `attn.wk_dv.weight`
    pub wv: Vec<u16>,     // [Hkv·D, H]     `attn.wv_dv.weight`
    pub wr: Vec<u16>,     // [Hq·d_rel, H]  `attn.wr_du.weight`
    pub wo: Vec<u16>,     // [H, Hq·D]      `attn.wo_ud.weight`
    pub q_norm: Vec<f32>, // [D]
    pub k_norm: Vec<f32>, // [D]
}

pub struct AttentionLayer {
    pub dims: AttnDims,
    scale: f32,
    w: AttnWeights,
    k_sconv: ShortConv,
    v_sconv: ShortConv,
    relpos: RelPos,
    /// Cache rows per kv head: `window + rewind` (sliding) or `max_seq` (global).
    rows: usize,
    k: Vec<f32>, // [Hkv, rows, D]
    v: Vec<f32>, // [Hkv, rows, D]
    len: usize,
}

/// A saved attention state for prefix reuse: the cached k/v rows at positions
/// `[first, len)` (position-major, kv-head minor, `D` floats each) plus the
/// k/v conv histories. `first` is 0 for a global layer and the ring's oldest
/// valid position for a sliding one.
#[derive(Clone)]
pub struct AttnKv {
    len: usize,
    first: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    k_conv: ConvState,
    v_conv: ConvState,
}

impl AttnKv {
    /// Cached positions this snapshot covers.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Payload size in bytes.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
            + self.k_conv.bytes()
            + self.v_conv.bytes()
    }
}

impl AttentionLayer {
    /// Assemble a layer from its parts. `k_sconv` / `v_sconv` must have
    /// `c == Hkv·D`; `relpos.d_rel == dims.d_rel` (its `extent` is the layer's
    /// bias range: the window on sliding layers, `rel_extent` on global ones).
    pub fn from_parts(
        dims: AttnDims,
        w: AttnWeights,
        k_sconv: ShortConv,
        v_sconv: ShortConv,
        relpos: RelPos,
    ) -> Self {
        let (hd, hq, hkv, d, dr) = (
            dims.hidden,
            dims.n_heads,
            dims.n_kv_heads,
            dims.head_dim,
            dims.d_rel,
        );
        assert!(hq > 0 && hkv > 0 && d > 0, "attn: empty head dims");
        assert_eq!(
            hq % hkv,
            0,
            "attn: n_heads must be a multiple of n_kv_heads"
        );
        assert_eq!(w.wq.len(), hq * d * hd, "attn: wq shape");
        assert_eq!(w.wk.len(), hkv * d * hd, "attn: wk shape");
        assert_eq!(w.wv.len(), hkv * d * hd, "attn: wv shape");
        assert_eq!(w.wr.len(), hq * dr * hd, "attn: wr shape");
        assert_eq!(w.wo.len(), hd * hq * d, "attn: wo shape");
        assert_eq!(w.q_norm.len(), d, "attn: q_norm len != head_dim");
        assert_eq!(w.k_norm.len(), d, "attn: k_norm len != head_dim");
        assert_eq!(k_sconv.c, hkv * d, "attn: k_sconv channels != Hkv·D");
        assert_eq!(v_sconv.c, hkv * d, "attn: v_sconv channels != Hkv·D");
        assert_eq!(relpos.d_rel, dr, "attn: relpos d_rel mismatch");
        let rows = match dims.window {
            Some(win) => {
                assert!(win >= 1, "attn: sliding window must be >= 1");
                win + dims.rewind
            }
            None => {
                assert!(dims.max_seq >= 1, "attn: global layer needs max_seq >= 1");
                dims.max_seq
            }
        };
        Self {
            scale: 1.0 / d as f32,
            dims,
            w,
            k_sconv,
            v_sconv,
            relpos,
            rows,
            k: vec![0.0; hkv * rows * d],
            v: vec![0.0; hkv * rows * d],
            len: 0,
        }
    }

    /// Cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `Some(window)` for a sliding layer.
    pub fn window(&self) -> Option<usize> {
        self.dims.window
    }

    /// Bytes held by the KV cache + the two conv histories (allocated up front,
    /// independent of the current length): `2·Hkv·rows·D·4 + convs`, with
    /// `rows = window + rewind` (sliding) or `max_seq` (global).
    pub fn cache_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
            + self.k_sconv.cache_bytes()
            + self.v_sconv.cache_bytes()
    }

    #[inline]
    fn slot(&self, pos: usize) -> usize {
        match self.dims.window {
            Some(_) => pos % self.rows,
            None => pos,
        }
    }

    #[inline]
    fn kv_off(&self, kvh: usize, slot: usize) -> usize {
        (kvh * self.rows + slot) * self.dims.head_dim
    }

    /// Clear the KV cache and conv histories (new sequence).
    pub fn reset(&mut self) {
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.len = 0;
        self.k_sconv.reset();
        self.v_sconv.reset();
    }

    /// Roll back to `len` positions (spec-decode reject). O(1). A sliding
    /// layer can rewind at most `rewind` positions (its ring has overwritten
    /// older rows); the convs enforce the same bound.
    pub fn truncate(&mut self, len: usize) {
        assert!(
            len <= self.len,
            "attn truncate({len}) beyond current len {}",
            self.len
        );
        if self.dims.window.is_some() {
            assert!(
                self.len - len <= self.dims.rewind,
                "attn truncate: rewinding {} positions exceeds the sliding ring's rewind slack {}",
                self.len - len,
                self.dims.rewind
            );
        }
        self.len = len;
        self.k_sconv.truncate(len);
        self.v_sconv.truncate(len);
    }

    /// Snapshot the cached k/v (the valid window on a sliding layer, all
    /// positions on a global one) and the conv histories.
    pub fn snapshot(&self) -> AttnKv {
        let (hkv, d) = (self.dims.n_kv_heads, self.dims.head_dim);
        let first = match self.dims.window {
            Some(_) => self.len.saturating_sub(self.rows),
            None => 0,
        };
        let n = self.len - first;
        let mut k = Vec::with_capacity(n * hkv * d);
        let mut v = Vec::with_capacity(n * hkv * d);
        for p in first..self.len {
            let s = self.slot(p);
            for kvh in 0..hkv {
                let o = self.kv_off(kvh, s);
                k.extend_from_slice(&self.k[o..o + d]);
                v.extend_from_slice(&self.v[o..o + d]);
            }
        }
        AttnKv {
            len: self.len,
            first,
            k,
            v,
            k_conv: self.k_sconv.snapshot(),
            v_conv: self.v_sconv.snapshot(),
        }
    }

    /// Restore a snapshot (call after [`Self::reset`]); replaces the length,
    /// the cached rows and the conv histories. Snapshot and layer must share
    /// dims.
    pub fn restore(&mut self, kv: &AttnKv) {
        let (hkv, d) = (self.dims.n_kv_heads, self.dims.head_dim);
        let n = kv.len - kv.first;
        assert_eq!(kv.k.len(), n * hkv * d, "attn restore: k payload shape");
        assert_eq!(kv.v.len(), n * hkv * d, "attn restore: v payload shape");
        if self.dims.window.is_none() {
            assert!(
                kv.len <= self.rows,
                "attn restore: snapshot len {} exceeds max_seq {}",
                kv.len,
                self.rows
            );
        }
        self.len = kv.len;
        let start = kv.first.max(kv.len.saturating_sub(self.rows));
        for p in start..kv.len {
            let s = self.slot(p);
            for kvh in 0..hkv {
                let o = self.kv_off(kvh, s);
                let src = ((p - kv.first) * hkv + kvh) * d;
                self.k[o..o + d].copy_from_slice(&kv.k[src..src + d]);
                self.v[o..o + d].copy_from_slice(&kv.v[src..src + d]);
            }
        }
        self.k_sconv.restore(&kv.k_conv);
        self.v_sconv.restore(&kv.v_conv);
    }

    /// The four projections of one input-normed hidden `h` (`[H]`), each
    /// bf16-rounded: `(q, k_raw, v_raw, r)`.
    fn project(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (hd, hq, hkv, d, dr) = (
            self.dims.hidden,
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
        );
        let mut q = vec![0.0f32; hq * d];
        linear_bf16_w(h, &self.w.wq, hq * d, hd, &mut q);
        let mut kr = vec![0.0f32; hkv * d];
        linear_bf16_w(h, &self.w.wk, hkv * d, hd, &mut kr);
        let mut vr = vec![0.0f32; hkv * d];
        linear_bf16_w(h, &self.w.wv, hkv * d, hd, &mut vr);
        let mut r = vec![0.0f32; hq * dr];
        linear_bf16_w(h, &self.w.wr, hq * dr, hd, &mut r);
        (q, kr, vr, r)
    }

    /// The per-position core shared by decode and prefill: head-norm `q` and
    /// the conv'd `kc`, append k/v to the cache at position `self.len`, attend
    /// (with log scaling on global layers), and project out. `q`/`kc` are
    /// taken by value because they are normalised in place.
    fn attend(&mut self, mut q: Vec<f32>, mut kc: Vec<f32>, v: &[f32], r: &[f32]) -> Vec<f32> {
        let (hd, hq, hkv, d, dr) = (
            self.dims.hidden,
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
        );
        let p = self.len;
        if self.dims.window.is_none() {
            assert!(
                p < self.rows,
                "Inkling context length {} exceeds max_seq {}; raise the global layers' max_seq",
                p + 1,
                self.rows
            );
        }

        // Per-head RMSNorm on q and (conv'd) k, f32.
        rmsnorm_f32(&mut q, &self.w.q_norm, self.dims.eps);
        rmsnorm_f32(&mut kc, &self.w.k_norm, self.dims.eps);

        // Cache this position's k / v per kv head.
        let slot = self.slot(p);
        for kvh in 0..hkv {
            let o = self.kv_off(kvh, slot);
            self.k[o..o + d].copy_from_slice(&kc[kvh * d..(kvh + 1) * d]);
            self.v[o..o + d].copy_from_slice(&v[kvh * d..(kvh + 1) * d]);
        }
        self.len = p + 1;

        // Log scaling (global layers only): scales q and the position bias.
        let tau = match (self.dims.window, self.dims.n_floor) {
            (None, Some(nf)) => 1.0 + self.dims.alpha * ((p + 1) as f32 / nf).max(1.0).ln(),
            _ => 1.0,
        };
        if tau != 1.0 {
            for qi in q.iter_mut() {
                *qi *= tau;
            }
        }

        // Visible keys: global 0..=p; sliding p-window < j <= p.
        let j0 = match self.dims.window {
            Some(win) => (p + 1).saturating_sub(win),
            None => 0,
        };
        let n_keys = p + 1 - j0;
        let group = hq / hkv;
        let extent = self.relpos.extent;

        let mut ctx = vec![0.0f32; hq * d];
        let mut score = vec![0.0f32; n_keys];
        for h in 0..hq {
            let kvh = h / group;
            let qh = &q[h * d..(h + 1) * d];
            let prof = self.relpos.profile(&r[h * dr..(h + 1) * dr]);
            let mut smax = f32::NEG_INFINITY;
            for (i, j) in (j0..=p).enumerate() {
                let o = self.kv_off(kvh, self.slot(j));
                let dist = p - j;
                let bias = if dist < extent { prof[dist] * tau } else { 0.0 };
                let s = dot(qh, &self.k[o..o + d]) * self.scale + bias;
                score[i] = s;
                smax = smax.max(s);
            }
            let mut denom = 0.0f32;
            for s in score.iter_mut() {
                *s = (*s - smax).exp();
                denom += *s;
            }
            let ctx_h = &mut ctx[h * d..(h + 1) * d];
            for (i, j) in (j0..=p).enumerate() {
                let pj = score[i] / denom;
                let o = self.kv_off(kvh, self.slot(j));
                for (c, &x) in ctx_h.iter_mut().zip(&self.v[o..o + d]) {
                    *c += pj * x;
                }
            }
        }

        let mut out = vec![0.0f32; hd];
        linear_bf16_w(&ctx, &self.w.wo, hd, hq * d, &mut out);
        out
    }

    /// Attend one input-normed hidden `h` (`[H]`) at position `self.len`,
    /// appending to the cache. Returns the `Wo` output (`[H]`) BEFORE
    /// `attn_sconv` — the layer applies that conv and the residual.
    pub fn forward_token(&mut self, h: &[f32]) -> Vec<f32> {
        assert_eq!(h.len(), self.dims.hidden, "attn forward_token: h len");
        let (q, kr, vr, r) = self.project(h);
        let kc = self.k_sconv.decode(&kr);
        let v = self.v_sconv.decode(&vr);
        self.attend(q, kc, &v, &r)
    }

    /// Prefill `t` input-normed rows (`hs` = `[t, H]`) starting at position
    /// `self.len`; returns `[t, H]`. Projections run per row, the k/v convs as
    /// one batched prefill, attention per position (the causal cache must grow
    /// in order). Bit-identical to `t` calls of [`Self::forward_token`].
    pub fn forward_prefill(&mut self, hs: &[f32], t: usize) -> Vec<f32> {
        let (hd, hq, hkv, d, dr) = (
            self.dims.hidden,
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
        );
        assert_eq!(
            hs.len(),
            t * hd,
            "attn forward_prefill: hs len != t * hidden"
        );
        let (qd, kd, rd) = (hq * d, hkv * d, hq * dr);
        let mut q_all = vec![0.0f32; t * qd];
        let mut kr_all = vec![0.0f32; t * kd];
        let mut vr_all = vec![0.0f32; t * kd];
        let mut r_all = vec![0.0f32; t * rd];
        for (row, h) in hs.chunks_exact(hd).enumerate() {
            let (q, kr, vr, r) = self.project(h);
            q_all[row * qd..(row + 1) * qd].copy_from_slice(&q);
            kr_all[row * kd..(row + 1) * kd].copy_from_slice(&kr);
            vr_all[row * kd..(row + 1) * kd].copy_from_slice(&vr);
            r_all[row * rd..(row + 1) * rd].copy_from_slice(&r);
        }
        let kc_all = self.k_sconv.prefill(&kr_all, t);
        let v_all = self.v_sconv.prefill(&vr_all, t);
        let mut out = vec![0.0f32; t * hd];
        for row in 0..t {
            let o = self.attend(
                q_all[row * qd..(row + 1) * qd].to_vec(),
                kc_all[row * kd..(row + 1) * kd].to_vec(),
                &v_all[row * kd..(row + 1) * kd],
                &r_all[row * rd..(row + 1) * rd],
            );
            out[row * hd..(row + 1) * hd].copy_from_slice(&o);
        }
        out
    }
}
