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
//!
//! Ring validity follows [`ShortConv`](super::conv): `hwm` is one past the
//! furthest position written since the last `reset` / `restore`; a sliding
//! ring holds position `j` iff `j >= hwm - rows`. Attending at `p` reads keys
//! `[p + 1 - window, p]` (`p` itself is written first), inside that window iff
//! `hwm - p <= rewind + 1`; `truncate` enforces the uniform `hwm - len <=
//! rewind` bound shared with the convs. A global layer never wraps (slot =
//! position) and reads only `[0, p]`. Reads therefore never leave the rows
//! written since the last reset, so `reset` is O(1) with no zeroing.

use super::conv::{ConvState, ShortConv};
use super::relpos::RelPos;
use super::{rmsnorm_f32, DEFAULT_REWIND};
use crate::dsv4::math::{dot, linear_bf16_w};
use crate::dsv4::st::MappedBf16;

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
    /// Parked KV states for multi-stream decode (see [`Self::select`]);
    /// empty on the single-sequence path. The k/v convs keep their own pools.
    slots: Vec<AttnSlot>,
    /// Which parked slot the live `k/v/len/hwm` currently belong to.
    live: usize,
    /// Optional OpenVINO backend for the five projections (`(layer, backend)`);
    /// see [`super::ov_attn`].
    ov: Option<(u32, std::sync::Arc<super::ov_attn::OvAttn>)>,
    dims: AttnDims,
    scale: f32,
    w: AttnWeights,
    mapped_projections: Option<[MappedBf16; 5]>,
    k_sconv: ShortConv,
    v_sconv: ShortConv,
    relpos: RelPos,
    /// Cache rows per kv head: `window + rewind` (sliding) or `max_seq` (global).
    rows: usize,
    k: Vec<f32>, // [Hkv, rows, D]
    v: Vec<f32>, // [Hkv, rows, D]
    len: usize,
    /// Write high-water mark: one past the furthest position written since the
    /// last `reset` / `restore` (`>= len`; module docs).
    hwm: usize,
}

/// One parked sequence's attention state (multi-stream decode): the full
/// k/v cache buffers plus `len`/`hwm`, swapped whole with the live state by
/// [`AttentionLayer::select`] (pointer swaps — no copying).
struct AttnSlot {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    hwm: usize,
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
        Self::from_parts_with_mapped(dims, w, k_sconv, v_sconv, relpos, None)
    }

    pub(crate) fn from_parts_with_mapped(
        dims: AttnDims,
        w: AttnWeights,
        k_sconv: ShortConv,
        v_sconv: ShortConv,
        relpos: RelPos,
        mapped_projections: Option<[MappedBf16; 5]>,
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
        let owned = [&w.wq, &w.wk, &w.wv, &w.wr, &w.wo];
        let expected = [
            hq * d * hd,
            hkv * d * hd,
            hkv * d * hd,
            hq * dr * hd,
            hd * hq * d,
        ];
        for (i, size) in expected.into_iter().enumerate() {
            let weights = mapped_projections
                .as_ref()
                .map_or(owned[i].as_slice(), |m| m[i].as_slice());
            assert_eq!(weights.len(), size, "attn: projection {i} shape");
        }
        assert_eq!(w.q_norm.len(), d, "attn: q_norm len != head_dim");
        assert_eq!(w.k_norm.len(), d, "attn: k_norm len != head_dim");
        assert_eq!(k_sconv.c(), hkv * d, "attn: k_sconv channels != Hkv·D");
        assert_eq!(v_sconv.c(), hkv * d, "attn: v_sconv channels != Hkv·D");
        assert_eq!(relpos.d_rel(), dr, "attn: relpos d_rel mismatch");
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
            ov: None,
            slots: Vec::new(),
            live: 0,
            scale: 1.0 / d as f32,
            dims,
            w,
            mapped_projections,
            k_sconv,
            v_sconv,
            relpos,
            rows,
            k: vec![0.0; hkv * rows * d],
            v: vec![0.0; hkv * rows * d],
            len: 0,
            hwm: 0,
        }
    }

    /// Cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Shape + behaviour of this layer (read-only).
    pub fn dims(&self) -> &AttnDims {
        &self.dims
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

    /// Size the multi-stream slot pool to `n` sequences: each slot owns a full
    /// k/v cache (`cache_bytes` each) and conv histories. Slot 0 is the state
    /// live at the call. Growing keeps existing slots.
    pub fn ensure_slots(&mut self, n: usize) {
        while self.slots.len() < n {
            self.slots.push(AttnSlot {
                k: vec![0.0; self.k.len()],
                v: vec![0.0; self.v.len()],
                len: 0,
                hwm: 0,
            });
        }
        self.k_sconv.ensure_slots(n);
        self.v_sconv.ensure_slots(n);
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Make sequence `slot`'s KV cache and conv histories the live ones,
    /// parking the current sequence's. O(1) pointer swaps.
    pub fn select(&mut self, slot: usize) {
        assert!(
            slot < self.slots.len(),
            "AttentionLayer::select({slot}): pool holds {} slots",
            self.slots.len()
        );
        self.k_sconv.select(slot);
        self.v_sconv.select(slot);
        if slot == self.live {
            return;
        }
        let cur = self.live;
        std::mem::swap(&mut self.k, &mut self.slots[cur].k);
        std::mem::swap(&mut self.v, &mut self.slots[cur].v);
        std::mem::swap(&mut self.len, &mut self.slots[cur].len);
        std::mem::swap(&mut self.hwm, &mut self.slots[cur].hwm);
        std::mem::swap(&mut self.k, &mut self.slots[slot].k);
        std::mem::swap(&mut self.v, &mut self.slots[slot].v);
        std::mem::swap(&mut self.len, &mut self.slots[slot].len);
        std::mem::swap(&mut self.hwm, &mut self.slots[slot].hwm);
        self.live = slot;
    }

    /// Clear the KV cache and conv histories (new sequence). O(1): nothing is
    /// zeroed — with `hwm = 0` no cached row is inside the valid window, and
    /// attention only reads rows written since (module docs).
    pub fn reset(&mut self) {
        self.len = 0;
        self.hwm = 0;
        self.k_sconv.reset();
        self.v_sconv.reset();
    }

    /// Roll back to `len` positions (spec-decode reject). O(1). A sliding
    /// layer can rewind at most `rewind` positions below the write high-water
    /// mark — across any number of truncates, since the discarded positions
    /// have already overwritten the ring's older rows; the convs enforce the
    /// same bound. Panics beyond it.
    pub fn truncate(&mut self, len: usize) {
        assert!(
            len <= self.len,
            "attn truncate({len}) beyond current len {}",
            self.len
        );
        if self.dims.window.is_some() {
            let stale = self.hwm - len;
            assert!(
                stale <= self.dims.rewind || self.hwm <= self.rows,
                "attn truncate({len}): {stale} positions were written past it (high-water mark \
                 {}) which exceeds the sliding ring's rewind slack {} — the keys position {len} \
                 needs have been overwritten",
                self.hwm,
                self.dims.rewind
            );
        }
        self.len = len;
        self.k_sconv.truncate(len);
        self.v_sconv.truncate(len);
    }

    /// Snapshot the cached k/v (the valid ring window `[hwm - rows, len)` on a
    /// sliding layer, all positions on a global one) and the conv histories.
    pub fn snapshot(&self) -> AttnKv {
        let (hkv, d) = (self.dims.n_kv_heads, self.dims.head_dim);
        let first = match self.dims.window {
            Some(_) => self.hwm.saturating_sub(self.rows).min(self.len),
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

    /// Restore a snapshot; replaces the length, the cached rows, the conv
    /// histories and the high-water mark (so the rewind bound after a restore
    /// is what the snapshot's window supports). Snapshot and layer must share
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
        self.len = kv.len;
        // Valid window = [start, len): `hwm - rows == start`, or everything
        // from 0 when the restored rows fit without wrapping.
        self.hwm = if start == 0 {
            kv.len
        } else {
            start + self.rows
        };
        self.k_sconv.restore(&kv.k_conv);
        self.v_sconv.restore(&kv.v_conv);
    }

    /// The four projections of one input-normed hidden `h` (`[H]`), each
    /// bf16-rounded: `(q, k_raw, v_raw, r)`.
    /// Route the five projections through an OpenVINO backend (see
    /// [`super::ov_attn`]); `layer` is the global layer index its IRs are
    /// filed under.
    pub fn attach_ov(&mut self, layer: u32, ov: std::sync::Arc<super::ov_attn::OvAttn>) {
        self.ov = Some((layer, ov));
    }

    pub fn ov(&self) -> Option<(u32, &std::sync::Arc<super::ov_attn::OvAttn>)> {
        self.ov.as_ref().map(|(l, o)| (*l, o))
    }

    /// Compile this layer's projection IRs and take their first-call cost.
    pub fn warm_ov(&self) -> Option<bool> {
        let (lid, ov) = self.ov.as_ref()?;
        Some(ov.warm(
            *lid,
            self.dims.hidden,
            self.dims.n_heads * self.dims.head_dim,
        ))
    }

    /// `q`, `k`, `v`, `r` for `t` rows (`hs` = `[t, H]`): one backend call
    /// when attached (and it answers), else `t` Rust projections.
    /// Free the five bf16 projection tables once an OpenVINO backend serves
    /// them (`CASCADIA_INKLING_OV_ATTN_DROP_RUST=1`): 264 MB per layer that
    /// would otherwise sit next to the device copy in unified memory. Returns
    /// the bytes released. After this a refused backend call is fatal (there
    /// is nothing left to fall back to), which the projection paths report.
    pub fn release_rust_projections(&mut self) -> usize {
        assert!(
            self.ov.is_some(),
            "release_rust_projections without an OpenVINO attention backend"
        );
        let w = &mut self.w;
        let bytes = 2 * (w.wq.len() + w.wk.len() + w.wv.len() + w.wr.len() + w.wo.len())
            + self
                .mapped_projections
                .take()
                .map_or(0, |m| m.iter().map(|p| p.as_slice().len() * 2).sum());
        for t in [&mut w.wq, &mut w.wk, &mut w.wv, &mut w.wr, &mut w.wo] {
            *t = Vec::new();
        }
        bytes
    }

    fn rust_projections_released(&self) -> bool {
        self.mapped_projections.is_none() && self.w.wq.is_empty() && self.dims.hidden > 0
    }

    fn projection(&self, index: usize) -> &[u16] {
        self.mapped_projections.as_ref().map_or_else(
            || [&self.w.wq, &self.w.wk, &self.w.wv, &self.w.wr, &self.w.wo][index].as_slice(),
            |m| m[index].as_slice(),
        )
    }

    fn project_rows(&self, hs: &[f32], t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        if let Some((lid, ov)) = &self.ov {
            if let Some([q, k, v, r]) = ov.qkvr(*lid, hs, t) {
                return (q, k, v, r);
            }
            assert!(
                !self.rust_projections_released(),
                "inkling layer {lid}: the OpenVINO attention backend refused a call after \
                 the Rust projections were released (CASCADIA_INKLING_OV_ATTN_DROP_RUST=1)"
            );
        }
        let (hd, hq, hkv, d, dr) = (
            self.dims.hidden,
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
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
        (q_all, kr_all, vr_all, r_all)
    }

    /// The output projection for `t` context rows (`ctx` = `[t, Hq·D]`).
    fn project_out_rows(&self, ctx: &[f32], t: usize) -> Vec<f32> {
        if let Some((lid, ov)) = &self.ov {
            if let Some(y) = ov.o(*lid, ctx, t) {
                return y;
            }
            assert!(
                !self.rust_projections_released(),
                "inkling layer {lid}: the OpenVINO attention backend refused a call after \
                 the Rust projections were released (CASCADIA_INKLING_OV_ATTN_DROP_RUST=1)"
            );
        }
        let (hd, hq, d) = (self.dims.hidden, self.dims.n_heads, self.dims.head_dim);
        let mut out = vec![0.0f32; t * hd];
        for (row, c) in ctx.chunks_exact(hq * d).enumerate() {
            linear_bf16_w(
                c,
                self.projection(4),
                hd,
                hq * d,
                &mut out[row * hd..(row + 1) * hd],
            );
        }
        if let Some(maps) = &self.mapped_projections {
            for map in maps {
                map.trim_working_set();
            }
        }
        out
    }

    fn project(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (hd, hq, hkv, d, dr) = (
            self.dims.hidden,
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
        );
        let mut q = vec![0.0f32; hq * d];
        linear_bf16_w(h, self.projection(0), hq * d, hd, &mut q);
        let mut kr = vec![0.0f32; hkv * d];
        linear_bf16_w(h, self.projection(1), hkv * d, hd, &mut kr);
        let mut vr = vec![0.0f32; hkv * d];
        linear_bf16_w(h, self.projection(2), hkv * d, hd, &mut vr);
        let mut r = vec![0.0f32; hq * dr];
        linear_bf16_w(h, self.projection(3), hq * dr, hd, &mut r);
        (q, kr, vr, r)
    }

    /// The per-position core shared by decode and prefill: head-norm `q` and
    /// the conv'd `kc`, append k/v to the cache at position `self.len`, attend
    /// (with log scaling on global layers), and project out. `q`/`kc` are
    /// taken by value because they are normalised in place.
    fn attend(&mut self, q: Vec<f32>, kc: Vec<f32>, v: &[f32], r: &[f32]) -> Vec<f32> {
        let ctx = AttnCtx {
            dims: &self.dims,
            scale: self.scale,
            rows: self.rows,
            relpos: &self.relpos,
            q_norm: &self.w.q_norm,
            k_norm: &self.w.k_norm,
        };
        attend_state(
            &ctx,
            &mut self.k,
            &mut self.v,
            &mut self.len,
            &mut self.hwm,
            q,
            kc,
            v,
            r,
        )
    }
}

/// `CASCADIA_INKLING_PAR_ATTN` (default on): a frame's rows attend
/// concurrently. Each row belongs to a different sequence with its own KV
/// cache and conv histories, so the rows are independent; they used to run one
/// after another, which was invisible while the experts kept every core busy
/// and became a fifth of a frame once the experts moved to the iGPU.
fn row_parallel_attention() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("CASCADIA_INKLING_PAR_ATTN")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

impl AttentionLayer {
    /// The per-row part of [`Self::forward_rows`] with the rows concurrent:
    /// k/v convs, then attention, each on its own sequence's state. Bit for
    /// bit what the sequential loop computes per row. Returns `[t, Hq·D]`.
    fn attend_rows_parallel(
        &mut self,
        q_all: &[f32],
        kr_all: &[f32],
        vr_all: &[f32],
        r_all: &[f32],
        slots: &[usize],
    ) -> Vec<f32> {
        use rayon::prelude::*;
        let (hq, hkv, d, dr) = (
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
        );
        let (qd, kd, rd) = (hq * d, hkv * d, hq * dr);
        let kc_all = self.k_sconv.decode_slots(kr_all, slots);
        let v_all = self.v_sconv.decode_slots(vr_all, slots);
        // Every sequence's KV state into its slot entry (the live one sits in
        // the layer's own fields), back again afterwards.
        let live = self.live;
        std::mem::swap(&mut self.k, &mut self.slots[live].k);
        std::mem::swap(&mut self.v, &mut self.slots[live].v);
        std::mem::swap(&mut self.len, &mut self.slots[live].len);
        std::mem::swap(&mut self.hwm, &mut self.slots[live].hwm);
        let mut ctx_all = vec![0.0f32; slots.len() * qd];
        {
            let cx = AttnCtx {
                dims: &self.dims,
                scale: self.scale,
                rows: self.rows,
                relpos: &self.relpos,
                q_norm: &self.w.q_norm,
                k_norm: &self.w.k_norm,
            };
            let mut picked: Vec<Option<&mut AttnSlot>> = slots.iter().map(|_| None).collect();
            for (i, st) in self.slots.iter_mut().enumerate() {
                if let Some(row) = slots.iter().position(|&s| s == i) {
                    picked[row] = Some(st);
                }
            }
            picked
                .into_par_iter()
                .zip(ctx_all.par_chunks_mut(qd))
                .enumerate()
                .for_each(|(row, (st, out))| {
                    let st = st.expect("attention: slot out of range or listed twice");
                    let c = attend_state(
                        &cx,
                        &mut st.k,
                        &mut st.v,
                        &mut st.len,
                        &mut st.hwm,
                        q_all[row * qd..(row + 1) * qd].to_vec(),
                        kc_all[row * kd..(row + 1) * kd].to_vec(),
                        &v_all[row * kd..(row + 1) * kd],
                        &r_all[row * rd..(row + 1) * rd],
                    );
                    out.copy_from_slice(&c);
                });
        }
        std::mem::swap(&mut self.k, &mut self.slots[live].k);
        std::mem::swap(&mut self.v, &mut self.slots[live].v);
        std::mem::swap(&mut self.len, &mut self.slots[live].len);
        std::mem::swap(&mut self.hwm, &mut self.slots[live].hwm);
        ctx_all
    }
}

/// Private scratch for an opt-in overlap measurement. Borrows only weights;
/// no serving KV/conv state or projections are touched.
pub(super) struct CpuProbe<'a> {
    ctx: AttnCtx<'a>,
    slots: Vec<CpuProbeSlot>,
    context: usize,
    q: Vec<f32>,
    kr: Vec<f32>,
    vr: Vec<f32>,
    r: Vec<f32>,
}

struct CpuProbeSlot {
    k: Vec<f32>,
    v: Vec<f32>,
    kc: ShortConv,
    vc: ShortConv,
}

impl AttentionLayer {
    pub(super) fn cpu_probe(&self, rows: usize, context: usize) -> CpuProbe<'_> {
        assert!((1..=2).contains(&rows));
        let context = context.min(self.rows.saturating_sub(1));
        let cache_rows = context + 1;
        let kd = self.dims.n_kv_heads * self.dims.head_dim;
        let values = |n: usize| {
            (0..n)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
                .collect::<Vec<_>>()
        };
        let kr = values(kd);
        let vr = values(kd);
        let slots = (0..rows)
            .map(|_| {
                let mut kc = ShortConv::new(self.k_sconv.w().to_vec(), kd, self.k_sconv.k());
                let mut vc = ShortConv::new(self.v_sconv.w().to_vec(), kd, self.v_sconv.k());
                for _ in 0..3 {
                    kc.decode(&kr);
                    vc.decode(&vr);
                }
                CpuProbeSlot {
                    k: values(kd * cache_rows),
                    v: values(kd * cache_rows),
                    kc,
                    vc,
                }
            })
            .collect();
        CpuProbe {
            ctx: AttnCtx {
                dims: &self.dims,
                scale: self.scale,
                rows: cache_rows,
                relpos: &self.relpos,
                q_norm: &self.w.q_norm,
                k_norm: &self.w.k_norm,
            },
            slots,
            context,
            q: values(self.dims.n_heads * self.dims.head_dim),
            r: values(self.dims.n_heads * self.dims.d_rel),
            kr,
            vr,
        }
    }
}

impl CpuProbe<'_> {
    pub(super) fn step(&mut self) -> Vec<f32> {
        use rayon::prelude::*;
        let run = |slot: &mut CpuProbeSlot| {
            slot.kc.truncate(3);
            slot.vc.truncate(3);
            let kc = slot.kc.decode(&self.kr);
            let v = slot.vc.decode(&self.vr);
            let (mut len, mut hwm) = (self.context, self.context);
            attend_state(
                &self.ctx,
                &mut slot.k,
                &mut slot.v,
                &mut len,
                &mut hwm,
                self.q.clone(),
                kc,
                &v,
                &self.r,
            )
        };
        let outputs: Vec<Vec<f32>> = if self.slots.len() > 1 && row_parallel_attention() {
            self.slots.par_iter_mut().map(run).collect()
        } else {
            self.slots.iter_mut().map(run).collect()
        };
        outputs.into_iter().flatten().collect()
    }
}

/// What [`attend_state`] reads of a layer besides one sequence's state.
struct AttnCtx<'a> {
    dims: &'a AttnDims,
    scale: f32,
    rows: usize,
    relpos: &'a RelPos,
    q_norm: &'a [f32],
    k_norm: &'a [f32],
}

impl AttnCtx<'_> {
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
}

/// The per-position attention core on ONE sequence's state (`k`/`v` caches,
/// `len`, `hwm`), whichever sequence that is: the layer's live one
/// ([`AttentionLayer::attend`]) or a parked slot (row-parallel decode).
#[allow(clippy::too_many_arguments)]
fn attend_state(
    cx: &AttnCtx<'_>,
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    len: &mut usize,
    hwm: &mut usize,
    mut q: Vec<f32>,
    mut kc: Vec<f32>,
    v: &[f32],
    r: &[f32],
) -> Vec<f32> {
    {
        let (hd, hq, hkv, d, dr) = (
            cx.dims.hidden,
            cx.dims.n_heads,
            cx.dims.n_kv_heads,
            cx.dims.head_dim,
            cx.dims.d_rel,
        );
        let p = *len;
        if cx.dims.window.is_none() {
            assert!(
                p < cx.rows,
                "Inkling context length {} exceeds max_seq {}; raise the global layers' max_seq",
                p + 1,
                cx.rows
            );
        } else {
            debug_assert!(
                (p + 1).saturating_sub(cx.dims.window.unwrap_or(0))
                    >= (*hwm).saturating_sub(cx.rows),
                "attn ring invariant broken at position {p} (hwm {})",
                (*hwm)
            );
        }

        // Per-head RMSNorm on q and (conv'd) k, f32.
        rmsnorm_f32(&mut q, &cx.q_norm, cx.dims.eps);
        rmsnorm_f32(&mut kc, &cx.k_norm, cx.dims.eps);

        // Cache this position's k / v per kv head.
        let slot = cx.slot(p);
        for kvh in 0..hkv {
            let o = cx.kv_off(kvh, slot);
            k_cache[o..o + d].copy_from_slice(&kc[kvh * d..(kvh + 1) * d]);
            v_cache[o..o + d].copy_from_slice(&v[kvh * d..(kvh + 1) * d]);
        }
        *len = p + 1;
        *hwm = (*hwm).max(p + 1);

        // Log scaling (global layers only): scales q and the position bias.
        let tau = match (cx.dims.window, cx.dims.n_floor) {
            (None, Some(nf)) => 1.0 + cx.dims.alpha * ((p + 1) as f32 / nf).max(1.0).ln(),
            _ => 1.0,
        };
        if tau != 1.0 {
            for qi in q.iter_mut() {
                *qi *= tau;
            }
        }

        // Visible keys: global 0..=p; sliding p-window < j <= p.
        let j0 = match cx.dims.window {
            Some(win) => (p + 1).saturating_sub(win),
            None => 0,
        };
        let n_keys = p + 1 - j0;
        let group = hq / hkv;
        let extent = cx.relpos.extent();

        let mut ctx = vec![0.0f32; hq * d];
        let mut score = vec![0.0f32; n_keys];
        for h in 0..hq {
            let kvh = h / group;
            let qh = &q[h * d..(h + 1) * d];
            let prof = cx.relpos.profile(&r[h * dr..(h + 1) * dr]);
            let mut smax = f32::NEG_INFINITY;
            for (i, j) in (j0..=p).enumerate() {
                let o = cx.kv_off(kvh, cx.slot(j));
                let dist = p - j;
                let bias = if dist < extent { prof[dist] * tau } else { 0.0 };
                let s = dot(qh, &k_cache[o..o + d]) * cx.scale + bias;
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
                let o = cx.kv_off(kvh, cx.slot(j));
                for (c, &x) in ctx_h.iter_mut().zip(&v_cache[o..o + d]) {
                    *c += pj * x;
                }
            }
        }

        // The output projection is applied by the caller (batched for prefill).
        let _ = hd;
        ctx
    }
}

impl AttentionLayer {
    /// Attend one input-normed hidden `h` (`[H]`) at position `self.len`,
    /// appending to the cache. Returns the `Wo` output (`[H]`) BEFORE
    /// `attn_sconv` — the layer applies that conv and the residual.
    pub fn forward_token(&mut self, h: &[f32]) -> Vec<f32> {
        assert_eq!(h.len(), self.dims.hidden, "attn forward_token: h len");
        let (q, kr, vr, r) = self.project_rows(h, 1);
        let kc = self.k_sconv.decode(&kr);
        let v = self.v_sconv.decode(&vr);
        let ctx = self.attend(q, kc, &v, &r);
        self.project_out_rows(&ctx, 1)
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
        let (q_all, kr_all, vr_all, r_all) = self.project_rows(hs, t);
        let kc_all = self.k_sconv.prefill(&kr_all, t);
        let v_all = self.v_sconv.prefill(&vr_all, t);
        let mut ctx_all = vec![0.0f32; t * qd];
        for row in 0..t {
            let c = self.attend(
                q_all[row * qd..(row + 1) * qd].to_vec(),
                kc_all[row * kd..(row + 1) * kd].to_vec(),
                &v_all[row * kd..(row + 1) * kd],
                &r_all[row * rd..(row + 1) * rd],
            );
            ctx_all[row * qd..(row + 1) * qd].copy_from_slice(&c);
        }
        let _ = hd;
        self.project_out_rows(&ctx_all, t)
    }

    /// Multi-stream decode: `t` input-normed rows (`hs` = `[t, H]`), row `i`
    /// belonging to sequence `slots[i]` at that sequence's next position.
    /// Projections run as one batch (the weights are read once for all
    /// streams), then each row selects its slot and attends against its own
    /// cache; returns `[t, H]` before the layer's `attn_sconv`. Per row this
    /// is the same op sequence as [`Self::forward_token`] on that sequence
    /// alone (bit-identical on the CPU projections). A slot may appear once
    /// per call.
    pub fn forward_rows(&mut self, hs: &[f32], t: usize, slots: &[usize]) -> Vec<f32> {
        let (hd, hq, hkv, d, dr) = (
            self.dims.hidden,
            self.dims.n_heads,
            self.dims.n_kv_heads,
            self.dims.head_dim,
            self.dims.d_rel,
        );
        assert_eq!(slots.len(), t, "attn forward_rows: one slot per row");
        assert_eq!(hs.len(), t * hd, "attn forward_rows: hs len != t * hidden");
        let (qd, kd, rd) = (hq * d, hkv * d, hq * dr);
        let (q_all, kr_all, vr_all, r_all) = self.project_rows(hs, t);
        if t >= 2 && row_parallel_attention() {
            let ctx_all = self.attend_rows_parallel(&q_all, &kr_all, &vr_all, &r_all, slots);
            return self.project_out_rows(&ctx_all, t);
        }
        let mut ctx_all = vec![0.0f32; t * qd];
        for (row, &slot) in slots.iter().enumerate() {
            self.select(slot);
            let kc = self.k_sconv.decode(&kr_all[row * kd..(row + 1) * kd]);
            let v = self.v_sconv.decode(&vr_all[row * kd..(row + 1) * kd]);
            let c = self.attend(
                q_all[row * qd..(row + 1) * qd].to_vec(),
                kc,
                &v,
                &r_all[row * rd..(row + 1) * rd],
            );
            ctx_all[row * qd..(row + 1) * qd].copy_from_slice(&c);
        }
        self.project_out_rows(&ctx_all, t)
    }
}
