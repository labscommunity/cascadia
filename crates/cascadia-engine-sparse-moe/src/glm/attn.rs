//! GLM-5.2 MLA attention (classic DeepSeek-V3 MLA) — absorbed-latent decode.
//!
//! Per token at `pos`:
//!   qr   = rmsnorm(wq_a · x, q_a_ln)                       [q_lora]
//!   q    = wq_b · qr                                       [h·qk_head]  ([nope|pe] per head)
//!   rope(q_pe)                                             (last qk_rope of each head)
//!   comp = wkv_a · x                                       [kv_lora + qk_rope]
//!   Lc   = rmsnorm(comp[..kv_lora], kv_a_ln)               [kv_lora]   (cached latent)
//!   Rc   = rope(comp[kv_lora..])                           [qk_rope]   (cached k_pe, shared)
//! then, per head h (`W_UK_h` = kv_b rows `[rbase..rbase+nope]`,
//! `W_UV_h` = rows `[rbase+nope..rbase+nope+v]`, `rbase = h·(nope+v)`):
//!   qabs  = W_UK_hᵀ · q_nope                               [kv_lora]   (absorb the k up-proj)
//!   score[t] = (qabs·Lc[t] + q_pe·Rc[t]) · scale           over cached t
//!   p     = softmax(score)
//!   clat  = Σ_t p[t]·Lc[t]                                 [kv_lora]
//!   ctx_h = W_UV_h · clat                                  [v_head]    (absorb the v up-proj)
//!   out   = wo · concat_h(ctx_h)                           [hidden]
//!
//! `scale = qk_head^-0.5` (qk_head = qk_nope + qk_rope); no mscale (no YaRN),
//! no attention sink, no bias. Numeric contract (matches the CPU reference in
//! `tools/glm5_ref`): bf16 rounding after each linear / RMSNorm / rope; the
//! absorb core (qabs / score / softmax / clat / ctx) stays in f32.
//!
//! This is the decode path used for every position. Batched-naive prefill is a
//! later perf addition (M4); it is mathematically identical by linearity.
//!
//! DSA sparsity: with a [`Indexer`] attached ([`AttentionLayer::attach_indexer`]),
//! once the cached length exceeds `index_topk` the query attends only to the
//! indexer's top-`index_topk` positions (one set shared by all heads). At or
//! below the budget it is the full causal range — bit-identical to dense.

use half::bf16;

use super::indexer::Indexer;
use super::rope::apply_rope_row;
use crate::dsv4::math::{dot, dot_bf16w, linear_bf16_w, rmsnorm, round_bf16};
use crate::dsv4::rope::Freqs;

/// Widen a bf16-bit weight to f32 (bf16 is the top 16 bits of an f32).
#[inline]
fn widen(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The MLA latent (`lc`) / k_pe (`rc`) cache, held either exact f32 or — opt-in
/// via `CASCADIA_GLM5_BF16_KV` — bf16 bits, halving the KV RAM per rank (it scales
/// with context length, so this frees the most under long agentic context). bf16
/// rounds each cached latent, an attention-numerics change: validate greedy
/// parity before defaulting on. The absorb core still accumulates in f32; only
/// the stored latents are narrowed, read back through the same `widen`/`dot_bf16w`
/// path the bf16 projection weights already use.
enum KvStore {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl KvStore {
    fn zeros(n: usize, bf16_kv: bool) -> Self {
        if bf16_kv {
            KvStore::Bf16(vec![0u16; n])
        } else {
            KvStore::F32(vec![0.0; n])
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            KvStore::F32(v) => v.len(),
            KvStore::Bf16(v) => v.len(),
        }
    }

    fn fill0(&mut self) {
        match self {
            KvStore::F32(v) => v.fill(0.0),
            KvStore::Bf16(v) => v.fill(0),
        }
    }

    /// Write one f32 row (`dim` elems) at position `t`.
    #[inline]
    fn write_row(&mut self, t: usize, dim: usize, src: &[f32]) {
        match self {
            KvStore::F32(v) => v[t * dim..(t + 1) * dim].copy_from_slice(src),
            KvStore::Bf16(v) => {
                for (d, &s) in src.iter().enumerate() {
                    v[t * dim + d] = bf16::from_f32(s).to_bits();
                }
            }
        }
    }

    /// `dot(q, row t)` — `q` is f32 (`dim` elems). The bf16 arm reuses the same
    /// widen-and-FMA kernel as the bf16 projection weights.
    #[inline]
    fn dot_row(&self, t: usize, dim: usize, q: &[f32]) -> f32 {
        match self {
            KvStore::F32(v) => dot(q, &v[t * dim..(t + 1) * dim]),
            KvStore::Bf16(v) => dot_bf16w(&v[t * dim..(t + 1) * dim], q),
        }
    }

    /// `acc += p · row t` — `acc` is f32 (`dim` elems).
    #[inline]
    fn axpy_row(&self, t: usize, dim: usize, p: f32, acc: &mut [f32]) {
        match self {
            KvStore::F32(v) => {
                for (c, &l) in acc.iter_mut().zip(&v[t * dim..(t + 1) * dim]) {
                    *c += p * l;
                }
            }
            KvStore::Bf16(v) => {
                for (c, &l) in acc.iter_mut().zip(&v[t * dim..(t + 1) * dim]) {
                    *c += p * widen(l);
                }
            }
        }
    }

    /// The first `n` elements as f32 — the prefix-cache snapshot (kept f32 so a
    /// snapshot is dtype-independent).
    fn to_f32_prefix(&self, n: usize) -> Vec<f32> {
        match self {
            KvStore::F32(v) => v[..n].to_vec(),
            KvStore::Bf16(v) => v[..n].iter().map(|&b| widen(b)).collect(),
        }
    }

    /// Restore a prefix from an f32 snapshot (narrowed back to the store dtype).
    fn restore_prefix(&mut self, src: &[f32]) {
        match self {
            KvStore::F32(v) => v[..src.len()].copy_from_slice(src),
            KvStore::Bf16(v) => {
                for (d, &s) in src.iter().enumerate() {
                    v[d] = bf16::from_f32(s).to_bits();
                }
            }
        }
    }
}

/// MLA latent-norm eps. HF runs `q_a_layernorm` / `kv_a_layernorm` at the
/// `GlmMoeDsaRMSNorm` default (1e-6) — NOT `rms_norm_eps` (1e-5, which applies
/// only to input/post-attention/final norms). Verified against
/// `modeling_glm_moe_dsa.py`.
pub const MLA_LATENT_EPS: f32 = 1e-6;

/// Attention projection weights. GEMV weights are stored as bf16 bits (`u16`) —
/// batch-1 projections are bandwidth-bound and the model is bf16-native, so
/// halving the streamed bytes ~halves the projection cost. RMSNorm weights stay
/// f32.
pub struct AttnWeights {
    pub wq_a: Vec<u16>,    // [q_lora, hidden]
    pub q_a_ln: Vec<f32>,  // [q_lora]
    pub wq_b: Vec<u16>,    // [h*qk_head, q_lora]
    pub wkv_a: Vec<u16>,   // [kv_lora + qk_rope, hidden]
    pub kv_a_ln: Vec<f32>, // [kv_lora]
    pub wkv_b: Vec<u16>,   // [h*(qk_nope + v_head), kv_lora]
    pub wo: Vec<u16>,      // [hidden, h*v_head]
}

pub struct AttentionLayer {
    pub hidden: usize,
    pub h: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub kv_lora: usize,
    pub q_lora: usize,
    qk_head: usize, // qk_nope + qk_rope
    kv_out: usize,  // qk_nope + v_head (kv_b rows per head)
    scale: f32,
    w: AttnWeights,
    freqs: Freqs,
    lc: KvStore, // [max_seq, kv_lora]
    rc: KvStore, // [max_seq, qk_rope]
    len: usize,  // cached positions
    /// DSA lightning indexer. When attached and the cached length exceeds
    /// `index_topk`, the query attends only to the indexer's top-`index_topk`
    /// positions; otherwise attention is over the full causal range.
    indexer: Option<Indexer>,
    index_topk: usize,
    /// IndexShare `"shared"` layer: owns no indexer and reuses the top-k
    /// selection carried from the most recent `"full"` layer.
    is_shared: bool,
}

/// A saved KV snapshot for prefix reuse — the first `len` cached positions of
/// one attention layer (MLA latent + rope caches, and the DSA indexer keys when
/// present). Restoring it into a freshly-`reset` layer lets a request skip
/// re-prefilling a shared prompt prefix. Snapshot and layer must share dims
/// (same model / rank).
#[derive(Clone)]
pub struct AttnKv {
    len: usize,
    lc: Vec<f32>,         // len * kv_lora
    rc: Vec<f32>,         // len * qk_rope
    ic: Option<Vec<f32>>, // len * index_head_dim, if this layer has an indexer
}

impl AttnKv {
    /// Number of cached positions this snapshot covers.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A placeholder snapshot of the given length with no cache payload — for
    /// cache-bookkeeping tests only (not a restorable KV).
    #[doc(hidden)]
    pub fn empty(len: usize) -> Self {
        Self {
            len,
            lc: Vec::new(),
            rc: Vec::new(),
            ic: None,
        }
    }
}

impl AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        h: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_head: usize,
        kv_lora: usize,
        q_lora: usize,
        max_seq: usize,
        w: AttnWeights,
        freqs: Freqs,
    ) -> Self {
        let qk_head = qk_nope + qk_rope;
        let kv_out = qk_nope + v_head;
        assert_eq!(w.wq_a.len(), q_lora * hidden);
        assert_eq!(w.q_a_ln.len(), q_lora);
        assert_eq!(w.wq_b.len(), h * qk_head * q_lora);
        assert_eq!(w.wkv_a.len(), (kv_lora + qk_rope) * hidden);
        assert_eq!(w.kv_a_ln.len(), kv_lora);
        assert_eq!(w.wkv_b.len(), h * kv_out * kv_lora);
        assert_eq!(w.wo.len(), hidden * h * v_head);
        Self {
            hidden,
            h,
            qk_nope,
            qk_rope,
            v_head,
            kv_lora,
            q_lora,
            qk_head,
            kv_out,
            scale: (qk_head as f32).powf(-0.5),
            w,
            freqs,
            // KV starts exact f32; the loader flips it to bf16 via
            // `set_kv_precision` when the caller (StageOpts) asks. No env read
            // here, so the pin-budget reservation and this allocation can't
            // disagree.
            lc: KvStore::zeros(max_seq * kv_lora, false),
            rc: KvStore::zeros(max_seq * qk_rope, false),
            len: 0,
            indexer: None,
            index_topk: usize::MAX,
            is_shared: false,
        }
    }

    /// Select the KV-cache precision (f32 exact by default, or bf16 to halve KV
    /// RAM). Reallocates the freshly-zeroed latent/k_pe caches, so call this right
    /// after [`Self::new`], before any token is appended. Cheap (empty buffers).
    pub fn set_kv_precision(&mut self, bf16_kv: bool) {
        let max_seq = self.lc.len() / self.kv_lora;
        self.lc = KvStore::zeros(max_seq * self.kv_lora, bf16_kv);
        self.rc = KvStore::zeros(max_seq * self.qk_rope, bf16_kv);
    }

    /// Attach the DSA lightning indexer (with its `index_topk` budget). Without
    /// this the layer runs dense causal attention (correct for ctx ≤ index_topk).
    pub fn attach_indexer(&mut self, indexer: Indexer, index_topk: usize) {
        assert!(index_topk > 0, "index_topk must be positive");
        self.indexer = Some(indexer);
        self.index_topk = index_topk;
    }

    /// Mark this as an IndexShare `"shared"` layer: it owns no indexer and reuses
    /// the top-k selection carried from the most recent `"full"` layer.
    pub fn mark_shared(&mut self, index_topk: usize) {
        self.is_shared = true;
        self.index_topk = index_topk;
    }

    /// Clear the KV cache (and the indexer's key cache, if attached).
    pub fn reset(&mut self) {
        self.lc.fill0();
        self.rc.fill0();
        self.len = 0;
        if let Some(ix) = self.indexer.as_mut() {
            ix.reset();
        }
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Roll the KV (and indexer key) cache back to `len` positions — the
    /// speculative-decode reject path. O(1): the latent/k_pe slots at
    /// `[len, old_len)` stay allocated and are overwritten when decode resumes.
    /// `len` must not exceed the current length.
    pub fn truncate(&mut self, len: usize) {
        debug_assert!(len <= self.len, "attn truncate {len} > len {}", self.len);
        self.len = len;
        if let Some(ix) = self.indexer.as_mut() {
            ix.truncate(len);
        }
    }

    /// Snapshot this layer's KV (`[0, len)`) for prefix caching — copies out the
    /// latent/rope caches and the indexer keys.
    pub fn snapshot(&self) -> AttnKv {
        let n = self.len;
        AttnKv {
            len: n,
            lc: self.lc.to_f32_prefix(n * self.kv_lora),
            rc: self.rc.to_f32_prefix(n * self.qk_rope),
            ic: self.indexer.as_ref().map(|ix| ix.snapshot()),
        }
    }

    /// Restore a KV snapshot, replacing the current cache and length. Call after
    /// [`Self::reset`]; the cache tail past `kv.len` is stale but never read
    /// (attention only touches `[0, len)`), matching [`Self::truncate`].
    pub fn restore(&mut self, kv: &AttnKv) {
        self.len = kv.len;
        self.lc.restore_prefix(&kv.lc);
        self.rc.restore_prefix(&kv.rc);
        if let (Some(ix), Some(ic)) = (self.indexer.as_mut(), kv.ic.as_ref()) {
            ix.restore(kv.len, ic);
        }
    }

    /// The cached-latent (`lc`) / roped-k_pe (`rc`) mapping for one token: the
    /// exact computation an external prefill path (the OV iGPU graph) must
    /// reproduce bit-for-bit for [`Self::commit_prefill_rows`] to be a valid
    /// substitute for this cache write. `forward_token` is the only other
    /// caller — kept here so there is one place this math lives, not two.
    fn compute_lc_rc(&self, x: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        let (kvl, rope) = (self.kv_lora, self.qk_rope);
        let mut comp = vec![0.0f32; kvl + rope];
        linear_bf16_w(x, &self.w.wkv_a, kvl + rope, self.hidden, &mut comp);
        let mut lat = comp[..kvl].to_vec();
        rmsnorm(&mut lat, &self.w.kv_a_ln, MLA_LATENT_EPS);
        let mut kpe = comp[kvl..].to_vec();
        apply_rope_row(&mut kpe, &self.freqs, pos, rope, false);
        (lat, kpe)
    }

    /// Attend one token `x` (`[hidden]`) at absolute position `self.len`,
    /// appending its latent/k_pe to the cache. Returns `out` (`[hidden]`).
    /// Positions must be fed in order starting from 0.
    ///
    /// `carry` is the IndexShare selection channel: a `"full"` layer (has an
    /// indexer) computes this query's top-k and writes it into `carry`; a
    /// `"shared"` layer reads it back. Pass `&mut None` for a plain layer / the
    /// first token.
    pub fn forward_token(&mut self, x: &[f32], carry: &mut Option<Vec<usize>>) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        let (h, nope, rope) = (self.h, self.qk_nope, self.qk_rope);
        let (vh, kvl, qk) = (self.v_head, self.kv_lora, self.qk_head);
        let pos = self.len;
        // KV caches + the rope table are sized to max_seq at load; a position past
        // that would panic with an opaque slice-OOB. Fail with a diagnostic instead
        // (the DSA long-context regime is exactly where this bites).
        assert!(
            pos < self.lc.len() / kvl,
            "GLM context length {} exceeds max_seq {}; raise CASCADIA_GLM5_MAX_SEQ",
            pos + 1,
            self.lc.len() / kvl
        );

        // q = wq_b · rmsnorm(wq_a · x), rope on each head's pe tail.
        let mut qr = vec![0.0f32; self.q_lora];
        linear_bf16_w(x, &self.w.wq_a, self.q_lora, self.hidden, &mut qr);
        rmsnorm(&mut qr, &self.w.q_a_ln, MLA_LATENT_EPS);
        let mut q = vec![0.0f32; h * qk];
        linear_bf16_w(&qr, &self.w.wq_b, h * qk, self.q_lora, &mut q);
        for hi in 0..h {
            apply_rope_row(
                &mut q[hi * qk..(hi + 1) * qk],
                &self.freqs,
                pos,
                rope,
                false,
            );
        }

        // comp = wkv_a · x -> [latent | k_pe]; append normed latent + roped k_pe.
        let (lat, kpe) = self.compute_lc_rc(x, pos);
        self.lc.write_row(pos, kvl, &lat);
        self.rc.write_row(pos, rope, &kpe);
        self.len += 1;
        let n = self.len; // cached tokens, includes self (causal)

        // DSA: the lightning indexer picks which cached positions this query
        // attends to (top-`index_topk`, one set shared by all heads). Every token
        // still appends its key to the indexer cache. At or below the budget — or
        // with no indexer — `sel` is the full causal range `0..n`, so the result
        // is bit-identical to dense attention (existing goldens unaffected).
        let sel: Vec<usize> = if let Some(ix) = self.indexer.as_mut() {
            // "full" layer: compute this query's top-k and publish it so the
            // subsequent "shared" layers reuse the same selection.
            ix.append_key(x);
            let s: Vec<usize> = if n > self.index_topk {
                ix.select(&qr, x, n - 1, self.index_topk)
                    .into_iter()
                    .map(|t| t as usize)
                    .collect()
            } else {
                (0..n).collect()
            };
            *carry = Some(s.clone());
            s
        } else if self.is_shared {
            // "shared" layer: reuse the most recent full layer's selection
            // (carry is always set by then; None only in the not-yet-wired
            // cross-rank case, where we fall back to full causal).
            carry.clone().unwrap_or_else(|| (0..n).collect())
        } else {
            (0..n).collect()
        };
        let m = sel.len();

        // Absorbed attention, per head, over the selected tokens.
        let mut ctx = vec![0.0f32; h * vh];
        let mut qabs = vec![0.0f32; kvl];
        let mut score = vec![0.0f32; m];
        let mut clat = vec![0.0f32; kvl];
        for hi in 0..h {
            let qh = &q[hi * qk..(hi + 1) * qk];
            let qnope = &qh[..nope];
            let qpe = &qh[nope..];
            let rbase = hi * self.kv_out;

            // qabs = W_UK_hᵀ · q_nope  (Σ_d q_nope[d] · kv_b row (rbase+d)).
            qabs.iter_mut().for_each(|v| *v = 0.0);
            for (d, &qd) in qnope.iter().enumerate() {
                let row = &self.w.wkv_b[(rbase + d) * kvl..(rbase + d + 1) * kvl];
                for (a, &wb) in qabs.iter_mut().zip(row) {
                    *a += qd * widen(wb);
                }
            }

            // scores over the selected latents; softmax (f32).
            let mut smax = f32::NEG_INFINITY;
            for (j, &t) in sel.iter().enumerate() {
                let sn = self.lc.dot_row(t, kvl, &qabs);
                let sp = self.rc.dot_row(t, rope, qpe);
                let s = (sn + sp) * self.scale;
                score[j] = s;
                smax = smax.max(s);
            }
            let mut denom = 0.0f32;
            for s in score[..m].iter_mut() {
                *s = (*s - smax).exp();
                denom += *s;
            }

            // clat = Σ_j p[j]·Lc[sel[j]]; ctx_h = W_UV_h · clat.
            clat.iter_mut().for_each(|v| *v = 0.0);
            for (j, &t) in sel.iter().enumerate() {
                let p = score[j] / denom;
                self.lc.axpy_row(t, kvl, p, &mut clat);
            }
            let ctx_h = &mut ctx[hi * vh..(hi + 1) * vh];
            for (jj, o) in ctx_h.iter_mut().enumerate() {
                let row = &self.w.wkv_b[(rbase + nope + jj) * kvl..(rbase + nope + jj + 1) * kvl];
                *o = dot_bf16w(row, &clat);
            }
        }

        // out = wo · concat_h(ctx_h).
        let mut out = vec![0.0f32; self.hidden];
        linear_bf16_w(&ctx, &self.w.wo, self.hidden, h * vh, &mut out);
        out
    }

    /// Commit `rows` externally-computed cache rows — `lc_rows` `[rows,
    /// kv_lora]`, `rc_rows` `[rows, qk_rope]`, row-major, starting at absolute
    /// position `self.len` — into the live KV cache via the same
    /// `bf16_kv`-aware write path [`Self::forward_token`] uses, then advances
    /// `len` by `rows`. This is the landing site for an OV-computed prefill
    /// window's latent/k_pe rows; it does not itself compute them (see
    /// [`Self::compute_lc_rc`] for the mapping those rows must match).
    ///
    /// `rmsnorm`/rope always leave `forward_token`'s own rows on the bf16
    /// grid (see the module numeric contract), so every row is defensively
    /// re-rounded onto that grid before being written — a no-op for
    /// already-on-grid input, a correctness fix if a GPU plugin's Convert was
    /// silently elided the way one already was in this project (Task 2). A
    /// debug assertion catches off-grid input loudly in testing without
    /// costing anything in release, where the rounding is the only guard.
    ///
    /// Errors (never panics) on mismatched slice lengths or if `rows` would
    /// overflow the cache's capacity: the caller is a fallible OV path that
    /// must be able to fall back to the Rust loop instead of taking down a
    /// serving rank.
    pub fn commit_prefill_rows(
        &mut self,
        lc_rows: &[f32],
        rc_rows: &[f32],
        rows: usize,
    ) -> Result<(), String> {
        let (kvl, rope) = (self.kv_lora, self.qk_rope);
        if lc_rows.len() != rows * kvl {
            return Err(format!(
                "commit_prefill_rows: lc_rows len {} != rows {rows} * kv_lora {kvl}",
                lc_rows.len()
            ));
        }
        if rc_rows.len() != rows * rope {
            return Err(format!(
                "commit_prefill_rows: rc_rows len {} != rows {rows} * qk_rope {rope}",
                rc_rows.len()
            ));
        }
        let max_seq = self.lc.len() / kvl;
        if self.len + rows > max_seq {
            return Err(format!(
                "commit_prefill_rows: len {} + rows {rows} exceeds max_seq {max_seq}",
                self.len
            ));
        }
        let mut lat = vec![0.0f32; kvl];
        let mut kpe = vec![0.0f32; rope];
        for r in 0..rows {
            let pos = self.len + r;
            lat.copy_from_slice(&lc_rows[r * kvl..(r + 1) * kvl]);
            round_bf16(&mut lat);
            debug_assert_eq!(
                lat.as_slice(),
                &lc_rows[r * kvl..(r + 1) * kvl],
                "commit_prefill_rows: lc row {r} off the bf16 grid (OV contract violated)"
            );
            kpe.copy_from_slice(&rc_rows[r * rope..(r + 1) * rope]);
            round_bf16(&mut kpe);
            debug_assert_eq!(
                kpe.as_slice(),
                &rc_rows[r * rope..(r + 1) * rope],
                "commit_prefill_rows: rc row {r} off the bf16 grid (OV contract violated)"
            );
            self.lc.write_row(pos, kvl, &lat);
            self.rc.write_row(pos, rope, &kpe);
        }
        self.len += rows;
        Ok(())
    }

    /// Feed one row's post-`in_ln` normed activations (the same input
    /// `forward_token` takes as `x`, NOT the raw residual) to the DSA
    /// indexer's key cache — the bulk-commit counterpart of the
    /// `ix.append_key(x)` call inside `forward_token`. No-op on a layer with
    /// no indexer (plain layers and `"shared"` layers never hold one).
    pub fn indexer_append_normed(&mut self, normed_row: &[f32]) {
        if let Some(ix) = self.indexer.as_mut() {
            ix.append_key(normed_row);
        }
    }
}

#[cfg(test)]
mod kv_tests {
    use super::KvStore;

    /// bf16 KV must track the f32 path within bf16 tolerance across the three
    /// hot-path ops (write→dot, write→axpy, snapshot round-trip). The opt-in
    /// `CASCADIA_GLM5_BF16_KV` swap is a precision change, not a correctness one.
    #[test]
    fn kvstore_bf16_tracks_f32() {
        let (dim, rows) = (16usize, 3usize);
        let mut f = KvStore::zeros(rows * dim, false);
        let mut b = KvStore::zeros(rows * dim, true);
        let row: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13 - 0.7).sin()).collect();
        for t in 0..rows {
            f.write_row(t, dim, &row);
            b.write_row(t, dim, &row);
        }
        let q: Vec<f32> = (0..dim).map(|i| i as f32 * 0.05 - 0.2).collect();

        // dot_row: f32 accumulation, only the stored latent is bf16.
        let (df, db) = (f.dot_row(1, dim, &q), b.dot_row(1, dim, &q));
        assert!(
            (df - db).abs() <= df.abs() * 0.05 + 1e-2,
            "dot {df} vs {db}"
        );

        // axpy_row: weighted accumulate.
        let mut af = vec![0.0f32; dim];
        let mut ab = vec![0.0f32; dim];
        f.axpy_row(2, dim, 0.5, &mut af);
        b.axpy_row(2, dim, 0.5, &mut ab);
        for (x, y) in af.iter().zip(&ab) {
            assert!((x - y).abs() <= x.abs() * 0.02 + 1e-2, "axpy {x} vs {y}");
        }

        // snapshot round-trip: each element within one bf16 ULP.
        for (x, y) in f.to_f32_prefix(dim).iter().zip(&b.to_f32_prefix(dim)) {
            assert!((x - y).abs() <= x.abs() / 128.0 + 1e-3, "prefix {x} vs {y}");
        }
    }
}

#[cfg(test)]
mod commit_tests {
    use super::*;
    use crate::dsv4::rope::precompute_freqs;
    use crate::glm::indexer::IndexerWeights;

    /// A tiny xorshift-style LCG stands in for real weights/activations —
    /// these tests check `commit_prefill_rows`/`indexer_append_normed`
    /// against `forward_token` itself, not against an external reference, so
    /// the exact values don't matter as long as both sides of a comparison
    /// see the same ones.
    fn lcg(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / u16::MAX as f32 - 0.5
        }
    }

    fn bf16_bits(v: f32) -> u16 {
        bf16::from_f32(v).to_bits()
    }

    /// Deterministic tiny `AttentionLayer` — same seed every call, so two
    /// independently-built instances have byte-identical weights and can
    /// stand in for "the same layer" without needing `Clone`.
    fn tiny_layer(bf16_kv: bool, max_seq: usize) -> AttentionLayer {
        let (hidden, h, nope, rope, vh, kvl, ql) =
            (12usize, 2usize, 4usize, 4usize, 4usize, 6usize, 8usize);
        let mut rnd = lcg(7);
        let w = AttnWeights {
            wq_a: (0..ql * hidden).map(|_| bf16_bits(rnd())).collect(),
            q_a_ln: (0..ql).map(|_| 1.0 + rnd() * 0.1).collect(),
            wq_b: (0..h * (nope + rope) * ql)
                .map(|_| bf16_bits(rnd()))
                .collect(),
            wkv_a: (0..(kvl + rope) * hidden)
                .map(|_| bf16_bits(rnd()))
                .collect(),
            kv_a_ln: (0..kvl).map(|_| 1.0 + rnd() * 0.1).collect(),
            wkv_b: (0..h * (nope + vh) * kvl)
                .map(|_| bf16_bits(rnd()))
                .collect(),
            wo: (0..hidden * h * vh).map(|_| bf16_bits(rnd())).collect(),
        };
        let freqs = precompute_freqs(rope, max_seq, 0, 1.0e4, 1.0, 32.0, 1.0);
        let mut layer = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, max_seq, w, freqs);
        layer.set_kv_precision(bf16_kv);
        layer
    }

    fn rows(n: usize, hidden: usize, seed: u32) -> Vec<Vec<f32>> {
        let mut rnd = lcg(seed);
        (0..n)
            .map(|_| (0..hidden).map(|_| rnd()).collect())
            .collect()
    }

    fn kv_bytes_eq(a: &KvStore, b: &KvStore) -> bool {
        match (a, b) {
            (KvStore::F32(x), KvStore::F32(y)) => x == y,
            (KvStore::Bf16(x), KvStore::Bf16(y)) => x == y,
            _ => false,
        }
    }

    /// The equivalence this whole task exists to guarantee: writing rows via
    /// `commit_prefill_rows` must land byte-identical to what `forward_token`
    /// would have written itself, in BOTH KV precisions, and the committed
    /// state must go on to behave identically under further decode — not
    /// merely look similar right after the commit.
    #[test]
    fn commit_prefill_rows_matches_forward_token() {
        for bf16_kv in [false, true] {
            let (hidden, n, max_seq) = (12usize, 6usize, 16usize);
            let mut a = tiny_layer(bf16_kv, max_seq);
            let mut b = tiny_layer(bf16_kv, max_seq);
            let x_rows = rows(n, hidden, 123);

            let mut carry = None;
            for r in &x_rows {
                a.forward_token(r, &mut carry);
            }

            let mut lc_rows = Vec::new();
            let mut rc_rows = Vec::new();
            for (i, r) in x_rows.iter().enumerate() {
                let (lat, kpe) = b.compute_lc_rc(r, i);
                lc_rows.extend(lat);
                rc_rows.extend(kpe);
            }
            b.commit_prefill_rows(&lc_rows, &rc_rows, n)
                .expect("commit_prefill_rows");

            assert_eq!(a.len, b.len, "len diverged (bf16_kv={bf16_kv})");
            assert!(
                kv_bytes_eq(&a.lc, &b.lc),
                "lc cache diverged (bf16_kv={bf16_kv})"
            );
            assert!(
                kv_bytes_eq(&a.rc, &b.rc),
                "rc cache diverged (bf16_kv={bf16_kv})"
            );

            // Not just "looks the same" — the committed cache must be usable
            // going forward exactly like a forward_token-built one.
            let extra = &rows(1, hidden, 999)[0];
            let out_a = a.forward_token(extra, &mut None);
            let out_b = b.forward_token(extra, &mut None);
            assert_eq!(
                out_a, out_b,
                "post-commit forward_token diverged (bf16_kv={bf16_kv})"
            );
        }
    }

    /// Proves the on-grid debug guard actually fires rather than being
    /// present-but-vacuous: an off-grid `lc` row (a value bf16 rounding
    /// changes) must trip the debug assertion, the same way a GPU plugin
    /// silently defeating the graph's own rounding would.
    #[test]
    #[should_panic(expected = "off the bf16 grid")]
    fn commit_prefill_rows_debug_asserts_off_grid_input() {
        let (kvl, rope, max_seq) = (6usize, 4usize, 4usize);
        let mut layer = tiny_layer(false, max_seq);
        // 0.1f32 is not exactly representable in bf16 (7 mantissa bits), so
        // rounding it changes the value — the off-grid case this guards.
        let lc_row = vec![0.1f32; kvl];
        let rc_row = vec![0.0f32; rope];
        let _ = layer.commit_prefill_rows(&lc_row, &rc_row, 1);
    }

    #[test]
    fn commit_prefill_rows_rejects_bad_lengths_and_overflow() {
        let (hidden, kvl, rope, max_seq) = (12usize, 6usize, 4usize, 4usize);
        let mut layer = tiny_layer(false, max_seq);

        assert!(
            layer
                .commit_prefill_rows(&vec![0.0; kvl * 3], &vec![0.0; rope * 2], 3)
                .is_err(),
            "rc_rows length mismatch must error"
        );
        assert!(
            layer
                .commit_prefill_rows(&vec![0.0; kvl * 2], &vec![0.0; rope * 3], 3)
                .is_err(),
            "lc_rows length mismatch must error"
        );

        // Fill to capacity, then one more row must overflow rather than panic.
        let x_rows = rows(max_seq, hidden, 5);
        let mut carry = None;
        for r in &x_rows {
            layer.forward_token(r, &mut carry);
        }
        assert!(
            layer
                .commit_prefill_rows(&vec![0.0; kvl], &vec![0.0; rope], 1)
                .is_err(),
            "committing past max_seq must error, not panic"
        );
    }

    #[test]
    fn indexer_append_normed_noop_without_indexer_forwards_with_one() {
        let mut layer = tiny_layer(false, 8);
        let hidden = layer.hidden;

        // No indexer attached: must be a true no-op, not a panic.
        layer.indexer_append_normed(&vec![0.0f32; hidden]);

        let (q_lora, nh, hd, rope_dim) = (layer.q_lora, 2usize, 4usize, layer.qk_rope);
        let mut rnd = lcg(42);
        let iw = IndexerWeights {
            ix_wq: (0..nh * hd * q_lora).map(|_| bf16_bits(rnd())).collect(),
            ix_wk: (0..hd * hidden).map(|_| bf16_bits(rnd())).collect(),
            ix_wp: (0..nh * hidden).map(|_| bf16_bits(rnd())).collect(),
            k_norm_w: vec![1.0; hd],
            k_norm_b: vec![0.0; hd],
        };
        let freqs = precompute_freqs(rope_dim, 8, 0, 1.0e4, 1.0, 32.0, 1.0);
        let ix = Indexer::new(hidden, q_lora, nh, hd, rope_dim, 8, 1e-6, iw, freqs);
        layer.attach_indexer(ix, 4);

        let row: Vec<f32> = (0..hidden).map(|_| rnd()).collect();
        layer.indexer_append_normed(&row);
        assert_eq!(layer.indexer.as_ref().unwrap().len(), 1);
    }
}
