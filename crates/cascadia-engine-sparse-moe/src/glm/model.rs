//! GLM-5.2 transformer layer + (later) full model assembly.
//!
//! Layer forward is pre-norm:
//!   h   = x + attention(rmsnorm(x, input_layernorm))
//!   out = h + mlp(rmsnorm(h, post_attention_layernorm))
//! where `mlp` is the MoE block on sparse layers or a dense SwiGLU on the first
//! `first_k_dense_replace` (3) layers. Both norms are RMSNorm, eps
//! `rms_norm_eps` (1e-5).

use super::attn::{AttentionLayer, AttnKv};
use super::kv_cache::KvPrefixCache;
use super::moe::{AnyExpert, MoeLayer};
use super::mtp::MtpHead;
use super::ov_attn::OvRoute;
use super::prof;
use crate::dsv4::math::{linear_f32, rmsnorm};

/// The per-layer feed-forward: routed MoE, or a dense SwiGLU (first-k layers).
/// The dense MLP is a single SwiGLU FFN — stored as an [`AnyExpert`] so it
/// shares the Bf16/int4 storage dispatch with the routed experts.
pub enum LayerMlp {
    Moe(MoeLayer),
    Dense { w: AnyExpert, inter: usize },
}

pub struct GlmLayer {
    pub hidden: usize,
    pub eps: f32,
    in_ln: Vec<f32>,   // input_layernorm weight [hidden]
    post_ln: Vec<f32>, // post_attention_layernorm weight [hidden]
    attn: AttentionLayer,
    mlp: LayerMlp,
}

impl GlmLayer {
    pub fn new(
        hidden: usize,
        eps: f32,
        in_ln: Vec<f32>,
        post_ln: Vec<f32>,
        attn: AttentionLayer,
        mlp: LayerMlp,
    ) -> Self {
        assert_eq!(in_ln.len(), hidden);
        assert_eq!(post_ln.len(), hidden);
        Self {
            hidden,
            eps,
            in_ln,
            post_ln,
            attn,
            mlp,
        }
    }

    /// Clear the attention KV cache (new sequence).
    pub fn reset(&mut self) {
        self.attn.reset();
    }

    /// Roll this layer's KV cache back to `len` positions (spec-decode reject).
    pub fn truncate(&mut self, len: usize) {
        self.attn.truncate(len);
    }

    /// Snapshot this layer's KV for prefix caching.
    pub fn snapshot_kv(&self) -> AttnKv {
        self.attn.snapshot()
    }

    /// Restore a KV snapshot into this layer (call after [`Self::reset`]).
    pub fn restore_kv(&mut self, kv: &AttnKv) {
        self.attn.restore(kv);
    }

    /// This layer's MoE block, if it is a sparse (non-dense) layer — for the
    /// learned-pin machinery to enumerate/attach.
    pub fn moe(&self) -> Option<&MoeLayer> {
        match &self.mlp {
            LayerMlp::Moe(m) => Some(m),
            LayerMlp::Dense { .. } => None,
        }
    }

    pub fn moe_mut(&mut self) -> Option<&mut MoeLayer> {
        match &mut self.mlp {
            LayerMlp::Moe(m) => Some(m),
            LayerMlp::Dense { .. } => None,
        }
    }

    /// This layer's `post_attention_layernorm` weight — the norm the router
    /// input passes through. Used by the lookahead prefetch to build the
    /// next-layer proxy `rmsnorm(prev_out, post_ln)` before predicting its experts.
    pub fn post_ln(&self) -> &[f32] {
        &self.post_ln
    }

    /// The dense-layer MLP expert, if this is a dense (first-k) layer — for
    /// pinning the always-active weights.
    pub fn dense_expert(&self) -> Option<&AnyExpert> {
        match &self.mlp {
            LayerMlp::Dense { w, .. } => Some(w),
            LayerMlp::Moe(_) => None,
        }
    }

    /// Process one token at the next cached position. `x` is the residual-stream
    /// hidden `[hidden]`; returns the updated hidden after this layer. `carry`
    /// threads the IndexShare top-k selection (full layer writes, shared reads).
    pub fn forward_token(&mut self, x: &[f32], carry: &mut Option<Vec<usize>>) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        let t_wall = std::time::Instant::now();
        // h = x + attn(rmsnorm(x, in_ln))
        let mut nrm = x.to_vec();
        rmsnorm(&mut nrm, &self.in_ln, self.eps);
        let t_attn = std::time::Instant::now();
        let a = self.attn.forward_token(&nrm, carry);
        prof::add(prof::ATTN, t_attn);
        let mut h: Vec<f32> = x.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        // out = h + mlp(rmsnorm(h, post_ln))
        let mut nrm2 = h.clone();
        rmsnorm(&mut nrm2, &self.post_ln, self.eps);
        let f = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_token(&nrm2),
            LayerMlp::Dense { w, inter } => w.forward(&nrm2, self.hidden, *inter),
        };
        for (hi, &fi) in h.iter_mut().zip(&f) {
            *hi += fi;
        }
        prof::add(prof::WALL, t_wall);
        h
    }

    /// Batched prefill for `rows` tokens (`xs` = `[rows, hidden]`). Attention runs
    /// per position (the causal KV must grow in order); the MoE runs as one
    /// batch-union over all rows, so overlapping experts are loaded once. Returns
    /// `[rows, hidden]`, bit-identical to calling [`Self::forward_token`] per row.
    ///
    /// `route = None` is the untouched Rust path. `Some(..)` offers this
    /// window to the OpenVINO attention backend; the offer is refused (and the
    /// Rust loop runs from scratch, having committed nothing) for any of the
    /// [`OvSkipReason`] causes, all of which the route's tally records.
    pub fn forward_prefill(
        &mut self,
        xs: &[f32],
        rows: usize,
        carries: &mut [Option<Vec<usize>>],
        route: Option<OvRoute<'_>>,
    ) -> Vec<f32> {
        assert_eq!(xs.len(), rows * self.hidden);
        assert_eq!(carries.len(), rows, "one IndexShare carry per prompt row");
        let hd = self.hidden;
        // T2 spike (G2): attention's share of prefill layer time, cumulative
        // across layers and requests. Opt-in via CASCADIA_GLM5_ATTN_PROFILE=1;
        // zero cost when off. Denominator = layer forward time (embed/head and
        // cross-rank relay excluded — stated in the verdict doc).
        let prof = attn_profile_enabled();
        let t_all = prof.then(std::time::Instant::now);
        // h = x + attn(rmsnorm(x, in_ln)); nrm2 = rmsnorm(h, post_ln).  Sequential.
        let mut h = vec![0.0f32; rows * hd];
        let mut nrm2 = vec![0.0f32; rows * hd];
        let t_attn = prof.then(std::time::Instant::now);
        let on_ov = match route {
            Some(r) => self.attn_window_ov(xs, rows, carries, &mut h, &mut nrm2, r),
            None => false,
        };
        if !on_ov {
            for r in 0..rows {
                let x = &xs[r * hd..(r + 1) * hd];
                let mut nrm = x.to_vec();
                rmsnorm(&mut nrm, &self.in_ln, self.eps);
                let a = self.attn.forward_token(&nrm, &mut carries[r]);
                let hrow: Vec<f32> = x.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
                let mut n2 = hrow.clone();
                rmsnorm(&mut n2, &self.post_ln, self.eps);
                h[r * hd..(r + 1) * hd].copy_from_slice(&hrow);
                nrm2[r * hd..(r + 1) * hd].copy_from_slice(&n2);
            }
        }
        if let Some(t) = t_attn {
            ATTN_PREFILL_NS.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        // out = h + mlp(nrm2): MoE batched (dedup expert loads), dense per row.
        let f = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_batch(&nrm2, rows),
            LayerMlp::Dense { w, inter } => {
                let mut f = vec![0.0f32; rows * hd];
                for r in 0..rows {
                    let y = w.forward(&nrm2[r * hd..(r + 1) * hd], hd, *inter);
                    f[r * hd..(r + 1) * hd].copy_from_slice(&y);
                }
                f
            }
        };
        for (hi, &fi) in h.iter_mut().zip(&f) {
            *hi += fi;
        }
        if let Some(t) = t_all {
            use std::sync::atomic::Ordering::Relaxed;
            let total = LAYER_PREFILL_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed)
                + t.elapsed().as_nanos() as u64;
            let attn = ATTN_PREFILL_NS.load(Relaxed);
            tracing::info!(
                target: "cascadia::glm5",
                event = "attn_prefill_share",
                attn_ms = attn / 1_000_000,
                layer_ms = total / 1_000_000,
                share_pct = (attn as f64 / total.max(1) as f64) * 100.0,
            );
        }
        h
    }

    /// Try to serve this whole window's attention from the OpenVINO backend.
    /// Returns `true` only if the window was fully served AND committed; on
    /// `false` NOTHING has been written (no KV rows, no indexer keys, no
    /// carries), so the caller's Rust loop runs from scratch — the fallback is
    /// per-window and total by construction, which is why the commit is the
    /// last fallible step.
    ///
    /// On success this reproduces, per row, exactly what `forward_token` does:
    /// the KV write (via `commit_prefill_rows`), the indexer key append (fed
    /// the post-`in_ln` normed row, NOT the raw residual — see `attn.rs`
    /// `ix.append_key(x)`, whose `x` is `forward_token`'s already-normed
    /// input), the IndexShare carry, and the residual + post-norm composition.
    fn attn_window_ov(
        &mut self,
        xs: &[f32],
        rows: usize,
        carries: &mut [Option<Vec<usize>>],
        h: &mut [f32],
        nrm2: &mut [f32],
        route: OvRoute<'_>,
    ) -> bool {
        let hd = self.hidden;
        let ov = route.ov;
        let base = self.attn.len();
        // Spec-decode verify batches (`ForwardBatch`) enter this same seam with
        // g≈8 rows. Production `min_rows` is 64, so they are excluded here by
        // the rows floor — NOT by anything structural. Lowering `min_rows`
        // below the draft length would route decode-verify through OV.
        if let Some(reason) = ov_ineligible(ov, route.layer_local, base, rows) {
            route.tally.note_skip(reason);
            return false;
        }

        // Both consumers of the pre-norm row want the same buffer: the graph's
        // `x` input and (on a "full" layer) the DSA indexer key.
        let mut normed = vec![0.0f32; rows * hd];
        for r in 0..rows {
            let dst = &mut normed[r * hd..(r + 1) * hd];
            dst.copy_from_slice(&xs[r * hd..(r + 1) * hd]);
            rmsnorm(dst, &self.in_ln, self.eps);
        }

        let (past_lc, past_rc) = self.attn.past_rows();
        let Some(out) =
            ov.prefill_window(route.layer_local, &normed, rows, &past_lc, &past_rc, base)
        else {
            route
                .tally
                .note_skip(super::ov_attn::OvSkipReason::InferFailed);
            return false;
        };

        // Validate everything before touching live state.
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        if !finite(&out.attn_out) || !finite(&out.lc) || !finite(&out.rc) {
            tracing::warn!(
                target: "cascadia::glm5",
                event = "ov_attn_non_finite",
                layer_local = route.layer_local,
                base,
                rows,
            );
            route
                .tally
                .note_skip(super::ov_attn::OvSkipReason::NonFinite);
            return false;
        }
        // The bf16-grid guard lives inside `commit_prefill_rows` (round +
        // debug-assert). Deliberately NOT pre-rounded here: doing so would make
        // that assertion vacuous, which is the one thing standing between a
        // silently-elided Convert on the GPU and a contaminated live KV cache.
        if let Err(e) = self.attn.commit_prefill_rows(&out.lc, &out.rc, rows) {
            tracing::warn!(
                target: "cascadia::glm5",
                event = "ov_attn_commit_failed",
                layer_local = route.layer_local,
                base,
                rows,
                error = %e,
            );
            route
                .tally
                .note_skip(super::ov_attn::OvSkipReason::CommitFailed);
            return false;
        }

        // Past this point nothing can fail. Rows in position order, matching
        // the order `forward_token` would have appended them.
        let publishes_carry = self.attn.has_indexer();
        for r in 0..rows {
            self.attn
                .indexer_append_normed(&normed[r * hd..(r + 1) * hd]);
            // Carry semantics, mirroring `AttentionLayer::forward_token`: a
            // "full" layer publishes its selection; "shared"/plain layers read
            // `carry` and leave it untouched. Eligibility caps `base + rows` at
            // W == index_topk, so every OV-served query is at or below the DSA
            // budget and its selection is the full causal range `0..n` with
            // PER-ROW `n = base + r + 1`.
            if publishes_carry {
                carries[r] = Some((0..base + r + 1).collect());
            }
            let x = &xs[r * hd..(r + 1) * hd];
            let a = &out.attn_out[r * hd..(r + 1) * hd];
            let hrow: Vec<f32> = x.iter().zip(a).map(|(&xi, &ai)| xi + ai).collect();
            let mut n2 = hrow.clone();
            rmsnorm(&mut n2, &self.post_ln, self.eps);
            h[r * hd..(r + 1) * hd].copy_from_slice(&hrow);
            nrm2[r * hd..(r + 1) * hd].copy_from_slice(&n2);
        }
        route.tally.note_used();
        true
    }
}

/// The eligibility gate: `Some(reason)` means this window does not go to OV.
/// `rows > rows_cap` is folded into `WindowTooLarge` rather than left to
/// `prefill_window` — that would trip its caller-bug contract WARN, which is
/// meant for a stride bug, not for a legitimately oversized prompt window.
fn ov_ineligible(
    ov: &super::ov_attn::OvAttn,
    layer_local: usize,
    base: usize,
    rows: usize,
) -> Option<super::ov_attn::OvSkipReason> {
    use super::ov_attn::{OvAttnState, OvSkipReason};
    if ov.state() == OvAttnState::Poisoned {
        return Some(OvSkipReason::Poisoned);
    }
    if !ov.layer_compiled(layer_local) {
        return Some(OvSkipReason::LayerNotCompiled);
    }
    if base + rows > ov.w || rows > ov.rows_cap {
        return Some(OvSkipReason::WindowTooLarge);
    }
    if rows < ov.min_rows as usize {
        return Some(OvSkipReason::RowsBelowMin);
    }
    None
}

/// Cumulative prefill timers for the T2 attention-share measurement
/// (`CASCADIA_GLM5_ATTN_PROFILE=1`); process-wide across layers and requests.
static ATTN_PREFILL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAYER_PREFILL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn attn_profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CASCADIA_GLM5_ATTN_PROFILE").is_some())
}

/// Full GLM-5.2 model: embed → layers → final RMSNorm → lm_head. Single-stream
/// incremental decode (each `forward_token` advances the per-layer KV caches);
/// `lm_head` logits are f32 (argmax). Pipeline-parallel sharding is layered on
/// later at the engine level.
pub struct GlmModel {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    embed: Vec<f32>, // [vocab, hidden]
    layers: Vec<GlmLayer>,
    final_norm: Vec<f32>, // [hidden]
    lm_head: Vec<f32>,    // [vocab, hidden] (f32 -> f32 logits)
    /// Optional MTP draft head for native speculative decode ([`Self::generate_spec`]).
    mtp: Option<MtpHead>,
}

impl GlmModel {
    pub fn new(
        hidden: usize,
        vocab: usize,
        eps: f32,
        embed: Vec<f32>,
        layers: Vec<GlmLayer>,
        final_norm: Vec<f32>,
        lm_head: Vec<f32>,
    ) -> Self {
        assert_eq!(embed.len(), vocab * hidden);
        assert_eq!(final_norm.len(), hidden);
        assert_eq!(lm_head.len(), vocab * hidden);
        Self {
            hidden,
            vocab,
            eps,
            embed,
            layers,
            final_norm,
            lm_head,
            mtp: None,
        }
    }

    /// Attach the MTP draft head, enabling [`Self::generate_spec`]. The head must
    /// be built for this model's `hidden`/`vocab`; it shares `embed`, `final_norm`
    /// and `lm_head` (passed at draft time).
    pub fn set_mtp(&mut self, mtp: MtpHead) {
        assert_eq!(mtp.hidden, self.hidden, "MTP hidden mismatch");
        assert_eq!(mtp.vocab, self.vocab, "MTP vocab mismatch");
        self.mtp = Some(mtp);
    }

    /// Whether an MTP draft head is attached.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Clear all layer KV caches (new sequence).
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }

    /// Roll every layer's KV cache back to `len` positions (spec-decode reject).
    pub fn truncate(&mut self, len: usize) {
        for l in &mut self.layers {
            l.truncate(len);
        }
    }

    /// Snapshot every layer's KV at the current cached length — a reusable prompt
    /// prefix. Restore it (after [`Self::reset`]) to skip re-prefilling that
    /// prefix, then continue with prefill / [`Self::forward_token`] of the suffix.
    /// The per-layer int4 expert *weights* are unchanged; only the KV is saved,
    /// so this trades RAM for the expert-streaming a re-prefill would repeat.
    pub fn snapshot_prefix(&self) -> Vec<AttnKv> {
        self.layers.iter().map(|l| l.snapshot_kv()).collect()
    }

    /// Restore a prefix snapshot into every layer (replaces current KV). The
    /// snapshot must come from a model with this one's layer count and dims.
    pub fn restore_prefix(&mut self, snap: &[AttnKv]) {
        assert_eq!(
            snap.len(),
            self.layers.len(),
            "prefix snapshot layer count {} != {}",
            snap.len(),
            self.layers.len()
        );
        for (l, kv) in self.layers.iter_mut().zip(snap) {
            l.restore_kv(kv);
        }
    }

    /// Embed `token`, run all layers + final norm + lm_head; returns logits
    /// `[vocab]` at this position and advances the KV caches.
    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let t = token as usize;
        assert!(t < self.vocab, "token {t} >= vocab {}", self.vocab);
        let mut x = self.embed[t * self.hidden..(t + 1) * self.hidden].to_vec();
        let mut carry: Option<Vec<usize>> = None; // IndexShare selection, per token
        for l in &mut self.layers {
            x = l.forward_token(&x, &mut carry);
        }
        prof::dump("decode");
        rmsnorm(&mut x, &self.final_norm, self.eps);
        let mut logits = vec![0.0f32; self.vocab];
        linear_f32(&x, &self.lm_head, self.vocab, self.hidden, &mut logits);
        logits
    }

    /// Batched prefill of `prompt`: embed all tokens, run every layer with
    /// per-position attention + batch-union MoE, and return the logits at the
    /// LAST position (the next-token distribution). Advances the KV caches, so
    /// decode continues with [`Self::forward_token`]. Call [`Self::reset`] first
    /// for a fresh sequence. Bit-identical to looping `forward_token` over the
    /// prompt and keeping the last logits.
    pub fn prefill(&mut self, prompt: &[u32]) -> Vec<f32> {
        self.prefill_h(prompt).1
    }

    /// Like [`Self::prefill`] but also returns the LAST position's pre-final-norm
    /// hidden — the `hlast` the MTP draft head consumes (it applies `final_norm`
    /// itself). Returns `(hlast, logits)`.
    pub fn prefill_h(&mut self, prompt: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let rows = prompt.len();
        assert!(rows > 0, "prefill needs a non-empty prompt");
        let hd = self.hidden;
        let mut xs = vec![0.0f32; rows * hd];
        for (r, &t) in prompt.iter().enumerate() {
            let t = t as usize;
            assert!(t < self.vocab, "token {t} >= vocab {}", self.vocab);
            xs[r * hd..(r + 1) * hd].copy_from_slice(&self.embed[t * hd..(t + 1) * hd]);
        }
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows]; // per-row IndexShare
        for l in &mut self.layers {
            // Single-process `GlmModel` never routes through OV: the backend is
            // keyed by this-rank-local layer offsets, which only `GlmRunner`
            // owns. Keeping `None` here is also what makes this the reference
            // path the goldens compare against.
            xs = l.forward_prefill(&xs, rows, &mut carries, None);
        }
        let last = (rows - 1) * hd;
        let hlast = xs[last..last + hd].to_vec();
        let mut x = hlast.clone();
        rmsnorm(&mut x, &self.final_norm, self.eps);
        let mut logits = vec![0.0f32; self.vocab];
        linear_f32(&x, &self.lm_head, self.vocab, self.hidden, &mut logits);
        (hlast, logits)
    }

    /// Speculative-decode verify: run `tokens` through every layer as one
    /// batch-union forward (appending `tokens.len()` KV positions in order,
    /// starting at the current cached length), and return per-position
    /// `(argmax token, pre-final-norm hidden)`. Position `i`'s outputs are the
    /// model's greedy prediction for the token AFTER `tokens[i]` and the hidden
    /// that produced it. Bit-identical to feeding the tokens one at a time.
    fn forward_batch_h(&mut self, tokens: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
        let rows = tokens.len();
        assert!(rows > 0, "verify needs a non-empty batch");
        let hd = self.hidden;
        let mut xs = vec![0.0f32; rows * hd];
        for (r, &t) in tokens.iter().enumerate() {
            let t = t as usize;
            assert!(t < self.vocab, "token {t} >= vocab {}", self.vocab);
            xs[r * hd..(r + 1) * hd].copy_from_slice(&self.embed[t * hd..(t + 1) * hd]);
        }
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows];
        for l in &mut self.layers {
            xs = l.forward_prefill(&xs, rows, &mut carries, None);
        }
        let mut preds = Vec::with_capacity(rows);
        let mut hlasts = Vec::with_capacity(rows);
        let mut logits = vec![0.0f32; self.vocab];
        for r in 0..rows {
            let hlast = xs[r * hd..(r + 1) * hd].to_vec();
            let mut x = hlast.clone();
            rmsnorm(&mut x, &self.final_norm, self.eps);
            linear_f32(&x, &self.lm_head, self.vocab, self.hidden, &mut logits);
            preds.push(argmax(&logits) as u32);
            hlasts.push(hlast);
        }
        (preds, hlasts)
    }

    /// Native MTP speculative decode. Each round: the MTP head drafts `g` greedy
    /// tokens, the main model verifies `[next_tok, drafts…]` in one batch-union
    /// forward, and the longest draft prefix that matches the model's own greedy
    /// argmax is accepted; the first mismatch's KV is rewound. **Output is
    /// token-for-token identical to [`Self::generate`]** (every committed token is
    /// either the model's greedy pick or an accepted draft equal to it) — the win
    /// is purely fewer sequential forwards. Requires [`Self::set_mtp`]; panics
    /// otherwise. `g == 0` degenerates to plain greedy.
    pub fn generate_spec(&mut self, prompt: &[u32], n_gen: usize, g: usize) -> SpecOutput {
        assert!(self.mtp.is_some(), "generate_spec requires set_mtp");
        self.reset();
        let (mut hlast, logits) = self.prefill_h(prompt);
        let mut next_tok = argmax(&logits) as u32;
        let mut len = prompt.len(); // committed KV positions
        let mut out: Vec<u32> = Vec::with_capacity(n_gen);
        let mut forwards = 1usize; // the prefill
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        // Pull the head out so drafting (immutable self borrow) and verify
        // (mutable self borrow) don't overlap.
        let mut mtp = self.mtp.take().unwrap();
        while out.len() < n_gen {
            out.push(next_tok); // commit next_tok at position `len`
            if out.len() >= n_gen {
                break;
            }
            if g == 0 {
                // plain greedy: feed next_tok, read its prediction.
                let (preds, hlasts) = self.forward_batch_h(&[next_tok]);
                forwards += 1;
                next_tok = preds[0];
                hlast = hlasts[0].clone();
                len += 1;
                continue;
            }
            let drafts = mtp.draft(
                &hlast,
                next_tok,
                g,
                &self.embed,
                &self.final_norm,
                &self.lm_head,
            );
            drafted += g;
            // verify [next_tok, drafts…]: appends g+1 positions at [len, len+g].
            let mut verify_in = Vec::with_capacity(g + 1);
            verify_in.push(next_tok);
            verify_in.extend_from_slice(&drafts);
            let (preds, hlasts) = self.forward_batch_h(&verify_in);
            forwards += 1;
            // accept run: draft i is correct iff it equals the greedy token after
            // verify_in[i] (== preds[i]). Over-accepting past n_gen is harmless —
            // the final out.truncate trims it, and every accepted token equals
            // greedy anyway.
            let mut n_acc = 0usize;
            while n_acc < g && drafts[n_acc] == preds[n_acc] {
                n_acc += 1;
            }
            accepted += n_acc;
            for &d in &drafts[..n_acc] {
                out.push(d); // commit accepted drafts at positions [len+1, len+n_acc]
            }
            // next_tok = the greedy correction/bonus after the last accepted
            // position; its hidden drives the next draft. KV valid up to
            // len + 1 + n_acc (next_tok + accepted drafts); drop the rest.
            next_tok = preds[n_acc];
            hlast = hlasts[n_acc].clone();
            len += 1 + n_acc;
            self.truncate(len);
        }
        self.mtp = Some(mtp);
        let _ = len; // last write feeds the final truncate; silence unused-assign
        out.truncate(n_gen);
        SpecOutput {
            tokens: out,
            forwards,
            drafted,
            accepted,
        }
    }

    /// Grammar-constrained greedy generation with forced-run batching. Forced
    /// tokens (grammar admits exactly one) are emitted and their KV advanced as
    /// one batch-union forward; free positions run one forward and argmax over
    /// the grammar-allowed set. Returns the emitted tokens and the number of
    /// model forwards used (fewer than the token count exactly when the grammar
    /// forced runs — the throughput win). Stops at `max_new`, or when the grammar
    /// accepts with nothing forced.
    pub fn generate_grammar(
        &mut self,
        prompt: &[u32],
        grammar: &dyn super::grammar::Grammar,
        max_new: usize,
    ) -> super::grammar::GrammarOutput {
        self.reset();
        let mut logits = self.prefill(prompt);
        let mut forwards = 1usize;
        let mut out: Vec<u32> = Vec::new();
        while out.len() < max_new {
            let forced = grammar.forced_run(&out);
            if !forced.is_empty() {
                let take = forced.len().min(max_new - out.len());
                let run = &forced[..take];
                out.extend_from_slice(run);
                // Advance the KV for the whole forced run in one batched forward.
                logits = self.prefill(run);
                forwards += 1;
            } else if grammar.can_end(&out) {
                break;
            } else {
                let tok = super::grammar::masked_argmax(&logits, grammar, &out);
                out.push(tok);
                if grammar.can_end(&out) || out.len() >= max_new {
                    break;
                }
                logits = self.forward_token(tok);
                forwards += 1;
            }
        }
        super::grammar::GrammarOutput {
            tokens: out,
            forwards,
        }
    }

    /// Greedy generation: batched prefill of `prompt`, then `n_gen` argmax tokens.
    pub fn generate(&mut self, prompt: &[u32], n_gen: usize) -> Vec<u32> {
        self.reset();
        let mut logits = self.prefill(prompt);
        let mut gen = Vec::with_capacity(n_gen);
        for _ in 0..n_gen {
            let nxt = argmax(&logits) as u32;
            gen.push(nxt);
            logits = self.forward_token(nxt);
        }
        gen
    }

    /// Greedy generation with a [`KvPrefixCache`]: if a cached prompt prefix is
    /// found, restore its KV and prefill only the suffix (skipping the shared
    /// prefix's expert streaming); otherwise a full prefill. This prompt's KV is
    /// inserted for future reuse. Bit-identical to [`Self::generate`] (verified);
    /// returns the tokens plus how many leading positions were reused.
    pub fn generate_with_prefix_cache(
        &mut self,
        cache: &mut KvPrefixCache,
        prompt: &[u32],
        n_gen: usize,
    ) -> (Vec<u32>, usize) {
        assert!(
            !prompt.is_empty(),
            "generate_with_prefix_cache needs a non-empty prompt"
        );
        self.reset();
        // Restore the longest cached prefix, then prefill the suffix. We must
        // prefill at least the last prompt token to obtain its logits, so a
        // full-prompt hit is capped to `len-1` (its last position is re-derived).
        let mut reused = 0usize;
        if let Some((k, snap)) = cache.longest_prefix(prompt) {
            self.restore_prefix(snap);
            reused = k.min(prompt.len() - 1);
            if reused < k {
                self.truncate(reused);
            }
        }
        let mut logits = self.prefill(&prompt[reused..]);
        // Cache this prompt's full KV so a future extension of it reuses more.
        cache.insert(prompt.to_vec(), self.snapshot_prefix());
        let mut out = Vec::with_capacity(n_gen);
        for _ in 0..n_gen {
            let nxt = argmax(&logits) as u32;
            out.push(nxt);
            logits = self.forward_token(nxt);
        }
        (out, reused)
    }
}

/// Result of [`GlmModel::generate_spec`]: the generated tokens plus the
/// speculative-decode accounting. `forwards` is the number of main-model forward
/// passes (prefill + one per verify round); `accepted / drafted` is the MTP
/// acceptance rate. Fewer `forwards` than `tokens.len()` is the throughput win.
pub struct SpecOutput {
    pub tokens: Vec<u32>,
    pub forwards: usize,
    pub drafted: usize,
    pub accepted: usize,
}

/// Index of the first maximum (ties -> lowest index, matching `torch.argmax`).
fn argmax(v: &[f32]) -> usize {
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

#[cfg(test)]
mod ov_prefill_tests {
    //! Prefill-seam tests. Every OV route here is driven by `OvAttn::mock`,
    //! which returns canned window outputs *after* the real poison latch and
    //! argument-contract checks — so these exercise the production control
    //! flow on a machine with no OpenVINO SDK.

    use super::*;
    use crate::dsv4::math::to_bf16;
    use crate::dsv4::rope::precompute_freqs;
    use crate::glm::attn::AttnWeights;
    use crate::glm::indexer::{Indexer, IndexerWeights};
    use crate::glm::moe::ExpertW;
    use crate::glm::ov_attn::{AttnWindowOut, MockCfg, OvAttn, OvPrefillTally, OvSkipReason};
    use std::sync::{Arc, Mutex};

    const HIDDEN: usize = 12;
    const KV_LORA: usize = 6;
    const QK_ROPE: usize = 4;
    const Q_LORA: usize = 8;
    const INDEX_HD: usize = 4;
    const MAX_SEQ: usize = 32;
    /// Stands in for `index_topk` — the offload budget W the mock advertises.
    const W: usize = 16;

    fn lcg(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / u16::MAX as f32 - 0.5
        }
    }

    fn bits(v: f32) -> u16 {
        half::bf16::from_f32(v).to_bits()
    }

    /// Deterministic tiny layer — same seed every call, so two instances are
    /// byte-identical twins and can stand in for "the same layer" run twice.
    /// `indexer`: `None` = plain, `Some(true)` = IndexShare `"full"` (owns an
    /// indexer), `Some(false)` = `"shared"`.
    fn tiny_layer(indexer: Option<bool>) -> GlmLayer {
        let (h, nope, rope, vh, inter) = (2usize, 4usize, 4usize, 4usize, 8usize);
        let mut rnd = lcg(7);
        let aw = AttnWeights {
            wq_a: (0..Q_LORA * HIDDEN).map(|_| bits(rnd())).collect(),
            q_a_ln: (0..Q_LORA).map(|_| 1.0 + rnd() * 0.1).collect(),
            wq_b: (0..h * (nope + rope) * Q_LORA)
                .map(|_| bits(rnd()))
                .collect(),
            wkv_a: (0..(KV_LORA + rope) * HIDDEN)
                .map(|_| bits(rnd()))
                .collect(),
            kv_a_ln: (0..KV_LORA).map(|_| 1.0 + rnd() * 0.1).collect(),
            wkv_b: (0..h * (nope + vh) * KV_LORA)
                .map(|_| bits(rnd()))
                .collect(),
            wo: (0..HIDDEN * h * vh).map(|_| bits(rnd())).collect(),
        };
        let freqs = precompute_freqs(rope, MAX_SEQ, 0, 1.0e4, 1.0, 32.0, 1.0);
        let mut attn = AttentionLayer::new(
            HIDDEN, h, nope, rope, vh, KV_LORA, Q_LORA, MAX_SEQ, aw, freqs,
        );
        match indexer {
            Some(true) => {
                let (nh, hd) = (2usize, INDEX_HD);
                let iw = IndexerWeights {
                    ix_wq: (0..nh * hd * Q_LORA).map(|_| bits(rnd())).collect(),
                    ix_wk: (0..hd * HIDDEN).map(|_| bits(rnd())).collect(),
                    ix_wp: (0..nh * HIDDEN).map(|_| bits(rnd())).collect(),
                    k_norm_w: (0..hd).map(|_| 1.0 + rnd() * 0.1).collect(),
                    k_norm_b: (0..hd).map(|_| rnd() * 0.1).collect(),
                };
                let ifreqs = precompute_freqs(QK_ROPE, MAX_SEQ, 0, 1.0e4, 1.0, 32.0, 1.0);
                attn.attach_indexer(
                    Indexer::new(HIDDEN, Q_LORA, nh, hd, QK_ROPE, MAX_SEQ, 1e-6, iw, ifreqs),
                    W,
                );
            }
            Some(false) => attn.mark_shared(W),
            None => {}
        }
        let dense = ExpertW {
            wg: (0..inter * HIDDEN).map(|_| bits(rnd())).collect(),
            wu: (0..inter * HIDDEN).map(|_| bits(rnd())).collect(),
            wd: (0..HIDDEN * inter).map(|_| bits(rnd())).collect(),
        };
        // in_ln / post_ln deliberately NOT all-ones: a caller that fed the raw
        // residual where the normed row belongs must produce different numbers.
        GlmLayer::new(
            HIDDEN,
            1e-5,
            (0..HIDDEN).map(|_| 1.0 + rnd() * 0.4).collect(),
            (0..HIDDEN).map(|_| 1.0 + rnd() * 0.4).collect(),
            attn,
            LayerMlp::Dense {
                w: AnyExpert::Bf16(dense),
                inter,
            },
        )
    }

    fn rows_of(n: usize, seed: u32) -> Vec<f32> {
        let mut rnd = lcg(seed);
        (0..n * HIDDEN).map(|_| rnd()).collect()
    }

    /// Canned window outputs, on the bf16 grid so `commit_prefill_rows`'s
    /// on-grid debug assertion is satisfied. (An off-grid canned value is a
    /// test bug, not a scenario — the real off-grid case is Task 4's own test.)
    fn canned(rows: usize, seed: u32) -> AttnWindowOut {
        let mut rnd = lcg(seed);
        AttnWindowOut {
            attn_out: (0..rows * HIDDEN).map(|_| to_bf16(rnd())).collect(),
            lc: (0..rows * KV_LORA).map(|_| to_bf16(rnd())).collect(),
            rc: (0..rows * QK_ROPE).map(|_| to_bf16(rnd())).collect(),
        }
    }

    fn mock_cfg() -> MockCfg {
        MockCfg {
            compiled: vec![true],
            w: W,
            p_max: 8,
            rows_cap: 8,
            hidden: HIDDEN,
            kv_lora: KV_LORA,
            qk_rope: QK_ROPE,
            min_rows: 4,
        }
    }

    /// Every `(layer_local, rows, past_len)` the backend was asked for.
    type Calls = Arc<Mutex<Vec<(usize, usize, usize)>>>;

    fn recording_mock(
        cfg: MockCfg,
        out: impl Fn(usize) -> AttnWindowOut + Send + Sync + 'static,
    ) -> (OvAttn, Calls) {
        let calls: Calls = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&calls);
        let ov = OvAttn::mock(
            cfg,
            Box::new(move |ll, _x, rows, past| {
                sink.lock().unwrap().push((ll, rows, past));
                Some(out(rows))
            }),
        );
        (ov, calls)
    }

    fn run_ov(
        layer: &mut GlmLayer,
        ov: &OvAttn,
        xs: &[f32],
        rows: usize,
        carries: &mut [Option<Vec<usize>>],
    ) -> (Vec<f32>, OvPrefillTally) {
        let mut tally = OvPrefillTally::default();
        let out = layer.forward_prefill(
            xs,
            rows,
            carries,
            Some(crate::glm::ov_attn::OvRoute {
                ov,
                layer_local: 0,
                tally: &mut tally,
            }),
        );
        (out, tally)
    }

    // ---- gate 3: off means off --------------------------------------------

    /// `route = None` must be the pre-existing path, byte for byte — including
    /// the KV, the indexer keys and the carries, not just the returned hidden.
    #[test]
    fn route_none_is_byte_identical_to_forward_token() {
        let rows = 5usize;
        let xs = rows_of(rows, 11);
        let mut a = tiny_layer(Some(true));
        let mut b = tiny_layer(Some(true));

        let mut want = Vec::new();
        let mut carry_a: Option<Vec<usize>> = None;
        let mut carries_a = Vec::new();
        for r in 0..rows {
            want.extend(a.forward_token(&xs[r * HIDDEN..(r + 1) * HIDDEN], &mut carry_a));
            carries_a.push(carry_a.clone());
        }

        let mut carries_b: Vec<Option<Vec<usize>>> = vec![None; rows];
        let got = b.forward_prefill(&xs, rows, &mut carries_b, None);

        assert_eq!(got, want, "prefill output diverged from forward_token");
        assert_eq!(a.attn.len(), b.attn.len());
        assert_eq!(a.attn.past_rows(), b.attn.past_rows(), "KV diverged");
        assert_eq!(
            a.attn.indexer_keys(),
            b.attn.indexer_keys(),
            "indexer keys diverged"
        );
        assert_eq!(carries_a, carries_b, "IndexShare carries diverged");
    }

    // ---- the routed happy path --------------------------------------------

    #[test]
    fn routed_window_commits_the_graphs_rows_and_uses_its_attn_out() {
        let rows = 5usize;
        let xs = rows_of(rows, 11);
        let out = canned(rows, 91);
        let (want_lc, want_rc, want_attn) = (out.lc, out.rc, out.attn_out);
        let (ov, calls) = recording_mock(mock_cfg(), |n| canned(n, 91));

        let mut layer = tiny_layer(Some(true));
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows];
        let (got, tally) = run_ov(&mut layer, &ov, &xs, rows, &mut carries);

        assert_eq!(*calls.lock().unwrap(), vec![(0, rows, 0)]);
        assert!(tally.used());
        assert_eq!((tally.layers_ov, tally.layers_rust), (1, 0));
        assert_eq!(tally.skipped_reason(), "none");

        // The graph's lc/rc are what landed in the live KV cache.
        assert_eq!(layer.attn.len(), rows);
        assert_eq!(layer.attn.past_rows(), (want_lc, want_rc));

        // The graph's attn_out is what the residual used:
        //   h = x + attn_out;  out = h + mlp(rmsnorm(h, post_ln)).
        let post_ln = layer.post_ln().to_vec();
        let mut recomposed = vec![0.0f32; rows * HIDDEN];
        for r in 0..rows {
            let hrow: Vec<f32> = (0..HIDDEN)
                .map(|c| xs[r * HIDDEN + c] + want_attn[r * HIDDEN + c])
                .collect();
            let mut n2 = hrow.clone();
            rmsnorm(&mut n2, &post_ln, layer.eps);
            let f = match &layer.mlp {
                LayerMlp::Dense { w, inter } => w.forward(&n2, HIDDEN, *inter),
                LayerMlp::Moe(_) => unreachable!("tiny layer is dense"),
            };
            for c in 0..HIDDEN {
                recomposed[r * HIDDEN + c] = hrow[c] + f[c];
            }
        }
        assert_eq!(got, recomposed, "residual/post-ln composition diverged");
    }

    // ---- the test that matters most: indexer feeding ----------------------

    /// The DSA indexer must receive exactly what `forward_token` feeds it — the
    /// post-`in_ln` normed row — for every row, in position order.
    ///
    /// The canned `attn_out` is deliberately unrelated to anything the Rust
    /// path would produce: the indexer keys depend ONLY on the layer's input
    /// rows, so they must still match the Rust run bit for bit. Feeding the raw
    /// residual, or feeding rows out of order, breaks this — the two negative
    /// controls below prove the assertion is not vacuous.
    #[test]
    fn routed_window_feeds_the_indexer_the_normed_row_in_order() {
        let rows = 6usize;
        let xs = rows_of(rows, 11);

        let mut rust = tiny_layer(Some(true));
        let mut carries_rust: Vec<Option<Vec<usize>>> = vec![None; rows];
        rust.forward_prefill(&xs, rows, &mut carries_rust, None);
        let want = rust
            .attn
            .indexer_keys()
            .expect("full layer owns an indexer");

        let (ov, _) = recording_mock(mock_cfg(), |n| canned(n, 555));
        let mut routed = tiny_layer(Some(true));
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows];
        let (_, tally) = run_ov(&mut routed, &ov, &xs, rows, &mut carries);
        assert!(tally.used(), "window must have been routed");

        assert_eq!(
            routed.attn.indexer_keys().unwrap(),
            want,
            "OV route fed the indexer something other than what forward_token feeds it"
        );
        assert_eq!(want.len(), rows * INDEX_HD, "one key per row, no more");

        // Negative control 1: the RAW residual instead of the normed row.
        let mut raw = tiny_layer(Some(true));
        for r in 0..rows {
            raw.attn
                .indexer_append_normed(&xs[r * HIDDEN..(r + 1) * HIDDEN]);
        }
        assert_ne!(
            raw.attn.indexer_keys().unwrap(),
            want,
            "raw-residual feeding must NOT match — else the assertion above is vacuous"
        );

        // Negative control 2: correct rows, reversed order.
        let mut shuffled = tiny_layer(Some(true));
        let (in_ln, eps) = (shuffled.in_ln.clone(), shuffled.eps);
        for r in (0..rows).rev() {
            let mut n = xs[r * HIDDEN..(r + 1) * HIDDEN].to_vec();
            rmsnorm(&mut n, &in_ln, eps);
            shuffled.attn.indexer_append_normed(&n);
        }
        assert_ne!(
            shuffled.attn.indexer_keys().unwrap(),
            want,
            "out-of-order feeding must NOT match — the key's rope makes position load-bearing"
        );
    }

    /// A `"shared"` layer owns no indexer: the routed path must append no keys
    /// and must not touch the carry it only ever reads.
    #[test]
    fn routed_shared_layer_appends_no_keys_and_leaves_the_carry_alone() {
        let rows = 5usize;
        let xs = rows_of(rows, 11);
        let sentinel = || vec![Some(vec![0usize]); rows];

        let mut rust = tiny_layer(Some(false));
        let mut carries_rust = sentinel();
        rust.forward_prefill(&xs, rows, &mut carries_rust, None);

        let (ov, _) = recording_mock(mock_cfg(), |n| canned(n, 77));
        let mut routed = tiny_layer(Some(false));
        let mut carries = sentinel();
        let (_, tally) = run_ov(&mut routed, &ov, &xs, rows, &mut carries);

        assert!(tally.used());
        assert!(routed.attn.indexer_keys().is_none());
        assert_eq!(carries, carries_rust);
        assert_eq!(carries, sentinel(), "shared layer must not write the carry");
    }

    // ---- carry semantics ---------------------------------------------------

    /// A `"full"` layer publishes the same per-row carry the Rust path does:
    /// full causal `0..n` with PER-ROW `n = base + r + 1`. Run at a nonzero
    /// base so a `0..rows` off-by-base bug cannot pass.
    #[test]
    fn routed_carries_match_the_rust_path_at_a_nonzero_base() {
        let (warm, rows) = (3usize, 5usize);
        let xs = rows_of(warm + rows, 11);
        let (head, tail) = xs.split_at(warm * HIDDEN);

        let mut rust = tiny_layer(Some(true));
        let mut routed = tiny_layer(Some(true));
        for l in [&mut rust, &mut routed] {
            let mut c: Option<Vec<usize>> = None;
            for r in 0..warm {
                l.forward_token(&head[r * HIDDEN..(r + 1) * HIDDEN], &mut c);
            }
        }
        assert_eq!(routed.attn.len(), warm);

        let mut carries_rust: Vec<Option<Vec<usize>>> = vec![None; rows];
        rust.forward_prefill(tail, rows, &mut carries_rust, None);

        let (ov, calls) = recording_mock(mock_cfg(), |n| canned(n, 33));
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows];
        let (_, tally) = run_ov(&mut routed, &ov, tail, rows, &mut carries);

        assert!(tally.used());
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(0, rows, warm)],
            "past_len must be the layer's committed length, not 0"
        );
        assert_eq!(carries, carries_rust);
        for (r, c) in carries.iter().enumerate() {
            assert_eq!(
                c.as_deref(),
                Some((0..warm + r + 1).collect::<Vec<_>>().as_slice()),
                "row {r}"
            );
        }
    }

    // ---- fallback is per-window and total ----------------------------------

    /// Every failure mode must leave the layer exactly as the Rust path would —
    /// same output, same KV, same indexer keys, same carries — and must name
    /// itself in the tally.
    #[test]
    fn every_failure_mode_falls_back_totally_and_names_itself() {
        let rows = 5usize;
        let xs = rows_of(rows, 11);

        let mut rust = tiny_layer(Some(true));
        let mut carries_rust: Vec<Option<Vec<usize>>> = vec![None; rows];
        let want = rust.forward_prefill(&xs, rows, &mut carries_rust, None);
        let want_kv = rust.attn.past_rows();
        let want_ix = rust.attn.indexer_keys();

        let cases: Vec<(&str, OvAttn, OvSkipReason)> = vec![
            // prefill_window returned None.
            (
                "infer_failed",
                OvAttn::mock(mock_cfg(), Box::new(|_, _, _, _| None)),
                OvSkipReason::InferFailed,
            ),
            // A NaN in a real row.
            (
                "non_finite",
                OvAttn::mock(
                    mock_cfg(),
                    Box::new(|_, _, n, _| {
                        let mut o = canned(n, 5);
                        o.attn_out[0] = f32::NAN;
                        Some(o)
                    }),
                ),
                OvSkipReason::NonFinite,
            ),
            // Short lc: commit_prefill_rows rejects it before writing anything.
            (
                "commit_failed",
                OvAttn::mock(
                    mock_cfg(),
                    Box::new(|_, _, n, _| {
                        let mut o = canned(n, 5);
                        o.lc.truncate(o.lc.len() - KV_LORA);
                        Some(o)
                    }),
                ),
                OvSkipReason::CommitFailed,
            ),
            (
                "layer_not_compiled",
                OvAttn::mock(
                    MockCfg {
                        compiled: vec![false],
                        ..mock_cfg()
                    },
                    Box::new(|_, _, n, _| Some(canned(n, 5))),
                ),
                OvSkipReason::LayerNotCompiled,
            ),
            (
                "rows_below_min",
                OvAttn::mock(
                    MockCfg {
                        min_rows: (rows + 1) as u32,
                        ..mock_cfg()
                    },
                    Box::new(|_, _, n, _| Some(canned(n, 5))),
                ),
                OvSkipReason::RowsBelowMin,
            ),
            (
                "window_too_large_budget",
                OvAttn::mock(
                    MockCfg {
                        w: rows - 1,
                        ..mock_cfg()
                    },
                    Box::new(|_, _, n, _| Some(canned(n, 5))),
                ),
                OvSkipReason::WindowTooLarge,
            ),
            (
                "window_too_large_rows_cap",
                OvAttn::mock(
                    MockCfg {
                        rows_cap: rows - 1,
                        ..mock_cfg()
                    },
                    Box::new(|_, _, n, _| Some(canned(n, 5))),
                ),
                OvSkipReason::WindowTooLarge,
            ),
        ];

        for (label, ov, reason) in cases {
            let mut layer = tiny_layer(Some(true));
            let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows];
            let (got, tally) = run_ov(&mut layer, &ov, &xs, rows, &mut carries);
            assert_eq!(got, want, "{label}: output diverged from the Rust path");
            assert_eq!(layer.attn.past_rows(), want_kv, "{label}: KV diverged");
            assert_eq!(
                layer.attn.indexer_keys(),
                want_ix,
                "{label}: indexer keys diverged"
            );
            assert_eq!(carries, carries_rust, "{label}: carries diverged");
            assert!(!tally.used(), "{label}: must not report used");
            assert_eq!(tally.layers_rust, 1, "{label}");
            assert_eq!(tally.skipped_reason(), reason.as_str(), "{label}");
        }
    }

    /// A latched backend must report `poisoned` rather than fall back silently
    /// or be misreported as `infer_failed` — the "dead path that still says
    /// Active" defect this enum exists to prevent.
    #[test]
    fn a_poisoned_backend_reports_itself() {
        let rows = 5usize;
        let xs = rows_of(rows, 11);
        let ov = OvAttn::mock(mock_cfg(), Box::new(|_, _, _, _| None));
        // Latch it through the real bookkeeping: three consecutive failures.
        for _ in 0..3 {
            assert!(ov
                .prefill_window(0, &[0.0; HIDDEN], 1, &[], &[], 0)
                .is_none());
        }

        let mut layer = tiny_layer(Some(true));
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows];
        let (_, tally) = run_ov(&mut layer, &ov, &xs, rows, &mut carries);
        assert_eq!(
            tally.skipped_reason(),
            "poisoned",
            "a latched path must be named, not lumped into infer_failed"
        );
        assert_eq!(tally.layers_rust, 1);
        assert!(!tally.used());
    }

    /// The tally is a per-call aggregate over layers: both counts, and the
    /// FIRST skip reason.
    #[test]
    fn tally_aggregates_mixed_layers_and_keeps_the_first_reason() {
        let mut t = OvPrefillTally::default();
        assert_eq!(t.skipped_reason(), "none");
        t.note_used();
        t.note_skip(OvSkipReason::RowsBelowMin);
        t.note_skip(OvSkipReason::InferFailed);
        t.note_used();
        assert_eq!((t.layers_ov, t.layers_rust), (2, 2));
        assert!(t.used());
        assert_eq!(t.skipped_reason(), "rows_below_min");
    }
}
