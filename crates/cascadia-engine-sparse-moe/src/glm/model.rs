//! GLM-5.2 transformer layer + (later) full model assembly.
//!
//! Layer forward is pre-norm:
//!   h   = x + attention(rmsnorm(x, input_layernorm))
//!   out = h + mlp(rmsnorm(h, post_attention_layernorm))
//! where `mlp` is the MoE block on sparse layers or a dense SwiGLU on the first
//! `first_k_dense_replace` (3) layers. Both norms are RMSNorm, eps
//! `rms_norm_eps` (1e-5).

use super::attn::AttentionLayer;
use super::moe::{AnyExpert, MoeLayer};
use super::mtp::MtpHead;
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
        Self { hidden, eps, in_ln, post_ln, attn, mlp }
    }

    /// Clear the attention KV cache (new sequence).
    pub fn reset(&mut self) {
        self.attn.reset();
    }

    /// Roll this layer's KV cache back to `len` positions (spec-decode reject).
    pub fn truncate(&mut self, len: usize) {
        self.attn.truncate(len);
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
        // h = x + attn(rmsnorm(x, in_ln))
        let mut nrm = x.to_vec();
        rmsnorm(&mut nrm, &self.in_ln, self.eps);
        let a = self.attn.forward_token(&nrm, carry);
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
        h
    }

    /// Batched prefill for `rows` tokens (`xs` = `[rows, hidden]`). Attention runs
    /// per position (the causal KV must grow in order); the MoE runs as one
    /// batch-union over all rows, so overlapping experts are loaded once. Returns
    /// `[rows, hidden]`, bit-identical to calling [`Self::forward_token`] per row.
    pub fn forward_prefill(
        &mut self,
        xs: &[f32],
        rows: usize,
        carries: &mut [Option<Vec<usize>>],
    ) -> Vec<f32> {
        assert_eq!(xs.len(), rows * self.hidden);
        assert_eq!(carries.len(), rows, "one IndexShare carry per prompt row");
        let hd = self.hidden;
        // h = x + attn(rmsnorm(x, in_ln)); nrm2 = rmsnorm(h, post_ln).  Sequential.
        let mut h = vec![0.0f32; rows * hd];
        let mut nrm2 = vec![0.0f32; rows * hd];
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
        h
    }
}

/// Full GLM-5.2 model: embed → layers → final RMSNorm → lm_head. Single-stream
/// incremental decode (each `forward_token` advances the per-layer KV caches);
/// `lm_head` logits are f32 (argmax). Pipeline-parallel sharding is layered on
/// later at the engine level.
pub struct GlmModel {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    embed: Vec<f32>,      // [vocab, hidden]
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
        Self { hidden, vocab, eps, embed, layers, final_norm, lm_head, mtp: None }
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
            xs = l.forward_prefill(&xs, rows, &mut carries);
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
            xs = l.forward_prefill(&xs, rows, &mut carries);
        }
        let mut preds = Vec::with_capacity(rows);
        let mut hlasts = Vec::with_capacity(rows);
        let mut logits = vec![0.0f32; self.vocab];
        for r in 0..rows {
            let hlast = xs[r * hd..(r + 1) * hd].to_vec();
            let mut x = hlast.clone();
            rmsnorm(&mut x, &self.final_norm, self.eps);
            logits.iter_mut().for_each(|v| *v = 0.0);
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
            let drafts = mtp.draft(&hlast, next_tok, g, &self.embed, &self.final_norm, &self.lm_head);
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
        SpecOutput { tokens: out, forwards, drafted, accepted }
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
        super::grammar::GrammarOutput { tokens: out, forwards }
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
