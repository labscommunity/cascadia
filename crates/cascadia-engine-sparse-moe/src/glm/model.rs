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

    /// Process one token at the next cached position. `x` is the residual-stream
    /// hidden `[hidden]`; returns the updated hidden after this layer.
    pub fn forward_token(&mut self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        // h = x + attn(rmsnorm(x, in_ln))
        let mut nrm = x.to_vec();
        rmsnorm(&mut nrm, &self.in_ln, self.eps);
        let a = self.attn.forward_token(&nrm);
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
    pub fn forward_prefill(&mut self, xs: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(xs.len(), rows * self.hidden);
        let hd = self.hidden;
        // h = x + attn(rmsnorm(x, in_ln)); nrm2 = rmsnorm(h, post_ln).  Sequential.
        let mut h = vec![0.0f32; rows * hd];
        let mut nrm2 = vec![0.0f32; rows * hd];
        for r in 0..rows {
            let x = &xs[r * hd..(r + 1) * hd];
            let mut nrm = x.to_vec();
            rmsnorm(&mut nrm, &self.in_ln, self.eps);
            let a = self.attn.forward_token(&nrm);
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
        Self { hidden, vocab, eps, embed, layers, final_norm, lm_head }
    }

    /// Clear all layer KV caches (new sequence).
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }

    /// Embed `token`, run all layers + final norm + lm_head; returns logits
    /// `[vocab]` at this position and advances the KV caches.
    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let t = token as usize;
        assert!(t < self.vocab, "token {t} >= vocab {}", self.vocab);
        let mut x = self.embed[t * self.hidden..(t + 1) * self.hidden].to_vec();
        for l in &mut self.layers {
            x = l.forward_token(&x);
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
        let rows = prompt.len();
        assert!(rows > 0, "prefill needs a non-empty prompt");
        let hd = self.hidden;
        let mut xs = vec![0.0f32; rows * hd];
        for (r, &t) in prompt.iter().enumerate() {
            let t = t as usize;
            assert!(t < self.vocab, "token {t} >= vocab {}", self.vocab);
            xs[r * hd..(r + 1) * hd].copy_from_slice(&self.embed[t * hd..(t + 1) * hd]);
        }
        for l in &mut self.layers {
            xs = l.forward_prefill(&xs, rows);
        }
        let last = (rows - 1) * hd;
        let mut x = xs[last..last + hd].to_vec();
        rmsnorm(&mut x, &self.final_norm, self.eps);
        let mut logits = vec![0.0f32; self.vocab];
        linear_f32(&x, &self.lm_head, self.vocab, self.hidden, &mut logits);
        logits
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
