//! Grammar-constrained decoding with **forced-run batching**.
//!
//! When a grammar (JSON/tool-call schema, a template, …) admits exactly one
//! legal continuation, those tokens are *mandated*, not predicted — there is no
//! accept/reject. So instead of one model forward per forced token we emit the
//! whole forced run and advance the KV cache for it in a single batch-union
//! forward ([`crate::glm::model::GlmModel::prefill`] mid-stream). K forced
//! tokens then cost ~1 forward instead of K, which is exactly the win on
//! structured / tool-call output — and it needs no draft head, so it holds up
//! in the disk-bound regime.
//!
//! At a *free* position (more than one token allowed) the model runs one forward
//! and its logits are masked to the allowed set before argmax/sampling.

/// Result of a grammar-constrained generation: the emitted tokens and the
/// number of model forwards spent. `forwards < tokens.len()` exactly when the
/// grammar forced runs that were batched — the throughput the feature buys.
pub struct GrammarOutput {
    pub tokens: Vec<u32>,
    pub forwards: usize,
}

/// A constrained-decoding grammar over token ids. Implementations are the state
/// machine (a JSON PDA, a template, …); the engine drives them via this surface.
/// `emitted` is the generation so far (excluding the prompt).
pub trait Grammar {
    /// The maximal run of tokens the grammar FORCES next given `emitted` — i.e.
    /// the deterministic continuation where each position admits exactly one
    /// token. Empty when the next token is a free choice. These are emitted
    /// without sampling and their KV is advanced as one batch.
    fn forced_run(&self, emitted: &[u32]) -> Vec<u32>;

    /// Whether `token` is a legal next token at a free position given `emitted`.
    /// Used to mask the logits before argmax/sampling.
    fn allows(&self, emitted: &[u32], token: u32) -> bool;

    /// Whether generation may terminate after `emitted` (grammar reached an
    /// accepting state). The caller still honors its own max-token / EOS limits.
    fn can_end(&self, emitted: &[u32]) -> bool;
}

/// Index of the maximum logit among grammar-allowed tokens (ties → lowest index,
/// matching `torch.argmax`). Panics only if the grammar allows nothing, which is
/// a grammar bug the caller should avoid by checking `can_end` first.
pub fn masked_argmax(logits: &[f32], grammar: &dyn Grammar, emitted: &[u32]) -> u32 {
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in logits.iter().enumerate() {
        if grammar.allows(emitted, i as u32) {
            match best {
                Some((_, bv)) if x <= bv => {}
                _ => best = Some((i, x)),
            }
        }
    }
    best.expect("grammar allows no token at a free position").0 as u32
}
