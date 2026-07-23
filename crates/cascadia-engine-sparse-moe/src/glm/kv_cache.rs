//! Prompt-prefix KV cache for GLM-5.2 (single-process).
//!
//! Keyed by prompt token prefix, stores the per-layer KV snapshot ([`AttnKv`])
//! taken after prefilling that prefix. A later request that shares the prefix
//! restores it and prefills only the new suffix — skipping the expert-weight
//! streaming a full re-prefill repeats, which a profiled decode shows is ~78% of
//! the time on this NVMe-bound workload. Agentic loops are the target: the
//! system prompt + accumulated history is re-sent every step.
//!
//! Bounded LRU (MRU at the back). Snapshots cost RAM — but GLM's MLA latent KV
//! is compressed, so a cached prefix is far cheaper than a full-KV model would
//! be. This is the single-process store; the cross-rank per-stage snapshot
//! exchange (the `PipelineEngine` deployment) is layered on top.

use super::attn::AttnKv;

pub struct KvPrefixCache {
    /// `(prefix tokens, KV snapshot for [0, tokens.len()))`. LRU: MRU at the back.
    entries: Vec<(Vec<u32>, Vec<AttnKv>)>,
    cap: usize,
}

impl KvPrefixCache {
    /// A cache holding at most `cap` prefixes (>= 1).
    pub fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap: cap.max(1),
        }
    }

    /// The longest cached prefix of `prompt` and its KV snapshot, or `None`.
    /// Only returns a *strict-or-equal* token prefix, so restoring it and
    /// prefilling `prompt[k..]` is bit-identical to a full prefill.
    pub fn longest_prefix(&self, prompt: &[u32]) -> Option<(usize, &Vec<AttnKv>)> {
        let mut best: Option<(usize, &Vec<AttnKv>)> = None;
        for (toks, snap) in &self.entries {
            let k = toks.len();
            if k <= prompt.len() && prompt[..k] == toks[..] && best.map_or(true, |(b, _)| k > b) {
                best = Some((k, snap));
            }
        }
        best
    }

    /// Insert (or refresh to MRU) the snapshot for `tokens`. Evicts the LRU entry
    /// when over capacity.
    pub fn insert(&mut self, tokens: Vec<u32>, snap: Vec<AttnKv>) {
        if let Some(i) = self.entries.iter().position(|(t, _)| *t == tokens) {
            self.entries.remove(i);
        }
        self.entries.push((tokens, snap));
        while self.entries.len() > self.cap {
            self.entries.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
