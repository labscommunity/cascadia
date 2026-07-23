//! `StagedRunner` — the arch-agnostic surface the pipeline engine
//! ([`crate::engine::PipelineEngine`]) drives. One contiguous layer slice per
//! rank: rank 0 embeds + drives, mids relay the hidden state, the last rank
//! runs the head + sampler. Implemented by the dsv4 and glm5 Rust shells
//! (minimax keeps its own engine — different disconnect semantics).
//!
//! `generate` / `generate_argmax` are provided (default) methods built on the
//! required primitives, so every backend shares one single-stage sampling loop.

use crate::sampling::{init_rng, sample, SamplingConfig};

pub trait StagedRunner: Send + 'static {
    /// Short backend name for log lines (`"dsv4"`, `"glm5"`).
    fn arch_name(&self) -> &'static str;

    /// Inter-stage hidden width (the residual-stream width on the wire).
    fn hidden_size(&self) -> usize;

    /// Context budget the caches were sized for; the driver must not forward a
    /// token at an absolute position `>= max_seq`.
    fn max_seq(&self) -> usize;

    /// Stop-token ids (generation ends on any).
    fn eos_token_ids(&self) -> &[u32];

    /// Clear all per-generation state across this stage's layers.
    fn reset(&mut self);

    /// Rank-0 only: token id -> hidden.
    fn embed_token(&self, token: u32) -> Vec<f32>;

    /// Run this stage's layers for one token at absolute `pos`. `token` is
    /// available on the stage that needs raw ids (e.g. dsv4 hash gates, always
    /// rank 0); other backends/ranks ignore it.
    fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, token: Option<u32>) -> Vec<f32>;

    /// Run `rows` contiguous positions (`base..base+rows`) through this stage's
    /// layers as a batch — the prefill path. `hidden` is `[rows, hidden_size]`;
    /// returns `[rows, hidden_size]`. The default loops [`Self::forward_layers`]
    /// per row (bit-exact, correct for any runner); a backend may override to
    /// batch its MoE so overlapping experts are loaded once. Advances the same
    /// KV state as `rows` sequential `forward_layers` calls.
    fn forward_layers_batch(&mut self, hidden: Vec<f32>, base: usize, rows: usize) -> Vec<f32> {
        let hs = self.hidden_size();
        assert_eq!(
            hidden.len(),
            rows * hs,
            "forward_layers_batch: bad hidden length"
        );
        let mut out = vec![0.0f32; rows * hs];
        for r in 0..rows {
            let h = hidden[r * hs..(r + 1) * hs].to_vec();
            let o = self.forward_layers(h, base + r, None);
            out[r * hs..(r + 1) * hs].copy_from_slice(&o);
        }
        out
    }

    /// Whether [`Self::generate`] may prefill the prompt as one batch via
    /// [`Self::forward_layers_batch`] (dedup expert loads, one head GEMV). Off by
    /// default — a backend must opt in, and only when its batched path needs no
    /// per-position token id (e.g. dsv4 hash gates require the id, so it stays
    /// per-token). Bit-exact either way.
    fn supports_batched_prefill(&self) -> bool {
        false
    }

    /// Last-rank only: logits from the final hidden.
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32>;

    /// Distributed KV-prefix cache hooks (pipeline prefix reuse). Default:
    /// unsupported — only the glm5 runner implements them, so dsv4 / OV runners
    /// are unaffected. `restore_prefix` restores this rank's cached KV slice for
    /// `key` and returns the restored length (== new pos), or `None` on a miss.
    /// `cache_prefix` snapshots the current KV slice under `key`.
    fn prefix_cache_enabled(&self) -> bool {
        false
    }
    fn restore_prefix(&mut self, _key: u64) -> Option<usize> {
        None
    }
    fn cache_prefix(&mut self, _key: u64) {}

    /// Single-stage generation with sampling (greedy when the config says so).
    /// Prompt tokens drive the same per-token path as decode; sampling happens
    /// once after prefill (mirroring the pipeline). A prompt longer than the
    /// context budget is truncated to its first `max_seq` tokens.
    fn generate(&mut self, prompt: &[u32], max_new: usize, cfg: &SamplingConfig) -> Vec<u32> {
        self.generate_reason(prompt, max_new, cfg).0
    }

    /// Like [`Self::generate`], but also reports whether decode stopped because
    /// the context window filled (`pos == max_seq`) rather than by the token cap
    /// or an EOS. The caller needs this to set the OpenAI `finish_reason`: a run
    /// cut off by the window is `length`, not `stop`, even below `max_new`.
    fn generate_reason(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
    ) -> (Vec<u32>, bool) {
        self.reset();
        if prompt.is_empty() {
            return (Vec::new(), false);
        }
        let max_seq = self.max_seq();
        let mut rng = init_rng(cfg.seed);
        let mut history: Vec<i64> = Vec::new();
        let rows = prompt.len().min(max_seq);
        let last_logits: Vec<f32> = if self.supports_batched_prefill() {
            // Batch-union prefill: embed all rows, run the layers once as a batch,
            // and take the head only at the final position. Bit-identical to the
            // per-token loop; the batched MoE just loads overlapping experts once.
            let hs = self.hidden_size();
            let mut batch = vec![0.0f32; rows * hs];
            for (r, &t) in prompt[..rows].iter().enumerate() {
                batch[r * hs..(r + 1) * hs].copy_from_slice(&self.embed_token(t));
            }
            let h = self.forward_layers_batch(batch, 0, rows);
            self.head_logits(&h[(rows - 1) * hs..rows * hs])
        } else {
            // Per-token prefill (backends whose layers need the position's token
            // id, e.g. dsv4 hash gates).
            let mut ll = Vec::new();
            for (pos, &t) in prompt.iter().take(rows).enumerate() {
                let h = self.embed_token(t);
                let h = self.forward_layers(h, pos, Some(t));
                ll = self.head_logits(&h);
            }
            ll
        };
        let mut next = sample(&last_logits, &history, cfg, &mut rng);
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len().min(max_seq);
        let mut hit_context_cap = false;
        loop {
            let tok = next as u32;
            out.push(tok);
            history.push(next);
            if out.len() >= max_new || self.eos_token_ids().contains(&tok) {
                break;
            }
            // Stop before forwarding at an absolute position the caches can't
            // hold (== max_seq): that write would index past the cache rows.
            // Checked after the push so the token sampled from the last in-range
            // position is still emitted. This is a truncation, not a natural
            // stop -> the caller reports `length`.
            if pos >= max_seq {
                hit_context_cap = true;
                break;
            }
            let h = self.embed_token(tok);
            let h = self.forward_layers(h, pos, Some(tok));
            let logits = self.head_logits(&h);
            next = sample(&logits, &history, cfg, &mut rng);
            pos += 1;
        }
        (out, hit_context_cap)
    }

    /// Single-stage greedy convenience (warmup / tests).
    fn generate_argmax(&mut self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        self.generate(prompt, max_new, &SamplingConfig::default())
    }
}
