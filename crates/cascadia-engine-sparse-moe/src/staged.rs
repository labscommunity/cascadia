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

    /// Positions this runner's state currently covers, if it tracks that.
    ///
    /// A cache entry has to be keyed by EXACTLY the tokens its snapshot covers.
    /// Key it by fewer and a later restore reports a reuse shorter than the state
    /// it installed, so every subsequent position is written one slot early —
    /// silently wrong output rather than a failure. The pipeline path derives
    /// this from its own bookkeeping; a single-stage caller has none, so it asks.
    fn covered_positions(&self) -> Option<usize> {
        None
    }

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
        self.generate_reason_from(prompt, None, max_new, cfg)
    }

    /// [`Self::generate_reason`], optionally starting from a cached prefix.
    ///
    /// `prefix` is a key to restore AFTER the reset — the reset is what clears
    /// the state the restore then fills, so the order is not negotiable. The
    /// restored length decides how much of the prompt is already accounted for
    /// and gets skipped; a restore that misses falls back to prefilling all of
    /// it, which is exactly the behaviour with no cache at all.
    fn generate_reason_from(
        &mut self,
        prompt: &[u32],
        prefix: Option<u64>,
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
        // Take the runner's word for how much it restored rather than the
        // caller's: those two disagreeing is what writes later positions into
        // the wrong slots. Clamp below `rows` so the final prompt token is always
        // forwarded — its logits are what the first sampled token comes from.
        let base = prefix
            .and_then(|key| self.restore_prefix(key))
            .unwrap_or(0)
            .min(rows.saturating_sub(1));
        let last_logits: Vec<f32> = if self.supports_batched_prefill() {
            // Batch-union prefill: embed all rows, run the layers once as a batch,
            // and take the head only at the final position. Bit-identical to the
            // per-token loop; the batched MoE just loads overlapping experts once.
            let hs = self.hidden_size();
            let n = rows - base;
            let mut batch = vec![0.0f32; n * hs];
            for (r, &t) in prompt[base..rows].iter().enumerate() {
                batch[r * hs..(r + 1) * hs].copy_from_slice(&self.embed_token(t));
            }
            let h = self.forward_layers_batch(batch, base, n);
            self.head_logits(&h[(n - 1) * hs..n * hs])
        } else {
            // Per-token prefill (backends whose layers need the position's token
            // id, e.g. dsv4 hash gates).
            let mut ll = Vec::new();
            for (i, &t) in prompt[base..rows].iter().enumerate() {
                let h = self.embed_token(t);
                let h = self.forward_layers(h, base + i, Some(t));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A runner whose "state" is just the absolute positions it has forwarded,
    /// which is what prefix reuse has to get right. Logits pick the token id
    /// equal to the number of positions covered, so the sampled output is a
    /// direct readout of position bookkeeping: if a restore lets prefill start
    /// at the wrong base, the tokens change.
    struct PosRunner {
        seen: Vec<usize>,
        snapshot: Option<(u64, Vec<usize>)>,
        batched: bool,
        enabled: bool,
    }

    impl PosRunner {
        fn new(batched: bool) -> Self {
            Self {
                seen: Vec::new(),
                snapshot: None,
                batched,
                enabled: true,
            }
        }
    }

    impl StagedRunner for PosRunner {
        fn arch_name(&self) -> &'static str {
            "pos"
        }
        fn hidden_size(&self) -> usize {
            1
        }
        fn max_seq(&self) -> usize {
            64
        }
        fn eos_token_ids(&self) -> &[u32] {
            &[]
        }
        fn reset(&mut self) {
            self.seen.clear();
        }
        fn embed_token(&self, token: u32) -> Vec<f32> {
            vec![token as f32]
        }
        fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, _t: Option<u32>) -> Vec<f32> {
            self.seen.push(pos);
            hidden
        }
        fn supports_batched_prefill(&self) -> bool {
            self.batched
        }
        fn head_logits(&self, _hidden: &[f32]) -> Vec<f32> {
            // Keyed on the LAST ABSOLUTE POSITION forwarded, not on how many
            // were: a base that ignores the restore still forwards the right
            // COUNT of rows, just at the wrong offsets, so a count-based readout
            // reports success for exactly the bug this is meant to catch.
            let mut v = vec![0.0f32; 64];
            let last = self.seen.last().copied().unwrap_or(0);
            v[(last + 1).min(63)] = 1.0;
            v
        }
        fn prefix_cache_enabled(&self) -> bool {
            self.enabled
        }
        fn restore_prefix(&mut self, key: u64) -> Option<usize> {
            let (k, seen) = self.snapshot.as_ref()?;
            if *k != key {
                return None;
            }
            self.seen = seen.clone();
            Some(self.seen.len())
        }
        fn cache_prefix(&mut self, key: u64) {
            self.snapshot = Some((key, self.seen.clone()));
        }
        fn covered_positions(&self) -> Option<usize> {
            Some(self.seen.len())
        }
    }

    fn run(batched: bool) -> (Vec<u32>, Vec<u32>, Vec<usize>) {
        let cfg = SamplingConfig::default();
        let prompt: Vec<u32> = (0..6).collect();

        // An earlier, SHORTER request leaves a snapshot behind. That is the only
        // shape the engine can hand back, because prefix_reuse takes a strict
        // prefix — a snapshot covering as much as the prompt, or more, is never
        // offered.
        let mut r0 = PosRunner::new(batched);
        r0.generate_reason_from(&prompt[..3], None, 1, &cfg);
        r0.cache_prefix(7);
        let snapshot = r0.snapshot.clone();
        let covered = r0.covered_positions().unwrap();

        // the longer prompt, once cold and once resuming that prefix
        let mut r1 = PosRunner::new(batched);
        let full = r1.generate_reason_from(&prompt, None, 3, &cfg).0;

        let mut r2 = PosRunner::new(batched);
        r2.snapshot = snapshot;
        let reused = r2.generate_reason_from(&prompt, Some(7), 3, &cfg).0;
        (full, reused, vec![covered, r2.seen.len()])
    }

    #[test]
    fn restoring_a_prefix_does_not_change_what_is_generated() {
        // The failure this guards is not a crash: a base that disagrees with the
        // restored state writes every later position one slot off, and the run
        // is wrong without saying so. Identical output is the only cheap proof
        // that the skip and the state agree.
        for batched in [false, true] {
            let (full, reused, _) = run(batched);
            assert_eq!(full, reused, "batched={batched}: reuse changed the output");
        }
    }

    #[test]
    fn a_resumed_prefill_forwards_the_positions_the_prefix_did_not() {
        // Output equality alone is weak here: forwarding the right NUMBER of
        // rows at the wrong offsets can still agree by luck. Assert the absolute
        // positions directly.
        for batched in [false, true] {
            let cfg = SamplingConfig::default();
            let prompt: Vec<u32> = (0..6).collect();

            let mut r0 = PosRunner::new(batched);
            r0.generate_reason_from(&prompt[..3], None, 1, &cfg);
            r0.cache_prefix(7);
            let covered = r0.covered_positions().unwrap();
            let snapshot = r0.snapshot.clone();

            let mut r2 = PosRunner::new(batched);
            r2.snapshot = snapshot;
            r2.generate_reason_from(&prompt, Some(7), 1, &cfg);

            // the restored rows, then every remaining prompt position, in order
            let expect: Vec<usize> = (0..covered).chain(covered..prompt.len()).collect();
            assert_eq!(
                &r2.seen[..prompt.len()],
                &expect[..],
                "batched={batched}: resumed prefill covered the wrong positions"
            );
        }
    }

    #[test]
    fn a_restore_that_misses_falls_back_to_a_full_prefill() {
        // A key the runner does not hold must not be treated as coverage. If it
        // were, the caller would skip tokens it never forwarded.
        let cfg = SamplingConfig::default();
        let prompt: Vec<u32> = (0..6).collect();
        let mut r = PosRunner::new(true);
        let base = r.generate_reason_from(&prompt, None, 2, &cfg).0;

        let mut r2 = PosRunner::new(true);
        let missed = r2.generate_reason_from(&prompt, Some(999), 2, &cfg).0;
        assert_eq!(base, missed, "a miss must behave exactly like no cache");
    }

    #[test]
    fn the_last_prompt_token_is_always_forwarded() {
        // Its logits are where the first sampled token comes from, so a restore
        // covering the WHOLE prompt must still leave one row to prefill.
        let cfg = SamplingConfig::default();
        let prompt: Vec<u32> = (0..4).collect();
        let mut r = PosRunner::new(true);
        r.generate_reason_from(&prompt, None, 1, &cfg);
        r.cache_prefix(1);
        let snap = r.snapshot.clone();

        let mut r2 = PosRunner::new(true);
        r2.snapshot = snap;
        r2.generate_reason_from(&prompt, Some(1), 1, &cfg);
        assert!(
            r2.seen.len() >= prompt.len(),
            "prefill skipped the final prompt token: {:?}",
            r2.seen
        );
    }
}
