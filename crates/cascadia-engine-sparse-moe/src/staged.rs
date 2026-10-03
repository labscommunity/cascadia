//! `StagedRunner` — the arch-agnostic surface the pipeline engine
//! ([`crate::engine::PipelineEngine`]) drives. One contiguous layer slice per
//! rank: rank 0 embeds + drives, mids relay the hidden state, the last rank
//! runs the head + sampler. Implemented by the dsv4 and glm5 Rust shells
//! (minimax keeps its own engine — different disconnect semantics).
//!
//! `generate` / `generate_argmax` are provided (default) methods built on the
//! required primitives, so every backend shares one single-stage sampling loop.

use crate::sampling::{init_rng, sample, SamplingConfig};

/// Wall-clock split of one generation: the prompt forward plus the first
/// sample, then the decode steps (each a full forward + head + sample) after
/// it. `decode_steps` is the number of timed steps, i.e. tokens minus one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenTiming {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub decode_steps: usize,
}

impl GenTiming {
    /// Decode tokens per second over the timed steps (0 without any).
    pub fn decode_tok_s(&self) -> f64 {
        if self.decode_s > 0.0 && self.decode_steps > 0 {
            self.decode_steps as f64 / self.decode_s
        } else {
            0.0
        }
    }
}

/// What a runner measured about its own layers, for the stage profile the
/// pipeline engine logs (`CASCADIA_STAGE_PROFILE_SECS`). Cumulative since
/// [`StagedRunner::enable_profile`], except the `_mib` gauges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunnerProfile {
    /// Decode steps: attention branch / MLP branch, summed over my layers.
    pub decode_attn_ns: u64,
    pub decode_mlp_ns: u64,
    /// Prefill blocks: the same split.
    pub prefill_attn_ns: u64,
    pub prefill_mlp_ns: u64,
    /// Expert cache: lookups served from RAM / read from the model files.
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_retained_mib: u64,
    pub cache_capacity_mib: u64,
    /// Offloaded attention projections and output head (OpenVINO), inside the
    /// attention / head time above.
    pub ov_attn_calls: u64,
    pub ov_attn_ns: u64,
    pub ov_head_calls: u64,
    pub ov_head_ns: u64,
    /// Fused MoE layers on the device: calls, time, calls handed back to the
    /// CPU path, and of those the ones that overflowed half precision.
    pub ov_moe_calls: u64,
    pub ov_moe_ns: u64,
    pub ov_moe_fallbacks: u64,
    pub ov_moe_nonfinite: u64,
    /// `CASCADIA_INKLING_OV_PERF=1` only: of the offloaded time, the part
    /// inside `infer()` and the part the device spent executing.
    pub ov_attn_infer_ns: u64,
    pub ov_attn_device_ns: u64,
    pub ov_moe_infer_ns: u64,
    pub ov_moe_device_ns: u64,
}

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

    /// Last-rank only: logits for `rows` final hiddens (`[rows, hidden]` ->
    /// `[rows, vocab]`). Default loops [`Self::head_logits`]; a backend with a
    /// batched head (one table read per step) overrides.
    fn head_logits_rows(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        let hs = self.hidden_size();
        assert_eq!(
            hidden.len(),
            rows * hs,
            "head_logits_rows: bad hidden length"
        );
        let mut out = Vec::new();
        for r in 0..rows {
            out.extend(self.head_logits(&hidden[r * hs..(r + 1) * hs]));
        }
        out
    }

    // ---- multi-stream decode (continuous batching) ----------------------
    //
    // A runner that keeps one KV/sequence state per *stream slot* can decode
    // several sequences per step: each step takes one token per active
    // stream, runs attention per stream and the MoE as one batch. All
    // default to "unsupported" (`stream_capacity() == 0`), which keeps every
    // other backend on the one-sequence path.

    /// Stream slots this runner holds (0 = single-sequence only).
    fn stream_capacity(&self) -> usize {
        0
    }

    /// Allocate `n` stream slots (KV + conv state each). Returns `false` if
    /// the runner cannot batch streams.
    fn configure_streams(&mut self, _n: usize) -> bool {
        false
    }

    /// Take a free slot for a new sequence (state cleared); `None` when all
    /// slots are busy.
    fn open_stream(&mut self) -> Option<usize> {
        None
    }

    /// Open a specific slot (a worker rank following rank 0's choice); the
    /// slot's state is cleared even if it was busy. `false` if out of range.
    fn open_stream_at(&mut self, _slot: usize) -> bool {
        false
    }

    /// Release a slot.
    fn close_stream(&mut self, _slot: usize) {}

    /// Positions consumed on `slot` (its next token's absolute position).
    fn stream_pos(&self, _slot: usize) -> usize {
        0
    }

    /// Prefill `rows` prompt positions of `slot` (`hidden` = `[rows, hidden]`,
    /// positions `stream_pos(slot)..+rows`); returns `[rows, hidden]`.
    fn prefill_stream(&mut self, _slot: usize, _hidden: Vec<f32>, _rows: usize) -> Vec<f32> {
        unimplemented!("this runner does not batch streams")
    }

    /// Prefill several slots' prompts in one pass (`segs[i] = (slot, rows)`,
    /// rows end to end in `hidden`), sharing whatever the runner can share
    /// between them. Default: one [`Self::prefill_stream`] per slot.
    fn prefill_streams(&mut self, segs: &[(usize, usize)], hidden: Vec<f32>) -> Vec<f32> {
        let h = self.hidden_size();
        let mut out = Vec::with_capacity(hidden.len());
        let mut at = 0usize;
        for &(slot, rows) in segs {
            out.extend(self.prefill_stream(slot, hidden[at * h..(at + rows) * h].to_vec(), rows));
            at += rows;
        }
        out
    }

    /// Decode one token on each of `slots` (`hidden` = `[slots.len(), hidden]`,
    /// row `i` at `stream_pos(slots[i])`); returns `[slots.len(), hidden]` and
    /// advances every listed slot by one. A slot appears at most once.
    fn decode_streams(&mut self, _hidden: Vec<f32>, _slots: &[usize]) -> Vec<f32> {
        unimplemented!("this runner does not batch streams")
    }

    /// Start collecting the runner-side counters [`Self::profile`] reports.
    /// Off until called (the per-layer clocks cost two `Instant::now` each).
    fn enable_profile(&mut self) {}

    /// Runner-side counters for the engine's periodic stage profile; `None`
    /// when the runner keeps none. All fields but the gauges are cumulative.
    fn profile(&self) -> Option<RunnerProfile> {
        None
    }

    /// Roll `slot` back to `len` positions (a speculated token was wrong).
    /// `false` when the runner cannot (the caller must not speculate then).
    fn truncate_stream(&mut self, _slot: usize, _len: usize) -> bool {
        false
    }

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
        let (out, cap, _) = self.generate_reason_timed(prompt, max_new, cfg);
        (out, cap)
    }

    /// [`Self::generate_reason`] plus wall-clock timing split into the prefill
    /// (prompt forward + first sample) and the decode steps after it, so a
    /// serving log can report decode tok/s the way the benchmarks do instead
    /// of folding a 25 s prefill into a 16-token request.
    fn generate_reason_timed(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
    ) -> (Vec<u32>, bool, GenTiming) {
        let started = std::time::Instant::now();
        self.reset();
        if prompt.is_empty() {
            return (Vec::new(), false, GenTiming::default());
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
        let prefill_s = started.elapsed().as_secs_f64();
        let decode_started = std::time::Instant::now();
        let mut decode_steps = 0usize;
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
            decode_steps += 1;
        }
        let timing = GenTiming {
            prefill_s,
            decode_s: decode_started.elapsed().as_secs_f64(),
            decode_steps,
        };
        (out, hit_context_cap, timing)
    }

    /// Single-stage greedy convenience (warmup / tests).
    fn generate_argmax(&mut self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        self.generate(prompt, max_new, &SamplingConfig::default())
    }
}
