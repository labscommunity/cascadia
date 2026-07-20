//! dsv4 stage runner — the engine-facing wrapper the sparse-moe `Builder`
//! constructs when a model dir's manifest says `arch = "deepseek_v4"`.
//! Mirrors the `OvMoeRunner` surface the MiniMax-M2 engine drives: one
//! contiguous layer slice per rank, token-by-token forwarding (prompt
//! included — validated equivalent to batch prefill by
//! `token_by_token_drive_matches_and_reset_reuses`), flattened HC copies
//! (`hc * dim` f32) as the inter-stage hidden.
//!
//! Token ids exist only on rank 0 (embed + hash-gate layers); workers
//! forward hidden states without them.

use std::path::Path;

use super::loader::{load_stage_mode, ExpertsMode, LoadError, Manifest};
use super::model::DsV4Model;
use crate::sampling::{init_rng, sample, SamplingConfig};

/// Default context budget for cache sizing when the caller doesn't pass one
/// (the checkpoint itself allows up to 1M; caches scale with this).
pub const DSV4_DEFAULT_MAX_SEQ: usize = 4096;

pub struct Dsv4Runner {
    model: DsV4Model,
    eos: Vec<u32>,
    hidden: usize,  // hc * dim (wire width)
    max_seq: usize, // context budget the caches were sized for
    pub rank: u32,
    pub total: u32,
}

/// Contiguous even split of `n` layers across `total` ranks.
pub fn even_layer_split(n: usize, rank: u32, total: u32) -> (usize, usize) {
    let total = total.max(1) as usize;
    let rank = (rank as usize).min(total - 1);
    let base = n / total;
    let rem = n % total;
    let lo = rank * base + rank.min(rem);
    let hi = lo + base + usize::from(rank < rem);
    (lo, hi)
}

impl Dsv4Runner {
    /// Load rank `rank` of `total`. `layer_start/layer_end` from the
    /// ShardSpec override the even split when nonzero.
    pub fn load_staged(
        model_dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> Result<Self, LoadError> {
        let m: Manifest =
            serde_json::from_str(&std::fs::read_to_string(model_dir.join("manifest.json"))?)
                .map_err(|e| LoadError::Manifest(e.to_string()))?;
        let n = m.num_layers;
        let total = total.max(1);
        let rank = rank.min(total - 1);
        let (lo, hi) = if layer_end > 0 {
            (layer_start as usize, layer_end as usize)
        } else {
            even_layer_split(n, rank, total)
        };
        let first = rank == 0;
        let last = rank == total - 1;
        // Real-model expert sets don't fit in RAM dequantized; tiny/dev ones
        // are faster eager. CASCADIA_DSV4_EXPERTS=eager|mmap overrides.
        let mode = match std::env::var("CASCADIA_DSV4_EXPERTS").as_deref() {
            Ok("eager") => ExpertsMode::Eager,
            Ok("mmap") => ExpertsMode::Mmap,
            _ if m.n_routed_experts > 32 => ExpertsMode::Mmap,
            _ => ExpertsMode::Eager,
        };
        let model = load_stage_mode(model_dir, max_seq, lo, hi, first, last, mode)?;
        let eos = m.eos_token_ids.iter().map(|&e| e as u32).collect();
        let hidden = m.hc_mult * m.hidden_size;
        Ok(Self {
            model,
            eos,
            hidden,
            max_seq,
            rank,
            total,
        })
    }

    /// Wire hidden width: hc * dim.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Context budget the caches were sized for. The driver must not forward a
    /// token at an absolute position >= this: the rope table, KV, compressed
    /// and indexer caches all have exactly this many rows.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Clear all per-generation state across this stage's layers.
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Rank-0 only: token -> flattened HC copies.
    pub fn embed_token(&self, token: u32) -> Vec<f32> {
        self.model
            .embed_ids(&[token as usize])
            .pop()
            .expect("embed_ids returns one entry per id")
    }

    /// Run this stage's layers for one token at absolute `pos`. `token` is
    /// required on the stage holding hash-gate layers (always rank 0, which
    /// knows it); workers pass None.
    pub fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, token: Option<u32>) -> Vec<f32> {
        self.model
            .forward_layers_decode(hidden, pos, token.map(|t| t as usize))
    }

    /// Last-rank only: logits from the final hidden.
    pub fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.model.logits(hidden)
    }

    /// Rank-0 batched prefill: embed the whole prompt, run this stage's layers
    /// over ALL positions at once (attention per position, MoE batch-union so
    /// overlapping experts decode once), and return the flattened
    /// `[rows * hidden]` output batch to ship downstream. Positions are
    /// `0..prompt.len()` — call [`Self::reset`] first. Bit-identical to feeding
    /// the prompt token-by-token through [`Self::forward_layers`].
    pub fn prefill_batch_first(&mut self, prompt: &[u32]) -> Vec<f32> {
        let ids: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
        let copies = self.model.embed_ids(&ids);
        let out = self.model.forward_layers_prefill(copies, Some(&ids));
        out.into_iter().flatten().collect()
    }

    /// Worker batched prefill: run this stage's layers over `rows` received
    /// hidden states (flattened `[rows * hidden]`, one `hidden`-wide HC-copy row
    /// per prompt position), batched. Workers hold no hash-gate layers, so no
    /// token ids. Returns the flattened `[rows * hidden]` output batch.
    pub fn forward_layers_prefill_batch(&mut self, hiddens: Vec<f32>, rows: usize) -> Vec<f32> {
        debug_assert_eq!(hiddens.len(), rows * self.hidden);
        let copies: Vec<Vec<f32>> = (0..rows)
            .map(|r| hiddens[r * self.hidden..(r + 1) * self.hidden].to_vec())
            .collect();
        let out = self.model.forward_layers_prefill(copies, None);
        out.into_iter().flatten().collect()
    }

    /// Single-stage generation with sampling (greedy when the config says
    /// so). Prompt tokens drive the same per-token path as decode.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, cfg: &SamplingConfig) -> Vec<u32> {
        self.reset();
        if prompt.is_empty() {
            return Vec::new();
        }
        let max_seq = self.max_seq;
        let mut rng = init_rng(cfg.seed);
        let mut history: Vec<i64> = Vec::new();
        // Prefill: forward each prompt token at its absolute position, keeping
        // only the LAST forwarded token's logits. We sample exactly once, after
        // prefill — mirroring the pipeline, where intermediate prompt tokens go
        // as ForwardNoSample and only the last one seeds+draws, so seeded
        // temperature>0 output is identical single-stage vs pipelined. (Greedy
        // is unchanged: sample() returns the argmax of these same last logits
        // without touching the RNG.) A prompt longer than the context budget is
        // truncated to its first max_seq tokens — raise CASCADIA_DSV4_MAX_SEQ.
        let mut last_logits: Vec<f32> = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            if pos >= max_seq {
                break;
            }
            let h = self.embed_token(t);
            let h = self.forward_layers(h, pos, Some(t));
            last_logits = self.head_logits(&h);
        }
        let mut next = sample(&last_logits, &history, cfg, &mut rng);
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len().min(max_seq);
        loop {
            let tok = next as u32;
            out.push(tok);
            history.push(next);
            // Stop before forwarding at an absolute position the caches can't
            // hold (== max_seq): that write would index past the rope/KV/
            // compressed/indexer rows. Checked after the push so the token
            // sampled from the last in-range position is still emitted.
            if out.len() >= max_new || self.eos.contains(&tok) || pos >= max_seq {
                break;
            }
            let h = self.embed_token(tok);
            let h = self.forward_layers(h, pos, Some(tok));
            let logits = self.head_logits(&h);
            next = sample(&logits, &history, cfg, &mut rng);
            pos += 1;
        }
        out
    }

    /// Single-stage greedy convenience (warmup / tests).
    pub fn generate_argmax(&mut self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        self.generate(prompt, max_new, &SamplingConfig::default())
    }
}
