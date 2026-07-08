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
    hidden: usize, // hc * dim (wire width)
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
            rank,
            total,
        })
    }

    /// Wire hidden width: hc * dim.
    pub fn hidden_size(&self) -> usize {
        self.hidden
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

    /// Single-stage generation with sampling (greedy when the config says
    /// so). Prompt tokens drive the same per-token path as decode.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, cfg: &SamplingConfig) -> Vec<u32> {
        self.reset();
        let mut rng = init_rng(cfg.seed);
        let mut history: Vec<i64> = Vec::new();
        let mut next: i64 = -1;
        for (pos, &t) in prompt.iter().enumerate() {
            let h = self.embed_token(t);
            let h = self.forward_layers(h, pos, Some(t));
            let logits = self.head_logits(&h);
            next = sample(&logits, &history, cfg, &mut rng);
        }
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        loop {
            let tok = next as u32;
            out.push(tok);
            history.push(next);
            if out.len() >= max_new || self.eos.contains(&tok) {
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
