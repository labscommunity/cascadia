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
}

/// The engine-facing staged surface. Single-stage `generate`/`generate_argmax`
/// come from the trait defaults (identical loop to the former inherent ones,
/// gated by `dsv4_sampling_parity`); the 7 required methods delegate to the
/// model. The inherent accessors above are kept for the wire tests that drive
/// the runner directly.
impl crate::staged::StagedRunner for Dsv4Runner {
    fn arch_name(&self) -> &'static str {
        "dsv4"
    }
    fn hidden_size(&self) -> usize {
        self.hidden
    }
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    fn reset(&mut self) {
        self.model.reset();
    }
    fn embed_token(&self, token: u32) -> Vec<f32> {
        self.model
            .embed_ids(&[token as usize])
            .pop()
            .expect("embed_ids returns one entry per id")
    }
    fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, token: Option<u32>) -> Vec<f32> {
        self.model
            .forward_layers_decode(hidden, pos, token.map(|t| t as usize))
    }
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.model.logits(hidden)
    }
}
