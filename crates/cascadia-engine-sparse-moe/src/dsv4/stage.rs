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

use tracing::warn;

use super::loader::{load_stage_mode, ExpertsMode, LoadError, Manifest};
use super::model::DsV4Model;

/// Default context budget for cache sizing when the caller doesn't pass one
/// (the checkpoint itself allows up to 1M; caches scale with this).
pub const DSV4_DEFAULT_MAX_SEQ: usize = 4096;

pub struct Dsv4Runner {
    model: DsV4Model,
    eos: Vec<u32>,
    hidden: usize,             // hc * dim (wire width)
    max_seq: usize,            // context budget the caches were sized for
    experts_mode: ExpertsMode, // resolved mode the experts were loaded in
    pub rank: u32,
    pub total: u32,
}

/// Expert storage mode precedence: explicit `override_val` → `env_val` (the
/// caller's read of `CASCADIA_DSV4_EXPERTS`) → the `n_routed_experts > 32`
/// size heuristic (real-model expert sets don't fit in RAM dequantized;
/// tiny/dev ones are faster eager). An unrecognized `override_val` is logged
/// and treated as absent rather than panicking. `override_val` is trimmed (it
/// arrives from a config file); `env_val` is not, matching glm5.
///
/// Takes the env value as a parameter (rather than reading `std::env` itself)
/// so it's a pure function tests can exercise without mutating process-global
/// state (`set_var` is `unsafe` under edition 2024 and racy under a parallel
/// test runner regardless).
pub fn resolve_experts_mode(
    override_val: Option<&str>,
    env_val: Option<&str>,
    n_routed_experts: usize,
) -> ExpertsMode {
    let from_env_or_heuristic = || match env_val {
        Some("eager") => ExpertsMode::Eager,
        Some("mmap") => ExpertsMode::Mmap,
        _ if n_routed_experts > 32 => ExpertsMode::Mmap,
        _ => ExpertsMode::Eager,
    };
    match override_val.map(str::trim) {
        Some("eager") => ExpertsMode::Eager,
        Some("mmap") => ExpertsMode::Mmap,
        Some(other) => {
            warn!(
                value = other,
                "unrecognized dsv4 experts_mode override; falling back to env/heuristic"
            );
            from_env_or_heuristic()
        }
        None => from_env_or_heuristic(),
    }
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
    ///
    /// Resolves the experts mode from `CASCADIA_DSV4_EXPERTS`, else the size
    /// heuristic. Kept signature-stable for its many call sites; use
    /// [`Self::load_staged_with_experts`] to pass a config-first override.
    pub fn load_staged(
        model_dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> Result<Self, LoadError> {
        Self::load_staged_with_experts(
            model_dir,
            max_seq,
            rank,
            total,
            layer_start,
            layer_end,
            None,
        )
    }

    /// [`Self::load_staged`] with an explicit experts-mode override
    /// (`"eager"` | `"mmap"`), taking precedence over both
    /// `CASCADIA_DSV4_EXPERTS` and the size heuristic. An unrecognized value
    /// is logged and treated as `None` — the env/heuristic path decides
    /// instead of panicking.
    ///
    /// Exists because in-process hosts cannot set the environment for a
    /// single engine (`set_var` is `unsafe` under edition 2024, and
    /// process-global).
    pub fn load_staged_with_experts(
        model_dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
        experts_override: Option<&str>,
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
        let mode = resolve_experts_mode(
            experts_override,
            std::env::var("CASCADIA_DSV4_EXPERTS").ok().as_deref(),
            m.n_routed_experts,
        );
        let model = load_stage_mode(model_dir, max_seq, lo, hi, first, last, mode)?;
        let eos = m.eos_token_ids.iter().map(|&e| e as u32).collect();
        let hidden = m.hc_mult * m.hidden_size;
        Ok(Self {
            model,
            eos,
            hidden,
            max_seq,
            experts_mode: mode,
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

    /// The expert storage mode this stage actually resolved to. Exposed so a
    /// caller's override can be observed — otherwise both modes just load.
    pub fn experts_mode(&self) -> ExpertsMode {
        self.experts_mode
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
