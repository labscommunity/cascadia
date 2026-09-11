//! Inkling stage runner — the engine-facing wrapper `SparseMoEBuilder`
//! constructs when a model dir's manifest says `arch = "inkling"`. One
//! contiguous layer slice per rank; rank 0 embeds (+ embed norm) and drives,
//! mids relay the `hidden_size`-wide residual stream, the last rank runs the
//! final norm / mup / unembed. Implements [`StagedRunner`] so the generic
//! [`crate::engine::PipelineEngine`] drives it exactly like glm5 / dsv4.

use std::path::Path;

use super::loader::{load_stage, read_manifest, InklingManifest};
use super::model::{Layer, WideTable};
use super::rmsnorm_f32;
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::staged::StagedRunner;

/// Default context budget for the global layers' KV caches (sliding layers
/// keep a fixed `window + rewind` ring regardless). Override with
/// `CASCADIA_INKLING_MAX_SEQ` or `--max-seq`.
pub const INKLING_DEFAULT_MAX_SEQ: usize = 4096;

/// Contiguous even split of `n` layers across `total` ranks.
pub fn even_layer_split(n: usize, rank: u32, total: u32) -> (usize, usize) {
    crate::dsv4::stage::even_layer_split(n, rank, total)
}

/// The layer range `[lo, hi)` that rank `rank` of `total` owns. Single source
/// of truth for the split (the ShardDescriptor a scheduler publishes must
/// agree with what the rank loads). Ends are exclusive; rank 0 starts at 0,
/// the last rank ends at `num_layers`, no rank owns zero layers.
pub fn layer_split(m: &InklingManifest, rank: u32, total: u32) -> Result<(usize, usize), String> {
    let n = m.num_layers;
    let total = total.max(1);
    if rank >= total {
        return Err(format!("rank {rank} out of range for total {total}"));
    }
    let (lo, hi) = even_layer_split(n, rank, total);
    if lo >= hi {
        return Err(format!(
            "rank {rank} of {total} owns zero layers [{lo}, {hi}) — total exceeds the \
             model's {n} layers; reduce --total"
        ));
    }
    Ok((lo, hi))
}

pub struct InklingRunner {
    embed: Option<(WideTable, Vec<f32>)>, // (table, embed_norm) on rank 0
    layers: Vec<Layer>,                   // this rank's slice
    head: Option<(Vec<f32>, WideTable)>,  // (norm, unembed) on the last rank
    hidden: usize,
    unpadded_vocab: usize,
    eps: f32,
    mup: f32,
    max_seq: usize,
    eos: Vec<u32>,
    pos: usize,
    pub rank: u32,
    pub total: u32,
    pub lo: usize,
    pub hi: usize,
}

impl InklingRunner {
    /// Load rank `rank` of `total`. `layer_start/layer_end` from the ShardSpec
    /// override the even split when nonzero. `experts_mode` (`"eager"` |
    /// `"mmap"`) falls back to `CASCADIA_INKLING_EXPERTS`, then to mmap for any
    /// real-sized expert set (> 32 experts).
    pub fn load_staged(
        dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
        experts_mode: Option<String>,
    ) -> Result<Self, LoadError> {
        let m = read_manifest(dir)?;
        let n = m.num_layers;
        let total = total.max(1);
        let rank = rank.min(total - 1);
        let (lo, hi) = if layer_end > 0 {
            (layer_start as usize, layer_end as usize)
        } else {
            layer_split(&m, rank, total).map_err(LoadError::Manifest)?
        };
        if lo >= hi || hi > n {
            return Err(LoadError::Manifest(format!(
                "rank {rank} of {total} owns an invalid layer range [{lo}, {hi}) of {n}"
            )));
        }
        let first = rank == 0;
        let last = rank == total - 1;
        let experts_mode = experts_mode.or_else(|| std::env::var("CASCADIA_INKLING_EXPERTS").ok());
        let mode = match experts_mode.as_deref() {
            Some("eager") => ExpertsMode::Eager,
            Some("mmap") => ExpertsMode::Mmap,
            _ if m.num_experts > 32 => ExpertsMode::Mmap,
            _ => ExpertsMode::Eager,
        };
        let s = load_stage(dir, max_seq, lo, hi, first, last, mode)?;
        let cache_bytes: usize = s.layers.iter().map(Layer::cache_bytes).sum();
        tracing::info!(
            rank,
            total,
            lo,
            hi,
            max_seq,
            experts = ?mode,
            cache_mib = cache_bytes >> 20,
            "inkling stage loaded"
        );
        Ok(Self {
            embed: s.embed,
            layers: s.layers,
            head: s.head,
            hidden: m.hidden_size,
            unpadded_vocab: m.unpadded_vocab(),
            eps: m.rms_norm_eps,
            mup: m.logits_mup_width_multiplier,
            max_seq,
            eos: m.eos_token_ids.clone(),
            pos: 0,
            rank,
            total,
            lo,
            hi,
        })
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Current sequence position (tokens folded so far).
    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl StagedRunner for InklingRunner {
    fn arch_name(&self) -> &'static str {
        "inkling"
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
    fn supports_batched_prefill(&self) -> bool {
        true // layers ignore the token id; batch-union MoE prefill is bit-exact
    }
    fn reset(&mut self) {
        self.pos = 0;
        for l in &mut self.layers {
            l.reset();
        }
    }
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let (table, norm) = self
            .embed
            .as_ref()
            .expect("embed_token on a non-first rank");
        let mut x = table.row(token as usize, self.hidden);
        rmsnorm_f32(&mut x, norm, self.eps);
        x
    }
    fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, _token: Option<u32>) -> Vec<f32> {
        assert_eq!(
            pos, self.pos,
            "inkling stage position desync (expected {}, got {pos})",
            self.pos
        );
        let mut x = hidden;
        for l in &mut self.layers {
            x = l.forward_token(&x);
        }
        self.pos += 1;
        x
    }
    fn forward_layers_batch(&mut self, hidden: Vec<f32>, base: usize, rows: usize) -> Vec<f32> {
        assert_eq!(
            base, self.pos,
            "inkling stage batch position desync (expected {}, got {base})",
            self.pos
        );
        assert_eq!(
            hidden.len(),
            rows * self.hidden,
            "inkling batch: bad hidden length"
        );
        let mut x = hidden;
        for l in &mut self.layers {
            x = l.forward_prefill(&x, rows);
        }
        self.pos += rows;
        x
    }
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (norm, unembed) = self.head.as_ref().expect("head_logits on a non-last rank");
        let mut x = hidden.to_vec();
        rmsnorm_f32(&mut x, norm, self.eps);
        let inv = 1.0 / self.mup;
        for v in &mut x {
            *v *= inv;
        }
        let mut logits = vec![0.0f32; self.unpadded_vocab];
        unembed.matvec_f32(&x, self.hidden, &mut logits);
        logits
    }
}
