//! GLM-5.2 stage runner — the [`StagedRunner`] the pipeline engine drives.
//! One contiguous layer slice per rank; `GlmRunner` itself is the staged
//! container (embed only on rank 0, head only on the last rank), so `GlmModel`
//! stays the untouched single-process form its goldens validate.
//!
//! Position: GLM's `AttentionLayer` appends KV at its own internal counter and
//! ignores the wire `pos` — the two stay in sync only via reset +
//! exactly-one-advance-per-forward. `forward_layers` asserts `pos == self.pos`
//! so a dropped/replayed frame is a loud worker death, never silent garbage.

use std::path::Path;

use super::loader::{load_stage, read_manifest};
use super::model::GlmLayer;
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::dsv4::math::{linear_f32, rmsnorm};
use crate::dsv4::stage::even_layer_split;
use crate::staged::StagedRunner;

/// Default context budget when the caller passes none (the checkpoint allows up
/// to 1M; the KV caches scale with this).
pub const GLM5_DEFAULT_MAX_SEQ: usize = 4096;

pub struct GlmRunner {
    embed: Option<Vec<f32>>,             // [vocab, hidden] on rank 0
    layers: Vec<GlmLayer>,               // this rank's slice
    head: Option<(Vec<f32>, Vec<f32>)>,  // (final_norm, lm_head) on the last rank
    hidden: usize,
    vocab: usize,
    eps: f32,
    max_seq: usize,
    eos: Vec<u32>,
    pos: usize,
    pub rank: u32,
    pub total: u32,
}

impl GlmRunner {
    /// Load rank `rank` of `total`. `layer_start/layer_end` from the ShardSpec
    /// override the even split when nonzero.
    pub fn load_staged(
        dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> Result<Self, LoadError> {
        let m = read_manifest(dir)?;
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
        // Real-model expert sets can't be held dequantized; tiny/dev ones are
        // faster eager. CASCADIA_GLM5_EXPERTS=eager|mmap overrides.
        let mode = match std::env::var("CASCADIA_GLM5_EXPERTS").as_deref() {
            Ok("eager") => ExpertsMode::Eager,
            Ok("mmap") => ExpertsMode::Mmap,
            _ if m.num_experts > 32 => ExpertsMode::Mmap,
            _ => ExpertsMode::Eager,
        };
        let s = load_stage(dir, max_seq, lo, hi, first, last, mode)?;
        Ok(Self {
            embed: s.embed,
            layers: s.layers,
            head: s.head,
            hidden: s.hidden,
            vocab: s.vocab,
            eps: s.eps,
            max_seq,
            eos: s.eos,
            pos: 0,
            rank,
            total,
        })
    }
}

impl StagedRunner for GlmRunner {
    fn arch_name(&self) -> &'static str {
        "glm5"
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
        self.pos = 0;
        for l in &mut self.layers {
            l.reset();
        }
    }
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let e = self.embed.as_ref().expect("embed_token on a non-first rank");
        let t = token as usize;
        e[t * self.hidden..(t + 1) * self.hidden].to_vec()
    }
    fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, _token: Option<u32>) -> Vec<f32> {
        assert_eq!(
            pos, self.pos,
            "glm5 stage position desync (expected {}, got {pos})",
            self.pos
        );
        let mut x = hidden;
        for l in &mut self.layers {
            x = l.forward_token(&x);
        }
        self.pos += 1;
        x
    }
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (final_norm, lm_head) = self.head.as_ref().expect("head_logits on a non-last rank");
        let mut x = hidden.to_vec();
        rmsnorm(&mut x, final_norm, self.eps);
        let mut logits = vec![0.0f32; self.vocab];
        linear_f32(&x, lm_head, self.vocab, self.hidden, &mut logits);
        logits
    }
}
