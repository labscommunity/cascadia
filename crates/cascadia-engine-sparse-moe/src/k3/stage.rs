//! `K3Runner` — one pipeline stage of a K3 model, implementing
//! [`crate::staged::StagedRunner`].
//!
//! Rank 0 embeds and drives, mid ranks relay, the last rank runs the output
//! AttnRes mixture, the final norm and the head.
//!
//! The inter-stage activation is the WIDENED wire from [`crate::k3::model`]:
//! `prefix_sum` followed by `max_blocks` block-residual slots. An even layer
//! split is used — unlike glm5's index-aligned split there is no boundary to
//! snap to, because the AttnRes mixture reads every prior block regardless of
//! which rank produced it, so the stack travels either way.

use std::path::Path;

use crate::dsv4::math::{linear_bf16_w, rmsnorm};
use crate::dsv4::stage::even_layer_split;
use crate::k3::attn_res::apply_attn_res;
use crate::k3::expert_fp4;
use crate::k3::loader::{load_embed, load_head, load_layers, K3Head, K3LoadError, K3Manifest};
use crate::k3::model::LayerFfn;
use crate::k3::model::{
    blocks_at, forward_slice, forward_slice_batch, max_blocks, K3Dims, K3Layer, LayerState,
};
use crate::k3::moe::MmapExperts;
use crate::k3::prof;
use crate::k3::residency;
use crate::staged::StagedRunner;

/// Default context budget when `CASCADIA_K3_MAX_SEQ` is unset. K3's
/// `max_position_embeddings` is 1M; sizing caches from that would preallocate
/// TB-scale state, so the deployment picks the real budget.
pub const K3_DEFAULT_MAX_SEQ: usize = 4096;

pub struct K3Runner {
    m: K3Manifest,
    d: K3Dims,
    layers: Vec<K3Layer<MmapExperts>>,
    states: Vec<LayerState>,
    embed: Option<Vec<f32>>,
    head: Option<K3Head>,
    /// First layer index of this rank's slice — fixes how many block slots are
    /// already live when the wire arrives.
    lo: usize,
    /// This rank's id, so profiler lines identify which slice they describe.
    rank: u32,
    hidden: usize,
    wire: usize,
    max_blocks: usize,
    max_seq: usize,
    eos: Vec<u32>,
    /// Where the learned routing histogram is kept.
    usage_path: std::path::PathBuf,
    pinned: usize,
}

impl Drop for K3Runner {
    fn drop(&mut self) {
        // only write back when pinning is in use, so a plain run never touches
        // the model dir
        if self.pinned > 0 || std::env::var_os("CASCADIA_K3_AUTOPIN").is_some() {
            self.save_usage();
        }
    }
}

impl K3Runner {
    /// Load rank `rank` of `total` from an export directory.
    pub fn load(dir: &Path, rank: u32, total: u32, max_seq: usize) -> Result<Self, K3LoadError> {
        let m = K3Manifest::load(dir)?;
        let (lo, hi) = even_layer_split(m.num_hidden_layers, rank, total);
        let first = rank == 0;
        let last = rank == total.max(1) - 1;

        let (layers, states) = load_layers(dir, &m, lo, hi)?;
        let embed = if first { Some(load_embed(dir)?) } else { None };
        let head = if last { Some(load_head(dir)?) } else { None };

        let hidden = m.hidden_size;
        let mb = max_blocks(m.num_hidden_layers, m.attn_res_block_size);
        let eos = if m.eos_token_ids.is_empty() {
            vec![0]
        } else {
            m.eos_token_ids.clone()
        };
        // Learned residency: throughput is decided by what stays resident, so
        // the pin set comes from recorded traffic. Off unless
        // CASCADIA_K3_AUTOPIN is set — an over-large pin set evicts the cache
        // serving the cold tail and is worse than no pinning.
        let usage_p = residency::usage_path(dir);
        if usage_p.exists() {
            if let Err(e) = residency::load_global(&usage_p) {
                tracing::warn!("k3: could not read {}: {e}", usage_p.display());
            }
        }
        let expert_bytes =
            expert_fp4::expert_bytes(m.routed_expert_hidden_size, m.moe_intermediate_size) as u64;
        // reserve for the bf16 shell this rank holds plus its KV/recurrent state
        let reserve = layers.len() as u64 * 1_200_000_000;
        let budget = residency::autopin_budget(expert_bytes, reserve);
        let mut pinned = 0usize;
        if budget > 0 {
            crate::dsv4::expert_mmap::reserve_lockable(budget * expert_bytes as usize);
            let u = residency::snapshot();
            // spread the budget evenly over this rank's MoE layers
            let moe_layers = layers
                .iter()
                .filter(|l| matches!(l.ffn, LayerFfn::Moe(..)))
                .count()
                .max(1);
            let per_layer = budget / moe_layers;
            if per_layer > 0 && !u.is_empty() {
                for l in layers.iter() {
                    if let LayerFfn::Moe(_, _, ex) = &l.ffn {
                        for e in u.hottest_for(l.idx as u32, per_layer) {
                            if ex.pin_expert(e as usize) {
                                pinned += 1;
                            }
                        }
                    }
                }
            }
            tracing::info!(
                rank,
                budget,
                pinned,
                "k3 autopin: mlock'd {pinned} of a {budget}-expert budget \
                 ({} recorded selections)",
                u.total()
            );
        }

        Ok(Self {
            d: m.dims(),
            m,
            layers,
            states,
            embed,
            head,
            lo,
            rank,
            hidden,
            wire: (1 + mb) * hidden,
            max_blocks: mb,
            max_seq,
            eos,
            usage_path: usage_p,
            pinned,
        })
    }

    /// Persist the routing histogram so the next run starts warm. Best-effort.
    pub fn save_usage(&self) {
        if let Err(e) = residency::save_global(&self.usage_path) {
            tracing::debug!("k3: could not persist {}: {e}", self.usage_path.display());
        }
    }

    /// Experts this rank managed to lock into RAM.
    pub fn pinned_experts(&self) -> usize {
        self.pinned
    }

    /// Fold this rank's expert-map residency into the profiler. Sampled, not
    /// exhaustive — probing every page of a 15.7 GB layer per token would cost
    /// more than the decode.
    fn sample_residency(&self) {
        const SAMPLES_PER_LAYER: usize = 64;
        for l in &self.layers {
            if let crate::k3::model::LayerFfn::Moe(_, _, ex) = &l.ffn {
                let (r, p) = ex.resident_pages_sampled(SAMPLES_PER_LAYER);
                prof::record_residency(r, p);
            }
        }
    }

    /// Split a wire buffer into `(prefix_sum, blocks)`.
    fn split_wire(&self, w: &mut [f32]) -> (usize, usize) {
        debug_assert_eq!(w.len(), self.wire);
        (self.hidden, self.max_blocks * self.hidden)
    }
}

impl StagedRunner for K3Runner {
    fn arch_name(&self) -> &'static str {
        "k3"
    }

    /// The widened wire, not the residual width: prefix sum + block stack.
    fn hidden_size(&self) -> usize {
        self.wire
    }

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    fn reset(&mut self) {
        for s in self.states.iter_mut() {
            s.clear();
        }
        prof::new_sequence();
    }

    /// Rank 0: the embedding row becomes the prefix sum; the stack starts empty.
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let e = self
            .embed
            .as_ref()
            .expect("embed_token on a non-first rank");
        let h = self.hidden;
        let mut w = vec![0.0f32; self.wire];
        w[..h].copy_from_slice(&e[token as usize * h..(token as usize + 1) * h]);
        w
    }

    fn forward_layers(&mut self, hidden: Vec<f32>, _pos: usize, _token: Option<u32>) -> Vec<f32> {
        assert_eq!(hidden.len(), self.wire, "k3: bad wire width");
        let mut w = hidden;
        let (h, _) = self.split_wire(&mut w);
        let (prefix, blocks) = w.split_at_mut(h);
        // how many slots are already live when this rank's slice begins
        let nb = blocks_at(self.lo, self.m.attn_res_block_size);
        let t = std::time::Instant::now();
        forward_slice(
            &mut self.layers,
            &mut self.states,
            self.d,
            prefix,
            blocks,
            nb,
        );
        prof::add(prof::WALL, t);
        if prof::enabled() {
            self.sample_residency();
            prof::dump(&format!("rank{}", self.rank));
        }
        w
    }

    /// Batch-union prefill is mandatory at K3's scale, not an optimisation:
    /// per-token prefill re-streams the whole active expert set at every
    /// position, which at realistic residency is tens of TB for a few-thousand
    /// token prompt.
    fn supports_batched_prefill(&self) -> bool {
        true
    }

    fn forward_layers_batch(&mut self, hidden: Vec<f32>, _base: usize, rows: usize) -> Vec<f32> {
        assert_eq!(hidden.len(), rows * self.wire, "k3: bad batch wire width");
        let h = self.hidden;
        let mb = self.max_blocks;

        // the wire interleaves [prefix | blocks] per row; the batched layer loop
        // wants them contiguous, so split into row-major planes and rejoin after
        let mut prefix = vec![0.0f32; rows * h];
        let mut blocks = vec![0.0f32; rows * mb * h];
        for r in 0..rows {
            let src = r * self.wire;
            prefix[r * h..(r + 1) * h].copy_from_slice(&hidden[src..src + h]);
            blocks[r * mb * h..(r + 1) * mb * h].copy_from_slice(&hidden[src + h..src + self.wire]);
        }

        let nb = blocks_at(self.lo, self.m.attn_res_block_size);
        forward_slice_batch(
            &mut self.layers,
            &mut self.states,
            self.d,
            &mut prefix,
            &mut blocks,
            rows,
            mb,
            nb,
        );

        let mut out = vec![0.0f32; rows * self.wire];
        for r in 0..rows {
            let dst = r * self.wire;
            out[dst..dst + h].copy_from_slice(&prefix[r * h..(r + 1) * h]);
            out[dst + h..dst + self.wire].copy_from_slice(&blocks[r * mb * h..(r + 1) * mb * h]);
        }
        out
    }

    /// Last rank: the model-level AttnRes mixture, final norm, then lm_head.
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let head = self.head.as_ref().expect("head_logits on a non-last rank");
        assert_eq!(hidden.len(), self.wire, "k3: bad wire width");
        let h = self.hidden;
        let (prefix, blocks) = hidden.split_at(h);
        let nb = self.max_blocks;

        let mut x = vec![0.0f32; h];
        apply_attn_res(
            prefix,
            &blocks[..nb * h],
            &head.out_res_proj,
            &head.out_res_norm,
            self.m.rms_norm_eps,
            &mut x,
        );
        rmsnorm(&mut x, &head.norm, self.m.rms_norm_eps);
        let mut logits = vec![0.0f32; self.m.vocab_size];
        linear_bf16_w(&x, &head.lm_head, self.m.vocab_size, h, &mut logits);
        logits
    }
}
