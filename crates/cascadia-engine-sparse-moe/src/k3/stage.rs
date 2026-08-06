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
    /// Tokens this rank's state has consumed.
    ///
    /// K3 otherwise carries position implicitly — `forward_layers` ignores the
    /// `pos` argument because the KDA recurrence and the MLA cache each advance
    /// themselves. The prefix cache needs it explicitly: a rank slice can be all
    /// KDA, and KDA state has no length to read back off it.
    pos: usize,
    /// Cached post-prefill state, keyed by the caller's prefix key.
    ///
    /// Prefill is the expensive half — batch-union touches up to `rows * top_k`
    /// distinct experts per layer, measured at ~155 GB for a 5-token prompt — so
    /// a shared prefix is worth not re-paying: 2.45-2.60x on a next-turn
    /// request. Budgeted from free RAM by default, or exactly by
    /// `CASCADIA_K3_PREFIX_CACHE`, which also disables it at 0.
    prefix: PrefixStore,
}

/// A tiny LRU over whole-model state snapshots, bounded by bytes.
///
/// One entry is every layer's state at some position, so entries are large and
/// few: the budget is in bytes rather than count because a long-context entry can
/// be orders of magnitude bigger than a short one.
#[derive(Default)]
struct PrefixStore {
    budget: usize,
    bytes: usize,
    /// `(key, states, covered_tokens)`, insertion order (see `get`).
    entries: Vec<(u64, Vec<LayerState>, usize)>,
}

/// Share of currently-available RAM the prefix cache may hold by default.
///
/// Small on purpose. This memory competes directly with the expert page cache,
/// which is what decides decode throughput, so the cache has to earn its keep
/// out of headroom rather than out of residency. A few entries is all the reuse
/// pattern needs — the engine caps the key index at a handful — so a large
/// budget would buy nothing and cost the thing that matters.
const PREFIX_CACHE_FRACTION: f64 = 0.05;

impl PrefixStore {
    fn new() -> Self {
        Self {
            budget: Self::budget_from_env_or_ram(),
            ..Default::default()
        }
    }

    /// Exact bytes from `CASCADIA_K3_PREFIX_CACHE`, else a slice of free RAM.
    ///
    /// Deriving it rather than defaulting to zero is what makes the cache work
    /// out of the box, and deriving it from what is FREE is what keeps it from
    /// hurting: on a node running K3 at 29 GB of 32 the share is a few tens of
    /// MB, no snapshot fits, and `put` declines before it clones anything. On a
    /// host with room it lands in the GBs and the cache does its job.
    ///
    /// `CASCADIA_K3_PREFIX_CACHE=0` still turns it off outright.
    fn budget_from_env_or_ram() -> usize {
        if let Some(b) = crate::k3::knobs::get().prefix_cache_bytes {
            return b as usize;
        }
        if let Ok(v) = std::env::var("CASCADIA_K3_PREFIX_CACHE") {
            return v.trim().parse::<usize>().unwrap_or(0);
        }
        (residency::mem_available() as f64 * PREFIX_CACHE_FRACTION) as usize
    }

    fn enabled(&self) -> bool {
        self.budget > 0
    }

    /// Clone rather than take-and-reinsert: a hit must NOT reorder the LRU.
    /// Rank 0's token index and every rank's store evict in lockstep, which only
    /// holds if eviction is pure insertion order with no refresh on read — the
    /// same constraint glm's slice cache documents.
    fn get(&self, key: u64) -> Option<(Vec<LayerState>, usize)> {
        let (_, states, covered) = self.entries.iter().find(|(k, ..)| *k == key)?;
        Some((states.clone(), *covered))
    }

    fn put(&mut self, key: u64, states: &[LayerState], covered: usize) {
        if !self.enabled() || self.entries.iter().any(|(k, ..)| *k == key) {
            return;
        }
        let sz: usize = states.iter().map(|s| s.approx_bytes()).sum();
        if sz > self.budget {
            return; // one entry alone blows the budget; caching it is pointless
        }
        while self.bytes + sz > self.budget {
            match self.entries.first() {
                Some(_) => {
                    let old = self.entries.remove(0);
                    self.bytes -= old.1.iter().map(|s| s.approx_bytes()).sum::<usize>();
                }
                None => break,
            }
        }
        self.entries.push((key, states.to_vec(), covered));
        self.bytes += sz;
    }
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
    pub fn load(
        dir: &Path,
        rank: u32,
        total: u32,
        max_seq: usize,
        top_k_override: Option<u32>,
    ) -> Result<Self, K3LoadError> {
        let mut m = K3Manifest::load(dir)?;
        // Applied before the layers are built: `moe_dims` is evaluated once per
        // layer at load, so overriding here is what makes every consumer —
        // routing, the batch-union slot layout, the residency histogram — agree
        // on one k. Setting it later would leave them disagreeing.
        let eff = K3Manifest::effective_top_k(m.top_k, top_k_override);
        if eff != m.top_k {
            tracing::info!(
                config_top_k = m.top_k,
                effective_top_k = eff,
                "k3: top-k override; routed bytes per token scale with this"
            );
            m.top_k = eff;
        }
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
        let raw_budget = residency::autopin_budget(expert_bytes, reserve);
        let u = residency::snapshot();
        // Earn the budget: a histogram with few observations picks near-arbitrary
        // experts and spends the whole allowance on them, evicting page-cache
        // entries that were doing more good. Ramp with confidence instead.
        let budget = residency::autopin_count(u.total(), raw_budget);
        let mut pinned = 0usize;
        if budget > 0 {
            crate::dsv4::expert_mmap::reserve_lockable(budget * expert_bytes as usize);
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
                raw_budget,
                observations = u.total(),
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
            pos: 0,
            prefix: PrefixStore::new(),
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

    /// Routed experts per token, as every MoE layer on this rank currently sees it.
    pub fn top_k(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| match &l.ffn {
                LayerFfn::Moe(_, md, _) => Some(md.top_k),
                LayerFfn::Dense { .. } => None,
            })
            .unwrap_or(self.m.top_k)
    }

    /// Change the routed width after load, clamped the same way as the flag.
    ///
    /// Exists for sweeping k without paying the load again. K3 takes ~1000 s to
    /// load and a sweep wants several points, so reloading per point would be
    /// most of the measurement. Returns the value actually applied.
    ///
    /// Updates EVERY MoE layer: a partial update would leave the rank routing
    /// different widths per layer, which is not a configuration anyone wants to
    /// interpret. Serving would also want this to retune without a restart, but
    /// nothing calls it on that path today.
    pub fn set_top_k(&mut self, k: usize) -> usize {
        let eff = K3Manifest::effective_top_k(self.m.top_k, Some(k as u32));
        for l in self.layers.iter_mut() {
            if let LayerFfn::Moe(_, md, _) = &mut l.ffn {
                md.top_k = eff;
            }
        }
        eff
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
        self.pos = 0;
        prof::new_sequence();
    }

    fn prefix_cache_enabled(&self) -> bool {
        self.prefix.enabled()
    }

    /// Restore every layer's state as it stood after the cached prefix, and
    /// report how many prompt tokens that covers so the caller can skip them.
    ///
    /// Restoring is a whole-model swap because K3 carries state in both layer
    /// kinds — the KDA recurrence and the MLA latent cache — and they must agree
    /// on position or the run is silently wrong.
    /// Restore every layer's state as it stood after the cached prefix and report
    /// how many tokens that covers, so the caller prefills only the suffix.
    ///
    /// This is a whole-slice swap because K3 carries state in both layer kinds —
    /// the KDA recurrence and the MLA latent cache — and they must agree on
    /// position, or the run is silently wrong rather than obviously broken.
    fn restore_prefix(&mut self, key: u64) -> Option<usize> {
        let (states, covered) = self.prefix.get(key)?;
        if states.len() != self.states.len() {
            return None; // a different rank slice; not ours to restore
        }
        self.states = states;
        self.pos = covered;
        prof::note_prefix_hit(covered);
        Some(covered)
    }

    fn cache_prefix(&mut self, key: u64) {
        if self.pos == 0 {
            return;
        }
        let states = std::mem::take(&mut self.states);
        self.prefix.put(key, &states, self.pos);
        self.states = states;
    }

    fn covered_positions(&self) -> Option<usize> {
        Some(self.pos)
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
        self.pos += 1;
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
        if let Some(line) = crate::k3::gate_probe::report() {
            eprintln!("{line}");
        }
        if let Some(line) = crate::k3::chess_probe::report() {
            eprintln!("{line}");
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
        self.pos += rows;
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
        // Prefill costs real wall time and streams real bytes. Without this the
        // byte counters still advance but the clock does not, so the next dump
        // divides prefill's bytes by decode's time and reports a rate that was
        // never achieved.
        let t = std::time::Instant::now();
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
        prof::add(prof::WALL, t);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_budget_wins_over_the_derived_one() {
        // The env is the escape hatch for both directions: a deployment that
        // knows its own headroom can raise it, and 0 must still mean off, or
        // there is no way to take the cache out of the picture when debugging.
        let derived = PrefixStore::budget_from_env_or_ram();
        assert!(
            derived > 0,
            "a machine running tests has free RAM; deriving 0 would leave the \
             cache off everywhere by accident"
        );

        let store = PrefixStore {
            budget: 0,
            ..Default::default()
        };
        assert!(!store.enabled(), "0 must disable the cache outright");
    }

    #[test]
    fn a_budget_smaller_than_one_entry_declines_without_copying() {
        // The constrained-node case, and the reason deriving from FREE RAM is
        // safe: on a node with no headroom the share is tiny, nothing fits, and
        // `put` has to bail BEFORE cloning. If it ever cloned first, enabling
        // this by default would spend memory precisely where there is none.
        let mut store = PrefixStore {
            budget: 1,
            ..Default::default()
        };
        let states = vec![LayerState::Kda(Box::new(crate::k3::model::KdaState {
            recurrent: vec![0.0; 4096],
            conv_q: vec![0.0; 256],
            conv_k: vec![0.0; 256],
            conv_v: vec![0.0; 256],
        }))];
        let before = store.bytes;
        store.put(7, &states, 1);
        assert!(
            store.entries.is_empty(),
            "an oversized entry must not be kept"
        );
        assert_eq!(store.bytes, before, "declining must not charge the budget");
        assert!(store.get(7).is_none());
    }

    #[test]
    fn the_derived_share_leaves_most_of_ram_to_the_page_cache() {
        // This memory competes with expert residency, which is what decides
        // throughput. If the fraction ever grows to where it could evict a
        // meaningful slice of the expert cache, that trade needs measuring
        // first -- it is not obviously the right one.
        assert!(
            PREFIX_CACHE_FRACTION > 0.0 && PREFIX_CACHE_FRACTION <= 0.10,
            "fraction {PREFIX_CACHE_FRACTION} is outside the measured-safe range"
        );
    }
}
