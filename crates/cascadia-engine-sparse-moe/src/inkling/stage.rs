//! Inkling stage runner — the engine-facing wrapper `SparseMoEBuilder`
//! constructs when a model dir's manifest says `arch = "inkling"`. One
//! contiguous layer slice per rank; rank 0 embeds (+ embed norm) and drives,
//! mids relay the `hidden_size`-wide residual stream, the last rank runs the
//! final norm / mup / unembed. Implements [`StagedRunner`] so the generic
//! [`crate::engine::PipelineEngine`] drives it exactly like glm5 / dsv4.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::ep::EpClient;
use super::loader::{load_stage, read_manifest, ExpertSet, InklingManifest};
use super::model::{Head, Layer, WideTable};
use super::rmsnorm_f32;
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::staged::StagedRunner;

/// Default context budget for the global layers' KV caches (sliding layers
/// keep a fixed `window + rewind` ring regardless). Override with
/// `CASCADIA_INKLING_MAX_SEQ` (or `SparseMoEBuilderConfig::max_seq` for
/// in-process hosts).
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
    head: Option<Head>,                   // norm / mup / unembed on the last rank
    hidden: usize,
    eps: f32,
    max_seq: usize,
    eos: Vec<u32>,
    pos: usize,
    /// Multi-stream slots: `Some(pos)` while a sequence owns the slot, `None`
    /// when free. Empty on the single-sequence path.
    streams: Vec<Option<usize>>,
    pub rank: u32,
    pub total: u32,
    pub lo: usize,
    pub hi: usize,
    /// Per-layer branch clocks for the stage profile; `None` until
    /// [`StagedRunner::enable_profile`].
    profile: Option<Arc<ProfileClocks>>,
}

/// Decode / prefill time per branch, summed over this rank's layers.
#[derive(Default)]
struct ProfileClocks {
    decode_attn_ns: AtomicU64,
    decode_mlp_ns: AtomicU64,
    prefill_attn_ns: AtomicU64,
    prefill_mlp_ns: AtomicU64,
}

impl InklingRunner {
    /// Load rank `rank` of `total`. `layer_start/layer_end` from the ShardSpec
    /// override the even split when nonzero. `experts_mode` (`"eager"` |
    /// `"mmap"`) falls back to `CASCADIA_INKLING_EXPERTS`, then to mmap for any
    /// real-sized expert set (> 32 experts). `remote` makes this rank an
    /// expert-parallel DRIVER: no expert bins are opened (`ExpertSet::None`)
    /// and every MoE layer dispatches to the client's workers, keyed by its
    /// absolute layer index; the client's dims must match the manifest.
    #[allow(clippy::too_many_arguments)]
    pub fn load_staged(
        dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
        experts_mode: Option<String>,
        remote: Option<Arc<EpClient>>,
    ) -> Result<Self, LoadError> {
        let m = read_manifest(dir)?;
        let n = m.num_layers;
        let total = total.max(1);
        let rank = rank.min(total - 1);
        if let Some(c) = remote.as_ref() {
            if c.n_workers() == 0 {
                return Err(LoadError::Manifest(
                    "expert-parallel driver: the expert client has no workers".into(),
                ));
            }
            if c.hidden() != m.hidden_size
                || c.n_routed() != m.num_experts
                || c.n_shared() != m.n_shared_experts
            {
                return Err(LoadError::Manifest(format!(
                    "expert-parallel driver: client dims (hidden {}, routed {}, shared {}) do not \
                     match the manifest (hidden {}, routed {}, shared {})",
                    c.hidden(),
                    c.n_routed(),
                    c.n_shared(),
                    m.hidden_size,
                    m.num_experts,
                    m.n_shared_experts
                )));
            }
        }
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
        let experts = if remote.is_some() {
            ExpertSet::None
        } else {
            ExpertSet::All
        };
        let mut s = load_stage(dir, max_seq, lo, hi, first, last, mode, experts)?;
        if let Some(client) = remote.as_ref() {
            for (i, l) in s.layers.iter_mut().enumerate() {
                if let Some(moe) = l.moe_mut() {
                    moe.attach_remote((lo + i) as u32, Arc::clone(client));
                }
            }
        }
        let cache_bytes: usize = s.layers.iter().map(Layer::cache_bytes).sum();
        tracing::info!(
            rank,
            total,
            lo,
            hi,
            max_seq,
            experts = ?mode,
            expert_workers = remote.as_ref().map_or(0, |c| c.n_workers()),
            cache_mib = cache_bytes >> 20,
            "inkling stage loaded"
        );
        Ok(Self {
            embed: s.embed,
            layers: s.layers,
            head: s.head,
            hidden: m.hidden_size,
            eps: m.rms_norm_eps,
            max_seq,
            eos: m.eos_token_ids.clone(),
            pos: 0,
            streams: Vec::new(),
            rank,
            total,
            lo,
            hi,
            profile: None,
        })
    }
}

impl InklingRunner {
    /// Expert-cache counters summed over this rank's MoE layers.
    pub fn expert_cache_stats_total(&self) -> super::ExpertCacheStats {
        let mut total = super::ExpertCacheStats::default();
        for l in &self.layers {
            if let Some(m) = l.moe() {
                total.add(m.expert_cache_stats());
            }
        }
        total
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
        // The one head implementation (`Head::logits`) shared with `Model` —
        // `x / mup`, not `x * (1 / mup)` (a 1-ULP drift at mup 24).
        self.head
            .as_ref()
            .expect("head_logits on a non-last rank")
            .logits(hidden)
    }
    fn head_logits_rows(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        self.head
            .as_ref()
            .expect("head_logits_rows on a non-last rank")
            .logits_rows(hidden, rows)
    }

    // ---- multi-stream decode ---------------------------------------------
    fn enable_profile(&mut self) {
        if self.profile.is_some() {
            return;
        }
        let clocks = Arc::new(ProfileClocks::default());
        for l in &mut self.layers {
            let c = Arc::clone(&clocks);
            l.set_timing_observer(Some(Arc::new(move |t: super::model::LayerTiming| {
                let (attn, mlp) = if t.prefill {
                    (&c.prefill_attn_ns, &c.prefill_mlp_ns)
                } else {
                    (&c.decode_attn_ns, &c.decode_mlp_ns)
                };
                attn.fetch_add(t.attention.as_nanos() as u64, Ordering::Relaxed);
                mlp.fetch_add(t.mlp.as_nanos() as u64, Ordering::Relaxed);
            })));
        }
        self.profile = Some(clocks);
    }
    fn profile(&self) -> Option<crate::staged::RunnerProfile> {
        let c = self.profile.as_ref()?;
        let cache = self.expert_cache_stats_total();
        let attn = self
            .layers
            .iter()
            .find_map(|l| l.ov_attn())
            .map(|o| o.stats());
        let head = self.head.as_ref().and_then(|h| h.ov()).map(|o| o.stats());
        Some(crate::staged::RunnerProfile {
            decode_attn_ns: c.decode_attn_ns.load(Ordering::Relaxed),
            decode_mlp_ns: c.decode_mlp_ns.load(Ordering::Relaxed),
            prefill_attn_ns: c.prefill_attn_ns.load(Ordering::Relaxed),
            prefill_mlp_ns: c.prefill_mlp_ns.load(Ordering::Relaxed),
            cache_hits: cache.hits,
            cache_misses: cache.misses,
            cache_retained_mib: (cache.retained_bytes >> 20) as u64,
            cache_capacity_mib: (cache.capacity_bytes >> 20) as u64,
            ov_attn_calls: attn.map_or(0, |a| a.calls),
            ov_attn_ns: attn.map_or(0, |a| a.call_ns),
            ov_head_calls: head.map_or(0, |h| h.calls),
            ov_head_ns: head.map_or(0, |h| h.call_ns),
        })
    }
    fn stream_capacity(&self) -> usize {
        self.streams.len()
    }
    fn configure_streams(&mut self, n: usize) -> bool {
        if n == 0 {
            return false;
        }
        for l in &mut self.layers {
            l.ensure_slots(n);
        }
        if self.streams.len() < n {
            self.streams.resize(n, None);
        }
        let per_slot: usize = self.layers.iter().map(Layer::slot_bytes).sum();
        tracing::info!(
            rank = self.rank,
            streams = n,
            slot_mib = per_slot >> 20,
            pool_mib = (per_slot * n) >> 20,
            "inkling stream slots allocated"
        );
        true
    }
    fn open_stream(&mut self) -> Option<usize> {
        let slot = self.streams.iter().position(Option::is_none)?;
        for l in &mut self.layers {
            l.select_slot(slot);
            l.reset();
        }
        self.streams[slot] = Some(0);
        Some(slot)
    }
    fn open_stream_at(&mut self, slot: usize) -> bool {
        if slot >= self.streams.len() {
            return false;
        }
        for l in &mut self.layers {
            l.select_slot(slot);
            l.reset();
        }
        self.streams[slot] = Some(0);
        true
    }
    fn close_stream(&mut self, slot: usize) {
        if let Some(s) = self.streams.get_mut(slot) {
            *s = None;
        }
    }
    fn stream_pos(&self, slot: usize) -> usize {
        self.streams.get(slot).copied().flatten().unwrap_or(0)
    }
    fn prefill_stream(&mut self, slot: usize, hidden: Vec<f32>, rows: usize) -> Vec<f32> {
        let pos = self.streams[slot].expect("prefill_stream on a free slot");
        assert_eq!(
            hidden.len(),
            rows * self.hidden,
            "inkling prefill_stream: bad hidden length"
        );
        let mut x = hidden;
        for l in &mut self.layers {
            l.select_slot(slot);
            x = l.forward_prefill(&x, rows);
        }
        self.streams[slot] = Some(pos + rows);
        x
    }
    fn decode_streams(&mut self, hidden: Vec<f32>, slots: &[usize]) -> Vec<f32> {
        let rows = slots.len();
        assert_eq!(
            hidden.len(),
            rows * self.hidden,
            "inkling decode_streams: bad hidden length"
        );
        for (i, &s) in slots.iter().enumerate() {
            assert!(
                self.streams[s].is_some(),
                "decode_streams: slot {s} is free"
            );
            assert!(
                !slots[..i].contains(&s),
                "decode_streams: slot {s} listed twice in one step"
            );
        }
        let mut x = hidden;
        for l in &mut self.layers {
            x = l.forward_rows(&x, rows, slots);
        }
        for &s in slots {
            if let Some(p) = self.streams[s].as_mut() {
                *p += 1;
            }
        }
        x
    }
}
