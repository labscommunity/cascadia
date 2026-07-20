//! GLM-5.2 stage runner — the [`StagedRunner`] the pipeline engine drives.
//! One contiguous layer slice per rank; `GlmRunner` itself is the staged
//! container (embed only on rank 0, head only on the last rank), so `GlmModel`
//! stays the untouched single-process form its goldens validate.
//!
//! Position: GLM's `AttentionLayer` appends KV at its own internal counter and
//! ignores the wire `pos` — the two stay in sync only via reset +
//! exactly-one-advance-per-forward. `forward_layers` asserts `pos == self.pos`
//! so a dropped/replayed frame is a loud worker death, never silent garbage.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::loader::{load_stage, read_manifest};
use super::model::GlmLayer;
use super::moe::AnyExpert;
use super::residency::{self, UsageStats};
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
    /// Shared learned-pin routing histogram (attached to every owned MoE layer).
    usage: Arc<Mutex<UsageStats>>,
    /// Where [`Self::save_usage`] persists the histogram (`<dir>/.coli_usage`).
    usage_path: PathBuf,
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
        let mut s = load_stage(dir, max_seq, lo, hi, first, last, mode)?;

        // --- learned-pin residency (cap_for_ram budget + AUTOPIN) -----------
        // Attach the routing histogram to every owned MoE layer (records which
        // experts fire), and mlock the hottest known experts in this slice up to
        // the RAM budget. No-op on the eager path (as_mmap -> None).
        let usage = Arc::new(Mutex::new(UsageStats::new()));
        let usage_path = dir.join(".coli_usage");
        let _ = usage.lock().unwrap().load(&usage_path);

        for (i, layer) in s.layers.iter_mut().enumerate() {
            if let Some(ml) = layer.moe_mut() {
                ml.attach_usage((lo + i) as u32, Arc::clone(&usage));
            }
        }

        if mode == ExpertsMode::Mmap {
            let budget = Self::pin_budget(&m, lo, hi, first, last, max_seq);
            let (total, hot) = {
                let u = usage.lock().unwrap();
                (u.total, u.hottest(residency::autopin_count(u.total, budget)))
            };
            let n_pin = hot.len();
            let mut pinned = 0usize;
            for (gl, e) in hot {
                let gl = gl as usize;
                if gl < lo || gl >= hi {
                    continue; // another rank owns this layer
                }
                if let Some(mm) =
                    s.layers[gl - lo].moe().and_then(|ml| ml.experts().get(e as usize)).and_then(AnyExpert::as_mmap)
                {
                    if mm.pin().is_ok() {
                        pinned += 1;
                    }
                }
            }
            if n_pin > 0 {
                eprintln!(
                    "[glm5] rank {rank}: mlock'd {pinned}/{n_pin} hot experts (budget {budget}, history {total})"
                );
            }
        }

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
            usage,
            usage_path,
        })
    }

    /// RAM budget (in whole experts) this rank may `mlock`, after reserving its
    /// bf16 dense shells, KV latent caches, the batch-union working set, and the
    /// page-cache reserve.
    fn pin_budget(
        m: &super::loader::GlmManifest,
        lo: usize,
        hi: usize,
        first: bool,
        last: bool,
        max_seq: usize,
    ) -> usize {
        let owned = hi - lo;
        let heads = m.num_attention_heads;
        let qk_head = m.qk_nope_head_dim + m.qk_rope_head_dim;
        // Per-layer attention shell params (bf16): wq_a, wq_b, wkv_a, wkv_b, wo.
        let attn = m.q_lora_rank * m.hidden_size
            + heads * qk_head * m.q_lora_rank
            + (m.kv_lora_rank + m.qk_rope_head_dim) * m.hidden_size
            + heads * (m.qk_nope_head_dim + m.v_head_dim) * m.kv_lora_rank
            + m.hidden_size * heads * m.v_head_dim;
        let mut resident = (owned * attn) as u64 * 2;
        // Router projections (f32) on owned MoE layers.
        let moe_owned = (lo..hi).filter(|li| !m.dense_layers.contains(li)).count();
        resident += (moe_owned * m.num_experts * m.hidden_size) as u64 * 4;
        // Embedding / lm_head bf16 tables live on the edge ranks.
        if first {
            resident += (m.vocab_size * m.hidden_size) as u64 * 2;
        }
        if last {
            resident += (m.vocab_size * m.hidden_size) as u64 * 2;
        }
        // Absorbed-MLA latent KV cache: (kv_lora + qk_rope) f32 per token per layer.
        let kv = owned as u64 * max_seq as u64 * (m.kv_lora_rank + m.qk_rope_head_dim) as u64 * 4;
        let eb = residency::int4_expert_bytes(m.hidden_size, m.expert_intermediate);
        residency::pin_expert_count(residency::mem_available(), resident, kv, eb)
    }

    /// Persist the learned-pin routing histogram to `<dir>/.coli_usage` so the
    /// next run mlocks a better initial set ("faster the more you use it").
    /// Each node writes its own file (it only records its own layers); best-effort.
    pub fn save_usage(&self) -> std::io::Result<()> {
        self.usage.lock().unwrap().save(&self.usage_path)
    }
}

impl Drop for GlmRunner {
    /// Persist the routing histogram on teardown (save-on-exit), so a
    /// run's routing feeds the next run's pin set. Skipped when nothing was ever
    /// recorded (an unused runner must not clobber existing history with empties).
    fn drop(&mut self) {
        let has_history = self.usage.lock().map(|u| u.total > 0).unwrap_or(false);
        if has_history {
            let _ = self.save_usage();
        }
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
    fn supports_batched_prefill(&self) -> bool {
        true // glm layers ignore the token id; batch-union prefill is bit-exact
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
        // IndexShare carry threads within this rank's slice. (Cross-rank carry —
        // a "shared" layer at a rank boundary reusing an upstream full layer's
        // selection — is the not-yet-wired distributed case; it falls back to
        // full causal, correct for a single-rank run.)
        let mut carry: Option<Vec<usize>> = None;
        for l in &mut self.layers {
            x = l.forward_token(&x, &mut carry);
        }
        self.pos += 1;
        x
    }
    fn forward_layers_batch(&mut self, hidden: Vec<f32>, base: usize, rows: usize) -> Vec<f32> {
        assert_eq!(
            base, self.pos,
            "glm5 stage batch position desync (expected {}, got {base})",
            self.pos
        );
        assert_eq!(hidden.len(), rows * self.hidden, "glm5 batch: bad hidden length");
        // Each layer runs per-position attention (KV in order) + batch-union MoE.
        let mut x = hidden;
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows]; // per-row IndexShare
        for l in &mut self.layers {
            x = l.forward_prefill(&x, rows, &mut carries);
        }
        self.pos += rows;
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
