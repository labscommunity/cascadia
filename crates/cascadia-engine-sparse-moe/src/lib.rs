//! Sparse-MoE engine for Kimi K2.6-style models.
//!
//! Kimi K2.6 has 60 MoE layers with 384 experts each and top-8 routing.
//! Naive OV traced exports include all 384 experts as a weighted sum
//! every forward pass, which runs ~50x the compute a real sparse-routed
//! model would. This crate runs the pre-exported per-expert IRs the
//! exporter produces and dispatches only the top-k experts the router
//! actually selected.
//!
//! Model directory layout:
//!
//! ```text
//! <model_dir>/
//!     manifest.json
//!     tokenizer.json | tiktoken.model
//!     layer0/openvino_model.{xml,bin}      # embed + dense layer 0 (stateless)
//!     head/openvino_model.{xml,bin}        # final RMSNorm + lm_head
//!     shells/layer_NN/openvino_model.{xml,bin}    # MoE shell w/ KV cache
//!     experts/layer_NN/expert_XXX/openvino_model.{xml,bin}
//! ```
//!
//! Wiring per forward step (decode):
//!
//! ```text
//! h = layer0(input_ids)                              # full prefix, stateless
//! for L in 1..num_layers:
//!     attn_out, residual, shared_out, ids, weights, new_k, new_v
//!         = shell_L(h[:, -1:, :], past_k[L], past_v[L], mask, past_len)
//!     moe = sum_k( weights[k] * experts[L][ids[k]](attn_out) )
//!     h_L_out = residual + shared_out + moe
//!     past_k[L] = concat(past_k[L], new_k); past_v[L] = concat(past_v[L], new_v)
//!     h = h_L_out
//! logits = head(h[:, -1:, :])
//! next_token = argmax(logits)
//! ```
//!
//! The expert IRs are lazy-loaded on first use into an LRU cache.

pub mod dist;
pub mod dsv4;
pub mod engine;
pub mod glm;
pub mod inkling;
#[cfg(feature = "kv_coord")]
pub mod kv_coordination;
pub mod kv_prefix_cache;
pub mod manifest;
pub mod ngram_draft;
pub mod ov_kv_cache;
#[cfg(feature = "kv_coord")]
pub mod ov_kv_coordination;
pub mod ov_moe;
pub mod runner;
pub mod sampling;
pub mod spec_decode;
pub mod staged;
pub mod tensors;

#[doc(hidden)]
pub use engine::{prepare_resume, ResumeSeed};

/// Size rayon's global pool to the physical cores when `RAYON_NUM_THREADS`
/// is unset. The row-parallel int4 GEMV kernels gain nothing from a core's
/// second hyperthread and lose a lot to it on macOS (Mac Pro, 28c/56t:
/// 33 → 10 ms per Inkling MoE layer); Linux is indifferent (miner,
/// 24c/48t: 14.2 → 13.4 ms). Called at engine load and by the bench
/// examples; a no-op once a pool exists (a host that built its own keeps
/// it) and whenever the operator set `RAYON_NUM_THREADS`.
pub fn init_thread_pool() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_some() {
            return;
        }
        let phys = num_cpus::get_physical();
        if phys == 0 {
            return;
        }
        if rayon::ThreadPoolBuilder::new()
            .num_threads(phys)
            .build_global()
            .is_ok()
        {
            tracing::info!(threads = phys, "rayon pool sized to the physical cores");
        }
    });
}
pub use engine::{SparseMoEBuilder, SparseMoEBuilderConfig, SparseMoEEngine};
pub use kv_prefix_cache::{KvPrefixCache, KvSnapshot, LayerKvSlice, ModelFingerprint};
pub use manifest::Manifest;
pub use ngram_draft::{Draft, DEFAULT_DRAFT_K, MAX_NGRAM, MIN_NGRAM};
pub use ov_moe::{GenStats, OvMoeError, OvMoeRunner};
pub use runner::{Runner, RunnerError, RunnerOptions};
pub use sampling::SamplingConfig;
pub use spec_decode::{count_accepted, reconcile_after_round, RoundReconcile};
