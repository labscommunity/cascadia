//! Sparse-MoE engine for Kimi K2.6-style models.
//!
//! Kimi K2.6 has 60 MoE layers with 384 experts each and top-8 routing.
//! Naive OV traced exports include all 384 experts as a weighted sum
//! every forward pass, which runs ~50x the compute a real sparse-routed
//! model would (DISCOVERIES — rainier, 2026-04). This crate runs the
//! pre-exported per-expert IRs the rainier exporter produces and
//! dispatches only the top-k experts the router actually selected.
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
pub mod engine;
pub mod manifest;
pub mod runner;
pub mod sampling;
pub mod startup_profile;
pub mod tensors;

pub use engine::{SparseMoEBuilder, SparseMoEBuilderConfig, SparseMoEEngine};
pub use manifest::Manifest;
pub use runner::{Runner, RunnerError};
pub use sampling::SamplingConfig;
pub use startup_profile::{drain_report, format_report, PhaseRecord, PhaseTimer};
