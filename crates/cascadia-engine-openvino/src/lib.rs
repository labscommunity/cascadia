//! OpenVINO engines for cascadia.
//!
//! Currently exposes [`OvGenaiBuilder`] / [`OvGenaiEngine`] — the Rust
//! port of `cascadia/worker/engines/openvino/genai_engine.py`. Wraps
//! `cascadia-ov-genai-shim` (FFI to openvino-genai) under the
//! [`cascadia_engine::Engine`] and [`cascadia_engine::Builder`] traits.
//!
//! Two further engines from the Python tree (`ov-runtime`,
//! `ov-dist-spec`) are deferred — they need the lower-level
//! `openvino` crate (Core/CompiledModel/InferRequest) and the
//! distributed spec-decode protocol; tracked separately.

pub mod dist_spec;
pub mod gemma3_vision;
pub mod gemma4;
pub mod gemma4_vision;
pub mod genai;
pub mod image_cache;
pub mod internvl_vision;
#[cfg(feature = "kv_coord")]
pub mod kv_coordination;
pub mod llava_next_vision;
pub mod packed;
mod packed_exec;
pub mod phi3v_vision;
pub mod pixtral_vision;
pub mod prefix_cache;
pub mod qwen35_vision;
pub mod qwen36;
pub mod qwen_vl_vision;
pub mod rotary;
pub mod runtime;
pub mod vision_kv;
mod warn_limit;

pub use dist_spec::{
    DistributedMaskedReq, FrameKind, MaskedReq, OvDistSpecBuilder, OvDistSpecEngine,
    OvDistSpecWorkerBuilder, OvDistSpecWorkerEngine, SpecDecodeStats,
};
pub use gemma4::{Gemma4Builder, Gemma4Engine};
pub use genai::{OvGenaiBuilder, OvGenaiCbEngine, OvGenaiEngine};
pub use qwen36::{Qwen36Builder, Qwen36Engine};
pub use rotary::{ModelTextConfig, RopeScalingConfig, Rotary};
pub use runtime::{OvRuntimeBuilder, OvRuntimeEngine};

/// Tests that touch the process-global activation-timeout knob (or read a
/// value derived from it under a precondition) must hold this lock: three
/// runtime.rs tests set it to 1 s and back while gemma4's ceiling test asserts
/// the budget exceeds the ceiling — racing them made the full-workspace suite
/// flake with "transport budget (10s) must exceed the ceiling".
#[cfg(test)]
pub(crate) fn timeout_knob_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

/// Exclusive id bound for `validate_resume_ids`: max assigned id + 1 — NOT
/// `get_vocab_size(true)`, which returns the ENTRY COUNT and under-counts the
/// bound on a sparse vocab (added tokens with gapped ids), wrongly rejecting
/// legitimate resume ids. Falls back to the count for an empty vocab map.
pub(crate) fn resume_vocab_bound(tok: &tokenizers::Tokenizer) -> u32 {
    tok.get_vocab(true)
        .values()
        .max()
        .map(|&m| m.saturating_add(1))
        .unwrap_or(0)
        .max(tok.get_vocab_size(true) as u32)
}

/// `max_position_embeddings` from `dir/config.json` (`text_config` when nested). `None` when the
/// tree does not state it — the caller's prompt-window guard is then off, so say which of the
/// three ways that happened rather than letting an unreadable config disable it silently.
/// `engine` prefixes the warning.
pub(crate) fn read_max_ctx(dir: &std::path::Path, engine: &str) -> Option<usize> {
    let raw = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(raw) => raw,
        Err(e) => {
            tracing::warn!(error = %e, "{engine}: no readable config.json; the prompt-window guard is OFF");
            return None;
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "{engine}: config.json is not valid JSON; the prompt-window guard is OFF");
            return None;
        }
    };
    let cfg = if v.get("text_config").is_some() {
        &v["text_config"]
    } else {
        &v
    };
    let max = cfg["max_position_embeddings"]
        .as_u64()
        .filter(|&m| m > 0)
        .map(|m| m as usize);
    if max.is_none() {
        tracing::warn!(
            "{engine}: config.json states no max_position_embeddings; the prompt-window guard is OFF"
        );
    }
    max
}
