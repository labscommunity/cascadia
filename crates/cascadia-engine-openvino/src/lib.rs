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
pub mod gemma4;
pub mod genai;
#[cfg(feature = "kv_coord")]
pub mod kv_coordination;
pub mod packed;
mod packed_exec;
pub mod prefix_cache;
pub mod qwen36;
pub mod rotary;
pub mod runtime;
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

/// `max_position_embeddings` (top level or `text_config`) from a pipeline's HF config where its
/// stages load it, `tokenizer/` then the pipeline root. `None` when neither records one.
pub(crate) fn pipeline_model_window(pipeline_dir: &std::path::Path) -> Option<u32> {
    [pipeline_dir.join("tokenizer"), pipeline_dir.to_path_buf()]
        .iter()
        .find_map(|dir| {
            let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
            let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
            let positions = |c: &serde_json::Value| c.get("max_position_embeddings")?.as_u64();
            positions(&v)
                .or_else(|| positions(v.get("text_config")?))
                .filter(|n| *n > 0)
                .and_then(|n| u32::try_from(n).ok())
        })
}

/// Refuse a prompt over a pipeline stage's model window at submit, counted as admission encodes
/// it. Nothing else bounds these stages' prompts, so an unrefused one prefills past what the
/// model was built for. No window or no count admits it.
pub(crate) fn refuse_over_model_window(
    tokenizer: Option<&tokenizers::Tokenizer>,
    prompt: &str,
    window: Option<u32>,
) -> cascadia_engine::EngineResult<()> {
    let (Some(tokenizer), Some(window)) = (tokenizer, window) else {
        return Ok(());
    };
    let Ok(encoding) = tokenizer.encode(prompt, false) else {
        return Ok(());
    };
    let tokens = u32::try_from(encoding.get_ids().len()).unwrap_or(u32::MAX);
    match genai::prompt_over_limit(tokens, window) {
        Some(msg) => Err(cascadia_engine::EngineError::PromptTooLong(msg)),
        None => Ok(()),
    }
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

#[cfg(test)]
mod model_window_tests {
    use super::{pipeline_model_window, refuse_over_model_window};
    use cascadia_engine::EngineError;

    fn word_tokenizer() -> tokenizers::Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"hi":0},"unk_token":"hi"}}"#;
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("tokenizer")
    }

    #[test]
    fn refuses_only_a_counted_prompt_past_the_window() {
        let tok = word_tokenizer();
        assert!(refuse_over_model_window(Some(&tok), "hi hi hi", Some(3)).is_ok());
        let err = refuse_over_model_window(Some(&tok), "hi hi hi hi", Some(3)).unwrap_err();
        match err {
            EngineError::PromptTooLong(msg) => {
                assert!(
                    msg.contains("prompt is 4 tokens") && msg.contains("at most 3"),
                    "{msg}"
                )
            }
            other => panic!("expected PromptTooLong, got {other:?}"),
        }
        assert!(refuse_over_model_window(Some(&tok), "hi hi hi hi", None).is_ok());
        assert!(refuse_over_model_window(None, "hi hi hi hi", Some(3)).is_ok());
    }

    #[test]
    fn pipeline_window_follows_where_the_stage_loads_its_config() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(pipeline_model_window(dir.path()), None);
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config":{"max_position_embeddings":32768}}"#,
        )
        .unwrap();
        assert_eq!(pipeline_model_window(dir.path()), Some(32768));
        std::fs::create_dir(dir.path().join("tokenizer")).unwrap();
        std::fs::write(
            dir.path().join("tokenizer/config.json"),
            r#"{"max_position_embeddings":4096}"#,
        )
        .unwrap();
        assert_eq!(
            pipeline_model_window(dir.path()),
            Some(4096),
            "tokenizer/ wins"
        );
        std::fs::write(
            dir.path().join("tokenizer/config.json"),
            r#"{"max_position_embeddings":0}"#,
        )
        .unwrap();
        assert_eq!(
            pipeline_model_window(dir.path()),
            Some(32768),
            "0 is not a window"
        );
    }
}
