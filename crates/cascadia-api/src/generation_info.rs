//! What a model directory says about generation: its context window and its
//! recommended sampling values. Read once at startup by the CLI to resolve
//! the server's [`crate::GenerationDefaults`].

use std::path::Path;

/// Generation facts read from a model directory. Every field is `None` when
/// the files are missing, malformed, or do not carry the key.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelGenerationInfo {
    /// The model's context window from `config.json`.
    pub context_window: Option<u32>,
    /// The compiled KV size of a static-shape (NPU) `cascadia shard` export:
    /// the smallest `static_context` over the `stage_<r>/stage_config.json`
    /// files whose `stateful` is false. Only the ov-runtime engine uses it.
    pub static_context: Option<u32>,
    /// `generation_config.json` `temperature`.
    pub temperature: Option<f32>,
    /// `generation_config.json` `top_p`.
    pub top_p: Option<f32>,
}

/// Context-window keys in `config.json`, first match wins (vLLM's list).
const WINDOW_KEYS: [&str; 7] = [
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "max_sequence_length",
    "max_seq_length",
    "seq_len",
];

/// Reads `generation_config.json` and `config.json` from `model_dir`, then
/// from `model_dir/tokenizer/` for any value the root does not give, plus the
/// static (NPU) context of a shard tree. Never fails: a missing or malformed
/// file just leaves its fields `None`.
pub fn read_model_generation_info(model_dir: &Path) -> ModelGenerationInfo {
    let dirs = [model_dir.to_path_buf(), model_dir.join("tokenizer")];
    let sampling = |key: &str| {
        first(&dirs, "generation_config.json", |v| {
            v.get(key)
                .and_then(|x| x.as_f64())
                .filter(|x| x.is_finite() && *x >= 0.0)
                .map(|x| x as f32)
        })
    };
    ModelGenerationInfo {
        context_window: first(&dirs, "config.json", |v| {
            window_of(v).or_else(|| v.get("text_config").and_then(window_of))
        }),
        static_context: static_context(model_dir),
        temperature: sampling("temperature"),
        top_p: sampling("top_p"),
    }
}

/// The first value `get` finds in `file` across `dirs`, in order. A missing
/// or malformed file, or one without the value, falls through to the next.
fn first<T>(
    dirs: &[std::path::PathBuf],
    file: &str,
    get: impl Fn(&serde_json::Value) -> Option<T>,
) -> Option<T> {
    dirs.iter()
        .filter_map(|d| read_json(&d.join(file)))
        .find_map(|v| get(&v))
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The first positive window key in `v` that fits a `u32`.
fn window_of(v: &serde_json::Value) -> Option<u32> {
    WINDOW_KEYS.iter().find_map(|k| {
        v.get(*k)
            .and_then(|x| x.as_u64())
            .filter(|&w| w > 0)
            .and_then(|w| u32::try_from(w).ok())
    })
}

/// Smallest `static_context` over `stage_*/stage_config.json` with
/// `"stateful": false` (an absent `stateful` means stateful, as the engine
/// reads it).
fn static_context(model_dir: &Path) -> Option<u32> {
    std::fs::read_dir(model_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("stage_"))
        .filter_map(|e| read_json(&e.path().join("stage_config.json")))
        .filter(|v| v.get("stateful").and_then(|s| s.as_bool()) == Some(false))
        .filter_map(|v| v.get("static_context")?.as_u64())
        .filter_map(|c| u32::try_from(c).ok())
        .filter(|&c| c > 0)
        .min()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    #[test]
    fn empty_dir_gives_nothing() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(
            read_model_generation_info(d.path()),
            ModelGenerationInfo::default()
        );
    }

    #[test]
    fn missing_dir_gives_nothing() {
        let d = tempfile::tempdir().unwrap();
        let info = read_model_generation_info(&d.path().join("absent"));
        assert_eq!(info, ModelGenerationInfo::default());
    }

    #[test]
    fn config_without_a_window_key_gives_no_window() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "config.json", r#"{"hidden_size": 4096}"#);
        assert_eq!(read_model_generation_info(d.path()).context_window, None);
    }

    #[test]
    fn window_from_max_position_embeddings() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "config.json",
            r#"{"max_position_embeddings": 32768}"#,
        );
        assert_eq!(
            read_model_generation_info(d.path()).context_window,
            Some(32768)
        );
    }

    #[test]
    fn window_keys_follow_the_vllm_order() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "config.json",
            r#"{"seq_len": 1, "n_positions": 2048}"#,
        );
        assert_eq!(
            read_model_generation_info(d.path()).context_window,
            Some(2048)
        );
    }

    #[test]
    fn window_under_text_config() {
        // Gemma keeps the window under `text_config`.
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "config.json",
            r#"{"text_config": {"max_position_embeddings": 131072}}"#,
        );
        assert_eq!(
            read_model_generation_info(d.path()).context_window,
            Some(131072)
        );
    }

    #[test]
    fn root_wins_and_tokenizer_dir_is_the_fallback() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "tokenizer/config.json",
            r#"{"max_position_embeddings": 8192}"#,
        );
        write(
            d.path(),
            "tokenizer/generation_config.json",
            r#"{"temperature": 0.6}"#,
        );
        let info = read_model_generation_info(d.path());
        assert_eq!(info.context_window, Some(8192));
        assert_eq!(info.temperature, Some(0.6));

        write(
            d.path(),
            "config.json",
            r#"{"max_position_embeddings": 4096}"#,
        );
        write(
            d.path(),
            "generation_config.json",
            r#"{"temperature": 0.3}"#,
        );
        let info = read_model_generation_info(d.path());
        assert_eq!(info.context_window, Some(4096));
        assert_eq!(info.temperature, Some(0.3));
    }

    #[test]
    fn sampling_values_from_generation_config() {
        let d = tempfile::tempdir().unwrap();
        // `do_sample` is not applied (D3).
        write(
            d.path(),
            "generation_config.json",
            r#"{"do_sample": false, "temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
        );
        let info = read_model_generation_info(d.path());
        assert_eq!(info.temperature, Some(0.7));
        assert_eq!(info.top_p, Some(0.8));
    }

    #[test]
    fn generation_config_without_sampling_keys_gives_none() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "generation_config.json", r#"{"eos_token_id": 2}"#);
        let info = read_model_generation_info(d.path());
        assert_eq!(info.temperature, None);
        assert_eq!(info.top_p, None);
    }

    #[test]
    fn malformed_json_gives_none_never_a_panic() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "config.json", "{not json");
        write(d.path(), "generation_config.json", "");
        write(d.path(), "stage_0/stage_config.json", "[");
        assert_eq!(
            read_model_generation_info(d.path()),
            ModelGenerationInfo::default()
        );
        // Wrong types are ignored too.
        write(
            d.path(),
            "config.json",
            r#"{"max_position_embeddings": "lots"}"#,
        );
        write(
            d.path(),
            "generation_config.json",
            r#"{"temperature": "hot", "top_p": null}"#,
        );
        assert_eq!(
            read_model_generation_info(d.path()),
            ModelGenerationInfo::default()
        );
    }

    #[test]
    fn malformed_root_file_falls_back_to_tokenizer_dir() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "config.json", "{oops");
        write(
            d.path(),
            "tokenizer/config.json",
            r#"{"max_position_embeddings": 2048}"#,
        );
        assert_eq!(
            read_model_generation_info(d.path()).context_window,
            Some(2048)
        );
    }

    #[test]
    fn static_context_is_the_smallest_stateless_stage() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "pipeline_config.json",
            r#"{"model_id": "m", "num_stages": 3}"#,
        );
        write(
            d.path(),
            "stage_0/stage_config.json",
            r#"{"stateful": false, "static_context": 2048}"#,
        );
        write(
            d.path(),
            "stage_1/stage_config.json",
            r#"{"stateful": false, "static_context": 1024}"#,
        );
        // A stateful stage's static_context is not a compiled KV size.
        write(
            d.path(),
            "stage_2/stage_config.json",
            r#"{"stateful": true, "static_context": 512}"#,
        );
        assert_eq!(
            read_model_generation_info(d.path()).static_context,
            Some(1024)
        );
    }

    #[test]
    fn stateful_or_unmarked_stages_have_no_static_context() {
        let d = tempfile::tempdir().unwrap();
        // Absent `stateful` = stateful (the engine's default).
        write(
            d.path(),
            "stage_0/stage_config.json",
            r#"{"static_context": 1024}"#,
        );
        write(
            d.path(),
            "stage_1/stage_config.json",
            r#"{"stateful": true}"#,
        );
        assert_eq!(read_model_generation_info(d.path()).static_context, None);
    }
}
