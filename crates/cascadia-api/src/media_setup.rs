//! Which media a served model takes: the [`MediaConfig`] for an export or
//! pipeline tree on a given engine, read from its files. One copy for every
//! caller that serves a model: `cascadia worker` / `cascadia run`, and any
//! gateway that decodes media on the serving node itself.

use std::path::Path;

use crate::MediaConfig;

/// The engine serving a model, as far as its media is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaEngine {
    /// `ov-genai` on a VLM-layout export (GenAI's VLMPipeline).
    OvGenai,
    /// `ov-runtime` on a pipeline tree with a `vision` block.
    OvRuntime,
    /// `gemma4` on a Gemma 4 E2B / E4B tree.
    Gemma4,
    /// `qwen35` on a Qwen3.5 / 3.6 tree carrying the vision IRs.
    Qwen35,
    /// `ov-dist-spec` on a causal 1-D-position target tree.
    OvDistSpec,
    /// Every other engine: text only.
    Other,
}

/// What [`media_setup`] found for a served model.
#[derive(Clone, Debug, Default)]
pub struct MediaSetup {
    /// The media config to serve with (`None`: text only). The caller sets
    /// its own limits on it, and [`attach_ffmpeg`] for video.
    pub config: Option<MediaConfig>,
    /// Why a model with a vision tower still serves text only, for the 400
    /// that refuses its images (an ov-genai Pixtral export today).
    pub no_media_reason: Option<String>,
    /// Operator-facing notes worth a log line (a tree without `<|video|>`, a
    /// vision family without an API image tag, ...).
    pub notes: Vec<String>,
}

/// The media a model at `model_dir` takes when served by `engine`: images on
/// an ov-genai VLM export ([`MediaConfig::ov_genai`], by its `model_type`), on
/// a qwen35 tree with the vision IRs, and on a pipeline tree whose
/// `pipeline_config.json` has a `vision` block (ov-runtime, gemma4,
/// ov-dist-spec), by its vision family. Video rides on images (as frames) once
/// ffmpeg is attached, or whole where the engine takes it natively.
pub fn media_setup(engine: MediaEngine, model_dir: &Path) -> MediaSetup {
    let mut out = MediaSetup::default();
    let has_vision = model_dir
        .join("openvino_vision_embeddings_model.xml")
        .exists();
    if !has_vision {
        return out;
    }
    // GenAI mishandles some families' images (Pixtral): text only there, and
    // the 400 that refuses them names the alternative.
    let genai_gap = (engine == MediaEngine::OvGenai)
        .then(|| cascadia_engine::genai_unsupported_vision(model_dir))
        .flatten();
    if let Some(why) = genai_gap {
        out.notes.push(format!(
            "ov-genai: serving this VLM export text only: {why}"
        ));
        out.no_media_reason = Some(why);
        return out;
    }
    out.config = match engine {
        // GenAI also reads the family's native tags as image slots, so the
        // API refuses them in caller text by the export's model type.
        MediaEngine::OvGenai => Some(MediaConfig::ov_genai(
            export_model_type(model_dir).as_deref(),
        )),
        // Rank 0 also takes each video whole (Qwen3.5's frame pairs and
        // timestamps), by the export's model type.
        MediaEngine::Qwen35 => Some(
            MediaConfig::qwen35_native().staged_native_video(
                export_model_type(model_dir)
                    .as_deref()
                    .unwrap_or("qwen3_5_moe"),
            ),
        ),
        MediaEngine::OvRuntime | MediaEngine::Gemma4 | MediaEngine::OvDistSpec => {
            match pipeline_vision_family(model_dir) {
                Some(f) => {
                    let gemma4 = f == "gemma4"
                        && matches!(engine, MediaEngine::Gemma4 | MediaEngine::OvRuntime);
                    let video_token = gemma4 && tree_has_video_token(model_dir);
                    if gemma4 && !video_token {
                        out.notes.push(
                            "gemma4 tree tokenizer has no <|video|>; videos arrive as frames"
                                .into(),
                        );
                    }
                    let media = vision_tree_media(engine, &f, video_token);
                    if media.is_none() {
                        out.notes.push(format!(
                            "pipeline vision family `{f}` has no API image tag; serving text only"
                        ));
                    }
                    media
                }
                None => None,
            }
        }
        MediaEngine::Other => None,
    };
    out
}

/// Turn video input on: `explicit` (an operator-named ffmpeg) must work, else
/// one found on the PATH is used. `Ok(None)` when it is on, `Ok(Some(note))`
/// when no ffmpeg was found and video stays off (images still work), `Err`
/// when the named one does not work.
pub fn attach_ffmpeg(
    config: &mut MediaConfig,
    explicit: Option<&Path>,
) -> Result<Option<String>, String> {
    match (crate::Ffmpeg::locate(explicit), explicit) {
        (Ok(f), _) => {
            config.ffmpeg = Some(f);
            Ok(None)
        }
        (Err(why), Some(_)) => Err(why),
        (Err(why), None) => Ok(Some(format!("video input off: no ffmpeg found ({why})"))),
    }
}

/// The API's image settings for a vision tree of `family` served by `engine`
/// (the tag rank 0 expands), or `None` where that engine has no image path for
/// the family. `video_token`: the tree's tokenizer has `<|video|>`
/// ([`tree_has_video_token`]), which a Gemma 4 tree needs to take video.
pub(crate) fn vision_tree_media(
    engine: MediaEngine,
    family: &str,
    video_token: bool,
) -> Option<MediaConfig> {
    match (engine, family) {
        // A Gemma 4 tree's rank 0 also takes each video whole (Gemma 4's
        // per-frame `mm:ss` markup) when its tokenizer has `<|video|>`, the
        // engines' own condition (gemma4.rs and runtime.rs load): the gemma4
        // engine's E2B / E4B trees and ov-runtime's 26B / 31B trees (each
        // frame its own bidirectional block). Otherwise a video arrives as
        // frames.
        (MediaEngine::Gemma4 | MediaEngine::OvRuntime, "gemma4") if video_token => {
            Some(MediaConfig::gemma4_native().staged_native_video(family))
        }
        (MediaEngine::Gemma4 | MediaEngine::OvRuntime, "gemma4") => {
            Some(MediaConfig::gemma4_native())
        }
        (MediaEngine::OvRuntime, "gemma3") => Some(MediaConfig::gemma3_native()),
        (MediaEngine::OvRuntime | MediaEngine::OvDistSpec, "pixtral" | "mistral3") => {
            Some(MediaConfig::pixtral_native())
        }
        // Causal families with 1-D positions: staged ov-runtime and
        // ov-dist-spec (whose draft stays text-only) alike.
        (MediaEngine::OvRuntime | MediaEngine::OvDistSpec, "phi3_v") => {
            Some(MediaConfig::phi3v_native())
        }
        (MediaEngine::OvRuntime | MediaEngine::OvDistSpec, "internvl_chat") => {
            Some(MediaConfig::internvl_native())
        }
        (MediaEngine::OvRuntime | MediaEngine::OvDistSpec, "llava_next") => {
            Some(MediaConfig::llava_next_native())
        }
        // Qwen2.5-VL / Qwen3-VL: Qwen's native tags, as the staged qwen35
        // engine; rank 0 expands `<|image_pad|>` and takes each video whole
        // behind `<|video_pad|>`. ov-runtime only: dist-spec feeds 1-D
        // positions and refuses these trees.
        (MediaEngine::OvRuntime, "qwen2_5_vl" | "qwen3_vl") => {
            Some(MediaConfig::qwen35_native().staged_native_video(family))
        }
        _ => None,
    }
}

/// The image family of a pipeline dir: its `pipeline_config.json` `vision`
/// block's `family`, or `gemma4` for a block written by the Gemma 4 vision
/// surgery before the field existed. `None` without a `vision` block.
pub(crate) fn pipeline_vision_family(model: &Path) -> Option<String> {
    let v: serde_json::Value = std::fs::read(model.join("pipeline_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())?;
    let block = v.get("vision").filter(|x| x.is_object())?;
    match block.get("family").and_then(serde_json::Value::as_str) {
        Some(f) => Some(f.to_string()),
        None if v.get("export_version").and_then(serde_json::Value::as_str)
            == Some("gemma4_vision_surgery_v1") =>
        {
            Some("gemma4".into())
        }
        None => Some(String::new()),
    }
}

/// Whether a tree's `tokenizer/tokenizer.json` knows `<|video|>` (an added
/// token or a vocabulary entry), as the gemma4 and ov-runtime engines check
/// at load before a Gemma 4 tree takes video whole.
pub(crate) fn tree_has_video_token(model: &Path) -> bool {
    const TOKEN: &str = "<|video|>";
    let path = model.join("tokenizer").join("tokenizer.json");
    let Some(v) = std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    else {
        return false;
    };
    let added = v
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|a| {
            a.iter()
                .any(|t| t.get("content").and_then(serde_json::Value::as_str) == Some(TOKEN))
        });
    added
        || v.pointer("/model/vocab")
            .and_then(|x| x.get(TOKEN))
            .is_some()
}

/// An export's `config.json` `model_type` (`qwen2_5_vl`, `gemma3`, ...),
/// which picks the native image tags ov-genai reads in a prompt.
pub(crate) fn export_model_type(model: &Path) -> Option<String> {
    let v: serde_json::Value = std::fs::read(model.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())?;
    v.get("model_type")?.as_str().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_tree_media_per_engine() {
        use super::{vision_tree_media, MediaEngine as EngineKind};
        let tag = |e, f| vision_tree_media(e, f, true).map(|m| m.image_placeholder);
        assert_eq!(
            tag(EngineKind::OvRuntime, "gemma4").as_deref(),
            Some("<|image|>")
        );
        assert_eq!(
            tag(EngineKind::Gemma4, "gemma4").as_deref(),
            Some("<|image|>")
        );
        assert_eq!(
            tag(EngineKind::OvRuntime, "pixtral").as_deref(),
            Some("[IMG]")
        );
        assert_eq!(
            tag(EngineKind::OvDistSpec, "pixtral").as_deref(),
            Some("[IMG]")
        );
        assert_eq!(
            tag(EngineKind::OvDistSpec, "mistral3").as_deref(),
            Some("[IMG]")
        );
        assert_eq!(
            tag(EngineKind::OvRuntime, "phi3_v").as_deref(),
            Some("<|image_{n}|>\n")
        );
        assert_eq!(
            tag(EngineKind::OvRuntime, "internvl_chat").as_deref(),
            Some("<img><IMG_CONTEXT></img>\n")
        );
        // dist-spec takes the causal 1-D-position families too.
        assert_eq!(
            tag(EngineKind::OvDistSpec, "phi3_v").as_deref(),
            Some("<|image_{n}|>\n")
        );
        assert_eq!(
            tag(EngineKind::OvDistSpec, "internvl_chat").as_deref(),
            Some("<img><IMG_CONTEXT></img>\n")
        );
        assert_eq!(
            tag(EngineKind::OvDistSpec, "llava_next"),
            tag(EngineKind::OvRuntime, "llava_next")
        );
        assert!(tag(EngineKind::OvDistSpec, "llava_next").is_some());
        assert_eq!(
            tag(EngineKind::OvRuntime, "gemma3").as_deref(),
            Some("<start_of_image>")
        );
        // dist-spec refuses Gemma 3 / 4 targets at load; the gemma4 engine
        // serves only its own family.
        assert_eq!(tag(EngineKind::OvDistSpec, "gemma3"), None);
        assert_eq!(tag(EngineKind::Gemma4, "gemma3"), None);
        assert_eq!(tag(EngineKind::OvDistSpec, "gemma4"), None);
        assert_eq!(tag(EngineKind::Gemma4, "pixtral"), None);
        assert_eq!(tag(EngineKind::OvDistSpec, "qwen35"), None);
        let qwen = "<|vision_start|><|image_pad|><|vision_end|>";
        assert_eq!(
            tag(EngineKind::OvRuntime, "qwen2_5_vl").as_deref(),
            Some(qwen)
        );
        assert_eq!(
            tag(EngineKind::OvRuntime, "qwen3_vl").as_deref(),
            Some(qwen)
        );
        assert_eq!(tag(EngineKind::OvDistSpec, "qwen2_5_vl"), None);
        assert_eq!(tag(EngineKind::OvDistSpec, "qwen3_vl"), None);
        assert_eq!(
            tag(EngineKind::OvRuntime, "llava_next").as_deref(),
            Some("<image>\n")
        );
        // Native video: the ov-runtime Qwen VL trees and Gemma 4 trees on
        // either engine, behind Gemma 4's own `<|video|>`.
        let native = |e, f| vision_tree_media(e, f, true).is_some_and(|m| m.native_video);
        assert!(native(EngineKind::OvRuntime, "qwen2_5_vl"));
        assert!(native(EngineKind::OvRuntime, "qwen3_vl"));
        for e in [EngineKind::Gemma4, EngineKind::OvRuntime] {
            assert!(native(e, "gemma4"));
            assert_eq!(
                vision_tree_media(e, "gemma4", true)
                    .map(|m| (m.video_placeholder, m.video_frame_multiple)),
                Some(("<|video|>".to_string(), 1))
            );
            // Without `<|video|>` in its tokenizer the engine refuses native
            // video, so the API keeps sending frames (images still served).
            let plain = vision_tree_media(e, "gemma4", false).unwrap();
            assert!(!plain.native_video);
            assert_eq!(plain.image_placeholder, "<|image|>");
        }
        for f in ["gemma3", "pixtral", "phi3_v", "internvl_chat", "llava_next"] {
            assert!(!native(EngineKind::OvRuntime, f), "{f}");
        }
        assert!(!native(EngineKind::OvDistSpec, "pixtral"));
    }

    #[test]
    fn pipeline_vision_family_is_read() {
        let dir = std::env::temp_dir().join(format!("cascadia-pv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.as_path();
        let write = |s: &str| std::fs::write(dir.join("pipeline_config.json"), s).unwrap();
        assert_eq!(pipeline_vision_family(d), None, "no pipeline_config.json");
        write(r#"{"num_stages":2}"#);
        assert_eq!(pipeline_vision_family(d), None);
        write(r#"{"vision":{"family":"gemma4"}}"#);
        assert_eq!(pipeline_vision_family(d).as_deref(), Some("gemma4"));
        write(r#"{"export_version":"gemma4_vision_surgery_v1","vision":{"text_embeddings":"t"}}"#);
        assert_eq!(
            pipeline_vision_family(d).as_deref(),
            Some("gemma4"),
            "pre-field trees"
        );
        write(r#"{"vision":{"family":"mistral3"}}"#);
        assert_eq!(pipeline_vision_family(d).as_deref(), Some("mistral3"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ov_genai_media_reads_the_export_model_type() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path();
        assert_eq!(export_model_type(model), None);
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen2_5_vl", "vision_config": {"model_type": "x"}}"#,
        )
        .unwrap();
        let t = export_model_type(model);
        assert_eq!(t.as_deref(), Some("qwen2_5_vl"));
        let media = MediaConfig::ov_genai(t.as_deref());
        assert!(media.reserved_in("<|image_pad|>").is_some());
        assert!(media.reserved_in("<image>").is_none());
    }

    #[test]
    fn video_token_is_read_from_the_tree_tokenizer() {
        let dir = std::env::temp_dir().join(format!("cascadia-vt-{}", std::process::id()));
        let tok = dir.join("tokenizer");
        std::fs::create_dir_all(&tok).unwrap();
        let d = dir.as_path();
        let write = |s: &str| std::fs::write(tok.join("tokenizer.json"), s).unwrap();
        assert!(!tree_has_video_token(d), "no tokenizer.json");
        write(r#"{"added_tokens":[{"id":1,"content":"<|image|>"}],"model":{"vocab":{"a":0}}}"#);
        assert!(!tree_has_video_token(d));
        write(r#"{"added_tokens":[{"id":7,"content":"<|video|>"}],"model":{"vocab":{}}}"#);
        assert!(tree_has_video_token(d));
        write(r#"{"added_tokens":[],"model":{"vocab":{"<|video|>":7}}}"#);
        assert!(tree_has_video_token(d));
        write("not json");
        assert!(!tree_has_video_token(d));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// One call decides a served model's media: text only without a vision
    /// encoder, an ov-genai export's tags by its model type, the Pixtral gap
    /// with its reason, and a pipeline tree's family per engine.
    #[test]
    fn media_setup_per_model_and_engine() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        let write = |name: &str, body: &str| std::fs::write(p.join(name), body).unwrap();
        let s = media_setup(MediaEngine::OvGenai, p);
        assert!(
            s.config.is_none() && s.no_media_reason.is_none(),
            "no vision encoder"
        );
        write("openvino_vision_embeddings_model.xml", "<net/>");
        write("config.json", r#"{"model_type": "qwen2_5_vl"}"#);
        let s = media_setup(MediaEngine::OvGenai, p);
        assert!(s.config.unwrap().reserved_in("<|image_pad|>").is_some());
        write(
            "config.json",
            r#"{"model_type": "llava", "vision_config": {"model_type": "pixtral"}}"#,
        );
        let s = media_setup(MediaEngine::OvGenai, p);
        assert!(s.config.is_none());
        assert!(s.no_media_reason.unwrap().contains("ov-runtime"));
        write(
            "pipeline_config.json",
            r#"{"vision": {"family": "pixtral"}}"#,
        );
        for e in [MediaEngine::OvRuntime, MediaEngine::OvDistSpec] {
            assert!(media_setup(e, p).config.is_some(), "{e:?}");
        }
        assert!(media_setup(MediaEngine::Other, p).config.is_none());
        assert!(
            media_setup(MediaEngine::Gemma4, p).config.is_none(),
            "gemma4 serves its own family only"
        );
    }
}
