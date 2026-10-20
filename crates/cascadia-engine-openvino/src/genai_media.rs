//! Sizing an `ov-genai` prompt's images and videos before GenAI runs it.
//!
//! GenAI expands each `<ov_genai_image_N>` / `<ov_genai_video_N>` tag inside
//! `generate` and reports the expanded prompt's length only after the run, so
//! the engine cannot ask it how long a media prompt is. [`MediaSizer`]
//! answers from the export's own processor configs instead, with the same
//! per-family token math the staged engines' vision front ends use (each
//! pinned to Hugging Face's processor by its `tests/*_vision_parity.rs`, and
//! equal to GenAI's prompt counts on the published exports: see
//! `docs/rfcs/vlm-support.md`, "Every GenAI-only family across machines").
//!
//! The count is a floor of what GenAI builds, never more, so the window check
//! never refuses a prompt that fits: it counts each item's soft tokens only
//! (no begin/end-of-image framing, no Qwen3-VL / Gemma 4 timestamp text), the
//! prompt text with the tags removed, and gives back one token per removed tag
//! for any merge of the text on either side of it.

use std::path::Path;

use cascadia_types::{MediaInput, MediaKind};

/// One export family's soft-token math, read from its processor configs.
#[derive(Clone, Debug)]
pub(crate) enum MediaSizer {
    QwenVl(crate::qwen_vl_vision::Layout),
    Gemma3 {
        /// Soft tokens per image (`image_seq_length`).
        per_image: usize,
    },
    Gemma4 {
        image: crate::gemma4_vision::PreprocessConfig,
        video: crate::gemma4_vision::video::VideoConfig,
    },
    InternVL(crate::internvl_vision::PreprocessConfig),
    Phi3V(crate::phi3v_vision::PreprocessConfig),
    LlavaNext(crate::llava_next_vision::PreprocessConfig),
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

impl MediaSizer {
    /// The sizer for the export at `dir`, by its `config.json` `model_type`.
    /// `Ok(None)` for a family it has no math for; `Err` when the family is
    /// known but its configs cannot be read as the staged front ends read
    /// them. Either way the caller sizes nothing up front.
    pub(crate) fn from_export(dir: &Path) -> Result<Option<Self>, String> {
        let cfg = read_json(&dir.join("config.json"))?;
        let model_type = cfg
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let sizer = match model_type {
            "qwen2_5_vl" => Self::QwenVl(crate::qwen_vl_vision::Layout::load(
                dir,
                crate::qwen_vl_vision::Kind::Qwen25Vl,
                &cfg,
            )?),
            "qwen3_vl" => Self::QwenVl(crate::qwen_vl_vision::Layout::load(
                dir,
                crate::qwen_vl_vision::Kind::Qwen3Vl,
                &cfg,
            )?),
            "gemma3" => {
                let p = crate::gemma3_vision::PreprocessConfig::from_model_dir(dir)?;
                if let Some(n) = cfg
                    .get("mm_tokens_per_image")
                    .and_then(serde_json::Value::as_u64)
                {
                    if n != u64::from(p.image_seq_length) {
                        return Err(format!(
                            "config mm_tokens_per_image {n} != processor image_seq_length {}",
                            p.image_seq_length
                        ));
                    }
                }
                Self::Gemma3 {
                    per_image: p.image_seq_length as usize,
                }
            }
            "gemma4" => Self::Gemma4 {
                image: crate::gemma4_vision::PreprocessConfig::from_model_dir(dir)?,
                video: crate::gemma4_vision::video::VideoConfig::from_model_dir(dir)?,
            },
            "internvl_chat" => {
                Self::InternVL(crate::internvl_vision::PreprocessConfig::from_configs(
                    &cfg,
                    &dir.join("preprocessor_config.json"),
                )?)
            }
            "phi3_v" => Self::Phi3V(crate::phi3v_vision::PreprocessConfig::from_json_file(
                &dir.join("preprocessor_config.json"),
            )?),
            "llava_next" => {
                let pre = read_json(&dir.join("preprocessor_config.json"))?;
                let proc_path = dir.join("processor_config.json");
                let proc = proc_path
                    .exists()
                    .then(|| read_json(&proc_path))
                    .transpose()?;
                Self::LlavaNext(crate::llava_next_vision::PreprocessConfig::from_export(
                    &pre,
                    proc.as_ref(),
                    &cfg,
                )?)
            }
            _ => return Ok(None),
        };
        Ok(Some(sizer))
    }

    /// The family, for the load log.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::QwenVl(l) => l.kind.name(),
            Self::Gemma3 { .. } => "Gemma 3",
            Self::Gemma4 { .. } => "Gemma 4",
            Self::InternVL(_) => "InternVL",
            Self::Phi3V(_) => "Phi-3.5-vision",
            Self::LlavaNext(_) => "LLaVA-NeXT",
        }
    }

    /// The soft tokens the turn's images and videos take, from their sizes
    /// alone (nothing decoded or encoded). `Err` for an item this family's
    /// math does not size (a video on a family without native video, a
    /// malformed item): the caller then leaves the turn to GenAI.
    pub(crate) fn soft_tokens(&self, media: &[MediaInput]) -> Result<usize, String> {
        if let Self::QwenVl(l) = self {
            return l.media_tokens(media);
        }
        let mut total = 0usize;
        for (i, m) in media.iter().enumerate() {
            let n = match (self, m.kind) {
                (Self::Gemma4 { video, .. }, MediaKind::Video) => {
                    let l = crate::gemma4_vision::video::layout(m, video)
                        .map_err(|e| format!("video {i}: {e}"))?;
                    l.frames.len() * l.soft_tokens
                }
                _ => {
                    crate::gemma4_vision::check_image(i, m)?;
                    self.image_tokens(m.width, m.height)
                        .map_err(|e| format!("image {i}: {e}"))?
                }
            };
            total += n;
        }
        Ok(total)
    }

    fn image_tokens(&self, width: u32, height: u32) -> Result<usize, String> {
        match self {
            Self::QwenVl(l) => {
                let [t, h, w] = l.grid_for_size(width, height)?;
                Ok((t * (h / l.merge) * (w / l.merge)) as usize)
            }
            Self::Gemma3 { per_image } => Ok(*per_image),
            Self::Gemma4 { image, .. } => {
                crate::gemma4_vision::soft_tokens_for_size(width, height, image)
            }
            Self::InternVL(c) => crate::internvl_vision::num_image_tokens(width, height, c),
            Self::Phi3V(c) => crate::phi3v_vision::num_image_tokens(width, height, c),
            Self::LlavaNext(c) => crate::llava_next_vision::image_tokens(height, width, c),
        }
    }
}

/// The prompt with every GenAI universal media tag (`<ov_genai_image_N>`,
/// `<ov_genai_video_N>`) removed, and how many it removed.
pub(crate) fn strip_media_tags(prompt: &str) -> (String, usize) {
    const TAGS: [&str; 2] = ["<ov_genai_image_", "<ov_genai_video_"];
    let mut out = String::with_capacity(prompt.len());
    let mut removed = 0;
    let mut rest = prompt;
    while let Some((at, tag)) = TAGS
        .iter()
        .filter_map(|t| rest.find(t).map(|at| (at, t)))
        .min_by_key(|(at, _)| *at)
    {
        let after = &rest[at + tag.len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after.as_bytes().get(digits) == Some(&b'>') {
            out.push_str(&rest[..at]);
            rest = &after[digits + 1..];
            removed += 1;
        } else {
            // Not a tag after all: keep its first byte and look past it.
            out.push_str(&rest[..at + 1]);
            rest = &rest[at + 1..];
        }
    }
    out.push_str(rest);
    (out, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn fixture(rel: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(rel);
        read_json(&path).unwrap()
    }

    fn export(files: &[(&str, Value)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, v) in files {
            std::fs::write(dir.path().join(name), v.to_string()).unwrap();
        }
        dir
    }

    fn image(width: u32, height: u32) -> MediaInput {
        MediaInput {
            kind: MediaKind::Image,
            pixels: vec![0u8; (width * height * 3) as usize].into(),
            num_frames: 1,
            width,
            height,
            timestamps_ms: None,
            sha256: [0; 32],
        }
    }

    fn video(frames: u32, width: u32, height: u32, timestamps_ms: Vec<u32>) -> MediaInput {
        MediaInput {
            kind: MediaKind::Video,
            pixels: vec![0u8; (frames * width * height * 3) as usize].into(),
            num_frames: frames,
            width,
            height,
            timestamps_ms: Some(timestamps_ms),
            sha256: [1; 32],
        }
    }

    fn size(v: &Value) -> (u32, u32) {
        let s = v["size"].as_array().unwrap();
        (s[0].as_u64().unwrap() as u32, s[1].as_u64().unwrap() as u32)
    }

    fn sizer(dir: &tempfile::TempDir) -> MediaSizer {
        MediaSizer::from_export(dir.path()).unwrap().unwrap()
    }

    #[test]
    fn strips_only_whole_universal_tags() {
        let (t, n) = strip_media_tags("a<ov_genai_image_0>b<ov_genai_video_12>c<ov_genai_image_1>");
        assert_eq!((t.as_str(), n), ("abc", 3));
        let (t, n) = strip_media_tags("<ov_genai_image_>x<ov_genai_image_3 y<ov_genai_image_9>");
        assert_eq!(
            (t.as_str(), n),
            ("<ov_genai_image_>x<ov_genai_image_3 y", 1)
        );
        assert_eq!(strip_media_tags("plain"), ("plain".to_string(), 0));
    }

    #[test]
    fn unknown_families_are_not_sized() {
        for t in ["llava", "minicpmv", "qwen2_vl", "qwen3_5_moe", "qwen3"] {
            let dir = export(&[("config.json", json!({ "model_type": t }))]);
            assert!(
                MediaSizer::from_export(dir.path()).unwrap().is_none(),
                "{t}"
            );
        }
    }

    /// A known family whose configs the front ends refuse is an error, which
    /// the engine logs and serves without sizing.
    #[test]
    fn unreadable_known_family_is_an_error() {
        let dir = export(&[("config.json", json!({ "model_type": "phi3_v" }))]);
        assert!(MediaSizer::from_export(dir.path()).is_err());
    }

    /// Qwen2.5-VL and Qwen3-VL images and videos: HF's pad counts
    /// (`tests/fixtures/qwen_vl_vision`), from the export's processor
    /// configs. The SDK check's 448x448 image is 256 / 196 of GenAI's 294 /
    /// 223 prompt tokens, its 4-frame 448x448 clip 512 / 392 of 565 / 447.
    #[test]
    fn qwen_vl_counts_match_hf() {
        let r = fixture("qwen_vl_vision/reference.json");
        let rv = fixture("qwen_vl_vision/video_reference.json");
        for (fam, f) in r["families"].as_object().unwrap() {
            let ids = &f["token_ids"];
            let cfg = json!({
                "model_type": fam,
                "image_token_id": ids["image"],
                "video_token_id": ids["video"],
                "vision_start_token_id": ids["vision_start"],
                "vision_end_token_id": ids["vision_end"],
                "vision_config": { "tokens_per_second": 2 },
            });
            let v = &rv["families"][fam];
            let dir = export(&[
                ("config.json", cfg),
                ("preprocessor_config.json", f["processor"].clone()),
                (
                    "video_preprocessor_config.json",
                    v["video_processor"].clone(),
                ),
            ]);
            let s = sizer(&dir);
            let merge = f["spatial_merge_size"].as_u64().unwrap() as usize;
            // Entries with a `source` rerun an image at another pixel budget.
            for (name, img) in f["images"].as_object().unwrap() {
                if img.get("source").is_some() {
                    continue;
                }
                let (w, h) = size(img);
                let g: Vec<usize> = img["fast"]["grid_thw"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let want = g[0] * g[1] * g[2] / (merge * merge);
                assert_eq!(s.soft_tokens(&[image(w, h)]).unwrap(), want, "{fam} {name}");
            }
            let one = s.soft_tokens(&[image(448, 448)]).unwrap();
            assert_eq!(one, if fam == "qwen2_5_vl" { 256 } else { 196 }, "{fam}");
            for (name, vid) in v["videos"].as_object().unwrap() {
                let frames = vid["frames"].as_u64().unwrap() as u32;
                let (w, h) = (
                    vid["width"].as_u64().unwrap() as u32,
                    vid["height"].as_u64().unwrap() as u32,
                );
                let ts: Vec<u32> = vid["timestamps_ms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let g: Vec<usize> = vid["video_grid_thw"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let want = g[0] * g[1] * g[2] / (merge * merge);
                let got = s.soft_tokens(&[video(frames, w, h, ts)]).unwrap();
                assert_eq!(got, want, "{fam} video {name}");
            }
            let clip = s
                .soft_tokens(&[video(4, 448, 448, vec![0, 1000, 2000, 3000])])
                .unwrap();
            assert_eq!(clip, if fam == "qwen2_5_vl" { 512 } else { 392 }, "{fam}");
        }
    }

    /// InternVL2: tiles x 256 as HF's `dynamic_preprocess` (thumbnail
    /// included); 256 of GenAI's 282 for the SDK check's image.
    #[test]
    fn internvl_counts_match_hf() {
        let r = fixture("internvl_vision/reference.json");
        let c = &r["config"];
        let dir = export(&[
            (
                "config.json",
                json!({
                    "model_type": "internvl_chat",
                    "force_image_size": c["image_size"],
                    "downsample_ratio": 0.5,
                    "min_dynamic_patch": c["min_tiles"],
                    "max_dynamic_patch": c["max_tiles"],
                    "use_thumbnail": c["use_thumbnail"],
                    "dynamic_image_size": true,
                    "vision_config": { "patch_size": 14, "image_size": 448 },
                }),
            ),
            (
                "preprocessor_config.json",
                json!({ "image_mean": c["image_mean"], "image_std": c["image_std"] }),
            ),
        ]);
        let s = sizer(&dir);
        for (name, img) in r["images"].as_object().unwrap() {
            let (w, h) = size(img);
            let want = img["num_tokens"].as_u64().unwrap() as usize;
            assert_eq!(s.soft_tokens(&[image(w, h)]).unwrap(), want, "{name}");
        }
        assert_eq!(s.soft_tokens(&[image(448, 448)]).unwrap(), 256);
    }

    /// Phi-3.5-vision: the HD transform's soft tokens; 757 of GenAI's 785.
    #[test]
    fn phi3v_counts_match_hf() {
        let r = fixture("phi3v_vision/reference.json");
        let dir = export(&[
            ("config.json", json!({ "model_type": "phi3_v" })),
            ("preprocessor_config.json", r["processor"].clone()),
        ]);
        let s = sizer(&dir);
        for (name, img) in r["images"].as_object().unwrap() {
            let (w, h) = size(img);
            let want = img["num_img_tokens"].as_u64().unwrap() as usize;
            assert_eq!(s.soft_tokens(&[image(w, h)]).unwrap(), want, "{name}");
        }
        assert_eq!(s.soft_tokens(&[image(448, 448)]).unwrap(), 757);
    }

    /// LLaVA-NeXT: base view plus unpadded tile rows and their newlines;
    /// 2928 of GenAI's 2956.
    #[test]
    fn llava_next_counts_match_hf() {
        let r = fixture("llava_next_vision/reference.json");
        let dir = export(&[
            (
                "config.json",
                json!({
                    "model_type": "llava_next",
                    "image_grid_pinpoints": r["processor"]["image_grid_pinpoints"],
                    "vision_config": { "image_size": 336, "patch_size": r["patch_size"] },
                }),
            ),
            ("preprocessor_config.json", r["processor"].clone()),
        ]);
        let s = sizer(&dir);
        for (name, img) in r["images"].as_object().unwrap() {
            let (w, h) = size(img);
            let want = img["tokens"].as_u64().unwrap() as usize;
            assert_eq!(s.soft_tokens(&[image(w, h)]).unwrap(), want, "{name}");
        }
        assert_eq!(s.soft_tokens(&[image(448, 448)]).unwrap(), 2928);
    }

    /// Gemma 3: `image_seq_length` per image whatever its size; 256 of
    /// GenAI's 286. Native video is not a Gemma 3 input.
    #[test]
    fn gemma3_counts_match_hf() {
        let r = fixture("gemma3_vision/reference.json");
        let dir = export(&[
            (
                "config.json",
                json!({ "model_type": "gemma3", "mm_tokens_per_image": 256 }),
            ),
            ("preprocessor_config.json", r["processor"].clone()),
        ]);
        let s = sizer(&dir);
        let all: Vec<MediaInput> = r["images"]
            .as_object()
            .unwrap()
            .values()
            .map(|img| {
                let (w, h) = size(img);
                image(w, h)
            })
            .collect();
        assert_eq!(s.soft_tokens(&all).unwrap(), 256 * all.len());
        assert!(s.soft_tokens(&[video(2, 8, 8, vec![0, 1000])]).is_err());
    }

    /// Gemma 4 (26B / 31B processor): each image's pooled patches and each
    /// video frame's, at the video budget; 256 of GenAI's 288 for the SDK
    /// image, 4 x 63 = 252 of its 336 for the SDK clip (448x336 frames).
    #[test]
    fn gemma4_counts_match_hf() {
        let r = fixture("gemma4_vision/reference.json");
        let rv = fixture("gemma4_vision/video_reference_26b_31b.json");
        let dir = export(&[
            ("config.json", json!({ "model_type": "gemma4" })),
            ("preprocessor_config.json", r["processor"].clone()),
            (
                "video_preprocessor_config.json",
                rv["video_processor"].clone(),
            ),
        ]);
        let s = sizer(&dir);
        // Entries with a `source` rerun an image at another soft-token budget.
        for (name, img) in r["images"].as_object().unwrap() {
            if img.get("source").is_some() {
                continue;
            }
            let (w, h) = size(img);
            let want = img["fast"]["num_soft_tokens"].as_u64().unwrap() as usize;
            assert_eq!(s.soft_tokens(&[image(w, h)]).unwrap(), want, "{name}");
        }
        assert_eq!(s.soft_tokens(&[image(448, 448)]).unwrap(), 256);
        for (name, v) in rv["videos"].as_object().unwrap() {
            let ts: Vec<u32> = v["timestamps_ms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let m = video(
                v["frames"].as_u64().unwrap() as u32,
                v["width"].as_u64().unwrap() as u32,
                v["height"].as_u64().unwrap() as u32,
                ts,
            );
            let kept = v["kept_frames"].as_array().unwrap().len();
            let want = kept * v["soft_tokens_per_frame"].as_u64().unwrap() as usize;
            assert_eq!(s.soft_tokens(&[m]).unwrap(), want, "video {name}");
        }
    }
}
