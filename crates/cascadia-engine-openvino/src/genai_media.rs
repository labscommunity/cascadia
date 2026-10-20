//! Sizing an `ov-genai` prompt's images and videos before GenAI runs it.
//!
//! GenAI expands each `<ov_genai_image_N>` / `<ov_genai_video_N>` tag inside
//! `generate` and reports the expanded prompt's length only after the run, so
//! the engine cannot ask it how long a media prompt is. [`MediaSizer`]
//! answers from the export's own processor configs instead.
//!
//! The answer must be a floor of what GenAI builds, or the window check would
//! refuse a prompt that fits. GenAI 2026.4.1 does not preprocess every family
//! as Hugging Face does, so the staged engines' HF-exact token math is not
//! enough on its own. Per family:
//!
//! - LLaVA-NeXT, InternVL2, Qwen2.5-VL, Qwen3-VL: each item counts the smaller
//!   of the HF count (the staged front ends' math) and a mirror of GenAI
//!   2026.4.1's own preprocessing (cited at each mirror below), so the check
//!   stays a floor if either one is what the running GenAI does. Where GenAI's
//!   choice depends on an order its sources leave unspecified (InternVL's
//!   unstable sort), the mirror takes the smallest choice it could make.
//! - Phi-3.5-vision, Gemma 3, Gemma 4: the HF math, which never exceeds
//!   GenAI's (its f32 HD transform for Phi-3.5, the fixed view for Gemma 3,
//!   the same aspect-preserving resize for Gemma 4). A Gemma 4 video keeps at
//!   most the frames GenAI keeps: GenAI takes every frame when the frame
//!   indices are given, as Cascadia gives them (`gemma4/classes.cpp:161-164`).
//!
//! "Never exceeds" is checked in the tests by sweeping image sizes against
//! literal transcriptions of the cited GenAI code, not against a running
//! GenAI.
//!
//! Each item counts its soft tokens only (no begin/end-of-image framing, no
//! Qwen3-VL / Gemma 4 timestamp text); the prompt text is counted with the
//! tags removed, less one token per removed tag for any merge of the text on
//! either side of it. An item a mirror cannot size (GenAI would refuse or
//! fail it) is left to GenAI.

use std::path::Path;

use cascadia_types::{MediaInput, MediaKind};

/// The Qwen VL settings GenAI 2026.4.1 resizes with, read as its
/// `ProcessorConfig` reads them (`processor_config.cpp:11-52`): `min_pixels`
/// / `max_pixels`, else `size.shortest_edge` / `size.longest_edge` when the
/// file has neither key, else the class defaults (`processor_config.hpp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QwenGenai {
    pub patch: u64,
    pub merge: u64,
    pub temporal: u64,
    pub min_pixels: u64,
    pub max_pixels: u64,
}

impl QwenGenai {
    fn from_json(v: &serde_json::Value) -> Self {
        let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        let mut c = Self {
            patch: u("patch_size").unwrap_or(14),
            merge: u("merge_size").unwrap_or(2),
            temporal: u("temporal_patch_size").unwrap_or(2),
            min_pixels: u("min_pixels").unwrap_or(3136),
            max_pixels: u("max_pixels").unwrap_or(12_845_056),
        };
        let set = |k: &str| v.get(k).is_some_and(|x| !x.is_null());
        if !set("min_pixels") && !set("max_pixels") {
            let size = |k: &str| {
                v.get("size")
                    .and_then(|s| s.get(k))
                    .and_then(serde_json::Value::as_u64)
            };
            if let Some(n) = size("shortest_edge") {
                c.min_pixels = n;
            }
            if let Some(n) = size("longest_edge") {
                c.max_pixels = n;
            }
        }
        c
    }

    /// The image and video settings, from the files GenAI reads
    /// (`VisionEncoder::resolve_processor_configs`, `vision_encoder.cpp:64-97`):
    /// `processor_config.json`'s `image_processor` and `video_processor` when it
    /// has both, else `preprocessor_config.json` and
    /// `video_preprocessor_config.json` (the image file when that is missing).
    fn from_export(dir: &Path) -> Result<(Self, Self), String> {
        let proc = dir.join("processor_config.json");
        if proc.exists() {
            let v = read_json(&proc)?;
            if let (Some(i), Some(vid)) = (v.get("image_processor"), v.get("video_processor")) {
                return Ok((Self::from_json(i), Self::from_json(vid)));
            }
        }
        let image = Self::from_json(&read_json(&dir.join("preprocessor_config.json"))?);
        let video_path = dir.join("video_preprocessor_config.json");
        let video = if video_path.exists() {
            Self::from_json(&read_json(&video_path)?)
        } else {
            image
        };
        Ok((image, video))
    }

    /// One image's (or one frame group's) pads: GenAI's
    /// `qwen2_vl_utils::smart_resize` (`qwen2vl/classes.cpp:263-282`: f32
    /// division, `std::round` half away from zero, `floor` / `ceil` against
    /// the budgets), then `grid_h * grid_w / merge²`
    /// (`qwen2vl/classes.cpp:744-748`, `calc_tokens_num` at 1014-1016).
    /// `None` where GenAI throws (aspect past 200).
    fn tokens(&self, height: u64, width: u64) -> Option<usize> {
        let (h, w) = (height, width);
        if h == 0 || w == 0 || self.patch == 0 || self.merge == 0 || h.max(w) / h.min(w) > 200 {
            return None;
        }
        let f = self.patch * self.merge;
        let mut hb = ((h as f32) / (f as f32)).round() as u64 * f;
        let mut wb = ((w as f32) / (f as f32)).round() as u64 * f;
        if hb * wb > self.max_pixels {
            let beta = ((h * w) as f64 / self.max_pixels as f64).sqrt();
            hb = (h as f64 / beta / f as f64).floor() as u64 * f;
            wb = (w as f64 / beta / f as f64).floor() as u64 * f;
        } else if hb * wb < self.min_pixels {
            let beta = (self.min_pixels as f64 / (h * w) as f64).sqrt();
            hb = (h as f64 * beta / f as f64).ceil() as u64 * f;
            wb = (w as f64 * beta / f as f64).ceil() as u64 * f;
        }
        Some(((hb / self.patch) * (wb / self.patch) / (self.merge * self.merge)) as usize)
    }

    /// A video's pads: GenAI encodes its frames `temporal` at a time, then each
    /// leftover frame on its own (`encode_frames_with_config`,
    /// `qwen2vl/classes.cpp:911-935`), every group resized as one frame.
    fn video_tokens(&self, frames: u64, height: u64, width: u64) -> Option<usize> {
        if self.temporal == 0 {
            return None;
        }
        let groups = frames / self.temporal + frames % self.temporal;
        Some(groups as usize * self.tokens(height, width)?)
    }
}

/// One export family's soft-token math, read from its processor configs.
#[derive(Clone, Debug)]
pub(crate) enum MediaSizer {
    QwenVl {
        /// HF's processors (the staged front end's).
        layout: crate::qwen_vl_vision::Layout,
        /// GenAI's image and video processor settings.
        genai_image: QwenGenai,
        genai_video: QwenGenai,
    },
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

/// InternVL2 tiles as GenAI 2026.4.1's `split_image_internvl` picks them
/// (`internvl_chat/classes.cpp:14-93`): the grids `i x j` with `1 <= i*j <=
/// 12` whatever the config says (`max_num = 12` and `use_thumbnail = true`
/// are the defaults its caller keeps, line 106), the aspect ratio and each
/// grid's in f32, the closest grid winning. Equally close grids are settled
/// by their order after an unstable `std::sort` by area (line 34) and an
/// area test, so the pick can be any grid at the smallest distance: this
/// takes the one with the fewest tiles. Returns the soft tokens, tiles plus
/// the thumbnail GenAI adds to more than one tile (line 87).
fn genai_internvl_tokens(width: u32, height: u32, tokens_per_tile: u32) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }
    let ar = width as f32 / height as f32;
    let mut best: Option<(f32, u32)> = None;
    for i in 1..=12u32 {
        for j in 1..=12u32 / i {
            let diff = (ar - i as f32 / j as f32).abs();
            best = match best {
                Some((d, n)) if d < diff || (d == diff && n <= i * j) => Some((d, n)),
                _ => Some((diff, i * j)),
            };
        }
    }
    let blocks = best?.1 as usize;
    let views = blocks + usize::from(blocks != 1);
    Some(views * tokens_per_tile as usize)
}

/// LLaVA-NeXT tokens as GenAI 2026.4.1 builds them. Its
/// `select_best_resolution` reads each `image_grid_pinpoints` pair as
/// `(width, height)` (`clip.cpp:332-357`; HF reads `(height, width)`), with
/// an f32 scale; the base view and `grid_w x grid_h` tiles follow
/// (`llava_next/classes.cpp:62-71`); the tiles' features are unpadded in f32
/// to the image's aspect (`unpad_image`, lines 77-121) and get a newline per
/// row (`add_image_newline`, 216-243). `None` where GenAI's own asserts fail.
fn genai_llava_next_tokens(
    width: u32,
    height: u32,
    cfg: &crate::llava_next_vision::PreprocessConfig,
) -> Option<usize> {
    let (ow, oh) = (width as i64, height as i64);
    if ow == 0 || oh == 0 || cfg.patch_size == 0 || cfg.tile == 0 {
        return None;
    }
    let mut best: Option<(i64, i64)> = None;
    let (mut max_eff, mut min_waste) = (0i64, i64::MAX);
    for &(a, b) in &cfg.pinpoints {
        let (w, h) = (i64::from(a), i64::from(b));
        let scale = (w as f32 / ow as f32).min(h as f32 / oh as f32);
        let dw = (ow as f32 * scale) as i64;
        let dh = (oh as f32 * scale) as i64;
        let eff = (dw * dh).min(ow * oh);
        let waste = w * h - eff;
        if eff > max_eff || (eff == max_eff && waste < min_waste) {
            max_eff = eff;
            min_waste = waste;
            best = Some((w, h));
        }
    }
    let (bw, bh) = best?;
    let side = (cfg.tile / cfg.patch_size) as i64;
    let tile = i64::from(cfg.tile);
    let (cur_h, cur_w) = (bh / tile * side, bw / tile * side);
    if cur_h == 0 || cur_w == 0 {
        return None;
    }
    let original = ow as f32 / oh as f32;
    let current = cur_w as f32 / cur_h as f32;
    let (rows, cols) = if original > current {
        let new_h = (oh as f32 * (cur_w as f32 / ow as f32)) as i64;
        let pad = (cur_h - new_h) / 2;
        if pad < 0 || cur_h <= pad * 2 {
            return None;
        }
        (cur_h - 2 * pad, cur_w)
    } else {
        let new_w = (ow as f32 * (cur_h as f32 / oh as f32)) as i64;
        let pad = (cur_w - new_w) / 2;
        if pad < 0 || cur_w <= pad * 2 {
            return None;
        }
        (cur_h, cur_w - 2 * pad)
    };
    Some((side * side + rows * (cols + 1)) as usize)
}

/// The smaller of two counts when both exist, else whichever one does: an
/// item neither sizes is an error.
fn floor_of(hf: Result<usize, String>, genai: Option<usize>) -> Result<usize, String> {
    match (hf, genai) {
        (Ok(a), Some(b)) => Ok(a.min(b)),
        (Ok(_), None) => Err("GenAI's preprocessing does not size this item".into()),
        (Err(e), _) => Err(e),
    }
}

impl MediaSizer {
    /// The sizer for the export at `dir`, by its `config.json` `model_type`.
    /// `Ok(None)` for a family it has no math for; `Err` when the family is
    /// known but its configs cannot be read. Either way the caller sizes
    /// nothing up front.
    pub(crate) fn from_export(dir: &Path) -> Result<Option<Self>, String> {
        let cfg = read_json(&dir.join("config.json"))?;
        let model_type = cfg
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let qwen = |kind| -> Result<Self, String> {
            let (genai_image, genai_video) = QwenGenai::from_export(dir)?;
            Ok(Self::QwenVl {
                layout: crate::qwen_vl_vision::Layout::load(dir, kind, &cfg)?,
                genai_image,
                genai_video,
            })
        };
        let sizer = match model_type {
            "qwen2_5_vl" => qwen(crate::qwen_vl_vision::Kind::Qwen25Vl)?,
            "qwen3_vl" => qwen(crate::qwen_vl_vision::Kind::Qwen3Vl)?,
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
            Self::QwenVl { layout, .. } => layout.kind.name(),
            Self::Gemma3 { .. } => "Gemma 3",
            Self::Gemma4 { .. } => "Gemma 4",
            Self::InternVL(_) => "InternVL",
            Self::Phi3V(_) => "Phi-3.5-vision",
            Self::LlavaNext(_) => "LLaVA-NeXT",
        }
    }

    /// The soft tokens the turn's images and videos take, from their sizes
    /// alone (nothing decoded or encoded): a floor of GenAI's count (see the
    /// module doc). `Err` for an item it does not size (a video on a family
    /// without native video, a malformed item, one GenAI would refuse): the
    /// caller then leaves the turn to GenAI.
    pub(crate) fn soft_tokens(&self, media: &[MediaInput]) -> Result<usize, String> {
        let mut total = 0usize;
        for (i, m) in media.iter().enumerate() {
            total += self
                .item_tokens(i, m)
                .map_err(|e| format!("media {i}: {e}"))?;
        }
        Ok(total)
    }

    fn item_tokens(&self, i: usize, m: &MediaInput) -> Result<usize, String> {
        if m.kind == MediaKind::Video {
            return self.video_tokens(m);
        }
        crate::gemma4_vision::check_image(i, m)?;
        self.image_tokens(m.width, m.height)
    }

    /// One image's soft tokens from its size.
    fn image_tokens(&self, w: u32, h: u32) -> Result<usize, String> {
        match self {
            Self::QwenVl {
                layout,
                genai_image,
                ..
            } => floor_of(
                layout
                    .grid_for_size(w, h)
                    .map(|[t, gh, gw]| (t * (gh / layout.merge) * (gw / layout.merge)) as usize),
                genai_image.tokens(u64::from(h), u64::from(w)),
            ),
            Self::Gemma3 { per_image } => Ok(*per_image),
            Self::Gemma4 { image, .. } => crate::gemma4_vision::soft_tokens_for_size(w, h, image),
            Self::InternVL(c) => floor_of(
                crate::internvl_vision::num_image_tokens(w, h, c),
                genai_internvl_tokens(w, h, c.tokens_per_tile),
            ),
            Self::Phi3V(c) => crate::phi3v_vision::num_image_tokens(w, h, c),
            Self::LlavaNext(c) => floor_of(
                crate::llava_next_vision::image_tokens(h, w, c),
                genai_llava_next_tokens(w, h, c),
            ),
        }
    }

    /// One native video's soft tokens; an error on a family without native
    /// video (the API sends those families frames as images).
    fn video_tokens(&self, m: &MediaInput) -> Result<usize, String> {
        match self {
            Self::QwenVl {
                layout,
                genai_image,
                genai_video,
            } => {
                let (n, h, w) = (
                    u64::from(m.num_frames),
                    u64::from(m.height),
                    u64::from(m.width),
                );
                // Qwen2.5-VL encodes frames with the image settings, Qwen3-VL
                // with the video ones (`qwen3_vl/classes.cpp:334-336`): the
                // smaller of the two covers either.
                let genai = match (
                    genai_video.video_tokens(n, h, w),
                    genai_image.video_tokens(n, h, w),
                ) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                floor_of(layout.media_tokens(std::slice::from_ref(m)), genai)
            }
            Self::Gemma4 { video, .. } => {
                let l = crate::gemma4_vision::video::layout(m, video)?;
                Ok(l.frames.len() * l.soft_tokens)
            }
            _ => Err(format!("{} takes no native video", self.name())),
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

    fn qwen_export(r: &Value, rv: &Value, fam: &str) -> tempfile::TempDir {
        let f = &r["families"][fam];
        let ids = &f["token_ids"];
        export(&[
            (
                "config.json",
                json!({
                    "model_type": fam,
                    "image_token_id": ids["image"],
                    "video_token_id": ids["video"],
                    "vision_start_token_id": ids["vision_start"],
                    "vision_end_token_id": ids["vision_end"],
                    "vision_config": { "tokens_per_second": 2 },
                }),
            ),
            ("preprocessor_config.json", f["processor"].clone()),
            (
                "video_preprocessor_config.json",
                rv["families"][fam]["video_processor"].clone(),
            ),
        ])
    }

    fn thw(v: &Value) -> usize {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .product()
    }

    /// A literal transcription of GenAI 2026.4.1's `qwen2_vl_utils::smart_resize`
    /// (`qwen2vl/classes.cpp:263-282`) and `calc_tokens_num` (1014-1016), kept
    /// apart from the sizer's: `None` where it throws.
    fn ref_genai_qwen(h: u64, w: u64, patch: u64, merge: u64, min: u64, max: u64) -> Option<usize> {
        if std::cmp::max(h, w) / std::cmp::min(h, w) > 200 {
            return None;
        }
        let factor = patch * merge;
        let mut h_bar = ((h as f32 / factor as f32).round() * factor as f32) as u64;
        let mut w_bar = ((w as f32 / factor as f32).round() * factor as f32) as u64;
        if h_bar * w_bar > max {
            let beta = ((h * w) as f64 / max as f64).sqrt();
            h_bar = ((h as f64 / beta / factor as f64).floor() * factor as f64) as u64;
            w_bar = ((w as f64 / beta / factor as f64).floor() * factor as f64) as u64;
        } else if h_bar * w_bar < min {
            let beta = (min as f64 / (h * w) as f64).sqrt();
            h_bar = ((h as f64 * beta / factor as f64).ceil() * factor as f64) as u64;
            w_bar = ((w as f64 * beta / factor as f64).ceil() * factor as f64) as u64;
        }
        Some(((h_bar / patch) * (w_bar / patch) / (merge * merge)) as usize)
    }

    /// Qwen2.5-VL and Qwen3-VL: never more than HF's pad counts
    /// (`tests/fixtures/qwen_vl_vision`, images and videos), equal to them where
    /// GenAI resizes alike. The SDK check's 448x448 image is 256 / 196 of
    /// GenAI's 294 / 223 prompt tokens, its 4-frame 448x448 clip 512 / 392 of
    /// 565 / 447.
    #[test]
    fn qwen_vl_counts_are_at_most_hf() {
        let r = fixture("qwen_vl_vision/reference.json");
        let rv = fixture("qwen_vl_vision/video_reference.json");
        for (fam, f) in r["families"].as_object().unwrap() {
            let s = sizer(&qwen_export(&r, &rv, fam));
            let merge = f["spatial_merge_size"].as_u64().unwrap() as usize;
            // Entries with a `source` rerun an image at another pixel budget.
            for (name, img) in f["images"].as_object().unwrap() {
                if img.get("source").is_some() {
                    continue;
                }
                let (w, h) = size(img);
                let hf = thw(&img["fast"]["grid_thw"]) / (merge * merge);
                let got = s.soft_tokens(&[image(w, h)]).unwrap();
                assert!(got <= hf, "{fam} {name}: {got} > HF {hf}");
            }
            let one = s.soft_tokens(&[image(448, 448)]).unwrap();
            assert_eq!(one, if fam == "qwen2_5_vl" { 256 } else { 196 }, "{fam}");
            for (name, vid) in rv["families"][fam]["videos"].as_object().unwrap() {
                let ts: Vec<u32> = vid["timestamps_ms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let m = video(
                    vid["frames"].as_u64().unwrap() as u32,
                    vid["width"].as_u64().unwrap() as u32,
                    vid["height"].as_u64().unwrap() as u32,
                    ts,
                );
                let hf = thw(&vid["video_grid_thw"]) / (merge * merge);
                let got = s.soft_tokens(&[m]).unwrap();
                assert!(got <= hf, "{fam} video {name}: {got} > HF {hf}");
            }
            let clip = s
                .soft_tokens(&[video(4, 448, 448, vec![0, 1000, 2000, 3000])])
                .unwrap();
            assert_eq!(clip, if fam == "qwen2_5_vl" { 512 } else { 392 }, "{fam}");
        }
    }

    /// Qwen3-VL 688x80 (either way round): HF's smart_resize rounds 2.5
    /// patches to even (72 pads), GenAI's `std::round` away from zero
    /// (`qwen2vl/classes.cpp:268-269`), which then trips the min-pixels
    /// branch: 66 pads. The sizer takes GenAI's 66.
    #[test]
    fn qwen3_vl_counts_genai_rounding() {
        let r = fixture("qwen_vl_vision/reference.json");
        let rv = fixture("qwen_vl_vision/video_reference.json");
        let s = sizer(&qwen_export(&r, &rv, "qwen3_vl"));
        assert_eq!(ref_genai_qwen(80, 688, 16, 2, 65_536, 16_777_216), Some(66));
        assert_eq!(s.soft_tokens(&[image(688, 80)]).unwrap(), 66);
        assert_eq!(s.soft_tokens(&[image(80, 688)]).unwrap(), 66);
        // Past GenAI's 200:1 limit it throws; the item is left to GenAI.
        assert!(s.soft_tokens(&[image(1206, 6)]).is_err());
    }

    /// GenAI reads `min_pixels` / `max_pixels`, and `size.*` only when the
    /// file has neither (`processor_config.cpp:35-52`); the class defaults
    /// otherwise.
    #[test]
    fn qwen_genai_settings_read_as_genai_does() {
        let c = QwenGenai::from_json(&json!({
            "patch_size": 16,
            "size": { "shortest_edge": 65536, "longest_edge": 16777216 },
        }));
        assert_eq!((c.patch, c.merge, c.temporal), (16, 2, 2));
        assert_eq!((c.min_pixels, c.max_pixels), (65_536, 16_777_216));
        let c = QwenGenai::from_json(&json!({
            "min_pixels": 1000,
            "size": { "shortest_edge": 65536, "longest_edge": 16777216 },
        }));
        assert_eq!(
            (c.patch, c.min_pixels, c.max_pixels),
            (14, 1000, 12_845_056)
        );
        let c = QwenGenai::from_json(&json!({ "min_pixels": null, "max_pixels": null,
            "size": { "shortest_edge": 7, "longest_edge": 8 } }));
        assert_eq!((c.min_pixels, c.max_pixels), (7, 8));
    }

    /// Every image size in the API's budget on a sweep, and native videos of
    /// 1..16 frames: the sizer never exceeds GenAI's count (the transcription
    /// above, with GenAI's image settings for images and the smaller of its
    /// image and video settings for videos, `qwen3_vl/classes.cpp:334-336`).
    #[test]
    fn qwen_vl_sweep_never_exceeds_genai() {
        let r = fixture("qwen_vl_vision/reference.json");
        let rv = fixture("qwen_vl_vision/video_reference.json");
        for fam in ["qwen2_5_vl", "qwen3_vl"] {
            let s = sizer(&qwen_export(&r, &rv, fam));
            let MediaSizer::QwenVl {
                genai_image: gi,
                genai_video: gv,
                ..
            } = &s
            else {
                panic!("{fam}")
            };
            let genai = |c: &QwenGenai, h: u32, w: u32| {
                ref_genai_qwen(
                    h.into(),
                    w.into(),
                    c.patch,
                    c.merge,
                    c.min_pixels,
                    c.max_pixels,
                )
            };
            for (w, h) in sweep() {
                let Some(g) = genai(gi, h, w) else { continue };
                // HF refuses some sizes GenAI takes (an aspect just past
                // 200:1); an unsized item is left to GenAI, which is safe.
                if let Ok(got) = s.image_tokens(w, h) {
                    assert!(got <= g, "{fam} {w}x{h}: {got} > GenAI {g}");
                }
            }
            for frames in [1u32, 2, 3, 4, 5, 8, 15, 16] {
                for (w, h) in sweep().step_by(97) {
                    if w < 32 || h < 32 || u64::from(frames) * u64::from(w * h) > 16 * 448 * 448 {
                        continue;
                    }
                    let groups = (frames / 2 + frames % 2) as usize;
                    let g = match (genai(gi, h, w), genai(gv, h, w)) {
                        (Some(a), Some(b)) => groups * a.min(b),
                        _ => continue,
                    };
                    let ts = (0..frames).map(|k| k * 500).collect();
                    let m = video(frames, w, h, ts);
                    if let Ok(got) = s.soft_tokens(&[m]) {
                        assert!(got <= g, "{fam} {frames}x{w}x{h}: {got} > GenAI {g}");
                    }
                }
            }
        }
    }

    /// Sizes for the sweeps: every size up to 200 a side, then a coarser grid
    /// up to 2048, within the API's 1 MP image budget.
    fn sweep() -> impl Iterator<Item = (u32, u32)> {
        let fine = (1..=200u32).flat_map(|w| (1..=200u32).map(move |h| (w, h)));
        let coarse = (200..=2048u32)
            .step_by(7)
            .flat_map(|w| (1..=2048u32).step_by(5).map(move |h| (w, h)));
        fine.chain(coarse.flat_map(|(w, h)| [(w, h), (h, w)]))
            .filter(|&(w, h)| u64::from(w) * u64::from(h) <= 1 << 20)
    }

    fn internvl_export() -> (Value, tempfile::TempDir) {
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
        (r, dir)
    }

    /// A literal transcription of GenAI 2026.4.1's `split_image_internvl`
    /// (`internvl_chat/classes.cpp:14-93`) at `image_size` 448: `target_ratios`
    /// built with its duplicates, sorted by area (std::sort is unstable, so the
    /// order inside an area is either way round here), the f32 search with its
    /// area tie-break, and the thumbnail. Returns tiles x 256.
    fn ref_genai_internvl(w: u32, h: u32, reverse_ties: bool) -> usize {
        let (min_num, max_num, image_size) = (1u32, 12u32, 448f64);
        let mut ratios = Vec::new();
        for n in min_num..=max_num {
            for i in 1..=n {
                for j in 1..=n {
                    if i * j <= max_num && i * j >= min_num {
                        ratios.push((i, j));
                    }
                }
            }
        }
        if reverse_ties {
            ratios.reverse();
        }
        ratios.sort_by_key(|&(i, j)| i * j);
        let ar = w as f32 / h as f32;
        let area = f64::from(w) * f64::from(h);
        let (mut best_diff, mut best) = (f32::MAX, (1u32, 1u32));
        for &(i, j) in &ratios {
            let diff = (ar - i as f32 / j as f32).abs();
            if diff < best_diff {
                best_diff = diff;
                best = (i, j);
            } else if diff == best_diff && area > 0.5 * image_size * image_size * f64::from(i * j) {
                best = (i, j);
            }
        }
        let blocks = (best.0 * best.1) as usize;
        (blocks + usize::from(blocks != 1)) * 256
    }

    /// InternVL2: never more than HF's `dynamic_preprocess` count (thumbnail
    /// included); 256 of GenAI's 282 for the SDK check's image.
    #[test]
    fn internvl_counts_are_at_most_hf() {
        let (r, dir) = internvl_export();
        let s = sizer(&dir);
        for (name, img) in r["images"].as_object().unwrap() {
            let (w, h) = size(img);
            let hf = img["num_tokens"].as_u64().unwrap() as usize;
            let got = s.soft_tokens(&[image(w, h)]).unwrap();
            assert!(got <= hf, "{name}: {got} > HF {hf}");
        }
        assert_eq!(s.soft_tokens(&[image(448, 448)]).unwrap(), 256);
    }

    /// The reviewer's counterexamples: 7:6 and its multiples sit exactly
    /// between 1:1 and 4:3 in f64, so HF (`internvl_vision`) takes 4:3's 12
    /// tiles plus the thumbnail (3328), while GenAI's f32 search
    /// (`internvl_chat/classes.cpp:22,44-45`) lands elsewhere: 2x2 + thumbnail
    /// (1280) for 700x600, one tile (256) for the others.
    #[test]
    fn internvl_counts_genai_ties() {
        let (_, dir) = internvl_export();
        let s = sizer(&dir);
        for (w, h, genai) in [
            (700, 600, 1280),
            (560, 480, 256),
            (84, 72, 256),
            (77, 66, 256),
        ] {
            let got = s.soft_tokens(&[image(w, h)]).unwrap();
            assert!(got <= genai, "{w}x{h}: {got} > GenAI {genai}");
            assert!(got <= ref_genai_internvl(w, h, false), "{w}x{h}");
            assert!(got <= ref_genai_internvl(w, h, true), "{w}x{h}");
        }
        assert_eq!(ref_genai_internvl(700, 600, false), 1280);
        assert_eq!(ref_genai_internvl(560, 480, false), 256);
    }

    #[test]
    fn internvl_sweep_never_exceeds_genai() {
        let (_, dir) = internvl_export();
        let s = sizer(&dir);
        for (w, h) in sweep() {
            let got = s.image_tokens(w, h).unwrap();
            for rev in [false, true] {
                let g = ref_genai_internvl(w, h, rev);
                assert!(got <= g, "{w}x{h} (reverse ties {rev}): {got} > GenAI {g}");
            }
        }
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

    /// A literal transcription of GenAI 2026.4.1's Phi-3.5-vision count:
    /// `HD_transform` in f32 (`phi3_vision/classes.cpp:49-76`), the short side
    /// padded to whole 336-px crops (`padding_336`, 24-46), then the global
    /// view's 12x(12+1) rows, `glb_GN` and the crops' rows with a `sub_GN` each
    /// (`hd_feature_transform`, 504-520, `add_image_newline` 465-500). `None` where it resizes to nothing.
    fn ref_genai_phi3v(w: u32, h: u32, num_crops: u32) -> Option<usize> {
        let (mut width, mut height) = (w as usize, h as usize);
        let trans = width < height;
        if trans {
            std::mem::swap(&mut width, &mut height);
        }
        let ratio = width as f32 / height as f32;
        let mut scale = 1u32;
        while scale as f32 * (scale as f32 / ratio).ceil() <= num_crops as f32 {
            scale += 1;
        }
        scale -= 1;
        let new_w = scale as usize * 336;
        let new_h = (new_w as f32 / ratio) as usize;
        if scale == 0 || new_h == 0 {
            return None;
        }
        let short = ((new_h as f32 / 336.0).ceil() * 336.0) as usize / 336;
        let long = new_w / 336;
        // A transposed image comes back `new_w` tall (line 71): its crop rows
        // are the long side's.
        let (rows, cols) = if trans { (long, short) } else { (short, long) };
        Some((rows * cols + 1) * 144 + 1 + (rows + 1) * 12)
    }

    #[test]
    fn phi3v_sweep_never_exceeds_genai() {
        let r = fixture("phi3v_vision/reference.json");
        let dir = export(&[
            ("config.json", json!({ "model_type": "phi3_v" })),
            ("preprocessor_config.json", r["processor"].clone()),
        ]);
        let s = sizer(&dir);
        for (w, h) in sweep() {
            let Some(g) = ref_genai_phi3v(w, h, 4) else {
                continue;
            };
            if let Ok(got) = s.image_tokens(w, h) {
                assert!(got <= g, "{w}x{h}: {got} > GenAI {g}");
            }
        }
    }

    fn llava_export() -> (Value, tempfile::TempDir) {
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
        (r, dir)
    }

    /// A literal transcription of GenAI 2026.4.1's LLaVA-NeXT count for the
    /// shipped settings (336 px tiles, 24x24 features): `select_best_resolution`
    /// with the pinpoints as (width, height) (`clip.cpp:332-357`), the grid
    /// (`llava_next/classes.cpp:62-71`), `unpad_image` (77-121) and a newline per
    /// row (216-243). `None` where its asserts fail.
    fn ref_genai_llava(w: u32, h: u32) -> Option<usize> {
        let pins: [(i32, i32); 5] = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];
        let (ow, oh) = (w as i32, h as i32);
        let (mut best, mut max_eff, mut min_waste) = ((0, 0), 0, i32::MAX);
        for (pw, ph) in pins {
            let scale = (pw as f32 / ow as f32).min(ph as f32 / oh as f32);
            let eff = ((ow as f32 * scale) as i32 * (oh as f32 * scale) as i32).min(ow * oh);
            let waste = pw * ph - eff;
            if eff > max_eff || (eff == max_eff && waste < min_waste) {
                max_eff = eff;
                min_waste = waste;
                best = (pw, ph);
            }
        }
        let (cur_h, cur_w) = (best.1 as usize / 336 * 24, best.0 as usize / 336 * 24);
        let (rows, cols) = if (w as f32 / h as f32) > (cur_w as f32 / cur_h as f32) {
            let new_h = (h as f32 * (cur_w as f32 / w as f32)) as usize;
            let pad = cur_h.checked_sub(new_h)? / 2;
            (cur_h > pad * 2).then_some((cur_h - pad * 2, cur_w))?
        } else {
            let new_w = (w as f32 * (cur_h as f32 / h as f32)) as usize;
            let pad = cur_w.checked_sub(new_w)? / 2;
            (cur_w > pad * 2).then_some((cur_h, cur_w - pad * 2))?
        };
        Some(576 + rows * (cols + 1))
    }

    /// LLaVA-NeXT: never more than HF's count (base view plus unpadded tile
    /// rows and their newlines); 2928 of GenAI's 2956 for the SDK image.
    #[test]
    fn llava_next_counts_are_at_most_hf() {
        let (r, dir) = llava_export();
        let s = sizer(&dir);
        for (name, img) in r["images"].as_object().unwrap() {
            let (w, h) = size(img);
            let hf = img["tokens"].as_u64().unwrap() as usize;
            match s.soft_tokens(&[image(w, h)]) {
                Ok(got) => assert!(got <= hf, "{name}: {got} > HF {hf}"),
                // The 1500x4 strip and 3x1200 pole unpad to nothing in
                // GenAI's f32 (its `unpad_image` assert fails): left to GenAI.
                Err(_) => assert_eq!(ref_genai_llava(w, h), None, "{name}"),
            }
        }
        assert_eq!(s.soft_tokens(&[image(448, 448)]).unwrap(), 2928);
    }

    /// The reviewer's counterexamples: an image no side over 336 ties every
    /// pinpoint and keeps the first, [336, 672], which HF reads as tall and
    /// GenAI as wide (`clip.cpp:341-342`), so a landscape image keeps more
    /// rows in HF than in GenAI. GenAI's counts (`ref_genai_llava`, and the
    /// reviewer's Python mirror): 40x30 1026, 300x200 976, 336x168 876.
    #[test]
    fn llava_next_counts_genai_pinpoints() {
        let (_, dir) = llava_export();
        let s = sizer(&dir);
        for (w, h, genai) in [(40, 30, 1026), (300, 200, 976), (336, 168, 876)] {
            assert_eq!(ref_genai_llava(w, h), Some(genai), "{w}x{h}");
            assert_eq!(s.soft_tokens(&[image(w, h)]).unwrap(), genai, "{w}x{h}");
        }
        // Portrait: HF's count is the smaller (30x40: HF 1032, GenAI 1376).
        assert_eq!(s.soft_tokens(&[image(30, 40)]).unwrap(), 1032);
    }

    #[test]
    fn llava_next_sweep_never_exceeds_genai() {
        let (_, dir) = llava_export();
        let s = sizer(&dir);
        for (w, h) in sweep() {
            let Some(g) = ref_genai_llava(w, h) else {
                continue;
            };
            match s.image_tokens(w, h) {
                Ok(got) => assert!(got <= g, "{w}x{h}: {got} > GenAI {g}"),
                Err(e) => panic!("{w}x{h}: GenAI sizes it ({g}) but the sizer does not: {e}"),
            }
        }
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

    /// A literal transcription of GenAI 2026.4.1's Gemma 4 sizing,
    /// `get_aspect_ratio_preserving_size` (`gemma4/classes.cpp:26-59`), and the
    /// soft tokens its patches pool to (233-234, the pooled patches GenAI keeps
    /// valid). `None` where it asserts.
    fn ref_genai_gemma4(
        w: u32,
        h: u32,
        patch: usize,
        max_soft: usize,
        pool: usize,
    ) -> Option<usize> {
        let (height, width) = (h as usize, w as usize);
        let max_patches = max_soft * pool * pool;
        let total_px = (height * width) as f64;
        let target_px = max_patches as f64 * (patch * patch) as f64;
        let factor = (target_px / total_px).sqrt();
        let side = pool * patch;
        let mut th = (factor * height as f64 / side as f64).floor() as usize * side;
        let mut tw = (factor * width as f64 / side as f64).floor() as usize * side;
        if th == 0 && tw == 0 {
            return None;
        }
        let max_side = (max_patches / (pool * pool)) * side;
        if th == 0 {
            th = side;
            tw = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
        } else if tw == 0 {
            tw = side;
            th = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
        }
        Some((th / patch) * (tw / patch) / (pool * pool))
    }

    #[test]
    fn gemma4_sweep_never_exceeds_genai() {
        let r = fixture("gemma4_vision/reference.json");
        let dir = export(&[
            ("config.json", json!({ "model_type": "gemma4" })),
            ("preprocessor_config.json", r["processor"].clone()),
        ]);
        let s = sizer(&dir);
        for (w, h) in sweep() {
            let Some(g) = ref_genai_gemma4(w, h, 16, 280, 3) else {
                continue;
            };
            if let Ok(got) = s.image_tokens(w, h) {
                assert!(got <= g, "{w}x{h}: {got} > GenAI {g}");
            }
        }
    }

    /// Gemma 3: GenAI resizes every image to the processor's one view
    /// (`size.height` x `size.width`, `processor_config.cpp:55-56`) and the IR
    /// returns `image_seq_length` rows for it, whatever the image's size.
    #[test]
    fn gemma3_sweep_is_constant() {
        let s = MediaSizer::Gemma3 { per_image: 256 };
        for (w, h) in sweep().step_by(101) {
            assert_eq!(s.image_tokens(w, h).unwrap(), 256, "{w}x{h}");
        }
    }
}
