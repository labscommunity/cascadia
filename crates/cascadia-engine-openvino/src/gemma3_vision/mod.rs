//! Host-side Gemma 3 image input (`Gemma3ForConditionalGeneration`, e.g.
//! OpenVINO/gemma-3-12b-it-int4-ov): the work Hugging Face's
//! `Gemma3ImageProcessor` and `Gemma3Processor` do before the vision IR runs,
//! for rank 0 of a staged `ov-runtime` tree.
//!
//! - [`preprocess`]: SigLIP's fixed-size resize (896x896, no aspect ratio
//!   kept, Pillow-exact bilinear from the shared resampler), rescale 1/255 in
//!   f64 cast to f32, normalise with mean/std 0.5 in f32, channels-first.
//!   Pan-and-scan (off by default in the processor) is refused at load.
//! - [`expand_prompt_text`]: `Gemma3Processor` replaces each
//!   `<start_of_image>` of the rendered template with `"\n\n" + boi +
//!   image_seq_length x <image_soft_token> + eoi + "\n\n"` on the TEXT, then
//!   tokenizes. The newlines merge with neighbouring text (`"user\n"` +
//!   `"\n\n"` is one token), so the expansion cannot be done on ids; rank 0
//!   expands the prompt text before tokenizing it, and [`check_expanded`]
//!   then verifies the ids.
//! - [`Frontend`]: the vision IR (`pixel_values [1, 3, 896, 896]` to
//!   `[1, 256, hidden]`: SigLIP, the 4x4 average pool, the soft-embedding norm
//!   and the projection are all inside the IR) plus the above.
//!
//! The language model takes `token_type_ids` (1 on soft tokens) and opens
//! each image both ways in its full and sliding masks
//! (`create_masks_for_vision_model` in transformers' modeling_gemma3.py), the
//! Gemma 4 machinery: the third wire frame, `image_safe_take`.
//!
//! Parity with transformers 5.14.1 is pinned by `tests/gemma3_vision_parity.rs`
//! against `tests/fixtures/gemma3_vision/`.

use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::resample::{self, Filter};

/// The image processor settings (`preprocessor_config.json`) and the
/// processor's `image_seq_length`.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    pub height: u32,
    pub width: u32,
    pub filter: Filter,
    pub rescale_factor: f64,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Soft tokens per image (`image_seq_length`, the config's
    /// `mm_tokens_per_image`).
    pub image_seq_length: u32,
}

impl PreprocessConfig {
    /// Gemma 3 (4B through 27B) as shipped.
    pub fn gemma3() -> Self {
        Self {
            height: 896,
            width: 896,
            filter: Filter::Bilinear,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            image_seq_length: 256,
        }
    }

    /// `preprocessor_config.json` of an export directory, with
    /// `processor_config.json`'s `image_seq_length` when it has one. Missing
    /// keys keep the Gemma 3 values; settings this port does not reproduce
    /// (pan-and-scan, another resampling filter) are refused.
    pub fn from_model_dir(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<Option<serde_json::Value>, String> {
            let path = dir.join(name);
            if !path.exists() {
                return Ok(None);
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            serde_json::from_str(&text)
                .map(Some)
                .map_err(|e| format!("parsing {}: {e}", path.display()))
        };
        let pre = read("preprocessor_config.json")?
            .ok_or_else(|| format!("{} has no preprocessor_config.json", dir.display()))?;
        let mut c = Self::from_json(&pre)?;
        if let Some(n) = read("processor_config.json")?.and_then(|v| {
            v.get("image_seq_length")
                .and_then(serde_json::Value::as_u64)
        }) {
            c.image_seq_length = n as u32;
        }
        if c.image_seq_length == 0 {
            return Err("image_seq_length is 0".into());
        }
        Ok(c)
    }

    fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let mut c = Self::gemma3();
        if let Some(s) = v.get("size") {
            let side = |k: &str| s.get(k).and_then(serde_json::Value::as_u64);
            if let (Some(h), Some(w)) = (side("height"), side("width")) {
                c.height = h as u32;
                c.width = w as u32;
            }
        }
        if let Some(r) = v.get("resample").and_then(serde_json::Value::as_u64) {
            c.filter = Filter::from_pil(r)
                .ok_or_else(|| format!("resample {r} is not bilinear (2) or bicubic (3)"))?;
        }
        if let Some(r) = v.get("rescale_factor").and_then(serde_json::Value::as_f64) {
            c.rescale_factor = r;
        }
        if let Some(b) = v.get("do_normalize").and_then(serde_json::Value::as_bool) {
            c.do_normalize = b;
        }
        let triple = |k: &str| -> Option<[f32; 3]> {
            let a = v.get(k)?.as_array()?;
            (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(f64::NAN) as f32))
        };
        if let Some(m) = triple("image_mean") {
            c.image_mean = m;
        }
        if let Some(s) = triple("image_std") {
            c.image_std = s;
        }
        if let Some(n) = v
            .get("image_seq_length")
            .and_then(serde_json::Value::as_u64)
        {
            c.image_seq_length = n as u32;
        }
        if v.get("do_pan_and_scan")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            return Err("do_pan_and_scan is on; only the single 896x896 view is served".into());
        }
        if c.height == 0 || c.width == 0 {
            return Err("zero image size".into());
        }
        Ok(c)
    }
}

/// The vision IR's input for one image: channels-first `[3, height, width]`.
#[derive(Clone, Debug)]
pub struct Pixels {
    pub pixel_values: Vec<f32>,
    pub height: u32,
    pub width: u32,
}

/// Preprocess one packed RGB8 image exactly as transformers' PIL Gemma 3
/// processor: resize to the fixed size, rescale (f64 multiply cast to f32),
/// normalise in f32.
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Pixels, String> {
    if width == 0 || height == 0 {
        return Err(format!("empty image {width}x{height}"));
    }
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(format!("image is not {width}x{height} RGB8"));
    }
    let (th, tw) = (cfg.height as usize, cfg.width as usize);
    let resized = resample::resize_rgb8(rgb, width as usize, height as usize, tw, th, cfg.filter);
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|v| {
            let x = (v as f64 * cfg.rescale_factor) as f32;
            if cfg.do_normalize {
                (x - cfg.image_mean[c]) / cfg.image_std[c]
            } else {
                x
            }
        })
    });
    let plane = th * tw;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in resized.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = lut[c][px[c] as usize];
        }
    }
    Ok(Pixels {
        pixel_values: out,
        height: cfg.height,
        width: cfg.width,
    })
}

/// The framing tokens, as ids and as the text the template and the
/// processor write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageTokens {
    /// `<image_soft_token>`: one per soft token after expansion.
    pub image: i64,
    /// `<start_of_image>`: the template's one-per-image placeholder.
    pub boi: i64,
    /// `<end_of_image>`.
    pub eoi: i64,
    pub image_text: String,
    pub boi_text: String,
    pub eoi_text: String,
}

/// `Gemma3Processor.full_image_sequence`.
pub fn full_image_sequence(t: &ImageTokens, seq_len: usize) -> String {
    format!(
        "\n\n{}{}{}\n\n",
        t.boi_text,
        t.image_text.repeat(seq_len),
        t.eoi_text
    )
}

/// `Gemma3Processor.__call__`'s expansion on the rendered prompt: every
/// `<start_of_image>` becomes [`full_image_sequence`].
pub fn expand_prompt_text(prompt: &str, t: &ImageTokens, seq_len: usize) -> String {
    prompt.replace(&t.boi_text, &full_image_sequence(t, seq_len))
}

/// Check that tokenized expanded `ids` carry `images` images, each `boi`,
/// exactly `seq_len` soft tokens, `eoi`, and no soft token outside one.
pub fn check_expanded(
    ids: &[i64],
    t: &ImageTokens,
    images: usize,
    seq_len: usize,
) -> Result<(), String> {
    let mut found = 0;
    let mut i = 0;
    while i < ids.len() {
        let x = ids[i];
        if x == t.boi {
            let run = ids[i + 1..].iter().take_while(|&&y| y == t.image).count();
            if run != seq_len || ids.get(i + 1 + run) != Some(&t.eoi) {
                return Err(format!(
                    "image {found}: {run} soft tokens between its begin and end markers, \
                     expected {seq_len}"
                ));
            }
            found += 1;
            i += run + 2;
            continue;
        }
        if x == t.image || x == t.eoi {
            return Err("an image soft token or end marker outside an image".into());
        }
        i += 1;
    }
    if found != images {
        return Err(format!(
            "prompt has {found} image placeholders for {images} images"
        ));
    }
    Ok(())
}

/// Rank 0's Gemma 3 image front end: the vision IR, its processor settings
/// and the framing tokens.
pub struct Frontend {
    rt: Runtime,
    /// Encoded images by content hash, so a chat's earlier images are not
    /// re-encoded every turn. Every image has the same token count, so no
    /// metadata.
    cache: crate::image_cache::EncodeCache<()>,
    pub preprocess: PreprocessConfig,
    pub tokens: ImageTokens,
}

impl Frontend {
    /// Compile `openvino_vision_embeddings_model.xml` from `dir` on the CPU.
    pub fn load(dir: &Path, tokens: ImageTokens) -> Result<Self, String> {
        let preprocess = PreprocessConfig::from_model_dir(dir)?;
        let path = dir.join("openvino_vision_embeddings_model.xml");
        let rt = Runtime::compile(&path.to_string_lossy(), "CPU", &PluginConfig::new())
            .map_err(|e| format!("compiling {}: {e}", path.display()))?;
        Ok(Self {
            rt,
            cache: crate::image_cache::EncodeCache::from_env(),
            preprocess,
            tokens,
        })
    }

    fn seq_len(&self) -> usize {
        self.preprocess.image_seq_length as usize
    }

    /// The rendered prompt as the tokenizer must see it (see the module doc).
    pub fn expand_text(&self, prompt: &str) -> String {
        expand_prompt_text(prompt, &self.tokens, self.seq_len())
    }

    /// `[soft_tokens, hidden]` rows for a preprocessed image, and `hidden`.
    pub fn encode(&mut self, p: &Pixels) -> Result<(Vec<f32>, usize), String> {
        let bytes: Vec<u8> = p
            .pixel_values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        self.rt
            .set_input(
                "pixel_values",
                DType::F32,
                &[1, 3, p.height as usize, p.width as usize],
                &bytes,
            )
            .and_then(|_| self.rt.infer())
            .map_err(|e| format!("vision embedding: {e}"))?;
        let (dtype, shape, bytes) = self
            .rt
            .output(0)
            .map_err(|e| format!("vision embedding: {e}"))?;
        if dtype != DType::F32 {
            return Err(format!(
                "vision embedding output is {dtype:?}, expected f32"
            ));
        }
        let (tokens, hidden) = match shape[..] {
            [1, t, h] | [t, h] => (t, h),
            _ => return Err(format!("vision embedding returned shape {shape:?}")),
        };
        if tokens != self.seq_len() {
            return Err(format!(
                "vision embedding returned {tokens} rows, expected {} per image",
                self.seq_len()
            ));
        }
        let v = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((v, hidden))
    }

    /// The prompt ids `prepare` would return (they are already expanded:
    /// see [`Frontend::expand_text`]), checked against `media` with no
    /// encoder work. Every image costs `image_seq_length` tokens whatever
    /// its size.
    pub fn expand_ids(
        &self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        for (i, m) in media.iter().enumerate() {
            crate::gemma4_vision::check_image(i, m)?;
        }
        check_expanded(prompt_ids, &self.tokens, media.len(), self.seq_len())?;
        Ok(prompt_ids.to_vec())
    }

    /// Encode a task's images for its expanded prompt ids. Returns the ids
    /// and each image's rows, in prompt order.
    pub fn prepare(
        &mut self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
        tenant: &str,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
        let ids = self.expand_ids(media, prompt_ids)?;
        let mut rows = Vec::with_capacity(media.len());
        let mut from_cache = 0;
        for (i, m) in media.iter().enumerate() {
            if let Some((emb, ())) = self.cache.get(tenant, &m.sha256) {
                rows.push(emb);
                from_cache += 1;
                continue;
            }
            let p = preprocess(&m.pixels, m.width, m.height, &self.preprocess)
                .map_err(|e| format!("image {i}: {e}"))?;
            let (emb, _) = self.encode(&p).map_err(|e| format!("image {i}: {e}"))?;
            self.cache.put(tenant, m.sha256, &emb, ());
            rows.push(emb);
        }
        self.cache.report(from_cache, media.len());
        Ok((ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks() -> ImageTokens {
        ImageTokens {
            image: 9,
            boi: 7,
            eoi: 8,
            image_text: "<S>".into(),
            boi_text: "<B>".into(),
            eoi_text: "<E>".into(),
        }
    }

    #[test]
    fn text_expansion_frames_each_image_with_blank_lines() {
        let t = toks();
        assert_eq!(full_image_sequence(&t, 2), "\n\n<B><S><S><E>\n\n");
        assert_eq!(
            expand_prompt_text("u\n<B>x<B>", &t, 1),
            "u\n\n\n<B><S><E>\n\nx\n\n<B><S><E>\n\n"
        );
        assert_eq!(expand_prompt_text("no image", &t, 3), "no image");
    }

    #[test]
    fn expanded_ids_are_checked() {
        let t = toks();
        assert!(check_expanded(&[1, 7, 9, 9, 8, 2, 7, 9, 9, 8], &t, 2, 2).is_ok());
        assert!(check_expanded(&[1, 2], &t, 0, 2).is_ok());
        // wrong count, short run, missing end, stray soft token
        assert!(check_expanded(&[1, 7, 9, 9, 8], &t, 2, 2).is_err());
        assert!(check_expanded(&[7, 9, 8], &t, 1, 2).is_err());
        assert!(check_expanded(&[7, 9, 9, 2], &t, 1, 2).is_err());
        assert!(check_expanded(&[7, 9, 9, 8, 9], &t, 1, 2).is_err());
        assert!(check_expanded(&[7, 9, 9], &t, 1, 2).is_err());
    }

    #[test]
    fn config_reads_size_and_refuses_what_is_not_ported() {
        let v = serde_json::json!({"size": {"height": 448, "width": 448}, "resample": 3,
            "image_mean": [0.1, 0.2, 0.3], "image_seq_length": 64});
        let c = PreprocessConfig::from_json(&v).unwrap();
        assert_eq!((c.height, c.width, c.filter), (448, 448, Filter::Bicubic));
        assert_eq!((c.image_mean, c.image_seq_length), ([0.1, 0.2, 0.3], 64));
        assert!(PreprocessConfig::from_json(&serde_json::json!({"resample": 1})).is_err());
        assert!(
            PreprocessConfig::from_json(&serde_json::json!({"do_pan_and_scan": true})).is_err()
        );
        assert!(PreprocessConfig::from_json(&serde_json::json!({"do_pan_and_scan": null})).is_ok());
        assert_eq!(
            PreprocessConfig::from_json(&serde_json::json!({})).unwrap(),
            PreprocessConfig::gemma3()
        );
    }

    #[test]
    fn preprocess_is_fixed_size_channels_first_and_normalised() {
        let c = PreprocessConfig {
            height: 4,
            width: 6,
            ..PreprocessConfig::gemma3()
        };
        let p = preprocess(&[255u8, 0, 51].repeat(10 * 3), 10, 3, &c).unwrap();
        assert_eq!((p.height, p.width, p.pixel_values.len()), (4, 6, 3 * 24));
        assert!(p.pixel_values[..24].iter().all(|&v| v == 1.0));
        assert!(p.pixel_values[24..48].iter().all(|&v| v == -1.0));
        let b = ((51.0f64 / 255.0) as f32 - 0.5) / 0.5;
        assert!(p.pixel_values[48..].iter().all(|&v| v == b));
        assert!(preprocess(&[0u8; 5], 1, 1, &c).is_err());
    }
}
