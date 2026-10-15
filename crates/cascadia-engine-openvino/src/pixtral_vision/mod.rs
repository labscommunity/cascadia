//! Host-side Pixtral image input (LLaVA-layout exports such as
//! OpenVINO/pixtral-12b-int4-ov): the work Hugging Face's
//! `PixtralImageProcessor` and `PixtralProcessor` do before the vision IR runs,
//! for rank 0 of a staged `ov-runtime` tree.
//!
//! - [`target_size`]: shrink so neither side exceeds `longest_edge` (floor),
//!   then round each side UP to a whole number of patches.
//! - [`preprocess`]: Pillow-exact bicubic resize (shared with Qwen3.5),
//!   rescale, normalise with the CLIP mean/std, channels-first `[3, H, W]`.
//! - [`expand_image_tokens`]: each `[IMG]` of the template becomes the image's
//!   rows of `[IMG] * cols` closed by `[IMG_BREAK]`, the last by `[IMG_END]`.
//! - [`Frontend`]: the vision IR (`pixel_values [1, 3, H, W]` to one row per
//!   patch) plus the above, for a task's images.
//!
//! Parity with transformers 5.14.1 is pinned by `tests/pixtral_vision_parity.rs`
//! against `tests/fixtures/pixtral_vision/`.

use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::resample;

/// `preprocessor_config.json` of a Pixtral export.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    pub patch_size: u32,
    pub longest_edge: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f64,
}

impl PreprocessConfig {
    /// Pixtral-12B as shipped.
    pub fn pixtral() -> Self {
        Self {
            patch_size: 16,
            longest_edge: 1024,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
            rescale_factor: 1.0 / 255.0,
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        let mut c = Self::pixtral();
        if let Some(p) = v
            .get("patch_size")
            .and_then(|p| p.get("height").or(Some(p)))
            .and_then(serde_json::Value::as_u64)
        {
            c.patch_size = p as u32;
        }
        if let Some(e) = v
            .get("size")
            .and_then(|s| s.get("longest_edge"))
            .and_then(serde_json::Value::as_u64)
        {
            c.longest_edge = e as u32;
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
        if let Some(r) = v.get("rescale_factor").and_then(serde_json::Value::as_f64) {
            c.rescale_factor = r;
        }
        if c.patch_size == 0 || c.longest_edge == 0 {
            return Err(format!("{}: zero patch size or edge", path.display()));
        }
        Ok(c)
    }
}

/// transformers' Pixtral `get_resize_output_image_size`: `(height, width)`.
pub fn target_size(height: u32, width: u32, cfg: &PreprocessConfig) -> Result<(u32, u32), String> {
    if height == 0 || width == 0 {
        return Err(format!("empty image {width}x{height}"));
    }
    let (mut h, mut w) = (f64::from(height), f64::from(width));
    let le = f64::from(cfg.longest_edge);
    let ratio = (h / le).max(w / le);
    if ratio > 1.0 {
        h = (h / ratio).floor();
        w = (w / ratio).floor();
    }
    let p = cfg.patch_size;
    // `_num_image_tokens`: (side - 1) // patch + 1, i.e. a ceiling.
    let tokens = |side: f64| ((side as u32).max(1) - 1) / p + 1;
    Ok((tokens(h) * p, tokens(w) * p))
}

/// The vision IR's input for one image.
#[derive(Clone, Debug)]
pub struct Pixels {
    /// Channels-first `[3, height, width]` f32.
    pub pixel_values: Vec<f32>,
    pub height: u32,
    pub width: u32,
    /// Patch grid: rows, columns.
    pub rows: u32,
    pub cols: u32,
}

impl Pixels {
    /// Soft tokens the image occupies: one per patch.
    pub fn patches(&self) -> usize {
        (self.rows * self.cols) as usize
    }
}

/// Preprocess one packed RGB8 image exactly as transformers' PIL Pixtral
/// processor (resize, rescale in f64 cast to f32, normalise in f32).
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Pixels, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(format!("image is not {width}x{height} RGB8"));
    }
    let (th, tw) = target_size(height, width, cfg)?;
    let resized = resample::resize_rgb8_bicubic(
        rgb,
        width as usize,
        height as usize,
        tw as usize,
        th as usize,
    );
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|v| {
            let x = (v as f64 * cfg.rescale_factor) as f32;
            (x - cfg.image_mean[c]) / cfg.image_std[c]
        })
    });
    let plane = th as usize * tw as usize;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in resized.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = lut[c][px[c] as usize];
        }
    }
    Ok(Pixels {
        pixel_values: out,
        height: th,
        width: tw,
        rows: th / cfg.patch_size,
        cols: tw / cfg.patch_size,
    })
}

/// Token ids that frame a Pixtral image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTokens {
    pub image: i64,
    pub image_break: i64,
    pub image_end: i64,
}

/// `PixtralProcessor`'s expansion on token ids: each `[IMG]` of the tokenized
/// template becomes, for its image's `(rows, cols)`, `rows` runs of
/// `[IMG] * cols` each closed by `[IMG_BREAK]`, the last by `[IMG_END]`.
pub fn expand_image_tokens(
    ids: &[i64],
    t: &ImageTokens,
    grids: &[(u32, u32)],
) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&x| x == t.image).count();
    if found != grids.len() {
        return Err(format!(
            "prompt has {found} image placeholders for {} images",
            grids.len()
        ));
    }
    if ids.iter().any(|&x| x == t.image_break || x == t.image_end) {
        return Err("prompt already contains image break/end tokens".into());
    }
    let extra: usize = grids.iter().map(|&(r, c)| (r * (c + 1)) as usize).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut images = grids.iter();
    for &x in ids {
        if x != t.image {
            out.push(x);
            continue;
        }
        let &(rows, cols) = images.next().expect("counted above");
        for r in 0..rows {
            out.extend(std::iter::repeat_n(t.image, cols as usize));
            out.push(if r + 1 == rows {
                t.image_end
            } else {
                t.image_break
            });
        }
    }
    Ok(out)
}

/// Rank 0's Pixtral image front end: the vision IR, its processor settings
/// and the framing token ids.
pub struct Frontend {
    rt: Runtime,
    pub preprocess: PreprocessConfig,
    pub tokens: ImageTokens,
}

impl Frontend {
    /// Compile `openvino_vision_embeddings_model.xml` from `dir` on the CPU.
    pub fn load(dir: &Path, tokens: ImageTokens) -> Result<Self, String> {
        let preprocess = PreprocessConfig::from_json_file(&dir.join("preprocessor_config.json"))?;
        let path = dir.join("openvino_vision_embeddings_model.xml");
        let rt = Runtime::compile(&path.to_string_lossy(), "CPU", &PluginConfig::new())
            .map_err(|e| format!("compiling {}: {e}", path.display()))?;
        Ok(Self {
            rt,
            preprocess,
            tokens,
        })
    }

    /// One row per patch, row-major, for a preprocessed image.
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
        if tokens != p.patches() {
            return Err(format!(
                "vision embedding returned {tokens} rows for {} patches",
                p.patches()
            ));
        }
        let v = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((v, hidden))
    }

    /// Encode a task's images and expand each `[IMG]` into its rows. Returns
    /// the expanded ids and each image's rows, in prompt order.
    pub fn prepare(
        &mut self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
        let mut rows = Vec::with_capacity(media.len());
        let mut grids = Vec::with_capacity(media.len());
        for (i, m) in media.iter().enumerate() {
            match m.kind {
                cascadia_types::MediaKind::Image => {}
                #[allow(unreachable_patterns)]
                other => return Err(format!("media {i}: {other:?} is not supported")),
            }
            if m.num_frames != 1 || m.pixels.len() != m.expected_len() {
                return Err(format!("image {i}: malformed pixel buffer"));
            }
            let p = preprocess(&m.pixels, m.width, m.height, &self.preprocess)
                .map_err(|e| format!("image {i}: {e}"))?;
            let (emb, _) = self.encode(&p).map_err(|e| format!("image {i}: {e}"))?;
            grids.push((p.rows, p.cols));
            rows.push(emb);
        }
        let ids = expand_image_tokens(prompt_ids, &self.tokens, &grids)?;
        Ok((ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_shrinks_then_rounds_up_to_patches() {
        let c = PreprocessConfig::pixtral();
        // Under the edge: only rounded up to 16.
        assert_eq!(target_size(448, 448, &c), Ok((448, 448)));
        assert_eq!(target_size(30, 40, &c), Ok((32, 48)));
        // 2048x1000 (w x h): ratio 2 -> 1024 x 500 -> 1024 x 512.
        assert_eq!(target_size(1000, 2048, &c), Ok((512, 1024)));
        // floor, then ceil to the patch: 1600x1200 -> ratio 1.5625 -> 1024 x 768.
        assert_eq!(target_size(1200, 1600, &c), Ok((768, 1024)));
    }

    #[test]
    fn expansion_closes_rows_with_break_and_the_image_with_end() {
        let t = ImageTokens {
            image: 10,
            image_break: 12,
            image_end: 13,
        };
        let e = expand_image_tokens(&[1, 10, 2], &t, &[(2, 3)]).unwrap();
        assert_eq!(e, vec![1, 10, 10, 10, 12, 10, 10, 10, 13, 2]);
        assert!(expand_image_tokens(&[10, 10], &t, &[(1, 1)]).is_err());
        assert!(expand_image_tokens(&[10, 12], &t, &[(1, 1)]).is_err());
    }

    #[test]
    fn preprocess_is_channels_first_and_normalised() {
        let c = PreprocessConfig::pixtral();
        let rgb = [255u8, 0, 0].repeat(16 * 32);
        let p = preprocess(&rgb, 32, 16, &c).unwrap();
        assert_eq!((p.height, p.width, p.rows, p.cols), (16, 32, 1, 2));
        let plane = 16 * 32;
        let red = (1.0f32 - c.image_mean[0]) / c.image_std[0];
        let green = (0.0f32 - c.image_mean[1]) / c.image_std[1];
        assert!(p.pixel_values[..plane].iter().all(|&v| v == red));
        assert!(p.pixel_values[plane..2 * plane].iter().all(|&v| v == green));
    }
}
