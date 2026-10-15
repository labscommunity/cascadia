//! Host-side Qwen3.5 vision input: the work Hugging Face's Qwen2-VL image
//! processor and `get_rope_index` do before the vision IRs and the language
//! model run. Rank 0 of the staged `qwen35` engine needs it because it calls
//! OpenVINO directly rather than through GenAI's `VLMPipeline`.
//!
//! - [`preprocess`]: `smart_resize` to a multiple of `patch * merge` within the
//!   pixel limits, Pillow-exact bicubic resize ([`resample`]), rescale and
//!   normalise exactly as the PIL backend does, pad frames to the temporal
//!   patch, and flatten into the `[patches, C * T_p * P * P]` rows the vision
//!   embeddings IR takes, with the grid.
//! - [`rope`]: the language model's M-RoPE positions for a prompt holding
//!   image/video tokens, and the `rope_delta` decode continues from.
//!
//! Inputs are frame stacks so video needs no new entry point: an image is one
//! frame. Parity with transformers 5.14.1 is pinned by
//! `tests/qwen35_vision_parity.rs` against `tests/fixtures/qwen35_vision/`.

pub mod encoder;
pub mod prompt;
pub mod resample;
pub mod rope;

use std::path::Path;

/// The image processor settings (`preprocessor_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub merge_size: u32,
    /// `size.shortest_edge`: the minimum pixel count after resize.
    pub min_pixels: u64,
    /// `size.longest_edge`: the maximum pixel count after resize.
    pub max_pixels: u64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f64,
}

impl PreprocessConfig {
    /// Qwen3.5/3.6 as shipped (35B-A3B and 397B-A17B configs are identical).
    pub fn qwen35() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// Read `preprocessor_config.json`. Missing keys keep the Qwen3.5 values;
    /// the legacy `min_pixels`/`max_pixels` keys are honoured when `size` is
    /// absent, as transformers does.
    pub fn from_json_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        let mut c = Self::qwen35();
        let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        if let Some(p) = u("patch_size") {
            c.patch_size = p as u32;
        }
        if let Some(p) = u("temporal_patch_size") {
            c.temporal_patch_size = p as u32;
        }
        if let Some(p) = u("merge_size") {
            c.merge_size = p as u32;
        }
        let size = v.get("size");
        if let Some(n) = size
            .and_then(|s| s.get("shortest_edge"))
            .and_then(|x| x.as_u64())
            .or(u("min_pixels"))
        {
            c.min_pixels = n;
        }
        if let Some(n) = size
            .and_then(|s| s.get("longest_edge"))
            .and_then(|x| x.as_u64())
            .or(u("max_pixels"))
        {
            c.max_pixels = n;
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
        Ok(c)
    }

    /// Side length every resized dimension is a multiple of.
    pub fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }
}

/// transformers' `smart_resize`: the closest `(height, width)` that are
/// multiples of `factor` and hold between `min_pixels` and `max_pixels`
/// pixels, keeping the aspect ratio. Python's `round` is round-half-to-even,
/// so is this.
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32), String> {
    if height == 0 || width == 0 {
        return Err("image has no pixels".into());
    }
    let (h, w, f) = (f64::from(height), f64::from(width), f64::from(factor));
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// The vision embeddings IR's input for one media item.
#[derive(Clone, Debug)]
pub struct Patches {
    /// Row-major `[rows, cols]` f32. Rows run over (temporal group, block
    /// row, block column, row and column inside the merge block); each row is
    /// (channel, temporal offset, pixel row, pixel column).
    pub pixel_values: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    /// `[t, h, w]` in patches, as `image_grid_thw` / `video_grid_thw`.
    pub grid_thw: [u32; 3],
}

impl Patches {
    /// Tokens this item occupies in the language model's prompt.
    pub fn llm_tokens(&self, merge_size: u32) -> usize {
        let [t, h, w] = self.grid_thw;
        (t * (h / merge_size) * (w / merge_size)) as usize
    }
}

/// Preprocess a frame stack of packed RGB8 frames, all `width x height` (an
/// image is one frame), exactly as transformers' PIL Qwen2-VL processor.
pub fn preprocess(
    frames: &[&[u8]],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Patches, String> {
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let frame_len = width as usize * height as usize * 3;
    if let Some(i) = frames.iter().position(|f| f.len() != frame_len) {
        return Err(format!("frame {i} is not {width}x{height} RGB8"));
    }
    let (rh, rw) = smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    let resized: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| {
            resample::resize_rgb8_bicubic(
                f,
                width as usize,
                height as usize,
                rw as usize,
                rh as usize,
            )
        })
        .collect();

    // rescale (f64 multiply, cast to f32) then normalise in f32, per channel,
    // as a lookup over the 256 possible inputs.
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|v| {
            let x = (v as f64 * cfg.rescale_factor) as f32;
            (x - cfg.image_mean[c]) / cfg.image_std[c]
        })
    });

    let (p, m, tp) = (
        cfg.patch_size as usize,
        cfg.merge_size as usize,
        cfg.temporal_patch_size as usize,
    );
    // Pad with copies of the last frame up to a multiple of the temporal patch.
    let n_frames = resized.len().div_ceil(tp) * tp;
    let frame = |i: usize| &resized[i.min(resized.len() - 1)];
    let (gt, gh, gw) = (n_frames / tp, rh as usize / p, rw as usize / p);
    let rows = gt * gh * gw;
    let cols = 3 * tp * p * p;
    let mut out = Vec::with_capacity(rows * cols);
    let row_w = rw as usize;
    for t in 0..gt {
        for hb in 0..gh / m {
            for wb in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        for c in 0..3 {
                            for ti in 0..tp {
                                let f = frame(t * tp + ti);
                                for ph in 0..p {
                                    let y = (hb * m + mh) * p + ph;
                                    let base = y * row_w;
                                    for pw in 0..p {
                                        let x = (wb * m + mw) * p + pw;
                                        out.push(lut[c][f[(base + x) * 3 + c] as usize]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), rows * cols);
    Ok(Patches {
        pixel_values: out,
        rows,
        cols,
        grid_thw: [gt as u32, gh as u32, gw as u32],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_hand_computed_cases() {
        let f = 32;
        // Already a multiple: unchanged.
        assert_eq!(
            smart_resize(448, 448, f, 65_536, 16_777_216),
            Ok((448, 448))
        );
        // 360/32 = 11.25 rounds to 11.
        assert_eq!(
            smart_resize(360, 640, f, 65_536, 16_777_216),
            Ok((352, 640))
        );
        // Too small: scaled up to min_pixels with ceil.
        assert_eq!(smart_resize(30, 40, f, 65_536, 16_777_216), Ok((224, 320)));
        // Python's round is half-to-even: 80/32 = 2.5 -> 2, 112/32 = 3.5 -> 4.
        assert_eq!(smart_resize(80, 112, f, 0, u64::MAX), Ok((64, 128)));
        // Too large: scaled down with floor.
        let (h, w) = smart_resize(4000, 3000, f, 65_536, 1 << 20).unwrap();
        assert!(u64::from(h) * u64::from(w) <= 1 << 20 && h % 32 == 0 && w % 32 == 0);
        assert!(smart_resize(10, 2010, f, 0, u64::MAX).is_err());
    }

    #[test]
    fn single_image_is_padded_to_the_temporal_patch() {
        let cfg = PreprocessConfig::qwen35();
        let img = vec![255u8; 256 * 256 * 3];
        let p = preprocess(&[&img], 256, 256, &cfg).unwrap();
        assert_eq!(p.grid_thw, [1, 16, 16]);
        assert_eq!((p.rows, p.cols), (256, 3 * 2 * 16 * 16));
        assert!(p.pixel_values.iter().all(|&v| v == 1.0));
        assert_eq!(p.llm_tokens(cfg.merge_size), 64);
    }

    #[test]
    fn reads_preprocessor_config() {
        let dir = std::env::temp_dir().join(format!("q35pp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("preprocessor_config.json");
        std::fs::write(
            &path,
            r#"{"size":{"longest_edge":1000000,"shortest_edge":4096},"patch_size":14,
               "image_mean":[0.1,0.2,0.3],"image_std":[1,1,1]}"#,
        )
        .unwrap();
        let c = PreprocessConfig::from_json_file(&path).unwrap();
        assert_eq!(
            (c.patch_size, c.min_pixels, c.max_pixels),
            (14, 4096, 1_000_000)
        );
        assert_eq!(c.image_mean, [0.1, 0.2, 0.3]);
        assert_eq!(c.merge_size, 2);
        std::fs::remove_dir_all(dir).ok();
    }
}
