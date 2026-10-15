//! Host-side Gemma 4 image input: the work Hugging Face's `Gemma4ImageProcessor`
//! does before the vision IR runs. Rank 0 of a staged Gemma 4 vision tree
//! needs it because it calls OpenVINO directly rather than through GenAI's
//! `VLMPipeline`.
//!
//! - [`target_size`]: the largest aspect-preserving size whose sides are
//!   multiples of `pooling * patch` and whose patch count fits the budget
//!   (`get_aspect_ratio_preserving_size`, edge cases included).
//! - [`preprocess`]: Pillow-exact bicubic resize (shared with Qwen3.5,
//!   [`crate::qwen35_vision::resample`]), rescale to `[0, 1]` (Gemma 4 does
//!   not normalise), patchify into `(row, col)` rows of `(ph, pw, c)` values,
//!   and pad to `max_patches` rows with zero pixels and position `-1`.
//! - [`encoder`]: the single vision IR, `pixel_values` + `image_position_ids`
//!   to one embedding row per soft token.
//!
//! Parity with transformers 5.14.1 is pinned by `tests/gemma4_vision_parity.rs`
//! against `tests/fixtures/gemma4_vision/`.

pub mod encoder;

use std::path::Path;

use crate::qwen35_vision::resample;

/// Soft-token budgets the processor accepts (`_SUPPORTED_SOFT_TOKENS`).
pub const SUPPORTED_SOFT_TOKENS: [u32; 5] = [70, 140, 280, 560, 1120];

/// The image processor settings (`processor_config.json` `image_processor`,
/// or a standalone `preprocessor_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    pub patch_size: u32,
    pub max_soft_tokens: u32,
    pub pooling_kernel_size: u32,
    pub rescale_factor: f64,
    /// Gemma 4 ships `do_normalize: false` (identity mean 0 / std 1).
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl PreprocessConfig {
    /// Gemma 4 as shipped (E2B through 31B share it).
    pub fn gemma4() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            rescale_factor: 1.0 / 255.0,
            do_normalize: false,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }

    /// Read the processor settings from an export directory: the
    /// `image_processor` object of `processor_config.json` when present,
    /// else `preprocessor_config.json`. Missing keys keep the Gemma 4 values.
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
        let v = match read("processor_config.json")?.and_then(|v| v.get("image_processor").cloned())
        {
            Some(v) => v,
            None => read("preprocessor_config.json")?.ok_or_else(|| {
                format!(
                    "{} has neither processor_config.json nor preprocessor_config.json",
                    dir.display()
                )
            })?,
        };
        Self::from_json(&v)
    }

    fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let mut c = Self::gemma4();
        let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        if let Some(p) = u("patch_size") {
            c.patch_size = p as u32;
        }
        if let Some(p) = u("max_soft_tokens") {
            c.max_soft_tokens = p as u32;
        }
        if let Some(p) = u("pooling_kernel_size") {
            c.pooling_kernel_size = p as u32;
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
        if !SUPPORTED_SOFT_TOKENS.contains(&c.max_soft_tokens) {
            return Err(format!(
                "max_soft_tokens {} is not one of {SUPPORTED_SOFT_TOKENS:?}",
                c.max_soft_tokens
            ));
        }
        if c.patch_size == 0 || c.pooling_kernel_size == 0 {
            return Err("patch_size and pooling_kernel_size must be positive".into());
        }
        Ok(c)
    }

    /// Patch rows the vision IR takes: `max_soft_tokens * pooling²`.
    pub fn max_patches(&self) -> usize {
        (self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size) as usize
    }

    /// Values per patch row: `patch² * 3`.
    pub fn patch_dim(&self) -> usize {
        (self.patch_size * self.patch_size * 3) as usize
    }
}

/// transformers' `get_aspect_ratio_preserving_size`: `(height, width)`.
///
/// Scales the image to the patch budget, floors each side to a multiple of
/// `pooling * patch`, and when one side floors to zero gives it one multiple
/// and the other `floor(long / short)` multiples, capped at the longest side
/// the budget allows.
pub fn target_size(height: u32, width: u32, cfg: &PreprocessConfig) -> Result<(u32, u32), String> {
    if height == 0 || width == 0 {
        return Err(format!("empty image {width}x{height}"));
    }
    let (h, w) = (f64::from(height), f64::from(width));
    let max_patches = cfg.max_patches() as f64;
    let target_px = max_patches * f64::from(cfg.patch_size * cfg.patch_size);
    let factor = (target_px / (h * w)).sqrt();
    let side = cfg.pooling_kernel_size * cfg.patch_size;
    let floor_to = |ideal: f64| (ideal / f64::from(side)).floor() as u32 * side;
    let (mut th, mut tw) = (floor_to(factor * h), floor_to(factor * w));
    if th == 0 && tw == 0 {
        return Err(format!(
            "{width}x{height} resizes to 0x0 at multiples of {side}"
        ));
    }
    let k2 = cfg.pooling_kernel_size * cfg.pooling_kernel_size;
    let max_side = (cfg.max_patches() as u32 / k2) * side;
    if th == 0 {
        th = side;
        tw = ((w / h).floor() as u32 * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((h / w).floor() as u32 * side).min(max_side);
    }
    if f64::from(th) * f64::from(tw) > target_px || th == 0 || tw == 0 {
        return Err(format!(
            "resizing {width}x{height} to {tw}x{th} exceeds {} patches",
            cfg.max_patches()
        ));
    }
    Ok((th, tw))
}

/// The vision IR's input for one image.
#[derive(Clone, Debug)]
pub struct Patches {
    /// Row-major `[max_patches, patch² * 3]` f32: real rows first, in
    /// `(patch row, patch column)` order, each `(pixel row, pixel column,
    /// channel)`; then zero rows up to `max_patches`.
    pub pixel_values: Vec<f32>,
    /// Row-major `[max_patches, 2]`: `(column, row)` per real patch, then
    /// `(-1, -1)` for padding.
    pub position_ids: Vec<i64>,
    pub rows: usize,
    pub cols: usize,
    /// Real (unpadded) patch rows.
    pub num_patches: usize,
    /// `(height, width)` after the resize.
    pub resized_hw: (u32, u32),
}

impl Patches {
    /// Tokens this image occupies in the language model's prompt (between
    /// its begin- and end-of-image tokens).
    pub fn soft_tokens(&self, pooling_kernel_size: u32) -> usize {
        self.num_patches / (pooling_kernel_size * pooling_kernel_size) as usize
    }
}

/// Preprocess one packed RGB8 image (`width x height`) exactly as
/// transformers' PIL Gemma 4 image processor.
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Patches, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(format!("image is not {width}x{height} RGB8"));
    }
    let (rh, rw) = target_size(height, width, cfg)?;
    let resized = resample::resize_rgb8_bicubic(
        rgb,
        width as usize,
        height as usize,
        rw as usize,
        rh as usize,
    );

    // rescale (f64 multiply, cast to f32), then normalise in f32 if asked,
    // as a lookup over the 256 possible inputs.
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

    let p = cfg.patch_size as usize;
    let (gh, gw) = (rh as usize / p, rw as usize / p);
    let num_patches = gh * gw;
    let rows = cfg.max_patches();
    let cols = cfg.patch_dim();
    if num_patches > rows {
        return Err(format!("{num_patches} patches exceed the budget of {rows}"));
    }
    let mut pixel_values = vec![0f32; rows * cols];
    let mut position_ids = vec![-1i64; rows * 2];
    let row_w = rw as usize * 3;
    for i in 0..gh {
        for j in 0..gw {
            let r = i * gw + j;
            let out = &mut pixel_values[r * cols..(r + 1) * cols];
            for ph in 0..p {
                let src = &resized[(i * p + ph) * row_w + j * p * 3..][..p * 3];
                for (k, &v) in src.iter().enumerate() {
                    out[ph * p * 3 + k] = lut[k % 3][v as usize];
                }
            }
            position_ids[r * 2] = j as i64;
            position_ids[r * 2 + 1] = i as i64;
        }
    }
    Ok(Patches {
        pixel_values,
        position_ids,
        rows,
        cols,
        num_patches,
        resized_hw: (rh, rw),
    })
}

/// Token ids that frame an image in the language model's prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTokens {
    /// `<|image|>`: the one-per-image placeholder the chat template renders,
    /// and each soft token after expansion.
    pub image: i64,
    /// `<|image>`, begin of image.
    pub boi: i64,
    /// `<image|>`, end of image.
    pub eoi: i64,
    /// What the soft-token positions are embedded as before the image rows
    /// overwrite them (HF uses the pad id).
    pub pad: i64,
}

/// `Gemma4Processor`'s placeholder expansion on token ids: each image token
/// of the tokenized template becomes `boi, image * soft_tokens[i], eoi`, in
/// order. The counts must match the placeholders one for one.
pub fn expand_image_tokens(
    ids: &[i64],
    t: &ImageTokens,
    soft_tokens: &[usize],
) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&x| x == t.image).count();
    if found != soft_tokens.len() {
        return Err(format!(
            "prompt has {found} image placeholders for {} images",
            soft_tokens.len()
        ));
    }
    if ids.iter().any(|&x| x == t.boi || x == t.eoi) {
        return Err("prompt already contains begin/end-of-image tokens".into());
    }
    let extra: usize = soft_tokens.iter().map(|n| n + 1).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut images = soft_tokens.iter();
    for &x in ids {
        if x == t.image {
            let n = *images.next().expect("counted above");
            out.push(t.boi);
            out.extend(std::iter::repeat_n(t.image, n));
            out.push(t.eoi);
        } else {
            out.push(x);
        }
    }
    Ok(out)
}

/// Ids to embed with the text embedding table: image soft tokens become the
/// pad id (their rows are overwritten by [`splice_images`]).
pub fn text_embedding_ids(expanded: &[i64], t: &ImageTokens) -> Vec<i64> {
    expanded
        .iter()
        .map(|&x| if x == t.image { t.pad } else { x })
        .collect()
}

/// Overwrite the image soft-token rows of row-major `[len, hidden]`
/// `embeds` with the images' embedding rows, in prompt order.
pub fn splice_images(
    embeds: &mut [f32],
    hidden: usize,
    expanded: &[i64],
    image_token: i64,
    images: &[&[f32]],
) -> Result<(), String> {
    if embeds.len() != expanded.len() * hidden {
        return Err(format!(
            "embeddings hold {} values, expected {} x {hidden}",
            embeds.len(),
            expanded.len()
        ));
    }
    let rows: Vec<usize> = (0..expanded.len())
        .filter(|&i| expanded[i] == image_token)
        .collect();
    let have: usize = images.iter().map(|im| im.len()).sum();
    if have != rows.len() * hidden {
        return Err(format!(
            "{} image rows in the prompt, but the images carry {} values ({} rows)",
            rows.len(),
            have,
            have / hidden.max(1)
        ));
    }
    let mut src = images.iter().flat_map(|im| im.chunks_exact(hidden));
    for r in rows {
        let row = src.next().expect("counted above");
        embeds[r * hidden..(r + 1) * hidden].copy_from_slice(row);
    }
    Ok(())
}

/// The language model's `token_type_ids` for a prompt: 1 on image soft
/// tokens, 0 elsewhere. Sliding-window layers attend bidirectionally inside
/// each run of 1s (one image), so a run must reach the model in one chunk.
pub fn token_type_ids(input_ids: &[i64], image_token_id: i64) -> Vec<i64> {
    input_ids
        .iter()
        .map(|&t| i64::from(t == image_token_id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_matches_hand_computed_cases() {
        let c = PreprocessConfig::gemma4();
        // 448² upscales: sqrt(2520*256/448²) * 448 = 803.2 -> 16 * 48.
        assert_eq!(target_size(448, 448, &c), Ok((768, 768)));
        // Already at a fitting multiple of 48 and within budget: 768² stays.
        assert_eq!(target_size(768, 768, &c), Ok((768, 768)));
        // 40x2000 is extreme but both sides survive the floor (96, 118*48).
        assert_eq!(target_size(40, 2000, &c), Ok((96, 5664)));
        // 2 tall, 1000 wide: height floors to 0 -> one multiple; width
        // floor(1000/2)=500 multiples, capped at 280*48.
        assert_eq!(target_size(2, 1000, &c), Ok((48, 13440)));
        assert_eq!(target_size(1000, 2, &c), Ok((13440, 48)));
        let (h, w) = target_size(1200, 1600, &c).unwrap();
        assert_eq!((h % 48, w % 48), (0, 0));
        assert!(u64::from(h / 16) * u64::from(w / 16) <= 2520);
    }

    #[test]
    fn config_reads_the_nested_image_processor_and_rejects_bad_budgets() {
        let v = serde_json::json!({"patch_size": 16, "max_soft_tokens": 70,
            "pooling_kernel_size": 3, "do_normalize": false});
        let c = PreprocessConfig::from_json(&v).unwrap();
        assert_eq!(
            (c.max_soft_tokens, c.max_patches(), c.patch_dim()),
            (70, 630, 768)
        );
        assert!(PreprocessConfig::from_json(&serde_json::json!({"max_soft_tokens": 100})).is_err());
    }

    #[test]
    fn patchify_layout_and_padding() {
        let c = PreprocessConfig {
            max_soft_tokens: 70,
            ..PreprocessConfig::gemma4()
        };
        let (w, h) = (96u32, 48u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let p = preprocess(&rgb, w, h, &c).unwrap();
        let (rh, rw) = p.resized_hw;
        let (gh, gw) = (rh as usize / 16, rw as usize / 16);
        assert_eq!(p.num_patches, gh * gw);
        assert_eq!(p.soft_tokens(3), gh * gw / 9);
        // Positions: (col, row) row-major, then -1 padding.
        assert_eq!(&p.position_ids[..4], &[0, 0, 1, 0]);
        let last = p.num_patches - 1;
        assert_eq!(
            &p.position_ids[last * 2..last * 2 + 2],
            &[gw as i64 - 1, gh as i64 - 1]
        );
        assert!(p.position_ids[p.num_patches * 2..].iter().all(|&v| v == -1));
        assert!(p.pixel_values[p.num_patches * p.cols..]
            .iter()
            .all(|&v| v == 0.0));
        assert!(p.pixel_values[..p.num_patches * p.cols]
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v)));
    }

    const T: ImageTokens = ImageTokens {
        image: 9,
        boi: 7,
        eoi: 8,
        pad: 0,
    };

    #[test]
    fn expansion_and_splice() {
        let ids = [2, 9, 5, 9, 6];
        let e = expand_image_tokens(&ids, &T, &[2, 1]).unwrap();
        assert_eq!(e, vec![2, 7, 9, 9, 8, 5, 7, 9, 8, 6]);
        assert_eq!(
            text_embedding_ids(&e, &T),
            vec![2, 7, 0, 0, 8, 5, 7, 0, 8, 6]
        );
        assert!(expand_image_tokens(&ids, &T, &[2]).is_err());
        assert!(expand_image_tokens(&[7, 9], &T, &[1]).is_err());

        let hidden = 2;
        let mut emb = vec![0f32; e.len() * hidden];
        let a = [1.0, 1.5, 2.0, 2.5];
        let b = [3.0, 3.5];
        splice_images(&mut emb, hidden, &e, T.image, &[&a, &b]).unwrap();
        assert_eq!(&emb[4..8], &a);
        assert_eq!(&emb[14..16], &b);
        assert!(emb[..4].iter().all(|&v| v == 0.0));
        assert!(splice_images(&mut emb, hidden, &e, T.image, &[&a]).is_err());
    }

    #[test]
    fn token_types_mark_image_tokens() {
        assert_eq!(token_type_ids(&[2, 9, 9, 3, 9], 9), vec![0, 1, 1, 0, 1]);
    }
}
