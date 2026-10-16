//! Host-side InternVL2 image input (`internvl_chat`, e.g.
//! OpenVINO/InternVL2-4B-int4-ov): the work the model card's `load_image`
//! and `InternVLChatModel.chat` do around the vision IR, for rank 0 of a
//! staged `ov-runtime` tree.
//!
//! The vision IR already holds InternViT, the pixel shuffle and the `mlp1`
//! projector (`pixel_values [N, 3, 448, 448]` to `[N, 256, 3072]`), so the
//! host part is the tiling and the prompt:
//!
//! - [`tiles_for_size`] / [`preprocess`]: `dynamic_preprocess`: pick the
//!   tile grid `(cols, rows)` (at most `max_dynamic_patch` tiles) whose
//!   aspect ratio is closest to the image's, resize to it (Pillow bicubic,
//!   `Image.resize`'s default), cut 448-px tiles row-major and, when there is
//!   more than one, add a 448x448 thumbnail of the whole image; normalise
//!   each with the ImageNet mean/std (torchvision `ToTensor` + `Normalize`).
//! - [`expand_image_tokens`]: the template carries `<img><IMG_CONTEXT></img>`
//!   per image; rank 0 widens the one `<IMG_CONTEXT>` to `256 * tiles`, the
//!   string `chat` builds (`<img>` + `<IMG_CONTEXT>` * 256 * tiles +
//!   `</img>`), tokenized the same way because all three are special tokens.
//!
//! Parity with the model card's code (Pillow 12.3, torchvision) is pinned by
//! `tests/internvl_vision_parity.rs` against `tests/fixtures/internvl_vision/`.

use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::resample::{self, Filter};

/// Python's `sorted(set(...), key=area)` order of the tile grids for
/// `min_num = 1, max_num = 12` (the shipped setting). Ties in area keep the
/// set's iteration order, which decides between equally close ratios.
const RATIOS_1_12: [(u32, u32); 35] = [
    (1, 1),
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (1, 6),
    (6, 1),
    (3, 2),
    (2, 3),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (1, 9),
    (3, 3),
    (9, 1),
    (2, 5),
    (5, 2),
    (10, 1),
    (1, 10),
    (11, 1),
    (1, 11),
    (12, 1),
    (3, 4),
    (4, 3),
    (1, 12),
    (6, 2),
    (2, 6),
];

/// The settings `load_image` / `chat` use, from the export's `config.json`
/// and `preprocessor_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    pub image_size: u32,
    pub min_tiles: u32,
    pub max_tiles: u32,
    pub use_thumbnail: bool,
    /// Soft tokens per tile: `(image_size / patch)^2 * downsample^2`.
    pub tokens_per_tile: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl PreprocessConfig {
    /// InternVL2-4B as shipped.
    pub fn internvl2() -> Self {
        Self {
            image_size: 448,
            min_tiles: 1,
            max_tiles: 12,
            use_thumbnail: true,
            tokens_per_tile: 256,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    /// From the VLM `config.json` (`force_image_size`, `vision_config`,
    /// `downsample_ratio`, `min/max_dynamic_patch`, `use_thumbnail`,
    /// `dynamic_image_size`) and `preprocessor_config.json` (mean/std).
    pub fn from_configs(vlm: &serde_json::Value, preprocessor: &Path) -> Result<Self, String> {
        let mut c = Self::internvl2();
        let u = |v: &serde_json::Value, k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        let vision = vlm.get("vision_config").cloned().unwrap_or_default();
        if let Some(s) = u(vlm, "force_image_size").or_else(|| u(&vision, "image_size")) {
            c.image_size = s as u32;
        }
        let patch = u(&vision, "patch_size").unwrap_or(14) as u32;
        let down = vlm
            .get("downsample_ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.5);
        if patch == 0 || !c.image_size.is_multiple_of(patch) {
            return Err(format!(
                "image size {} is not whole {patch}-px patches",
                c.image_size
            ));
        }
        let side = f64::from(c.image_size / patch) * down;
        c.tokens_per_tile = (side * side) as u32;
        if let Some(n) = u(vlm, "min_dynamic_patch") {
            c.min_tiles = n as u32;
        }
        if let Some(n) = u(vlm, "max_dynamic_patch") {
            c.max_tiles = n as u32;
        }
        if let Some(t) = vlm
            .get("use_thumbnail")
            .and_then(serde_json::Value::as_bool)
        {
            c.use_thumbnail = t;
        }
        if vlm
            .get("dynamic_image_size")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        {
            c.min_tiles = 1;
            c.max_tiles = 1;
        }
        if let Ok(text) = std::fs::read_to_string(preprocessor) {
            let v: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("parsing {}: {e}", preprocessor.display()))?;
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
        }
        if c.min_tiles == 0 || c.max_tiles < c.min_tiles || c.tokens_per_tile == 0 {
            return Err(format!(
                "tiles {}..{} with {} tokens each is not a usable setting",
                c.min_tiles, c.max_tiles, c.tokens_per_tile
            ));
        }
        Ok(c)
    }
}

/// The candidate grids `(cols, rows)` in the order `dynamic_preprocess`
/// visits them. For the shipped 1..12 this is Python's own order; otherwise
/// area order with ties in generation order (which can differ from Python's
/// set order only when two equally close ratios of one area tie).
pub fn target_ratios(min: u32, max: u32) -> Vec<(u32, u32)> {
    if (min, max) == (1, 12) {
        return RATIOS_1_12.to_vec();
    }
    let mut v: Vec<(u32, u32)> = Vec::new();
    for n in min..=max {
        for i in 1..=n {
            for j in 1..=n {
                let a = i * j;
                if a <= max && a >= min && !v.contains(&(i, j)) {
                    v.push((i, j));
                }
            }
        }
    }
    v.sort_by_key(|&(i, j)| i * j);
    v
}

/// `find_closest_aspect_ratio`.
pub fn closest_ratio(
    width: u32,
    height: u32,
    ratios: &[(u32, u32)],
    image_size: u32,
) -> (u32, u32) {
    let aspect = f64::from(width) / f64::from(height);
    let area = f64::from(width) * f64::from(height);
    let mut best_diff = f64::INFINITY;
    let mut best = (1, 1);
    for &(i, j) in ratios {
        let diff = (aspect - f64::from(i) / f64::from(j)).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff
            && area > 0.5 * f64::from(image_size) * f64::from(image_size) * f64::from(i * j)
        {
            best = (i, j);
        }
    }
    best
}

/// The tile grid `(cols, rows)` and the number of views the vision IR gets
/// (tiles, plus the thumbnail when there is more than one tile).
pub fn tiles_for_size(
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<((u32, u32), usize), String> {
    if width == 0 || height == 0 {
        return Err(format!("empty image {width}x{height}"));
    }
    let grid = closest_ratio(
        width,
        height,
        &target_ratios(cfg.min_tiles, cfg.max_tiles),
        cfg.image_size,
    );
    let blocks = (grid.0 * grid.1) as usize;
    let views = blocks + usize::from(cfg.use_thumbnail && blocks != 1);
    Ok((grid, views))
}

/// Soft tokens an image of `width x height` takes, from its size alone.
pub fn num_image_tokens(width: u32, height: u32, cfg: &PreprocessConfig) -> Result<usize, String> {
    let (_, views) = tiles_for_size(width, height, cfg)?;
    Ok(views * cfg.tokens_per_tile as usize)
}

/// The vision IR's input for one image: `[views, 3, size, size]` f32.
#[derive(Clone, Debug)]
pub struct Pixels {
    pub pixel_values: Vec<f32>,
    pub views: usize,
}

/// Preprocess one packed RGB8 image exactly as the model card's
/// `load_image` does.
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Pixels, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(format!("image is not {width}x{height} RGB8"));
    }
    let ((cols, rows), views) = tiles_for_size(width, height, cfg)?;
    let s = cfg.image_size as usize;
    let (tw, th) = (s * cols as usize, s * rows as usize);
    let resized = resample::resize_rgb8(
        rgb,
        width as usize,
        height as usize,
        tw,
        th,
        Filter::Bicubic,
    );
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|v| (v as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c])
    });
    let plane = s * s;
    let view = 3 * plane;
    let mut out = vec![0f32; views * view];
    let mut put = |idx: usize, img: &[u8], stride: usize, x0: usize, y0: usize| {
        let base = idx * view;
        for y in 0..s {
            for x in 0..s {
                let p = ((y0 + y) * stride + x0 + x) * 3;
                for c in 0..3 {
                    out[base + c * plane + y * s + x] = lut[c][img[p + c] as usize];
                }
            }
        }
    };
    let blocks = (cols * rows) as usize;
    for i in 0..blocks {
        let (cx, cy) = (i % cols as usize, i / cols as usize);
        put(i, &resized, tw, cx * s, cy * s);
    }
    if views > blocks {
        let thumb =
            resample::resize_rgb8(rgb, width as usize, height as usize, s, s, Filter::Bicubic);
        put(blocks, &thumb, s, 0, 0);
    }
    Ok(Pixels {
        pixel_values: out,
        views,
    })
}

/// Token ids that frame an InternVL image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTokens {
    /// `<IMG_CONTEXT>`: the soft token.
    pub context: i64,
    /// `<img>` / `</img>`.
    pub start: i64,
    pub end: i64,
}

/// Each `<img><IMG_CONTEXT></img>` of the prompt gets its image's
/// `counts[i]` context tokens. Any other use of the three ids is refused.
pub fn expand_image_tokens(
    ids: &[i64],
    t: &ImageTokens,
    counts: &[usize],
) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&x| x == t.context).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} image placeholders for {} images",
            counts.len()
        ));
    }
    for (i, &x) in ids.iter().enumerate() {
        let framed =
            |at: usize| at >= 1 && ids[at - 1] == t.start && ids.get(at + 1) == Some(&t.end);
        let ok = if x == t.context {
            framed(i)
        } else if x == t.start {
            ids.get(i + 1) == Some(&t.context)
        } else if x == t.end {
            i >= 1 && ids[i - 1] == t.context
        } else {
            true
        };
        if !ok {
            return Err("image tokens are not each <img><IMG_CONTEXT></img>".into());
        }
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut images = counts.iter();
    for &x in ids {
        if x == t.context {
            let &n = images.next().expect("counted above");
            out.extend(std::iter::repeat_n(t.context, n));
        } else {
            out.push(x);
        }
    }
    Ok(out)
}

/// Rank 0's InternVL front end: the vision IR, the tiling settings and the
/// framing ids.
pub struct Frontend {
    rt: Runtime,
    /// Encoded images by content hash. Metadata: the soft-token count.
    cache: crate::image_cache::EncodeCache<usize>,
    pub preprocess: PreprocessConfig,
    pub tokens: ImageTokens,
}

impl Frontend {
    /// Compile `openvino_vision_embeddings_model.xml` from `dir` on the CPU.
    pub fn load(
        dir: &Path,
        vlm_config: &serde_json::Value,
        tokens: ImageTokens,
    ) -> Result<Self, String> {
        let preprocess =
            PreprocessConfig::from_configs(vlm_config, &dir.join("preprocessor_config.json"))?;
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

    /// `[views * tokens_per_tile, hidden]` rows for a preprocessed image.
    pub fn encode(&mut self, p: &Pixels) -> Result<Vec<f32>, String> {
        let s = self.preprocess.image_size as usize;
        let bytes: Vec<u8> = p
            .pixel_values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        self.rt
            .set_input("pixel_values", DType::F32, &[p.views, 3, s, s], &bytes)
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
        let per = self.preprocess.tokens_per_tile as usize;
        match shape[..] {
            [n, t, _] if n == p.views && t == per => {}
            _ => {
                return Err(format!(
                    "vision embedding returned {shape:?} for {} tiles of {per}",
                    p.views
                ))
            }
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn counts(&self, media: &[cascadia_types::MediaInput]) -> Result<Vec<usize>, String> {
        media
            .iter()
            .enumerate()
            .map(|(i, m)| {
                crate::gemma4_vision::check_image(i, m)?;
                num_image_tokens(m.width, m.height, &self.preprocess)
                    .map_err(|e| format!("image {i}: {e}"))
            })
            .collect()
    }

    /// The expanded prompt `prepare` would return, from image sizes alone.
    pub fn expand_ids(
        &self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        expand_image_tokens(prompt_ids, &self.tokens, &self.counts(media)?)
    }

    /// Length of a `prompt_len`-token prompt once `media` expand.
    pub fn expanded_len(
        &self,
        prompt_len: usize,
        media: &[cascadia_types::MediaInput],
    ) -> Result<usize, String> {
        let extra: usize = self.counts(media)?.iter().sum();
        Ok((prompt_len + extra).saturating_sub(media.len()))
    }

    /// Encode a task's images and widen each `<IMG_CONTEXT>`. Returns the
    /// expanded ids and each image's rows, in prompt order.
    pub fn prepare(
        &mut self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
        tenant: &str,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
        let counts = self.counts(media)?;
        let ids = expand_image_tokens(prompt_ids, &self.tokens, &counts)?;
        let mut rows = Vec::with_capacity(media.len());
        let mut from_cache = 0;
        for (i, m) in media.iter().enumerate() {
            if let Some((emb, _)) = self.cache.get(tenant, &m.sha256) {
                rows.push(emb);
                from_cache += 1;
                continue;
            }
            let p = preprocess(&m.pixels, m.width, m.height, &self.preprocess)
                .map_err(|e| format!("image {i}: {e}"))?;
            let emb = self.encode(&p).map_err(|e| format!("image {i}: {e}"))?;
            self.cache.put(tenant, m.sha256, &emb, counts[i]);
            rows.push(emb);
        }
        self.cache.report(from_cache, media.len());
        Ok((ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_table_is_the_python_set_in_area_order() {
        let mut want: Vec<(u32, u32)> = Vec::new();
        for i in 1..=12u32 {
            for j in 1..=12u32 {
                if i * j <= 12 {
                    want.push((i, j));
                }
            }
        }
        let mut got = RATIOS_1_12.to_vec();
        assert!(got.windows(2).all(|w| w[0].0 * w[0].1 <= w[1].0 * w[1].1));
        got.sort();
        want.sort();
        assert_eq!(got, want);
        assert_eq!(target_ratios(1, 1), vec![(1, 1)]);
        assert_eq!(target_ratios(1, 2), vec![(1, 1), (1, 2), (2, 1)]);
    }

    #[test]
    fn tiles_follow_the_closest_ratio() {
        let c = PreprocessConfig::internvl2();
        // Square: (1, 1) wins over (2, 2) (area check fails), one tile.
        assert_eq!(tiles_for_size(448, 448, &c).unwrap(), ((1, 1), 1));
        assert_eq!(num_image_tokens(448, 448, &c).unwrap(), 256);
        // A big square: the area check picks the later equal ratio.
        assert_eq!(tiles_for_size(1600, 1600, &c).unwrap(), ((3, 3), 10));
        // 2:1 landscape: two tiles plus the thumbnail.
        assert_eq!(tiles_for_size(800, 400, &c).unwrap(), ((2, 1), 3));
        assert!(tiles_for_size(0, 3, &c).is_err());
    }

    #[test]
    fn expansion_widens_each_context_token() {
        let t = ImageTokens {
            context: 9,
            start: 7,
            end: 8,
        };
        let e = expand_image_tokens(&[1, 7, 9, 8, 2, 7, 9, 8], &t, &[2, 3]).unwrap();
        assert_eq!(e, vec![1, 7, 9, 9, 8, 2, 7, 9, 9, 9, 8]);
        assert!(expand_image_tokens(&[1, 9], &t, &[1]).is_err(), "unframed");
        assert!(
            expand_image_tokens(&[7, 9, 8, 8], &t, &[1]).is_err(),
            "stray </img>"
        );
        assert!(expand_image_tokens(&[7, 9, 8], &t, &[1, 1]).is_err());
    }

    #[test]
    fn preprocess_lays_out_tiles_then_thumbnail() {
        let c = PreprocessConfig::internvl2();
        // 896x448, left half black, right half white: two tiles + thumbnail.
        let mut rgb = vec![0u8; 896 * 448 * 3];
        for y in 0..448 {
            for x in 448..896 {
                let p = (y * 896 + x) * 3;
                rgb[p..p + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        let p = preprocess(&rgb, 896, 448, &c).unwrap();
        assert_eq!(p.views, 3);
        let view = 3 * 448 * 448;
        let black = (0.0 - c.image_mean[0]) / c.image_std[0];
        let white = (1.0 - c.image_mean[0]) / c.image_std[0];
        assert_eq!(p.pixel_values[0], black);
        assert_eq!(p.pixel_values[view], white);
        // Thumbnail: left edge black, right edge white.
        assert_eq!(p.pixel_values[2 * view], black);
        assert_eq!(p.pixel_values[2 * view + 447], white);
    }

    #[test]
    fn configs_derive_tokens_per_tile() {
        let dir = tempfile::tempdir().unwrap();
        let vlm = serde_json::json!({"force_image_size": 448, "downsample_ratio": 0.5,
            "vision_config": {"patch_size": 14, "image_size": 448},
            "max_dynamic_patch": 6, "use_thumbnail": true});
        let c = PreprocessConfig::from_configs(&vlm, &dir.path().join("none.json")).unwrap();
        assert_eq!((c.tokens_per_tile, c.max_tiles), (256, 6));
        let fixed = serde_json::json!({"dynamic_image_size": false});
        let c = PreprocessConfig::from_configs(&fixed, &dir.path().join("none.json")).unwrap();
        assert_eq!((c.min_tiles, c.max_tiles), (1, 1));
    }
}
