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
//! Mistral 3 (`Mistral3ForConditionalGeneration`, e.g. Ministral-3) uses the
//! same processor with `spatial_merge_size` 2: images resize to whole
//! `patch * merge` cells, each token stands for a merge x merge block of
//! patches, and between the vision IR (tower + projector norm, as optimum-intel
//! exports it) and the projector IR (patch merger linear + MLP) the host
//! regroups the patch rows like `torch.nn.functional.unfold` ([`merge_patches`]).
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
    /// Patches per token side (Mistral 3: 2; Pixtral: 1).
    pub spatial_merge: u32,
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
            spatial_merge: 1,
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
        if let Some(m) = v
            .get("spatial_merge_size")
            .and_then(serde_json::Value::as_u64)
        {
            c.spatial_merge = m as u32;
        }
        if c.patch_size == 0 || c.longest_edge == 0 || c.spatial_merge == 0 {
            return Err(format!("{}: zero patch size or edge", path.display()));
        }
        Ok(c)
    }
}

/// transformers' Pixtral `get_resize_output_image_size`: `(height, width)`.
/// `PixtralProcessor` passes `patch_size * spatial_merge_size` as the patch,
/// so each side is a whole number of tokens.
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
    let p = cfg.patch_size * cfg.spatial_merge;
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
    /// Token grid: rows, columns (one token per merged block of patches).
    pub rows: u32,
    pub cols: u32,
    /// Patch grid the vision IR sees: `rows * merge`, `cols * merge`.
    pub patch_rows: u32,
    pub patch_cols: u32,
}

impl Pixels {
    /// Soft tokens the image occupies.
    pub fn patches(&self) -> usize {
        (self.rows * self.cols) as usize
    }
}

/// `Mistral3PatchMerger`'s regrouping: `feat` is the vision IR's
/// `[patch_rows * patch_cols, d]` rows; each output row (token grid order)
/// concatenates its `merge x merge` block channel-major, element
/// `c * m*m + ki * m + kj`, as `unfold(kernel=m, stride=m)` lays it out.
pub fn merge_patches(
    feat: &[f32],
    d: usize,
    patch_rows: usize,
    patch_cols: usize,
    m: usize,
) -> Vec<f32> {
    let (rows, cols) = (patch_rows / m, patch_cols / m);
    let width = d * m * m;
    let mut out = vec![0f32; rows * cols * width];
    for r in 0..rows {
        for c in 0..cols {
            let o = &mut out[(r * cols + c) * width..(r * cols + c + 1) * width];
            for ki in 0..m {
                for kj in 0..m {
                    let src = ((r * m + ki) * patch_cols + (c * m + kj)) * d;
                    for ch in 0..d {
                        o[ch * m * m + ki * m + kj] = feat[src + ch];
                    }
                }
            }
        }
    }
    out
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
    let cell = cfg.patch_size * cfg.spatial_merge;
    Ok(Pixels {
        pixel_values: out,
        height: th,
        width: tw,
        rows: th / cell,
        cols: tw / cell,
        patch_rows: th / cfg.patch_size,
        patch_cols: tw / cfg.patch_size,
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

/// `spatial_merge_size` from `processor_config.json`, a processor (not image
/// processor) setting. `None` when the file or the key is absent, so the image
/// processor's value stands.
fn processor_spatial_merge(dir: &Path) -> Result<Option<u32>, String> {
    let Some(v) = std::fs::read(dir.join("processor_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    else {
        return Ok(None);
    };
    match v
        .get("spatial_merge_size")
        .and_then(serde_json::Value::as_u64)
    {
        Some(0) => Err("processor_config.json: spatial_merge_size is 0".into()),
        Some(m) => Ok(Some(m as u32)),
        None => Ok(None),
    }
}

/// Rank 0's Pixtral image front end: the vision IR, its processor settings
/// and the framing token ids.
pub struct Frontend {
    rt: Runtime,
    /// Mistral 3: `openvino_multi_modal_projector_model.xml` (patch merger
    /// linear + MLP), run on the merged rows. None on Pixtral, whose vision IR
    /// already ends in the projector.
    projector: Option<(Runtime, String)>,
    /// Encoded images by content hash, so a chat's earlier images are not
    /// re-encoded every turn. Metadata: the token grid.
    cache: crate::image_cache::EncodeCache<(u32, u32)>,
    pub preprocess: PreprocessConfig,
    pub tokens: ImageTokens,
}

impl Frontend {
    /// Compile `openvino_vision_embeddings_model.xml` from `dir` on the CPU.
    pub fn load(dir: &Path, tokens: ImageTokens) -> Result<Self, String> {
        let mut preprocess =
            PreprocessConfig::from_json_file(&dir.join("preprocessor_config.json"))?;
        if let Some(m) = processor_spatial_merge(dir)? {
            preprocess.spatial_merge = m;
        }
        let compile = |name: &str| {
            let path = dir.join(name);
            Runtime::compile(&path.to_string_lossy(), "CPU", &PluginConfig::new())
                .map_err(|e| format!("compiling {}: {e}", path.display()))
        };
        let rt = compile("openvino_vision_embeddings_model.xml")?;
        let projector = if preprocess.spatial_merge > 1 {
            let p = compile("openvino_multi_modal_projector_model.xml")?;
            let input = p
                .input_name(0)
                .map_err(|e| format!("projector input: {e}"))?;
            Some((p, input))
        } else {
            None
        };
        Ok(Self {
            rt,
            projector,
            cache: crate::image_cache::EncodeCache::from_env(),
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
        let want = (p.patch_rows * p.patch_cols) as usize;
        if tokens != want {
            return Err(format!(
                "vision embedding returned {tokens} rows for {want} patches"
            ));
        }
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let Some((proj, input)) = self.projector.as_mut() else {
            return Ok((v, hidden));
        };
        let m = self.preprocess.spatial_merge as usize;
        let merged = merge_patches(&v, hidden, p.patch_rows as usize, p.patch_cols as usize, m);
        let n = p.patches();
        let bytes: Vec<u8> = merged.iter().flat_map(|x| x.to_le_bytes()).collect();
        proj.set_input(input, DType::F32, &[n, hidden * m * m], &bytes)
            .and_then(|_| proj.infer())
            .map_err(|e| format!("multi-modal projector: {e}"))?;
        let (dtype, shape, bytes) = proj
            .output(0)
            .map_err(|e| format!("multi-modal projector: {e}"))?;
        if dtype != DType::F32 {
            return Err(format!("projector output is {dtype:?}, expected f32"));
        }
        let out_hidden = match shape[..] {
            [1, t, h] | [t, h] if t == n => h,
            _ => return Err(format!("projector returned {shape:?} for {n} tokens")),
        };
        let v = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((v, out_hidden))
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
        let mut from_cache = 0;
        for (i, m) in media.iter().enumerate() {
            match m.kind {
                cascadia_types::MediaKind::Image => {}
                #[allow(unreachable_patterns)]
                other => return Err(format!("media {i}: {other:?} is not supported")),
            }
            if m.num_frames != 1 || m.pixels.len() != m.expected_len() {
                return Err(format!("image {i}: malformed pixel buffer"));
            }
            if let Some((emb, grid)) = self.cache.get(&m.sha256) {
                grids.push(grid);
                rows.push(emb);
                from_cache += 1;
                continue;
            }
            let p = preprocess(&m.pixels, m.width, m.height, &self.preprocess)
                .map_err(|e| format!("image {i}: {e}"))?;
            let (emb, _) = self.encode(&p).map_err(|e| format!("image {i}: {e}"))?;
            self.cache.put(m.sha256, &emb, (p.rows, p.cols));
            grids.push((p.rows, p.cols));
            rows.push(emb);
        }
        self.cache.report(from_cache, media.len());
        let ids = expand_image_tokens(prompt_ids, &self.tokens, &grids)?;
        Ok((ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_config_overrides_merge_only_when_it_names_one() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(processor_spatial_merge(dir.path()), Ok(None));
        let write = |v: &str| std::fs::write(dir.path().join("processor_config.json"), v).unwrap();
        write(r#"{"processor_class":"PixtralProcessor"}"#);
        assert_eq!(
            processor_spatial_merge(dir.path()),
            Ok(None),
            "absent key keeps the default"
        );
        write(r#"{"spatial_merge_size":2}"#);
        assert_eq!(processor_spatial_merge(dir.path()), Ok(Some(2)));
        write(r#"{"spatial_merge_size":0}"#);
        assert!(processor_spatial_merge(dir.path()).is_err());
    }

    /// `unfold(kernel=2, stride=2)` on a 2x4 patch grid of 2 channels: token
    /// (0, c) takes patches (0,2c), (0,2c+1), (1,2c), (1,2c+1), channel-major.
    #[test]
    fn merge_patches_matches_unfold_layout() {
        let (d, pr, pc) = (2usize, 2usize, 4usize);
        // feat[(row * pc + col) * d + ch] = 100*row + 10*col + ch
        let feat: Vec<f32> = (0..pr * pc * d)
            .map(|i| {
                let (p, ch) = (i / d, i % d);
                (100 * (p / pc) + 10 * (p % pc) + ch) as f32
            })
            .collect();
        let out = merge_patches(&feat, d, pr, pc, 2);
        assert_eq!(out.len(), 2 * d * 4);
        // token 1 (patches cols 2,3): ch0 block then ch1 block, each (ki,kj) row-major
        let t1 = &out[d * 4..2 * d * 4];
        assert_eq!(t1, &[20.0, 30.0, 120.0, 130.0, 21.0, 31.0, 121.0, 131.0]);
    }

    #[test]
    fn merged_grid_counts_tokens_per_cell() {
        let c = PreprocessConfig {
            patch_size: 14,
            spatial_merge: 2,
            longest_edge: 1540,
            ..PreprocessConfig::pixtral()
        };
        // 448x448 fits; rounded up to 28-px cells: 16x16 tokens, 32x32 patches.
        assert_eq!(target_size(448, 448, &c).unwrap(), (448, 448));
        let px = preprocess(&vec![128u8; 30 * 50 * 3], 50, 30, &c).unwrap();
        assert_eq!((px.height, px.width), (56, 56));
        assert_eq!(
            (px.rows, px.cols, px.patch_rows, px.patch_cols),
            (2, 2, 4, 4)
        );
    }

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
