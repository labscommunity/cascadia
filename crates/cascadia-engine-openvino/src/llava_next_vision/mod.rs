//! Host-side LLaVA-NeXT image input (`llava_next`, e.g.
//! OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov) for rank 0 of a staged
//! `ov-runtime` tree: the work Hugging Face's `LlavaNextImageProcessor`,
//! `LlavaNextProcessor` and `LlavaNextModel.pack_image_features` do around the
//! vision IR.
//!
//! - [`select_best_resolution`]: the "anyres" pinpoint an image is tiled at
//!   (the largest effective resolution, then the least waste).
//! - [`preprocess`]: the base view (the whole image squashed to 336x336) and
//!   the pinpoint's 336-px tiles (the image resized into the pinpoint keeping
//!   its aspect ratio, centred on black), each bicubic-resized like Pillow,
//!   rescaled and CLIP-normalised, channels first.
//! - [`image_tokens`]: the soft tokens an image costs, from its size alone:
//!   576 for the base view plus the tile grid's rows after unpadding, each
//!   closed by an `image_newline` row (`LlavaNextProcessor._get_number_of_features`).
//! - [`pack_features`]: the vision IR's rows (one `[576, d]` block per view)
//!   in the order the language model sees them: the base view's rows, then
//!   the tiles stitched into one grid, the padding rows or columns dropped
//!   (`unpad_image`) and `image_newline` after each row.
//! - [`expand_image_tokens`]: each `<image>` of the prompt becomes that many
//!   `<image>` ids (newline rows included); the encoder supplies every row.
//!
//! The exported vision IR (`openvino_vision_embeddings_model.xml`) is the CLIP
//! tower with its penultimate layer, CLS dropped and the projector applied:
//! `pixel_values [n, 3, 336, 336]` to `[n, 576, 4096]`. `image_newline` is
//! not an IR: optimum-intel writes it into `config.json` (4096 floats).
//!
//! Parity with transformers 5.14.1 is pinned by
//! `tests/llava_next_vision_parity.rs` against `tests/fixtures/llava_next_vision/`.

use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::resample;

/// The image processor's and model's settings that shape an image's views and
/// tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    /// `(height, width)` pinpoints, each a whole number of tiles.
    pub pinpoints: Vec<(u32, u32)>,
    /// Side of the base view and of each tile (`crop_size`, the vision
    /// tower's `image_size`): 336.
    pub tile: u32,
    /// The vision tower's patch: 14, so a view is 24x24 tokens.
    pub patch_size: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f64,
}

impl PreprocessConfig {
    /// llava-v1.6 as shipped.
    pub fn llava_v16() -> Self {
        Self {
            pinpoints: vec![(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)],
            tile: 336,
            patch_size: 14,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// Tokens per view side (24).
    pub fn side(&self) -> u32 {
        self.tile / self.patch_size
    }

    /// From an export's `preprocessor_config.json`, `processor_config.json`
    /// and VLM `config.json` (`config.vlm.json` in a cut tree). Refuses a
    /// configuration this port does not reproduce (different base and tile
    /// sizes, the `full` feature strategy, pinpoints that disagree between
    /// processor and model).
    pub fn from_export(
        preprocessor: &serde_json::Value,
        processor: Option<&serde_json::Value>,
        vlm: &serde_json::Value,
    ) -> Result<Self, String> {
        let mut c = Self::llava_v16();
        let pins = |v: &serde_json::Value| -> Option<Vec<(u32, u32)>> {
            v.as_array()?
                .iter()
                .map(|p| {
                    let p = p.as_array()?;
                    Some((p.first()?.as_u64()? as u32, p.get(1)?.as_u64()? as u32))
                })
                .collect()
        };
        if let Some(p) = preprocessor.get("image_grid_pinpoints").and_then(pins) {
            c.pinpoints = p;
        }
        if let Some(model) = vlm.get("image_grid_pinpoints").and_then(pins) {
            if model != c.pinpoints {
                return Err(format!(
                    "the processor's grid pinpoints {:?} differ from the model's {model:?}",
                    c.pinpoints
                ));
            }
        }
        let u = |v: Option<&serde_json::Value>| v.and_then(serde_json::Value::as_u64);
        let crop = u(preprocessor.get("crop_size").and_then(|s| s.get("height")));
        let base = u(preprocessor
            .get("size")
            .and_then(|s| s.get("shortest_edge").or_else(|| s.get("height"))));
        let tower = vlm.get("vision_config");
        let image_size = u(tower.and_then(|t| t.get("image_size")));
        if let Some(t) = crop.or(image_size) {
            c.tile = t as u32;
        }
        for (what, v) in [
            ("size", base),
            ("crop_size", crop),
            ("vision image_size", image_size),
        ] {
            if let Some(v) = v {
                if v as u32 != c.tile {
                    return Err(format!(
                        "{what} {v} differs from the {} px tile; only one view size is supported",
                        c.tile
                    ));
                }
            }
        }
        if let Some(p) = u(processor.and_then(|p| p.get("patch_size")))
            .or_else(|| u(tower.and_then(|t| t.get("patch_size"))))
        {
            c.patch_size = p as u32;
        }
        let strategy = processor
            .and_then(|p| p.get("vision_feature_select_strategy"))
            .or_else(|| vlm.get("vision_feature_select_strategy"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        if strategy != "default" {
            return Err(format!(
                "vision_feature_select_strategy {strategy:?}; only \"default\" (CLS dropped) is supported"
            ));
        }
        let triple = |k: &str| -> Option<[f32; 3]> {
            let a = preprocessor.get(k)?.as_array()?;
            (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(f64::NAN) as f32))
        };
        if let Some(m) = triple("image_mean") {
            c.image_mean = m;
        }
        if let Some(s) = triple("image_std") {
            c.image_std = s;
        }
        if let Some(r) = preprocessor
            .get("rescale_factor")
            .and_then(serde_json::Value::as_f64)
        {
            c.rescale_factor = r;
        }
        if c.tile == 0 || c.patch_size == 0 || !c.tile.is_multiple_of(c.patch_size) {
            return Err(format!(
                "tile {} is not a whole number of {} px patches",
                c.tile, c.patch_size
            ));
        }
        if c.pinpoints.is_empty()
            || c.pinpoints.iter().any(|&(h, w)| {
                h == 0 || w == 0 || !h.is_multiple_of(c.tile) || !w.is_multiple_of(c.tile)
            })
        {
            return Err(format!(
                "grid pinpoints {:?} are not whole {} px tiles",
                c.pinpoints, c.tile
            ));
        }
        Ok(c)
    }
}

/// transformers' `select_best_resolution`: the `(height, width)` pinpoint
/// that keeps the most of a `height x width` image, ties to the least
/// padding.
pub fn select_best_resolution(height: u32, width: u32, pinpoints: &[(u32, u32)]) -> (u32, u32) {
    let (oh, ow) = (f64::from(height), f64::from(width));
    let mut best = pinpoints[0];
    let (mut max_eff, mut min_waste) = (0u64, u64::MAX);
    for &(ph, pw) in pinpoints {
        let scale = (f64::from(pw) / ow).min(f64::from(ph) / oh);
        let (dw, dh) = ((ow * scale) as u64, (oh * scale) as u64);
        let eff = (dw * dh).min(u64::from(width) * u64::from(height));
        let waste = (u64::from(pw) * u64::from(ph)).saturating_sub(eff);
        if eff > max_eff || (eff == max_eff && waste < min_waste) {
            max_eff = eff;
            min_waste = waste;
            best = (ph, pw);
        }
    }
    best
}

/// transformers' `get_patch_output_size`: the `(height, width)` an image
/// resizes to inside `target` keeping its aspect ratio.
pub fn patch_output_size(height: u32, width: u32, target: (u32, u32)) -> (u32, u32) {
    let (th, tw) = target;
    let scale_w = f64::from(tw) / f64::from(width);
    let scale_h = f64::from(th) / f64::from(height);
    if scale_w < scale_h {
        let nh = (f64::from(height) * scale_w).ceil() as u32;
        (nh.min(th), tw)
    } else {
        let nw = (f64::from(width) * scale_h).ceil() as u32;
        (th, nw.min(tw))
    }
}

/// Python's `int(round(x, 7))` for `x >= 0`: Rust's fixed-precision
/// formatting rounds the exact binary value correctly, as CPython's does.
fn int_round7(x: f64) -> i64 {
    format!("{x:.7}")
        .parse::<f64>()
        .map_or(0, |v| v.trunc() as i64)
}

/// The tile grid `(rows, cols)` of tokens kept after `unpad_image` for an
/// image of `height x width` whose tiles form a `cur_h x cur_w` token grid.
/// Either side can come out 0 for an extreme aspect ratio (a row of newline
/// rows then remains, or nothing at all).
pub fn unpadded_grid(height: u32, width: u32, cur_h: u32, cur_w: u32) -> (u32, u32) {
    let (oh, ow) = (f64::from(height), f64::from(width));
    let (ch, cw) = (f64::from(cur_h), f64::from(cur_w));
    if ow / oh > cw / ch {
        let new_h = int_round7(oh * (cw / ow));
        let pad = (i64::from(cur_h) - new_h).div_euclid(2);
        ((i64::from(cur_h) - 2 * pad).max(0) as u32, cur_w)
    } else {
        let new_w = int_round7(ow * (ch / oh));
        let pad = (i64::from(cur_w) - new_w).div_euclid(2);
        (cur_h, (i64::from(cur_w) - 2 * pad).max(0) as u32)
    }
}

/// An image's layout: the pinpoint, its tile grid and the kept token grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The chosen pinpoint `(height, width)`.
    pub best: (u32, u32),
    /// Tiles down and across.
    pub grid: (u32, u32),
    /// Token rows and columns kept after unpadding.
    pub kept: (u32, u32),
    /// Tokens per view (576).
    pub per_view: u32,
}

impl Layout {
    /// Views the vision IR encodes: the base view plus the tiles.
    pub fn views(&self) -> usize {
        1 + (self.grid.0 * self.grid.1) as usize
    }

    /// Soft tokens the image occupies: the base view, then each kept row
    /// closed by `image_newline`.
    pub fn tokens(&self) -> usize {
        self.per_view as usize + self.kept.0 as usize * (self.kept.1 as usize + 1)
    }
}

/// The layout of a `height x width` image, from its size alone.
pub fn layout(height: u32, width: u32, cfg: &PreprocessConfig) -> Result<Layout, String> {
    if height == 0 || width == 0 {
        return Err(format!("empty image {width}x{height}"));
    }
    let best = select_best_resolution(height, width, &cfg.pinpoints);
    let grid = (best.0 / cfg.tile, best.1 / cfg.tile);
    let side = cfg.side();
    let kept = unpadded_grid(height, width, grid.0 * side, grid.1 * side);
    Ok(Layout {
        best,
        grid,
        kept,
        per_view: side * side,
    })
}

/// Soft tokens a `height x width` image expands to (HF's
/// `_get_number_of_features` minus the CLS the `default` strategy drops).
pub fn image_tokens(height: u32, width: u32, cfg: &PreprocessConfig) -> Result<usize, String> {
    layout(height, width, cfg).map(|l| l.tokens())
}

/// Length of a `prompt_len`-token prompt once each image's `<image>` expands,
/// for images of the given `(height, width)`: computed before any image is
/// encoded, so an over-window prompt is refused without the encoder's cost.
pub fn expanded_prompt_len(
    prompt_len: usize,
    dims: &[(u32, u32)],
    cfg: &PreprocessConfig,
) -> Result<usize, String> {
    let mut n = prompt_len;
    for (i, &(h, w)) in dims.iter().enumerate() {
        let t = image_tokens(h, w, cfg).map_err(|e| format!("image {i}: {e}"))?;
        n = (n + t).saturating_sub(1);
    }
    Ok(n)
}

/// The vision IR's input for one image.
#[derive(Clone, Debug)]
pub struct Views {
    /// `[views, 3, tile, tile]` f32: the base view, then the tiles row-major.
    pub pixel_values: Vec<f32>,
    pub layout: Layout,
    /// The image's own size, which unpadding needs.
    pub height: u32,
    pub width: u32,
}

fn resize(rgb: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    // Pillow returns a copy for a same-size resize.
    if (w, h) == (nw, nh) {
        return rgb.to_vec();
    }
    resample::resize_rgb8_bicubic(rgb, w as usize, h as usize, nw as usize, nh as usize)
}

/// Preprocess one packed RGB8 image exactly as transformers' PIL
/// `LlavaNextImageProcessor`: the base view squashed to `tile x tile`, then
/// the image resized into its pinpoint (aspect kept, `ceil`), centred on
/// black (`divmod` padding, the odd pixel after), cut into tiles row-major;
/// every view rescaled in f64, cast to f32 and normalised in f32.
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Views, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(format!("image is not {width}x{height} RGB8"));
    }
    let lay = layout(height, width, cfg)?;
    let t = cfg.tile as usize;
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|v| {
            let x = (v as f64 * cfg.rescale_factor) as f32;
            (x - cfg.image_mean[c]) / cfg.image_std[c]
        })
    });
    let plane = t * t;
    let mut out = vec![0f32; lay.views() * 3 * plane];
    // One view from an RGB8 source of row stride `stride` pixels.
    let mut put = |view: usize, src: &[u8], stride: usize, x0: usize, y0: usize| {
        let o = &mut out[view * 3 * plane..(view + 1) * 3 * plane];
        for y in 0..t {
            for x in 0..t {
                let px = &src[((y0 + y) * stride + x0 + x) * 3..][..3];
                for c in 0..3 {
                    o[c * plane + y * t + x] = lut[c][px[c] as usize];
                }
            }
        }
    };
    let base = resize(rgb, width, height, cfg.tile, cfg.tile);
    put(0, &base, t, 0, 0);
    let (bh, bw) = lay.best;
    let (nh, nw) = patch_output_size(height, width, lay.best);
    let resized = resize(rgb, width, height, nw, nh);
    let (py, px) = ((bh - nh) / 2, (bw - nw) / 2);
    let (bh, bw) = (bh as usize, bw as usize);
    let mut padded = vec![0u8; bh * bw * 3];
    for y in 0..nh as usize {
        let dst = ((py as usize + y) * bw + px as usize) * 3;
        padded[dst..dst + nw as usize * 3]
            .copy_from_slice(&resized[y * nw as usize * 3..(y + 1) * nw as usize * 3]);
    }
    let mut view = 1;
    for ty in 0..lay.grid.0 as usize {
        for tx in 0..lay.grid.1 as usize {
            put(view, &padded, bw, tx * t, ty * t);
            view += 1;
        }
    }
    Ok(Views {
        pixel_values: out,
        layout: lay,
        height,
        width,
    })
}

/// `pack_image_features` for one image: `feat` holds the vision IR's rows,
/// `[views, per_view, d]` (base view first); returns the image's soft-token
/// rows, `[layout.tokens(), d]`: the base view's rows, then for each kept row
/// of the stitched tile grid its kept columns and `newline`.
pub fn pack_features(
    feat: &[f32],
    d: usize,
    lay: &Layout,
    newline: &[f32],
) -> Result<Vec<f32>, String> {
    let per = lay.per_view as usize;
    if feat.len() != lay.views() * per * d || newline.len() != d {
        return Err(format!(
            "{} feature values and a {}-wide newline for {} views of {per} x {d}",
            feat.len(),
            newline.len(),
            lay.views()
        ));
    }
    let side = (per as f64).sqrt() as usize;
    let (gh, gw) = (lay.grid.0 as usize, lay.grid.1 as usize);
    let (cur_h, cur_w) = (gh * side, gw * side);
    let (kh, kw) = (lay.kept.0 as usize, lay.kept.1 as usize);
    // unpad_image keeps rows [pad, cur - pad), with pad = (cur - kept) / 2.
    let (top, left) = ((cur_h - kh) / 2, (cur_w - kw) / 2);
    let mut out = Vec::with_capacity(lay.tokens() * d);
    out.extend_from_slice(&feat[..per * d]);
    for r in top..top + kh {
        let (ty, y) = (r / side, r % side);
        for c in left..left + kw {
            let (tx, x) = (c / side, c % side);
            let view = 1 + ty * gw + tx;
            let src = (view * per + y * side + x) * d;
            out.extend_from_slice(&feat[src..src + d]);
        }
        out.extend_from_slice(newline);
    }
    debug_assert_eq!(out.len(), lay.tokens() * d);
    Ok(out)
}

/// `LlavaNextProcessor`'s expansion on token ids: each `<image>` of the
/// tokenized template becomes `counts[k]` `<image>` ids for its image.
pub fn expand_image_tokens(
    ids: &[i64],
    image_token: i64,
    counts: &[usize],
) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&x| x == image_token).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} image placeholders for {} images",
            counts.len()
        ));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut each = counts.iter();
    for &x in ids {
        if x == image_token {
            let &n = each.next().expect("counted above");
            out.extend(std::iter::repeat_n(image_token, n));
        } else {
            out.push(x);
        }
    }
    Ok(out)
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let b = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_slice(&b).map_err(|e| format!("parsing {}: {e}", path.display()))
}

/// Rank 0's LLaVA-NeXT image front end: the vision IR (tower + projector),
/// `image_newline`, the processor settings and the image token id.
pub struct Frontend {
    rt: Runtime,
    newline: Vec<f32>,
    /// Encoded (packed) images by (tenant, pixel hash), so a chat's earlier
    /// images are not re-encoded every turn. Metadata: the soft-token count.
    cache: crate::image_cache::EncodeCache<usize>,
    pub preprocess: PreprocessConfig,
    pub image_token: i64,
}

impl Frontend {
    /// Compile `openvino_vision_embeddings_model.xml` from `dir` on the CPU
    /// and read the settings and `image_newline` from the export's configs
    /// (`config.vlm.json` in a cut tree, `config.json` in an export).
    pub fn load(dir: &Path, image_token: i64) -> Result<Self, String> {
        let pre = read_json(&dir.join("preprocessor_config.json"))?;
        let proc_path = dir.join("processor_config.json");
        let proc = proc_path
            .exists()
            .then(|| read_json(&proc_path))
            .transpose()?;
        let vlm_path = ["config.vlm.json", "config.json"]
            .map(|f| dir.join(f))
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| format!("{} has no config.vlm.json", dir.display()))?;
        let vlm = read_json(&vlm_path)?;
        let preprocess = PreprocessConfig::from_export(&pre, proc.as_ref(), &vlm)
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let newline: Vec<f32> = vlm
            .get("image_newline")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!(
                    "{} lacks image_newline (optimum-intel writes it there); re-export the model",
                    vlm_path.display()
                )
            })?
            .iter()
            .map(|x| x.as_f64().map(|v| v as f32))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{}: image_newline is not numeric", vlm_path.display()))?;
        if let Some(h) = vlm
            .get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(serde_json::Value::as_u64)
        {
            if newline.len() != h as usize {
                return Err(format!(
                    "image_newline has {} values for hidden size {h}",
                    newline.len()
                ));
            }
        }
        let path = dir.join("openvino_vision_embeddings_model.xml");
        let rt = Runtime::compile(&path.to_string_lossy(), "CPU", &PluginConfig::new())
            .map_err(|e| format!("compiling {}: {e}", path.display()))?;
        Ok(Self {
            rt,
            newline,
            cache: crate::image_cache::EncodeCache::from_env(),
            preprocess,
            image_token,
        })
    }

    /// The image's soft-token rows: every view through the vision IR in one
    /// inference, then packed.
    pub fn encode(&mut self, v: &Views) -> Result<Vec<f32>, String> {
        let t = self.preprocess.tile as usize;
        let n = v.layout.views();
        let bytes: Vec<u8> = v
            .pixel_values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        self.rt
            .set_input("pixel_values", DType::F32, &[n, 3, t, t], &bytes)
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
        let per = v.layout.per_view as usize;
        let d = match shape[..] {
            [b, p, d] if b == n && p == per => d,
            _ => {
                return Err(format!(
                    "vision embedding returned {shape:?} for {n} views of {per} tokens"
                ))
            }
        };
        let feat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        pack_features(&feat, d, &v.layout, &self.newline)
    }

    /// The expanded prompt ids `prepare` would return, from each image's
    /// size, with no encoder work.
    pub fn expand_ids(
        &self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        let mut counts = Vec::with_capacity(media.len());
        for (i, m) in media.iter().enumerate() {
            crate::gemma4_vision::check_image(i, m)?;
            counts.push(
                image_tokens(m.height, m.width, &self.preprocess)
                    .map_err(|e| format!("image {i}: {e}"))?,
            );
        }
        expand_image_tokens(prompt_ids, self.image_token, &counts)
    }

    /// [`expanded_prompt_len`] for `media` under this front end's processor.
    pub fn expanded_len(
        &self,
        prompt_len: usize,
        media: &[cascadia_types::MediaInput],
    ) -> Result<usize, String> {
        let dims: Vec<(u32, u32)> = media.iter().map(|m| (m.height, m.width)).collect();
        expanded_prompt_len(prompt_len, &dims, &self.preprocess)
    }

    /// Encode a task's images and expand each `<image>`. Returns the expanded
    /// ids and each image's rows, in prompt order.
    pub fn prepare(
        &mut self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
        tenant: &str,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
        let mut rows = Vec::with_capacity(media.len());
        let mut counts = Vec::with_capacity(media.len());
        let mut from_cache = 0;
        for (i, m) in media.iter().enumerate() {
            crate::gemma4_vision::check_image(i, m)?;
            if let Some((emb, n)) = self.cache.get(tenant, &m.sha256) {
                counts.push(n);
                rows.push(emb);
                from_cache += 1;
                continue;
            }
            let v = preprocess(&m.pixels, m.width, m.height, &self.preprocess)
                .map_err(|e| format!("image {i}: {e}"))?;
            let emb = self.encode(&v).map_err(|e| format!("image {i}: {e}"))?;
            let n = v.layout.tokens();
            self.cache.put(tenant, m.sha256, &emb, n);
            counts.push(n);
            rows.push(emb);
        }
        self.cache.report(from_cache, media.len());
        let ids = expand_image_tokens(prompt_ids, self.image_token, &counts)?;
        Ok((ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PreprocessConfig {
        PreprocessConfig::llava_v16()
    }

    #[test]
    fn best_resolution_prefers_coverage_then_less_padding() {
        let p = cfg().pinpoints;
        assert_eq!(select_best_resolution(448, 448, &p), (672, 672));
        assert_eq!(select_best_resolution(1000, 300, &p), (1008, 336));
        assert_eq!(select_best_resolution(4, 1500, &p), (336, 1008));
        // Small images all fit unscaled: equal coverage, least waste wins.
        assert_eq!(select_best_resolution(30, 40, &p), (336, 672));
    }

    #[test]
    fn layout_counts_the_base_view_and_newline_rows() {
        // 448x448 on 672x672: 48x48 tokens, no padding to drop.
        let l = layout(448, 448, &cfg()).unwrap();
        assert_eq!((l.grid, l.kept, l.views()), ((2, 2), (48, 48), 5));
        assert_eq!(l.tokens(), 576 + 48 * 49);
        // 1500x4 (w x h): every tile row is padding; only the base view stays.
        assert_eq!(image_tokens(4, 1500, &cfg()), Ok(576));
        // 3x1200: no column survives, but each of the 72 rows keeps its newline.
        assert_eq!(image_tokens(1200, 3, &cfg()), Ok(576 + 72));
        assert!(image_tokens(0, 3, &cfg()).is_err());
    }

    #[test]
    fn int_round7_is_pythons() {
        assert_eq!(int_round7(0.192), 0);
        assert_eq!(int_round7(47.999_999_99), 48);
        assert_eq!(int_round7(47.999_9), 47);
        assert_eq!(int_round7(36.0), 36);
    }

    #[test]
    fn expansion_repeats_the_token_per_image() {
        let e = expand_image_tokens(&[1, 9, 5, 9, 2], 9, &[2, 3]).unwrap();
        assert_eq!(e, vec![1, 9, 9, 5, 9, 9, 9, 2]);
        assert!(expand_image_tokens(&[9], 9, &[1, 1]).is_err());
        let dims = [(448u32, 448u32), (360, 640)];
        let n: usize = dims
            .iter()
            .map(|&(h, w)| image_tokens(h, w, &cfg()).unwrap())
            .sum();
        assert_eq!(expanded_prompt_len(10, &dims, &cfg()), Ok(10 - 2 + n));
    }

    #[test]
    fn pack_drops_padding_rows_and_closes_rows_with_newline() {
        // 2x2 tokens per view, a 1x2 tile grid.
        let lay = Layout {
            best: (2, 4),
            grid: (1, 2),
            kept: (1, 4),
            per_view: 4,
        };
        // feat[view][pos] = 10 * view + pos, d = 1.
        let feat: Vec<f32> = (0..3)
            .flat_map(|v| (0..4).map(move |p| (10 * v + p) as f32))
            .collect();
        let out = pack_features(&feat, 1, &lay, &[-1.0]).unwrap();
        // base 0..4; stitched grid rows: [10 11 20 21] / [12 13 22 23];
        // kept 1 of 2 rows: pad = (2 - 1) / 2 = 0, so the first.
        assert_eq!(out, vec![0., 1., 2., 3., 10., 11., 20., 21., -1.]);
    }

    #[test]
    fn preprocess_tiles_follow_the_layout() {
        let c = cfg();
        let rgb = [255u8, 0, 0].repeat(40 * 30);
        let v = preprocess(&rgb, 40, 30, &c).unwrap();
        assert_eq!(v.layout.views(), 3);
        assert_eq!(v.pixel_values.len(), 3 * 3 * 336 * 336);
        // The padded corner of the first tile is black.
        let black = (0.0f32 - c.image_mean[0]) / c.image_std[0];
        assert_eq!(v.pixel_values[336 * 336 * 3], black);
    }

    #[test]
    fn export_config_is_checked() {
        let pre = serde_json::json!({
            "crop_size": {"height": 336, "width": 336}, "size": {"shortest_edge": 336},
            "image_grid_pinpoints": [[336, 672], [672, 336]],
        });
        let vlm = serde_json::json!({
            "image_grid_pinpoints": [[336, 672], [672, 336]],
            "vision_config": {"image_size": 336, "patch_size": 14},
        });
        let proc =
            serde_json::json!({"patch_size": 14, "vision_feature_select_strategy": "default"});
        let c = PreprocessConfig::from_export(&pre, Some(&proc), &vlm).unwrap();
        assert_eq!(c.pinpoints, vec![(336, 672), (672, 336)]);
        let full = serde_json::json!({"vision_feature_select_strategy": "full"});
        assert!(PreprocessConfig::from_export(&pre, Some(&full), &vlm).is_err());
        let other = serde_json::json!({"image_grid_pinpoints": [[336, 672]]});
        assert!(PreprocessConfig::from_export(&pre, None, &other).is_err());
    }

    /// LLaVA-NeXT images key the KV cache like any other family: every soft
    /// token (newline rows included) is one image position.
    #[test]
    fn vision_kv_spans_cover_newline_rows() {
        let c = cfg();
        let counts = [
            image_tokens(30, 40, &c).unwrap(),
            image_tokens(4, 1500, &c).unwrap(),
        ];
        let ids = expand_image_tokens(&[1, 9, 5, 9, 2], 9, &counts).unwrap();
        let spans = crate::vision_kv::ImageSpans::new(&ids, 9, &counts).unwrap();
        assert!(spans.splits(3));
        assert!(!spans.splits(1 + counts[0]));
        assert!(!spans.splits(2 + counts[0]));
    }
}
