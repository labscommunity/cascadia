//! Host-side Phi-3.5-vision image input (`phi3_v`, e.g.
//! OpenVINO/Phi-3.5-vision-instruct-int4-ov): the work Hugging Face's
//! `Phi3VImageProcessor`, `Phi3VProcessor` and `Phi3ImageEmbedding` do around
//! the two vision IRs, for rank 0 of a staged `ov-runtime` tree.
//!
//! The export splits the image path in three. The vision IR is the CLIP
//! tower (`pixel_values [N, 3, 336, 336]` to `[N, 576, 1024]`: layer -2,
//! class token dropped); the projection IR is the MLP (`[1, T, 4096]` to
//! `[1, T, 3072]`). Everything between them is host work, done here:
//!
//! - [`hd_geometry`] / [`preprocess`]: the HD transform. Transpose a portrait
//!   image, scale the long side to the largest `k * 336` with
//!   `k * ceil(k / ratio) <= num_crops` (Pillow bilinear, as torchvision
//!   resizes a PIL image), pad the short side to whole 336-px crops with
//!   white (centred), transpose back; normalise with the CLIP mean/std. The
//!   global view is that image resized to 336x336 by torch's bicubic
//!   (`F.interpolate`, no antialias) on the normalised tensor. The vision IR
//!   sees the global view first, then the crops row-major.
//! - [`assemble`]: `hd_feature_transform`: each 24x24 patch grid merged 2x2
//!   into 12x12 rows of 4096, the crops tiled into one grid, each grid row
//!   closed by `sub_GN`, then `glb_GN`, then the global view's 12 rows (each
//!   closed by `sub_GN`). `glb_GN` / `sub_GN` are learned rows from
//!   `config.json`.
//! - [`encode_prompt`] / [`expand_image_tokens`]: `Phi3VProcessor` splits the
//!   prompt at each `<|image_N|>` and tokenizes every chunk on its own (each
//!   gets the tokenizer's BOS), with the image's soft tokens between. Rank 0
//!   stands one `<|image|>` in for each tag and expands it to the image's
//!   `(crops + 1) * 144 + 1 + (rows + 1) * 12` soft tokens.
//!
//! Parity with the remote-code processor (transformers 5.14.1, Pillow 12.3,
//! torch 2.14) is pinned by `tests/phi3v_vision_parity.rs` against
//! `tests/fixtures/phi3v_vision/`.

use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::resample::{self, Filter};

/// Side of one crop and of the global view.
pub const CROP: u32 = 336;
/// `<|image_` .. `|>`: the processor's per-image tag.
const TAG_OPEN: &str = "<|image_";
const TAG_CLOSE: &str = "|>";

/// `preprocessor_config.json` (and `config.json` `img_processor`) of a
/// Phi-3.5-vision export.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessConfig {
    /// HD transform budget: the image is cut into at most this many crops.
    pub num_crops: u32,
    /// Soft tokens per 336-px view after the 2x2 merge (12 x 12).
    pub num_img_tokens: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl PreprocessConfig {
    /// Phi-3.5-vision-instruct as shipped.
    pub fn phi35() -> Self {
        Self {
            num_crops: 4,
            num_img_tokens: 144,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_55, 0.261_302_6, 0.275_777_1],
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        let mut c = Self::phi35();
        if let Some(n) = v.get("num_crops").and_then(serde_json::Value::as_u64) {
            c.num_crops = n as u32;
        }
        if let Some(n) = v.get("num_img_tokens").and_then(serde_json::Value::as_u64) {
            c.num_img_tokens = n as u32;
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
        if c.num_crops == 0 || c.num_img_tokens != 144 {
            return Err(format!(
                "{}: num_crops {} / num_img_tokens {}: only 144 tokens per 336-px view \
                 (CLIP ViT-L/14-336 merged 2x2) is supported",
                path.display(),
                c.num_crops,
                c.num_img_tokens
            ));
        }
        Ok(c)
    }
}

/// `HD_transform`'s geometry for a `width x height` image. Sizes are in the
/// transform's own frame: a portrait image is transposed first, so `new_w`
/// is the long side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HdGeometry {
    pub transposed: bool,
    /// Resized size, transposed frame.
    pub new_w: u32,
    pub new_h: u32,
    /// Padded short side (a multiple of 336) and its top padding.
    pub padded_h: u32,
    pub top: u32,
}

impl HdGeometry {
    /// `(height, width)` of the padded image, original orientation (the
    /// processor's `image_sizes`).
    pub fn image_size(&self) -> (u32, u32) {
        if self.transposed {
            (self.new_w, self.padded_h)
        } else {
            (self.padded_h, self.new_w)
        }
    }

    /// Crop grid `(rows, cols)`.
    pub fn crops(&self) -> (u32, u32) {
        let (h, w) = self.image_size();
        (h / CROP, w / CROP)
    }

    /// Soft tokens the image occupies: `(rows * cols + 1) * 144 + 1 +
    /// (rows + 1) * 12` (the processor's `num_img_tokens`).
    pub fn num_tokens(&self) -> usize {
        let (r, c) = self.crops();
        ((r * c + 1) * 144 + 1 + (r + 1) * 12) as usize
    }
}

/// `calc_hd_transform_size` / `HD_transform` for `hd_num` crops.
pub fn hd_geometry(width: u32, height: u32, hd_num: u32) -> Result<HdGeometry, String> {
    if width == 0 || height == 0 {
        return Err(format!("empty image {width}x{height}"));
    }
    let transposed = width < height;
    let (w, h) = if transposed {
        (height, width)
    } else {
        (width, height)
    };
    let ratio = f64::from(w) / f64::from(h);
    let mut scale = 1u32;
    while f64::from(scale) * (f64::from(scale) / ratio).ceil() <= f64::from(hd_num) {
        scale += 1;
    }
    scale -= 1;
    let new_w = scale * CROP;
    let new_h = (f64::from(new_w) / ratio) as u32;
    if new_w == 0 || new_h == 0 {
        return Err(format!(
            "image {width}x{height} is too narrow for {hd_num} crops"
        ));
    }
    let padded_h = new_h.div_ceil(CROP) * CROP;
    let top = (padded_h - new_h) / 2;
    Ok(HdGeometry {
        transposed,
        new_w,
        new_h,
        padded_h,
        top,
    })
}

/// Soft tokens an image of `width x height` takes, from its size alone.
pub fn num_image_tokens(width: u32, height: u32, cfg: &PreprocessConfig) -> Result<usize, String> {
    Ok(hd_geometry(width, height, cfg.num_crops)?.num_tokens())
}

/// The vision IR's input for one image.
#[derive(Clone, Debug)]
pub struct Pixels {
    /// `[1 + rows * cols, 3, 336, 336]` f32: the global view, then the
    /// crops row-major.
    pub pixel_values: Vec<f32>,
    pub geometry: HdGeometry,
}

impl Pixels {
    pub fn views(&self) -> usize {
        let (r, c) = self.geometry.crops();
        1 + (r * c) as usize
    }
}

fn transpose_rgb8(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            let s = (y * w + x) * 3;
            let d = (x * h + y) * 3;
            out[d..d + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    out
}

/// torch's `get_cubic_upsample_coefficients` (A = -0.75), in f32.
fn cubic_coeffs(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let conv1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let conv2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let x2 = 1.0 - t;
    [conv2(t + 1.0), conv1(t), conv1(x2), conv2(x2 + 1.0)]
}

/// Per output index: the 4 clamped input indices and their weights, as
/// torch's `HelperInterpCubic` computes them (`align_corners=False`).
fn cubic_taps(in_size: usize, out_size: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = in_size as f32 / out_size as f32;
    (0..out_size)
        .map(|i| {
            let real = scale * (i as f32 + 0.5) - 0.5;
            let idx = (real.floor() as i64).min(in_size as i64 - 1);
            let lambda = (real - idx as f32).clamp(0.0, 1.0);
            let w = cubic_coeffs(lambda);
            let ids = [0i64, 1, 2, 3].map(|j| (idx - 1 + j).clamp(0, in_size as i64 - 1) as usize);
            (ids, w)
        })
        .collect()
}

/// `F.interpolate(x, size=(oh, ow), mode="bicubic")` on one channel plane,
/// as torch's CPU kernel computes it in f32: for each output, the sum over
/// the 4 source rows of the row weight times the 4-tap horizontal sum.
pub fn torch_bicubic_plane(src: &[f32], h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
    let ty = cubic_taps(h, oh);
    let tx = cubic_taps(w, ow);
    let mut out = vec![0f32; oh * ow];
    for (oy, (yi, yw)) in ty.iter().enumerate() {
        for (ox, (xi, xw)) in tx.iter().enumerate() {
            let mut acc = 0f32;
            for k in 0..4 {
                let row = &src[yi[k] * w..(yi[k] + 1) * w];
                let mut t = row[xi[0]] * xw[0];
                for j in 1..4 {
                    t += row[xi[j]] * xw[j];
                }
                if k == 0 {
                    acc = t * yw[0];
                } else {
                    acc += t * yw[k];
                }
            }
            out[oy * ow + ox] = acc;
        }
    }
    out
}

/// Preprocess one packed RGB8 image exactly as `Phi3VImageProcessor` does
/// (the global view to torch's bicubic within f32 rounding).
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Pixels, String> {
    if rgb.len() != width as usize * height as usize * 3 {
        return Err(format!("image is not {width}x{height} RGB8"));
    }
    let g = hd_geometry(width, height, cfg.num_crops)?;
    // HD_transform, in the transposed frame: resize, pad top/bottom white.
    let (fw, fh) = if g.transposed {
        (height as usize, width as usize)
    } else {
        (width as usize, height as usize)
    };
    let framed = if g.transposed {
        transpose_rgb8(rgb, width as usize, height as usize)
    } else {
        rgb.to_vec()
    };
    let resized = resample::resize_rgb8(
        &framed,
        fw,
        fh,
        g.new_w as usize,
        g.new_h as usize,
        Filter::Bilinear,
    );
    let row = g.new_w as usize * 3;
    let mut padded = vec![255u8; row * g.padded_h as usize];
    let top = g.top as usize * row;
    padded[top..top + resized.len()].copy_from_slice(&resized);
    let (h, w) = g.image_size();
    let (h, w) = (h as usize, w as usize);
    let hd = if g.transposed {
        transpose_rgb8(&padded, g.new_w as usize, g.padded_h as usize)
    } else {
        padded
    };
    // ToTensor (f32 / 255) and Normalize, channels first.
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|v| (v as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c])
    });
    let plane = h * w;
    let mut norm = vec![0f32; 3 * plane];
    for (i, px) in hd.chunks_exact(3).enumerate() {
        for c in 0..3 {
            norm[c * plane + i] = lut[c][px[c] as usize];
        }
    }
    let crop = CROP as usize;
    let (rows, cols) = (h / crop, w / crop);
    let view = 3 * crop * crop;
    let mut out = vec![0f32; (1 + rows * cols) * view];
    for c in 0..3 {
        let g = torch_bicubic_plane(&norm[c * plane..(c + 1) * plane], h, w, crop, crop);
        out[c * crop * crop..(c + 1) * crop * crop].copy_from_slice(&g);
    }
    for r in 0..rows {
        for col in 0..cols {
            let base = (1 + r * cols + col) * view;
            for c in 0..3 {
                for y in 0..crop {
                    let src = c * plane + (r * crop + y) * w + col * crop;
                    let dst = base + (c * crop + y) * crop;
                    out[dst..dst + crop].copy_from_slice(&norm[src..src + crop]);
                }
            }
        }
    }
    Ok(Pixels {
        pixel_values: out,
        geometry: g,
    })
}

/// `hd_feature_transform` minus the projection: the vision IR's rows
/// `feat` (`[1 + rows * cols, L, d]`, the global view first, `L` a square
/// patch grid) to the projection IR's input `[tokens, 4 * d]`.
pub fn assemble(
    feat: &[f32],
    d: usize,
    crops: (usize, usize),
    sub_gn: &[f32],
    glb_gn: &[f32],
) -> Result<Vec<f32>, String> {
    let (rows, cols) = crops;
    let views = 1 + rows * cols;
    if d == 0 || !feat.len().is_multiple_of(views * d) {
        return Err(format!(
            "{} vision values for {views} views of width {d}",
            feat.len()
        ));
    }
    let l = feat.len() / (views * d);
    let side = (l as f64).sqrt() as usize;
    if side * side != l || !side.is_multiple_of(2) {
        return Err(format!("{l} patches per view is not an even square grid"));
    }
    let width = 4 * d;
    if sub_gn.len() != width || glb_gn.len() != width {
        return Err(format!(
            "separators are {} / {} wide, expected {width}",
            sub_gn.len(),
            glb_gn.len()
        ));
    }
    let half = side / 2;
    // One merged token: patches (2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1).
    let token = |out: &mut Vec<f32>, view: usize, r: usize, c: usize| {
        for hp in 0..2 {
            for wp in 0..2 {
                let p = (2 * r + hp) * side + 2 * c + wp;
                let s = (view * l + p) * d;
                out.extend_from_slice(&feat[s..s + d]);
            }
        }
    };
    let total = rows * half * (cols * half + 1) + 1 + half * (half + 1);
    let mut out = Vec::with_capacity(total * width);
    for gr in 0..rows * half {
        for gc in 0..cols * half {
            let view = 1 + (gr / half) * cols + gc / half;
            token(&mut out, view, gr % half, gc % half);
        }
        out.extend_from_slice(sub_gn);
    }
    out.extend_from_slice(glb_gn);
    for r in 0..half {
        for c in 0..half {
            token(&mut out, 0, r, c);
        }
        out.extend_from_slice(sub_gn);
    }
    debug_assert_eq!(out.len(), total * width);
    Ok(out)
}

/// The prompt cut at each `<|image_N|>` tag: the text chunks (one more than
/// the tags) and each tag's N.
pub fn split_image_tags(prompt: &str) -> Result<(Vec<&str>, Vec<usize>), String> {
    let mut chunks = Vec::new();
    let mut ids = Vec::new();
    let mut rest = prompt;
    let mut start = 0usize;
    let mut cursor = 0usize;
    while let Some(at) = rest.find(TAG_OPEN) {
        let after = &rest[at + TAG_OPEN.len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after[digits..].starts_with(TAG_CLOSE) {
            let n: usize = after[..digits]
                .parse()
                .map_err(|e| format!("image tag number: {e}"))?;
            chunks.push(&prompt[start..cursor + at]);
            ids.push(n);
            let len = TAG_OPEN.len() + digits + TAG_CLOSE.len();
            start = cursor + at + len;
            cursor = start;
            rest = &prompt[start..];
        } else {
            let skip = at + TAG_OPEN.len();
            cursor += skip;
            rest = &rest[skip..];
        }
    }
    chunks.push(&prompt[start..]);
    Ok((chunks, ids))
}

/// `Phi3VProcessor`'s prompt layout on token ids: every chunk between the
/// `<|image_N|>` tags tokenized on its own by `encode` (which must add the
/// tokenizer's special tokens, i.e. its BOS, as the processor's
/// `tokenizer(chunk)` does), one `image` id standing in for each tag. Tags
/// must be `<|image_1|>` .. `<|image_n|>` in order, each once (what the API
/// writes). A prompt without tags is one chunk.
pub fn encode_prompt(
    prompt: &str,
    image: i64,
    mut encode: impl FnMut(&str) -> Result<Vec<i64>, String>,
) -> Result<Vec<i64>, String> {
    let (chunks, ids) = split_image_tags(prompt)?;
    if let Some((i, &n)) = ids.iter().enumerate().find(|&(i, &n)| n != i + 1) {
        return Err(format!(
            "image tag {} is <|image_{n}|>; tags must run <|image_1|>, <|image_2|>, ... in order",
            i + 1
        ));
    }
    let mut out = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 {
            out.push(image);
        }
        out.extend(encode(chunk)?);
    }
    Ok(out)
}

/// Each `image` id of an [`encode_prompt`] layout becomes its image's
/// `counts[i]` soft tokens.
pub fn expand_image_tokens(ids: &[i64], image: i64, counts: &[usize]) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&x| x == image).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} image placeholders for {} images",
            counts.len()
        ));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut images = counts.iter();
    for &x in ids {
        if x == image {
            let &n = images.next().expect("counted above");
            out.extend(std::iter::repeat_n(image, n));
        } else {
            out.push(x);
        }
    }
    Ok(out)
}

/// A learned separator row from the VLM `config.json` (`glb_GN`, `sub_GN`):
/// nested arrays flattened.
/// `embd_layer.hd_transform_order` of the export's `config.json`. Only
/// `sub_glb` (sub crops, `glb_GN`, then the global view: what `assemble`
/// builds, and what Phi-3.5-vision ships) is served. The reference embedding
/// defaults a missing key to `glb_sub`, so a config without it is refused too
/// rather than assembled in the other order.
pub fn check_hd_transform_order(cfg: &serde_json::Value) -> Result<(), String> {
    match cfg
        .get("embd_layer")
        .and_then(|e| e.get("hd_transform_order"))
        .and_then(serde_json::Value::as_str)
    {
        Some("sub_glb") => Ok(()),
        other => Err(format!(
            "config embd_layer.hd_transform_order is {}; only \"sub_glb\" is served (the \
             reference defaults a missing key to \"glb_sub\")",
            other.map_or_else(|| "missing".to_string(), |o| format!("{o:?}"))
        )),
    }
}

pub fn separator(cfg: &serde_json::Value, key: &str) -> Result<Vec<f32>, String> {
    fn flat(v: &serde_json::Value, out: &mut Vec<f32>) -> bool {
        match v {
            serde_json::Value::Array(a) => a.iter().all(|x| flat(x, out)),
            serde_json::Value::Number(n) => n.as_f64().map(|f| out.push(f as f32)).is_some(),
            _ => false,
        }
    }
    let v = cfg
        .get(key)
        .ok_or_else(|| format!("config lacks {key} (the HD separator rows)"))?;
    let mut out = Vec::new();
    if !flat(v, &mut out) || out.is_empty() {
        return Err(format!("config {key} is not a numeric array"));
    }
    Ok(out)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Rank 0's Phi-3.5-vision front end: the two vision IRs, the separators,
/// the processor settings and the soft-token id.
pub struct Frontend {
    vision: Runtime,
    projection: Runtime,
    projection_input: String,
    sub_gn: Vec<f32>,
    glb_gn: Vec<f32>,
    /// Encoded images by content hash. Metadata: the soft-token count.
    cache: crate::image_cache::EncodeCache<usize>,
    pub preprocess: PreprocessConfig,
    /// `<|image|>`: the id each image's soft tokens carry.
    pub image_token: i64,
}

impl Frontend {
    /// Compile `openvino_vision_embeddings_model.xml` and
    /// `openvino_vision_projection_model.xml` from `dir` on the CPU; the
    /// separators come from `vlm_config` (the export's `config.json`).
    pub fn load(
        dir: &Path,
        vlm_config: &serde_json::Value,
        image_token: i64,
    ) -> Result<Self, String> {
        let preprocess = PreprocessConfig::from_json_file(&dir.join("preprocessor_config.json"))?;
        check_hd_transform_order(vlm_config)?;
        let compile = |name: &str| {
            let path = dir.join(name);
            Runtime::compile(&path.to_string_lossy(), "CPU", &PluginConfig::new())
                .map_err(|e| format!("compiling {}: {e}", path.display()))
        };
        let vision = compile("openvino_vision_embeddings_model.xml")?;
        let projection = compile("openvino_vision_projection_model.xml")?;
        let projection_input = projection
            .input_name(0)
            .map_err(|e| format!("projection input: {e}"))?;
        let sub_gn = separator(vlm_config, "sub_GN")?;
        let glb_gn = separator(vlm_config, "glb_GN")?;
        if sub_gn.len() != glb_gn.len() {
            return Err(format!(
                "sub_GN has {} values, glb_GN {}",
                sub_gn.len(),
                glb_gn.len()
            ));
        }
        Ok(Self {
            vision,
            projection,
            projection_input,
            sub_gn,
            glb_gn,
            cache: crate::image_cache::EncodeCache::from_env(),
            preprocess,
            image_token,
        })
    }

    /// The projected rows (`[tokens, hidden]`) for a preprocessed image.
    pub fn encode(&mut self, p: &Pixels) -> Result<Vec<f32>, String> {
        let views = p.views();
        let crop = CROP as usize;
        self.vision
            .set_input(
                "pixel_values",
                DType::F32,
                &[views, 3, crop, crop],
                &f32_bytes(&p.pixel_values),
            )
            .and_then(|_| self.vision.infer())
            .map_err(|e| format!("vision embedding: {e}"))?;
        let (dtype, shape, bytes) = self
            .vision
            .output(0)
            .map_err(|e| format!("vision embedding: {e}"))?;
        if dtype != DType::F32 {
            return Err(format!(
                "vision embedding output is {dtype:?}, expected f32"
            ));
        }
        let d = match shape[..] {
            [n, _, d] if n == views => d,
            _ => {
                return Err(format!(
                    "vision embedding returned {shape:?} for {views} views"
                ))
            }
        };
        let (r, c) = p.geometry.crops();
        let merged = assemble(
            &f32s(&bytes),
            d,
            (r as usize, c as usize),
            &self.sub_gn,
            &self.glb_gn,
        )?;
        let n = p.geometry.num_tokens();
        self.projection
            .set_input(
                &self.projection_input,
                DType::F32,
                &[1, n, 4 * d],
                &f32_bytes(&merged),
            )
            .and_then(|_| self.projection.infer())
            .map_err(|e| format!("vision projection: {e}"))?;
        let (dtype, shape, bytes) = self
            .projection
            .output(0)
            .map_err(|e| format!("vision projection: {e}"))?;
        if dtype != DType::F32 {
            return Err(format!("projection output is {dtype:?}, expected f32"));
        }
        match shape[..] {
            [1, t, _] | [t, _] if t == n => Ok(f32s(&bytes)),
            _ => Err(format!("projection returned {shape:?} for {n} tokens")),
        }
    }

    /// The prompt's ids in the processor's layout (see [`encode_prompt`]).
    pub fn encode_prompt(
        &self,
        tok: &tokenizers::Tokenizer,
        prompt: &str,
    ) -> Result<Vec<i64>, String> {
        encode_prompt(prompt, self.image_token, |chunk| {
            tok.encode(chunk, true)
                .map(|e| e.get_ids().iter().map(|&u| i64::from(u)).collect())
                .map_err(|e| format!("tokenizer encode: {e}"))
        })
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

    /// The expanded prompt ids `prepare` would return, from each image's
    /// size, with no encoder work.
    pub fn expand_ids(
        &self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        expand_image_tokens(prompt_ids, self.image_token, &self.counts(media)?)
    }

    /// Length of a `prompt_len`-id [`encode_prompt`] layout once `media`
    /// expand.
    pub fn expanded_len(
        &self,
        prompt_len: usize,
        media: &[cascadia_types::MediaInput],
    ) -> Result<usize, String> {
        let extra: usize = self.counts(media)?.iter().sum();
        Ok((prompt_len + extra).saturating_sub(media.len()))
    }

    /// Encode a task's images and expand each `<|image|>` into its soft
    /// tokens. Returns the expanded ids and each image's rows, in prompt
    /// order.
    pub fn prepare(
        &mut self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
        tenant: &str,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
        // Refuse a malformed layout before any encoder work.
        let counts = self.counts(media)?;
        expand_image_tokens(prompt_ids, self.image_token, &counts)?;
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
        let ids = expand_image_tokens(prompt_ids, self.image_token, &counts)?;
        Ok((ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hd_geometry_matches_calc_hd_transform_size() {
        // (w, h) -> (padded h, w, tokens), from the processor's formulas.
        let g = hd_geometry(448, 448, 4).unwrap();
        assert_eq!((g.new_w, g.new_h, g.padded_h, g.top), (672, 672, 672, 0));
        assert_eq!(g.image_size(), (672, 672));
        assert_eq!(g.num_tokens(), 757);
        // Landscape 640x360: ratio 1.78, scale 2 -> 672 x 378 -> pad to 672.
        let g = hd_geometry(640, 360, 4).unwrap();
        assert_eq!((g.new_w, g.new_h, g.padded_h, g.top), (672, 378, 672, 147));
        assert_eq!(g.crops(), (2, 2));
        // Portrait 300x1000 is transposed: ratio 3.33, scale 3 (3*1 <= 4).
        let g = hd_geometry(300, 1000, 4).unwrap();
        assert!(g.transposed);
        assert_eq!((g.new_w, g.new_h, g.padded_h), (1008, 302, 336));
        assert_eq!(g.image_size(), (1008, 336));
        assert_eq!(g.crops(), (3, 1));
        assert_eq!(g.num_tokens(), (3 + 1) * 144 + 1 + 4 * 12);
        assert!(hd_geometry(0, 4, 4).is_err());
    }

    #[test]
    fn tags_split_the_prompt_and_numbers_must_run_in_order() {
        let (c, n) = split_image_tags("a<|image_1|>\nb<|image_2|>c").unwrap();
        assert_eq!(c, vec!["a", "\nb", "c"]);
        assert_eq!(n, vec![1, 2]);
        // Not a tag: no digits, or no close.
        let (c, n) = split_image_tags("x<|image_|> <|image_3 y").unwrap();
        assert_eq!(c, vec!["x<|image_|> <|image_3 y"]);
        assert!(n.is_empty());
        let enc = |s: &str| Ok(std::iter::once(1).chain(s.bytes().map(i64::from)).collect());
        // Every chunk, empty ones too, carries its own BOS (1).
        let ids = encode_prompt("<|image_1|>\nhi", 99, enc).unwrap();
        assert_eq!(ids, vec![1, 99, 1, 10, 104, 105]);
        assert!(encode_prompt("<|image_2|>", 99, enc).is_err());
        assert!(encode_prompt("<|image_1|><|image_1|>", 99, enc).is_err());
        assert_eq!(encode_prompt("plain", 99, enc).unwrap()[0], 1);
    }

    #[test]
    fn expansion_puts_each_images_count_in_place() {
        let e = expand_image_tokens(&[1, 9, 2, 9, 3], 9, &[2, 3]).unwrap();
        assert_eq!(e, vec![1, 9, 9, 2, 9, 9, 9, 3]);
        assert!(expand_image_tokens(&[1, 9], 9, &[1, 1]).is_err());
    }

    /// `reshape_hd_patches_2x2merge` + `add_image_newline` on a 2-wide grid
    /// of 4x4-patch views with d = 1: each token is (r, c) block's patches
    /// (2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1).
    #[test]
    fn assemble_tiles_crops_and_closes_rows() {
        let (side, d) = (4usize, 1usize);
        let views = 3; // global + 1x2 crops
        let feat: Vec<f32> = (0..views * side * side)
            .map(|i| (100 * (i / (side * side)) + i % (side * side)) as f32)
            .collect();
        let sub = [-1.0f32; 4];
        let glb = [-2.0f32; 4];
        let out = assemble(&feat, d, (1, 2), &sub, &glb).unwrap();
        // 2 grid rows of (2 crops * 2 tokens + newline), glb, 2 rows of 3.
        assert_eq!(out.len(), (2 * 5 + 1 + 2 * 3) * 4);
        // Row 0: crop 1 token (0,0), (0,1); crop 2 token (0,0), (0,1); sub.
        assert_eq!(&out[..4], &[100.0, 101.0, 104.0, 105.0]);
        assert_eq!(&out[4..8], &[102.0, 103.0, 106.0, 107.0]);
        assert_eq!(&out[8..12], &[200.0, 201.0, 204.0, 205.0]);
        assert_eq!(&out[16..20], &sub);
        // Then glb_GN and the global view's first token.
        assert_eq!(&out[40..44], &glb);
        assert_eq!(&out[44..48], &[0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn torch_bicubic_keeps_flat_planes_and_identity() {
        let flat = vec![0.25f32; 40 * 30];
        let out = torch_bicubic_plane(&flat, 30, 40, 7, 9);
        assert!(out.iter().all(|&v| (v - 0.25).abs() < 1e-6));
        let ramp: Vec<f32> = (0..64).map(|i| i as f32).collect();
        assert_eq!(torch_bicubic_plane(&ramp, 8, 8, 8, 8), ramp);
    }

    #[test]
    fn preprocess_shapes_views_and_pads_white() {
        let cfg = PreprocessConfig::phi35();
        // 640x200: ratio 3.2 -> scale 3? 3*ceil(3/3.2)=3 <= 4, 4*2=8 > 4.
        let p = preprocess(&vec![0u8; 640 * 200 * 3], 640, 200, &cfg).unwrap();
        assert_eq!(p.geometry.image_size(), (336, 1008));
        assert_eq!(p.views(), 4);
        assert_eq!(p.pixel_values.len(), 4 * 3 * 336 * 336);
        // Crop 1's first row is padding: white, normalised.
        let white = (1.0 - cfg.image_mean[0]) / cfg.image_std[0];
        let crop1 = 3 * 336 * 336;
        assert_eq!(p.pixel_values[crop1], white);
        let black = (0.0 - cfg.image_mean[0]) / cfg.image_std[0];
        assert_eq!(p.pixel_values[crop1 + 168 * 336], black);
        assert_eq!(
            num_image_tokens(640, 200, &cfg).unwrap(),
            p.geometry.num_tokens()
        );
    }

    #[test]
    fn separators_flatten_nested_arrays() {
        let v = serde_json::json!({"sub_GN": [[[[1.0, 2.0]]]], "glb_GN": [[[3.0]]]});
        assert_eq!(separator(&v, "sub_GN").unwrap(), vec![1.0, 2.0]);
        assert_eq!(separator(&v, "glb_GN").unwrap(), vec![3.0]);
        assert!(separator(&v, "nope").is_err());
    }

    #[test]
    fn only_the_sub_glb_hd_transform_order_is_served() {
        let order = |o: serde_json::Value| {
            check_hd_transform_order(&serde_json::json!({"embd_layer": {"hd_transform_order": o}}))
        };
        assert!(order(serde_json::json!("sub_glb")).is_ok());
        assert!(order(serde_json::json!("glb_sub")).is_err());
        assert!(order(serde_json::json!(1)).is_err());
        let e = check_hd_transform_order(&serde_json::json!({"embd_layer": {}})).unwrap_err();
        assert!(e.contains("missing"), "{e}");
        assert!(check_hd_transform_order(&serde_json::json!({})).is_err());
    }
}
