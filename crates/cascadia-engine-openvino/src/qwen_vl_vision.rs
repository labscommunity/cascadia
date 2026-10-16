//! Qwen2.5-VL and Qwen3-VL images on the staged `ov-runtime` engine: rank 0's
//! front end and the M-RoPE positions every stage is fed.
//!
//! Both families reuse the Qwen3.5 machinery ([`crate::qwen35_vision`]):
//! the same Qwen2-VL image processor (`smart_resize`, Pillow-exact bicubic,
//! temporal patch 2, merge 2; only patch size and mean/std differ and come
//! from `preprocessor_config.json`), the same pad expansion and the same
//! `get_rope_index` for images. What differs:
//!
//! - **Positions.** The language model takes HF's three rows
//!   `position_ids [3, B, T]` (temporal, height, width), not Qwen3.5's four.
//!   For an image, transformers 5.14.1 `Qwen2_5_VLModel.get_rope_index` and
//!   `Qwen3VLModel.get_rope_index` both give `t = start` (a still image has
//!   one temporal step, so Qwen2.5-VL's `tokens_per_second` spacing never
//!   applies), `h/w = start + row/col` and advance by `max(h, w) / merge`,
//!   which is what [`crate::qwen35_vision::rope::mrope_positions`] computes.
//!   Decode positions are `index + rope_delta` on all three rows (GenAI
//!   `get_generation_phase_position_ids`). How the rows split the rotary
//!   frequencies (Qwen2.5-VL `mrope_section [16, 24, 24]` in contiguous
//!   blocks, Qwen3-VL `[24, 20, 20]` interleaved) happens inside the
//!   language model IR, so the host feeds the same rows to both.
//! - **Qwen2.5-VL vision IRs.** Two IRs: patch embedding, then the merger
//!   (all 32 blocks, window attention and the 2x2 merger), which takes the
//!   per-frame `attention_mask`, a `window_attention_mask`, the `window_index`
//!   that reorders merge units into windows and `rotary_pos_emb`
//!   ([`window_index`], [`block_mask_bytes`], GenAI `qwen2_5_vl/classes.cpp`).
//! - **Qwen3-VL vision IRs** are Qwen3.5's three (patch, learned positions,
//!   merger) plus a second merger output, `deepstack_feature_lists
//!   [3, tokens, hidden]`. The language model IR adds row `k` of it to the
//!   image tokens' hidden states after decoder layer `k` (layers 0, 1, 2;
//!   `deepstack_visual_indexes [5, 11, 17]` are the *vision* layers the
//!   features come from), through two inputs, `visual_pos_masks [B, T]` and
//!   `deepstack_visual_embeds [3, N, hidden]`. Stage 0 holds layers 0-2 (the
//!   cut refuses otherwise) and runs on rank 0, so these never cross the wire.

use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::encoder as q35;
use crate::qwen35_vision::rope::{mrope_positions, MropePositions};
use crate::qwen35_vision::{preprocess, smart_resize, Patches, PreprocessConfig};

/// Rows of the language model's `position_ids` (temporal, height, width).
pub const MROPE_ROWS: usize = 3;

/// Which Qwen VLM a tree carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Qwen2.5-VL: window-attention vision tower, no DeepStack.
    Qwen25Vl,
    /// Qwen3-VL: Qwen3.5's vision tower plus DeepStack features.
    Qwen3Vl,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Qwen25Vl => "Qwen2.5-VL",
            Kind::Qwen3Vl => "Qwen3-VL",
        }
    }
}

/// `get_vision_window_index`: the order that groups merge units into
/// `merger_window x merger_window` windows (per frame, row-major windows,
/// row-major inside), and the cumulative patch counts of the windows.
/// Windows past the grid's edge are partial; empty ones are skipped, which is
/// what HF's `unique_consecutive` does to its padded windows.
pub fn window_index(
    grids: &[[u32; 3]],
    merge: usize,
    merger_window: usize,
) -> (Vec<i64>, Vec<i32>) {
    let unit = (merge * merge) as i32;
    let vw = merger_window.max(1);
    let mut index = Vec::new();
    let mut cu = vec![0i32];
    let mut base = 0usize;
    for &[t, h, w] in grids {
        let (t, lh, lw) = (t as usize, h as usize / merge, w as usize / merge);
        let (nwh, nww) = (lh.div_ceil(vw), lw.div_ceil(vw));
        for tt in 0..t {
            for wh in 0..nwh {
                for ww in 0..nww {
                    let mut count = 0;
                    for y in wh * vw..((wh + 1) * vw).min(lh) {
                        for x in ww * vw..((ww + 1) * vw).min(lw) {
                            index.push((base + tt * lh * lw + y * lw + x) as i64);
                            count += 1;
                        }
                    }
                    if count > 0 {
                        cu.push(cu.last().unwrap() + count * unit);
                    }
                }
            }
        }
        base += t * lh * lw;
    }
    (index, cu)
}

/// A `[1, N, N]` f32 block mask as the little-endian bytes the IR input
/// takes: 0 inside each `[cu[k], cu[k+1])` block, `-inf` elsewhere. Built
/// straight into the byte buffer, with no f32 copy: at ~1 MP (N = 5476
/// patches) one mask is 120 MB, and the merger takes two (per-frame
/// `attention_mask` from `cu_seqlens`, `window_attention_mask` from
/// `cu_window_seqlens`), so the encoder builds, feeds and drops them one at
/// a time. The shim copies each into its own OpenVINO tensor (at most 256 MiB
/// each, which the API's 1 MP default budget keeps well under).
pub fn block_mask_bytes(cu: &[i32]) -> Vec<u8> {
    let n = *cu.last().unwrap_or(&0) as usize;
    let ninf = f32::NEG_INFINITY.to_le_bytes();
    let mut out = Vec::with_capacity(n * n * 4);
    for _ in 0..n * n {
        out.extend_from_slice(&ninf);
    }
    for pair in cu.windows(2) {
        let (a, b) = (pair[0] as usize, pair[1] as usize);
        for row in a..b {
            out[(row * n + a) * 4..(row * n + b) * 4].fill(0);
        }
    }
    out
}

/// The Qwen2.5-VL `vision_config` fields the encoder needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowVisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub window_size: usize,
    pub patch_size: usize,
}

impl WindowVisionConfig {
    pub fn from_value(v: &serde_json::Value) -> Result<Self, String> {
        let vc = v
            .get("vision_config")
            .ok_or("config has no vision_config")?;
        let get = |k: &str| {
            vc.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        Ok(Self {
            hidden_size: get("hidden_size")?,
            num_heads: get("num_heads")?,
            spatial_merge_size: get("spatial_merge_size")?,
            out_hidden_size: get("out_hidden_size")?,
            window_size: get("window_size")?,
            patch_size: get("patch_size")?,
        })
    }

    /// Window side in merge units (112 / 2 / 14 = 4).
    pub fn merger_window(&self) -> usize {
        self.window_size / self.spatial_merge_size / self.patch_size
    }

    /// `rotary_pos_emb` width: half the attention head dimension (40).
    pub fn rotary_dim(&self) -> usize {
        self.hidden_size / self.num_heads / 2
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Qwen2.5-VL's two vision IRs (`openvino_vision_embeddings_model.xml`,
/// `openvino_vision_embeddings_merger_model.xml`) with their host inputs.
pub struct WindowEncoder {
    patch: Runtime,
    merger: Runtime,
    cfg: WindowVisionConfig,
}

impl WindowEncoder {
    pub fn load(
        dir: &Path,
        device: &str,
        plugin: &PluginConfig,
        cfg: WindowVisionConfig,
    ) -> Result<Self, String> {
        let compile = |suffix: &str| {
            let path = dir.join(format!("openvino_vision_embeddings{suffix}_model.xml"));
            Runtime::compile(&path.to_string_lossy(), device, plugin)
                .map_err(|e| format!("compiling {}: {e}", path.display()))
        };
        Ok(Self {
            patch: compile("")?,
            merger: compile("_merger")?,
            cfg,
        })
    }

    /// Image rows `[tokens, out_hidden]` for one preprocessed image.
    pub fn encode(&mut self, p: &Patches) -> Result<Vec<f32>, String> {
        let (n, d) = (p.rows, self.cfg.hidden_size);
        let merge = self.cfg.spatial_merge_size;
        let grids = [p.grid_thw];
        self.patch
            .set_input(
                "hidden_states",
                DType::F32,
                &[p.rows, p.cols],
                &f32_bytes(&p.pixel_values),
            )
            .and_then(|_| self.patch.infer())
            .map_err(|e| format!("patch embedding: {e}"))?;
        let (shape, patch_embeds) = q35::read_f32_at(&self.patch, 0, "patch embedding")?;
        if shape != [n, d] {
            return Err(format!(
                "patch embedding returned {shape:?}, expected [{n}, {d}]"
            ));
        }
        let rot_dim = self.cfg.rotary_dim();
        let rot = q35::rotary_pos_emb(&grids, merge, rot_dim, 10_000.0);
        let (index, cu_window) = window_index(&grids, merge, self.cfg.merger_window());
        let idx: Vec<u8> = index.iter().flat_map(|x| x.to_le_bytes()).collect();
        let m = &mut self.merger;
        m.set_input(
            "hidden_states",
            DType::F32,
            &[n, d],
            &f32_bytes(&patch_embeds),
        )
        // One dense mask alive at a time on the host (see `block_mask_bytes`).
        .and_then(|_| {
            let mask = block_mask_bytes(&q35::cu_seqlens(&grids));
            m.set_input("attention_mask", DType::F32, &[1, n, n], &mask)
        })
        .and_then(|_| {
            let mask = block_mask_bytes(&cu_window);
            m.set_input("window_attention_mask", DType::F32, &[1, n, n], &mask)
        })
        .and_then(|_| m.set_input("window_index", DType::I64, &[index.len()], &idx))
        .and_then(|_| {
            m.set_input(
                "rotary_pos_emb",
                DType::F32,
                &[n, rot_dim],
                &f32_bytes(&rot),
            )
        })
        .and_then(|_| m.infer())
        .map_err(|e| format!("merger: {e}"))?;
        let (shape, rows) = q35::read_f32_at(&self.merger, 0, "merger")?;
        let tokens = n / (merge * merge);
        if shape != [tokens, self.cfg.out_hidden_size] {
            return Err(format!(
                "merger returned {shape:?}, expected [{tokens}, {}]",
                self.cfg.out_hidden_size
            ));
        }
        Ok(rows)
    }
}

enum Encoder {
    Window(WindowEncoder),
    /// Qwen3.5's encoder, which reads the merger's DeepStack output too.
    Deepstack(q35::VisionEncoder),
}

/// The image token ids (`config.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTokens {
    pub image_pad: i64,
    pub video_pad: i64,
    pub vision_start: i64,
    pub vision_end: i64,
}

/// One turn's images, ready to prefill.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// Prompt ids with each `<|image_pad|>` expanded to its image's rows.
    pub ids: Vec<i64>,
    /// Each image's rows `[tokens_i, hidden]`, in prompt order.
    pub rows: Vec<Vec<f32>>,
    /// Qwen3-VL: each image's DeepStack rows `[layers, tokens_i, hidden]`,
    /// layer-major. Empty for Qwen2.5-VL.
    pub deepstack: Vec<Vec<f32>>,
    /// HF's positions over `ids`.
    pub mrope: MropePositions,
}

/// Rank 0's Qwen2.5-VL / Qwen3-VL image front end.
pub struct Frontend {
    pub kind: Kind,
    encoder: Encoder,
    /// Encoded images by (tenant, pixel hash): the rows, then (Qwen3-VL) the
    /// DeepStack rows. Metadata: the grid.
    cache: crate::image_cache::EncodeCache<[u32; 3]>,
    pub preprocess: PreprocessConfig,
    pub tokens: ImageTokens,
    pub merge: u32,
    /// The language model's hidden size (the vision output width).
    pub hidden: usize,
    /// DeepStack layers the language model takes (0 on Qwen2.5-VL).
    pub deepstack_layers: usize,
}

impl Frontend {
    /// Load from a dir holding the vision IRs and `preprocessor_config.json`
    /// (a cut tree's root or the export itself), given the VLM `config.json`
    /// (a tree keeps it as `config.vlm.json`), already parsed.
    pub fn load(dir: &Path, kind: Kind, vlm_config: &serde_json::Value) -> Result<Self, String> {
        let id = |k: &str| {
            vlm_config
                .get(k)
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| format!("config lacks {k}"))
        };
        let tokens = ImageTokens {
            image_pad: id("image_token_id")?,
            video_pad: id("video_token_id")?,
            vision_start: id("vision_start_token_id")?,
            vision_end: id("vision_end_token_id")?,
        };
        let preprocess = PreprocessConfig::from_json_file(&dir.join("preprocessor_config.json"))?;
        let plugin = PluginConfig::new();
        let (encoder, merge, hidden, deepstack_layers) = match kind {
            Kind::Qwen25Vl => {
                let cfg = WindowVisionConfig::from_value(vlm_config)?;
                if cfg.patch_size as u32 != preprocess.patch_size {
                    return Err(format!(
                        "vision patch {} differs from the processor's {}",
                        cfg.patch_size, preprocess.patch_size
                    ));
                }
                let enc = WindowEncoder::load(dir, "CPU", &plugin, cfg)?;
                (
                    Encoder::Window(enc),
                    cfg.spatial_merge_size,
                    cfg.out_hidden_size,
                    0,
                )
            }
            Kind::Qwen3Vl => {
                let cfg = q35::VisionConfig::from_value(vlm_config)?;
                let layers = vlm_config
                    .get("vision_config")
                    .and_then(|v| v.get("deepstack_visual_indexes"))
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                if layers == 0 {
                    return Err(
                        "Qwen3-VL config has no vision_config.deepstack_visual_indexes".into(),
                    );
                }
                let enc = q35::VisionEncoder::load(dir, "CPU", &plugin, cfg)?;
                if !enc.has_deepstack() {
                    return Err(
                        "the Qwen3-VL merger IR has no deepstack_feature_lists output".into(),
                    );
                }
                (
                    Encoder::Deepstack(enc),
                    cfg.spatial_merge_size,
                    cfg.out_hidden_size,
                    layers,
                )
            }
        };
        if merge as u32 != preprocess.merge_size {
            return Err(format!(
                "vision spatial_merge_size {merge} differs from the processor's {}",
                preprocess.merge_size
            ));
        }
        Ok(Self {
            kind,
            encoder,
            cache: crate::image_cache::EncodeCache::from_env(),
            preprocess,
            tokens,
            merge: merge as u32,
            hidden,
            deepstack_layers,
        })
    }

    /// The grid `[1, h, w]` (patches) an image of this size preprocesses to.
    pub fn grid_for_size(&self, width: u32, height: u32) -> Result<[u32; 3], String> {
        grid_for_size(width, height, &self.preprocess)
    }

    /// The expanded prompt `prepare` would return, from image sizes alone (no
    /// encode): each `<|image_pad|>` repeated for its image's tokens.
    pub fn expand_ids(
        &self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        let mut counts = Vec::with_capacity(media.len());
        for (i, m) in media.iter().enumerate() {
            crate::gemma4_vision::check_image(i, m)?;
            let [t, h, w] = self
                .grid_for_size(m.width, m.height)
                .map_err(|e| format!("image {i}: {e}"))?;
            counts.push((t * (h / self.merge) * (w / self.merge)) as usize);
        }
        expand_pads(prompt_ids, self.tokens.image_pad, &counts)
    }

    /// Encode a turn's images (or take them from the cache) and expand the
    /// prompt; positions cover the expanded prompt.
    pub fn prepare(
        &mut self,
        media: &[cascadia_types::MediaInput],
        prompt_ids: &[i64],
        tenant: &str,
    ) -> Result<Prepared, String> {
        let mut rows = Vec::with_capacity(media.len());
        let mut deepstack = Vec::with_capacity(media.len());
        let mut grids = Vec::with_capacity(media.len());
        let mut from_cache = 0;
        let layers = self.deepstack_layers;
        for (i, m) in media.iter().enumerate() {
            crate::gemma4_vision::check_image(i, m)?;
            let (mut all, grid) = match self.cache.get(tenant, &m.sha256) {
                Some(hit) => {
                    from_cache += 1;
                    hit
                }
                None => {
                    let p = preprocess(&[&m.pixels[..]], m.width, m.height, &self.preprocess)
                        .map_err(|e| format!("image {i}: {e}"))?;
                    let all = self.encode(&p).map_err(|e| format!("image {i}: {e}"))?;
                    self.cache.put(tenant, m.sha256, &all, p.grid_thw);
                    (all, p.grid_thw)
                }
            };
            let [t, h, w] = grid;
            let n = (t * (h / self.merge) * (w / self.merge)) as usize * self.hidden;
            if all.len() != n * (1 + layers) {
                return Err(format!(
                    "image {i}: {} encoded values for {} rows",
                    all.len(),
                    n / self.hidden.max(1)
                ));
            }
            deepstack.push(all.split_off(n));
            rows.push(all);
            grids.push(grid);
        }
        self.cache.report(from_cache, media.len());
        let counts: Vec<usize> = rows.iter().map(|r| r.len() / self.hidden).collect();
        let ids = expand_pads(prompt_ids, self.tokens.image_pad, &counts)?;
        let mrope = mrope_positions(
            &ids,
            self.tokens.image_pad,
            self.tokens.video_pad,
            &grids,
            &[],
            self.merge,
        )?;
        if layers == 0 {
            deepstack.clear();
        }
        Ok(Prepared {
            ids,
            rows,
            deepstack,
            mrope,
        })
    }

    /// One preprocessed image's rows `[tokens, hidden]` and (Qwen3-VL) its
    /// DeepStack rows `[layers, tokens, hidden]` (empty otherwise).
    pub fn encode_patches(&mut self, p: &Patches) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut all = self.encode(p)?;
        let n = p.llm_tokens(self.merge) * self.hidden;
        if all.len() < n {
            return Err(format!(
                "{} encoded values for {} rows",
                all.len(),
                n / self.hidden
            ));
        }
        let ds = all.split_off(n);
        Ok((all, ds))
    }

    /// One image's rows followed by its DeepStack rows (Qwen3-VL).
    fn encode(&mut self, p: &Patches) -> Result<Vec<f32>, String> {
        match &mut self.encoder {
            Encoder::Window(e) => e.encode(p),
            Encoder::Deepstack(e) => {
                let tr = e.encode_traced(p)?;
                let (layers, mut ds) = tr
                    .deepstack
                    .ok_or("the merger returned no DeepStack features")?;
                if layers != self.deepstack_layers {
                    return Err(format!(
                        "the merger returned {layers} DeepStack layers, the config names {}",
                        self.deepstack_layers
                    ));
                }
                let mut all = tr.image_embeds;
                all.append(&mut ds);
                Ok(all)
            }
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "{} images, {}-px patches merged {}x{}, {}..{} pixels{}",
            self.kind.name(),
            self.preprocess.patch_size,
            self.merge,
            self.merge,
            self.preprocess.min_pixels,
            self.preprocess.max_pixels,
            if self.deepstack_layers > 0 {
                format!(
                    ", DeepStack into the first {} layers",
                    self.deepstack_layers
                )
            } else {
                String::new()
            }
        )
    }
}

/// `smart_resize` then patches: the grid an image of `width x height` gets.
pub fn grid_for_size(width: u32, height: u32, cfg: &PreprocessConfig) -> Result<[u32; 3], String> {
    let (h, w) = smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    Ok([1, h / cfg.patch_size, w / cfg.patch_size])
}

/// [`crate::qwen35_vision::prompt::expand_pads`] over i64 ids.
pub fn expand_pads(ids: &[i64], pad: i64, counts: &[usize]) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&t| t == pad).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} image placeholders but {} images were attached",
            counts.len()
        ));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut next = counts.iter();
    for &t in ids {
        if t == pad {
            out.extend(std::iter::repeat_n(
                pad,
                *next.next().expect("counted above"),
            ));
        } else {
            out.push(t);
        }
    }
    Ok(out)
}

/// The `[3, t1 - t0]` row-major positions of absolute indices `[t0, t1)`:
/// the prompt's M-RoPE rows inside `mrope`, `index + rope_delta` after it.
/// `None` (a text-only turn) is the plain index on every row.
pub fn positions(mrope: Option<&MropePositions>, t0: usize, t1: usize) -> Vec<i64> {
    let n = t1.saturating_sub(t0);
    let mut out = vec![0i64; MROPE_ROWS * n];
    for (j, t) in (t0..t1).enumerate() {
        for r in 0..MROPE_ROWS {
            out[r * n + j] = match mrope {
                Some(m) if t < m.len() => m.thw[r][t],
                Some(m) => t as i64 + m.rope_delta,
                None => t as i64,
            };
        }
    }
    out
}

/// Stage 0's DeepStack inputs for the fed `ids`: `visual_pos_masks [T]`
/// (1 on image tokens, as i64; the cut converts it to the IR's boolean) and
/// `deepstack_visual_embeds [layers, N, hidden]` row-major, `N` the image
/// tokens fed, taken in prompt order from each image's layer-major rows.
/// With no image fed (a text turn, a decode step, even one that sampled the
/// pad id), the mask is all 0 and each layer gets one zero row (GenAI does the
/// same: the IR gathers row 0 and masks it out).
pub fn deepstack_inputs(
    ids: &[i64],
    image_pad: i64,
    images: &[Vec<f32>],
    layers: usize,
    hidden: usize,
) -> Result<(Vec<i64>, Vec<f32>, usize), String> {
    if images.is_empty() {
        return Ok((vec![0; ids.len()], vec![0f32; layers * hidden], 1));
    }
    let mask: Vec<i64> = ids.iter().map(|&t| i64::from(t == image_pad)).collect();
    let n: usize = mask.iter().filter(|&&m| m == 1).count();
    let per: Vec<usize> = images
        .iter()
        .map(|d| d.len() / (layers * hidden).max(1))
        .collect();
    if per.iter().sum::<usize>() != n
        || images
            .iter()
            .zip(&per)
            .any(|(d, &k)| d.len() != layers * k * hidden)
    {
        return Err(format!(
            "{n} image tokens fed, the DeepStack rows cover {}",
            per.iter().sum::<usize>()
        ));
    }
    let mut out = Vec::with_capacity(layers * n * hidden);
    for l in 0..layers {
        for (d, &k) in images.iter().zip(&per) {
            out.extend_from_slice(&d[l * k * hidden..(l + 1) * k * hidden]);
        }
    }
    Ok((mask, out, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x2-window grid: merge units are reordered window by window, and
    /// partial windows at the right and bottom edges keep their real units.
    #[test]
    fn window_index_groups_units_by_window() {
        // grid 1 x 6 x 10 patches -> 3 x 5 merge units, windows of 2.
        let (idx, cu) = window_index(&[[1, 6, 10]], 2, 2);
        assert_eq!(idx, vec![0, 1, 5, 6, 2, 3, 7, 8, 4, 9, 10, 11, 12, 13, 14]);
        assert_eq!(cu, vec![0, 16, 32, 40, 48, 56, 60]);
        // Two images: the second's units follow the first's.
        let (idx2, cu2) = window_index(&[[1, 4, 4], [1, 2, 2]], 2, 2);
        assert_eq!(idx2, vec![0, 1, 2, 3, 4]);
        assert_eq!(cu2, vec![0, 16, 20]);
    }

    #[test]
    fn window_mask_is_block_diagonal() {
        let b = block_mask_bytes(&[0, 2, 3]);
        let m: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(m.len(), 9);
        assert_eq!((m[0], m[1], m[2]), (0.0, 0.0, f32::NEG_INFINITY));
        assert_eq!((m[6], m[8]), (f32::NEG_INFINITY, 0.0));
    }

    /// The per-frame mask is Qwen3.5's `attention_mask` byte for byte.
    #[test]
    fn block_mask_bytes_match_the_f32_masks() {
        let grids = [[2u32, 4, 6]];
        let want: Vec<u8> = q35::attention_mask(&grids)
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(block_mask_bytes(&q35::cu_seqlens(&grids)), want);
        assert!(block_mask_bytes(&[0]).is_empty());
    }

    #[test]
    fn positions_follow_the_prompt_then_decode() {
        let ids = [1, 9, 9, 9, 9, 2];
        let m = mrope_positions(&ids, 9, 8, &[[1, 4, 4]], &[], 2).unwrap();
        let p = positions(Some(&m), 0, 7);
        // rows t, h, w; image at 1 is 2x2; text after it at 1 + 2 = 3.
        assert_eq!(&p[0..7], &[0, 1, 1, 1, 1, 3, 4]);
        assert_eq!(&p[7..14], &[0, 1, 1, 2, 2, 3, 4]);
        assert_eq!(&p[14..21], &[0, 1, 2, 1, 2, 3, 4]);
        // rope_delta = 3 + 1 - 6 = -2: decode index 6 sits at 4.
        assert_eq!(m.rope_delta, -2);
        assert_eq!(positions(Some(&m), 8, 9), vec![6, 6, 6]);
        assert_eq!(positions(None, 3, 5), vec![3, 4, 3, 4, 3, 4]);
    }

    #[test]
    fn deepstack_inputs_take_the_fed_images_layer_major() {
        // Two images of 1 and 2 tokens, 2 layers, hidden 1.
        let a = vec![10.0, 20.0]; // layer 0: 10, layer 1: 20
        let b = vec![1.0, 2.0, 3.0, 4.0]; // layer 0: 1, 2; layer 1: 3, 4
        let ids = [5, 9, 5, 9, 9];
        let (mask, e, n) = deepstack_inputs(&ids, 9, &[a, b], 2, 1).unwrap();
        assert_eq!(mask, vec![0, 1, 0, 1, 1]);
        assert_eq!(n, 3);
        assert_eq!(e, vec![10.0, 1.0, 2.0, 20.0, 3.0, 4.0]);
        // No image fed: one zero row per layer.
        let (mask, e, n) = deepstack_inputs(&[5, 6], 9, &[], 3, 2).unwrap();
        assert_eq!((mask, e, n), (vec![0, 0], vec![0.0; 6], 1));
        // A decode step that sampled the pad id is text, not an image.
        let (mask, _, n) = deepstack_inputs(&[9], 9, &[], 3, 2).unwrap();
        assert_eq!((mask, n), (vec![0], 1));
        // A mismatch fails rather than shifting features.
        assert!(deepstack_inputs(&[9, 9], 9, &[vec![1.0, 2.0]], 2, 1).is_err());
    }

    #[test]
    fn expansion_counts_must_match() {
        assert_eq!(
            expand_pads(&[1, 9, 2], 9, &[3]).unwrap(),
            vec![1, 9, 9, 9, 2]
        );
        assert!(expand_pads(&[1, 2], 9, &[3]).is_err());
    }

    #[test]
    fn grid_from_size_uses_the_processor_factor() {
        let mut c = PreprocessConfig::qwen35();
        c.patch_size = 14;
        c.min_pixels = 3136;
        c.max_pixels = 12_845_056;
        assert_eq!(grid_for_size(448, 448, &c).unwrap(), [1, 32, 32]);
    }
}
