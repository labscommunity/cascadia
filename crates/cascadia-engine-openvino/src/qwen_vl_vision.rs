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
//! - **Video** ([`crate::qwen35_vision::video`]): a `MediaKind::Video` item
//!   goes through the same IRs, frame pairs filling the temporal patch, one
//!   temporal group per encode. Its pads expand per family (Qwen3-VL: one
//!   run per group behind its timestamp text, which [`Frontend::prompt_text`]
//!   writes before tokenizing; Qwen2.5-VL: one run) and take their positions
//!   from the video grids. Once positions are fixed, rank 0 feeds every
//!   `<|video_pad|>` as `<|image_pad|>`: the language model never reads a
//!   pad's own embedding (its row is replaced by the vision row, as HF's
//!   `masked_scatter` does), and every path that finds vision rows by id
//!   (the splice, Qwen3-VL's `visual_pos_masks`, which HF sets on image and
//!   video tokens alike, and the KV / encode cache keys) then covers video
//!   rows with no second id to track.

use std::borrow::Cow;
use std::path::Path;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use crate::qwen35_vision::encoder as q35;
use crate::qwen35_vision::rope::{mrope_positions_with, MropePositions, VideoLayout};
use crate::qwen35_vision::video::{self, VideoResize};
use crate::qwen35_vision::{preprocess, smart_resize, Patches, PreprocessConfig};
use cascadia_types::{MediaInput, MediaKind};

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

    /// Which video processor's resize the family uses.
    pub fn video_resize(self) -> VideoResize {
        match self {
            Kind::Qwen25Vl => VideoResize::PerFrame,
            Kind::Qwen3Vl => VideoResize::Temporal,
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
/// takes: 0 inside each `[cu[k], cu[k+1])` block, [`q35::MASK_BLOCKED`]
/// (f16-safe, not `-inf`: the merger may run on the GPU in f16) elsewhere. Built
/// straight into the byte buffer, with no f32 copy: at ~1 MP (N = 5476
/// patches) one mask is 120 MB, and the merger takes two (per-frame
/// `attention_mask` from `cu_seqlens`, `window_attention_mask` from
/// `cu_window_seqlens`), so the encoder builds, feeds and drops them one at
/// a time. The shim copies each into its own OpenVINO tensor (at most 256 MiB
/// each, which the API's 1 MP default budget keeps well under).
pub fn block_mask_bytes(cu: &[i32]) -> Vec<u8> {
    let n = *cu.last().unwrap_or(&0) as usize;
    let blocked = q35::MASK_BLOCKED.to_le_bytes();
    let mut out = Vec::with_capacity(n * n * 4);
    for _ in 0..n * n {
        out.extend_from_slice(&blocked);
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

/// One turn's images and videos, ready to prefill.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// Prompt ids with each `<|image_pad|>` expanded to its image's rows and
    /// each video's pads to its rows, then fed as `<|image_pad|>` (see the
    /// module doc).
    pub ids: Vec<i64>,
    /// Each item's rows `[tokens_i, hidden]`, in prompt order (a video's
    /// temporal groups in order).
    pub rows: Vec<Vec<f32>>,
    /// Qwen3-VL: each item's DeepStack rows `[layers, tokens_i, hidden]`,
    /// layer-major. Empty for Qwen2.5-VL.
    pub deepstack: Vec<Vec<f32>>,
    /// HF's positions over the expanded prompt (with its real video ids).
    pub mrope: MropePositions,
}

/// [`Layout::prepare_ids`]: the fed ids, their positions, and each item's
/// grid in prompt order.
pub type PreparedIds = (Vec<i64>, MropePositions, Vec<[u32; 3]>);

/// A turn's media laid out from their sizes alone: what each pad expands to
/// and the grids `get_rope_index` takes.
struct MediaPlan {
    /// Tokens per image, in prompt order.
    image_counts: Vec<usize>,
    /// Tokens per `<|video_pad|>` in the prompt: per temporal group
    /// (Qwen3-VL), per video (Qwen2.5-VL).
    video_counts: Vec<usize>,
    image_grids: Vec<[u32; 3]>,
    video_grids: Vec<[u32; 3]>,
    layout: VideoLayout,
}

/// The processor half of the front end: everything from a turn's media
/// sizes to its expanded prompt and positions, with no IR involved (HF's
/// image and video processors, the processor's pad expansion and
/// `get_rope_index`).
#[derive(Clone, Debug)]
pub struct Layout {
    pub kind: Kind,
    pub preprocess: PreprocessConfig,
    /// The video processor's settings (its own pixel budget on Qwen3-VL).
    pub video: PreprocessConfig,
    pub tokens: ImageTokens,
    pub merge: u32,
    /// Qwen2.5-VL's `vision_config.tokens_per_second`, the temporal position
    /// step per second of video. `None` on Qwen3-VL.
    pub tokens_per_second: Option<u32>,
}

/// Rank 0's Qwen2.5-VL / Qwen3-VL image and video front end.
pub struct Frontend {
    pub layout: Layout,
    encoder: Encoder,
    /// Encoded items by (tenant, pixel hash): the rows, then (Qwen3-VL) the
    /// DeepStack rows. Metadata: the grid.
    cache: crate::image_cache::EncodeCache<[u32; 3]>,
    /// The language model's hidden size (the vision output width).
    pub hidden: usize,
    /// DeepStack layers the language model takes (0 on Qwen2.5-VL).
    pub deepstack_layers: usize,
}

impl Layout {
    /// Read from a dir holding `preprocessor_config.json` and the video
    /// processor's config (a cut tree's root or the export itself), given the
    /// VLM `config.json` (a tree keeps it as `config.vlm.json`), parsed.
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
        let video = PreprocessConfig::video_from_dir(dir)?;
        if (
            video.patch_size,
            video.merge_size,
            video.temporal_patch_size,
        ) != (
            preprocess.patch_size,
            preprocess.merge_size,
            preprocess.temporal_patch_size,
        ) {
            return Err(format!(
                "the video processor's patch {} / merge {} / temporal patch {} differ from the \
                 image processor's: the vision IRs take one layout",
                video.patch_size, video.merge_size, video.temporal_patch_size
            ));
        }
        let tokens_per_second = match kind {
            // Qwen2.5-VL ships 2 (as does GenAI's default); a config without
            // it keeps images working and videos on the shipped spacing.
            Kind::Qwen25Vl => Some(
                match vlm_config
                    .get("vision_config")
                    .and_then(|v| v.get("tokens_per_second"))
                    .and_then(serde_json::Value::as_u64)
                {
                    Some(t) => t as u32,
                    None => {
                        tracing::warn!(
                            "Qwen2.5-VL config lacks vision_config.tokens_per_second; \
                             video positions use 2"
                        );
                        2
                    }
                },
            ),
            Kind::Qwen3Vl => None,
        };
        Ok(Self {
            kind,
            merge: preprocess.merge_size,
            preprocess,
            video,
            tokens,
            tokens_per_second,
        })
    }

    /// The grid `[1, h, w]` (patches) an image of this size preprocesses to.
    pub fn grid_for_size(&self, width: u32, height: u32) -> Result<[u32; 3], String> {
        grid_for_size(width, height, &self.preprocess)
    }

    /// The rendered prompt as the tokenizer must see it: on Qwen3-VL each
    /// video's `<|video_pad|>` becomes its per-group timestamp markup
    /// ([`video::frame_markup`]), which HF's processor writes into the text
    /// before tokenizing. Unchanged without videos and on Qwen2.5-VL.
    pub fn prompt_text<'a>(&self, prompt: &'a str, media: &[MediaInput]) -> Cow<'a, str> {
        match self.kind {
            Kind::Qwen3Vl => {
                video::with_frame_markup(prompt, media, self.video.temporal_patch_size as usize)
            }
            Kind::Qwen25Vl => Cow::Borrowed(prompt),
        }
    }

    /// Every item's tokens and grid from its size alone (no encode).
    fn plan(&self, media: &[MediaInput]) -> Result<MediaPlan, String> {
        let m = self.merge;
        let mut p = MediaPlan {
            image_counts: Vec::new(),
            video_counts: Vec::new(),
            image_grids: Vec::new(),
            video_grids: Vec::new(),
            layout: VideoLayout::PerFrame,
        };
        let mut intervals = Vec::new();
        for (i, item) in media.iter().enumerate() {
            let [t, h, w] = self.grid_for(i, item)?;
            let per_group = ((h / m) * (w / m)) as usize;
            if item.kind == MediaKind::Image {
                p.image_counts.push(t as usize * per_group);
                p.image_grids.push([t, h, w]);
                continue;
            }
            p.video_grids.push([t, h, w]);
            match self.tokens_per_second {
                // Qwen2.5-VL: one run, its groups spaced in time.
                Some(tps) => {
                    p.video_counts.push(t as usize * per_group);
                    let ts = item.timestamps_ms.as_deref().unwrap_or_default();
                    intervals.push(video::temporal_step(
                        ts,
                        self.video.temporal_patch_size,
                        tps,
                    ));
                }
                // Qwen3-VL: one run per group, behind its timestamp.
                None => {
                    p.video_counts
                        .extend(std::iter::repeat_n(per_group, t as usize));
                }
            }
        }
        if self.tokens_per_second.is_some() {
            p.layout = VideoLayout::Temporal(intervals);
        }
        Ok(p)
    }

    /// Item `i`'s grid: an image's from the image processor, a video's from
    /// the video processor (frames in pairs). Refuses a malformed item.
    fn grid_for(&self, i: usize, m: &MediaInput) -> Result<[u32; 3], String> {
        match m.kind {
            MediaKind::Video => {
                check_video(i, m, self.kind == Kind::Qwen3Vl)?;
                video::video_grid(
                    m.num_frames,
                    m.width,
                    m.height,
                    &self.video,
                    self.kind.video_resize(),
                )
                .map_err(|e| format!("video {i}: {e}"))
            }
            _ => {
                crate::gemma4_vision::check_image(i, m)?;
                self.grid_for_size(m.width, m.height)
                    .map_err(|e| format!("image {i}: {e}"))
            }
        }
    }

    /// The prompt with every pad run widened, still with its real video ids.
    fn expand_true(&self, plan: &MediaPlan, prompt_ids: &[i64]) -> Result<Vec<i64>, String> {
        let ids = expand_runs(
            prompt_ids,
            self.tokens.image_pad,
            &plan.image_counts,
            "image",
        )?;
        let what = if self.tokens_per_second.is_some() {
            "video"
        } else {
            "video frame"
        };
        expand_runs(&ids, self.tokens.video_pad, &plan.video_counts, what)
    }

    /// The fed ids: video pads become image pads (see the module doc).
    fn feed_ids(&self, mut ids: Vec<i64>) -> Vec<i64> {
        for t in &mut ids {
            if *t == self.tokens.video_pad {
                *t = self.tokens.image_pad;
            }
        }
        ids
    }

    /// The expanded prompt `prepare` would return, from item sizes alone (no
    /// encode): each pad repeated for its item's tokens.
    pub fn expand_ids(&self, media: &[MediaInput], prompt_ids: &[i64]) -> Result<Vec<i64>, String> {
        let plan = self.plan(media)?;
        Ok(self.feed_ids(self.expand_true(&plan, prompt_ids)?))
    }

    /// [`Self::expand_ids`] and HF's positions over the expanded prompt, and
    /// each item's grid in prompt order.
    pub fn prepare_ids(
        &self,
        media: &[MediaInput],
        prompt_ids: &[i64],
    ) -> Result<PreparedIds, String> {
        let plan = self.plan(media)?;
        let ids = self.expand_true(&plan, prompt_ids)?;
        let mrope = mrope_positions_with(
            &ids,
            self.tokens.image_pad,
            self.tokens.video_pad,
            &plan.image_grids,
            &plan.video_grids,
            self.merge,
            &plan.layout,
        )?;
        let (mut images, mut videos) = (plan.image_grids.iter(), plan.video_grids.iter());
        let grids = media
            .iter()
            .map(|m| {
                *if m.kind == MediaKind::Video {
                    videos.next()
                } else {
                    images.next()
                }
                .expect("one grid per item")
            })
            .collect();
        Ok((self.feed_ids(ids), mrope, grids))
    }
}

impl Frontend {
    /// Load from a dir holding the vision IRs and the processor configs (a
    /// cut tree's root or the export itself), given the VLM `config.json`
    /// (a tree keeps it as `config.vlm.json`), already parsed.
    pub fn load(dir: &Path, kind: Kind, vlm_config: &serde_json::Value) -> Result<Self, String> {
        let layout = Layout::load(dir, kind, vlm_config)?;
        let preprocess = &layout.preprocess;
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
            layout,
            encoder,
            cache: crate::image_cache::EncodeCache::from_env(),
            hidden,
            deepstack_layers,
        })
    }

    pub fn kind(&self) -> Kind {
        self.layout.kind
    }

    /// [`Layout::prompt_text`].
    pub fn prompt_text<'a>(&self, prompt: &'a str, media: &[MediaInput]) -> Cow<'a, str> {
        self.layout.prompt_text(prompt, media)
    }

    /// [`Layout::expand_ids`].
    pub fn expand_ids(&self, media: &[MediaInput], prompt_ids: &[i64]) -> Result<Vec<i64>, String> {
        self.layout.expand_ids(media, prompt_ids)
    }

    /// Encode a turn's images and videos (or take them from the cache) and
    /// expand the prompt; positions cover the expanded prompt.
    pub fn prepare(
        &mut self,
        media: &[MediaInput],
        prompt_ids: &[i64],
        tenant: &str,
    ) -> Result<Prepared, String> {
        // Sized first: a prompt whose placeholders disagree costs no encode.
        let (ids, mrope, grids) = self.layout.prepare_ids(media, prompt_ids)?;
        let mut rows = Vec::with_capacity(media.len());
        let mut deepstack = Vec::with_capacity(media.len());
        let mut from_cache = 0;
        let layers = self.deepstack_layers;
        let merge = self.layout.merge;
        for ((i, m), &planned) in media.iter().enumerate().zip(&grids) {
            let what = if m.kind == MediaKind::Video {
                "video"
            } else {
                "image"
            };
            let (mut all, grid) = match self.cache.get(tenant, &m.sha256) {
                Some(hit) => {
                    from_cache += 1;
                    hit
                }
                None => {
                    let all = self
                        .encode_item(m)
                        .map_err(|e| format!("{what} {i}: {e}"))?;
                    self.cache.put(tenant, m.sha256, &all, planned);
                    (all, planned)
                }
            };
            if grid != planned {
                return Err(format!(
                    "{what} {i}: encoded grid {grid:?}, its size gives {planned:?}"
                ));
            }
            let [t, h, w] = grid;
            let n = (t * (h / merge) * (w / merge)) as usize * self.hidden;
            if all.len() != n * (1 + layers) {
                return Err(format!(
                    "{what} {i}: {} encoded values for {} rows",
                    all.len(),
                    n / self.hidden.max(1)
                ));
            }
            deepstack.push(all.split_off(n));
            rows.push(all);
        }
        self.cache.report(from_cache, media.len());
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

    /// One item's rows followed by (Qwen3-VL) its DeepStack rows. A video is
    /// encoded one temporal group at a time ([`video::temporal_group`]), its
    /// DeepStack rows regathered layer-major over the whole video.
    fn encode_item(&mut self, m: &MediaInput) -> Result<Vec<f32>, String> {
        if m.kind != MediaKind::Video {
            let p = preprocess(&[&m.pixels[..]], m.width, m.height, &self.layout.preprocess)?;
            return self.encode(&p);
        }
        let p = video::preprocess_video(
            &m.pixels,
            m.num_frames,
            m.width,
            m.height,
            &self.layout.video,
            self.layout.kind.video_resize(),
        )?;
        let groups = p.grid_thw[0] as usize;
        let per = p.llm_tokens(self.layout.merge) / groups.max(1) * self.hidden;
        let encoded = (0..groups)
            .map(|g| self.encode(&video::temporal_group(&p, g)))
            .collect::<Result<Vec<_>, _>>()?;
        join_groups(encoded, per, self.deepstack_layers)
    }

    /// One preprocessed image's rows `[tokens, hidden]` and (Qwen3-VL) its
    /// DeepStack rows `[layers, tokens, hidden]` (empty otherwise).
    pub fn encode_patches(&mut self, p: &Patches) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut all = self.encode(p)?;
        let n = p.llm_tokens(self.layout.merge) * self.hidden;
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
            "{} images and native video (frame pairs), {}-px patches merged {}x{}, \
             {}..{} pixels{}",
            self.layout.kind.name(),
            self.layout.preprocess.patch_size,
            self.layout.merge,
            self.layout.merge,
            self.layout.preprocess.min_pixels,
            self.layout.preprocess.max_pixels,
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

/// A video's encoded temporal groups (each its rows `per` values, then
/// `layers` DeepStack blocks of `per`) as one item: every group's rows in
/// order, then each DeepStack layer over every group, layer-major as one
/// image's.
fn join_groups(groups: Vec<Vec<f32>>, per: usize, layers: usize) -> Result<Vec<f32>, String> {
    let mut rows = Vec::with_capacity(per * groups.len() * (1 + layers));
    for (g, all) in groups.iter().enumerate() {
        if all.len() != per * (1 + layers) {
            return Err(format!(
                "temporal group {g}: {} encoded values, expected {}",
                all.len(),
                per * (1 + layers)
            ));
        }
        rows.extend_from_slice(&all[..per]);
    }
    for l in 1..=layers {
        for all in &groups {
            rows.extend_from_slice(&all[l * per..(l + 1) * per]);
        }
    }
    Ok(rows)
}

/// One video item must hold `num_frames >= 1` frames of its size, and (for
/// the timestamp markup) one time per frame when `needs_times`.
fn check_video(i: usize, m: &MediaInput, needs_times: bool) -> Result<(), String> {
    if m.num_frames == 0 || m.pixels.len() != m.expected_len() {
        return Err(format!("video {i}: malformed pixel buffer"));
    }
    match m.timestamps_ms.as_deref() {
        Some(ts) if ts.len() == m.num_frames as usize => Ok(()),
        _ if !needs_times => Ok(()),
        _ => Err(format!("video {i}: no time for every frame")),
    }
}

/// [`crate::qwen35_vision::prompt::expand_pads`] over i64 ids.
pub fn expand_pads(ids: &[i64], pad: i64, counts: &[usize]) -> Result<Vec<i64>, String> {
    expand_runs(ids, pad, counts, "image")
}

/// Expand the `k`-th single `pad` into `counts[k]` pads; `what` names the
/// item in the count-mismatch error.
fn expand_runs(ids: &[i64], pad: i64, counts: &[usize], what: &str) -> Result<Vec<i64>, String> {
    let found = ids.iter().filter(|&&t| t == pad).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} {what} placeholders but {} {what}s were attached",
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

/// Flatten per-image DeepStack rows (each `[layers, tokens_i, hidden]`,
/// layer-major) into one `[layers, total, hidden]` layer-major buffer in
/// prompt order, `total` the image tokens across all images. The static and
/// packed paths splice one image's rows into a prefill that may span several
/// inferences, so they slice this buffer by a running image-token cursor
/// rather than feeding whole images the way the stateful single-forward path
/// does.
pub fn flatten_deepstack(
    images: &[Vec<f32>],
    layers: usize,
    hidden: usize,
) -> Result<(Vec<f32>, usize), String> {
    if layers == 0 || hidden == 0 {
        return Ok((Vec::new(), 0));
    }
    let per: Vec<usize> = images.iter().map(|d| d.len() / (layers * hidden)).collect();
    for (d, &k) in images.iter().zip(&per) {
        if d.len() != layers * k * hidden {
            return Err(format!(
                "a DeepStack image has {} values, not a multiple of layers*hidden ({}*{})",
                d.len(),
                layers,
                hidden
            ));
        }
    }
    let total: usize = per.iter().sum();
    let mut out = vec![0f32; layers * total * hidden];
    for l in 0..layers {
        let mut off = 0usize;
        for (d, &k) in images.iter().zip(&per) {
            out[(l * total + off) * hidden..(l * total + off + k) * hidden]
                .copy_from_slice(&d[l * k * hidden..(l + 1) * k * hidden]);
            off += k;
        }
    }
    Ok((out, total))
}

/// One static/packed inference's DeepStack feed for stage 0: `visual_pos_masks`
/// (1 on the image tokens of `ids`, i64, length `width`) and
/// `deepstack_visual_embeds [layers, width, hidden]` whose first `k` rows are
/// the fed image tokens' DeepStack rows taken from `global[cursor..]` in
/// image-token order (the rest zero). The language model adds row
/// `max(0, cumsum(mask) - 1)` of the embeds to each image token, so the image
/// rows belong at the front and the padded tail is never gathered — `width`
/// may exceed `k`, which is what lets one static shape serve every inference.
/// Returns the mask, the embeds, and `k` (image tokens fed, to advance the
/// cursor). `k == 0` (a decode step, a text chunk) yields an all-zero feed.
#[allow(clippy::too_many_arguments)]
pub fn deepstack_window(
    ids: &[i64],
    image_pad: i64,
    global: &[f32],
    total: usize,
    cursor: usize,
    layers: usize,
    hidden: usize,
    width: usize,
) -> Result<(Vec<i64>, Vec<f32>, usize), String> {
    let mut mask = vec![0i64; width];
    let mut k = 0usize;
    for (i, &t) in ids.iter().enumerate().take(width) {
        if t == image_pad {
            mask[i] = 1;
            k += 1;
        }
    }
    if cursor + k > total {
        return Err(format!(
            "static DeepStack: {k} image tokens fed at cursor {cursor} run past the turn's {total}"
        ));
    }
    let mut embeds = vec![0f32; layers * width * hidden];
    for l in 0..layers {
        let src = &global[(l * total + cursor) * hidden..(l * total + cursor + k) * hidden];
        embeds[l * width * hidden..l * width * hidden + k * hidden].copy_from_slice(src);
    }
    Ok((mask, embeds, k))
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
        let x = q35::MASK_BLOCKED;
        assert_eq!((m[0], m[1], m[2]), (0.0, 0.0, x));
        assert_eq!((m[6], m[8]), (x, 0.0));
        assert!(m.iter().all(|v| v.is_finite()));
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
        use crate::qwen35_vision::rope::mrope_positions;
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
    fn flatten_deepstack_is_layer_major_in_prompt_order() {
        // Two images of 2 and 1 tokens, 2 layers, hidden 1.
        // image A layer0: 1,2  layer1: 3,4 ; image B layer0: 5  layer1: 6
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0];
        let (g, total) = flatten_deepstack(&[a, b], 2, 1).unwrap();
        assert_eq!(total, 3);
        // layer 0: A then B (1,2,5); layer 1: 3,4,6.
        assert_eq!(g, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
        assert_eq!(flatten_deepstack(&[], 2, 1).unwrap(), (vec![], 0));
        assert_eq!(flatten_deepstack(&[vec![1.0]], 0, 1).unwrap(), (vec![], 0));
    }

    /// A chunk boundary falling inside an image feeds that chunk only its own
    /// image tokens' rows (front-loaded), and the next chunk continues from the
    /// cursor — the same rows a whole-image chunk would get, since the language
    /// model gathers by the per-inference cumsum of the mask.
    #[test]
    fn deepstack_window_slices_by_cursor_across_a_split_image() {
        // One image of 3 tokens, 1 layer, hidden 1: rows 10, 20, 30.
        let (g, total) = flatten_deepstack(&[vec![10.0, 20.0, 30.0]], 1, 1).unwrap();
        assert_eq!(total, 3);
        // Whole image in one width-4 inference: ids [t, img, img, img].
        let (mask, emb, k) = deepstack_window(&[1, 9, 9, 9], 9, &g, total, 0, 1, 1, 4).unwrap();
        assert_eq!((mask, k), (vec![0, 1, 1, 1], 3));
        assert_eq!(emb, vec![10.0, 20.0, 30.0, 0.0]);
        // Split: chunk A = [t, img, img] (cursor 0), chunk B = [img, t] (cursor 2).
        let (ma, ea, ka) = deepstack_window(&[1, 9, 9], 9, &g, total, 0, 1, 1, 3).unwrap();
        assert_eq!((ma, ka), (vec![0, 1, 1], 2));
        assert_eq!(ea, vec![10.0, 20.0, 0.0]); // image rows front-loaded, padded
        let (mb, eb, kb) = deepstack_window(&[9, 1], 9, &g, total, ka, 1, 1, 2).unwrap();
        assert_eq!((mb, kb), (vec![1, 0], 1));
        assert_eq!(eb, vec![30.0, 0.0]);
        // Decode step (width 1, no image token): all-zero feed, cursor unmoved.
        let (md, ed, kd) = deepstack_window(&[5], 9, &g, total, 3, 1, 1, 1).unwrap();
        assert_eq!((md, ed, kd), (vec![0], vec![0.0], 0));
        // Feeding past the turn's image tokens is an error, not a bad slice.
        assert!(deepstack_window(&[9], 9, &g, total, 3, 1, 1, 1).is_err());
    }

    /// Two groups of 2 values with 2 DeepStack layers: rows of both groups,
    /// then layer 0 of both, then layer 1 of both.
    #[test]
    fn video_groups_join_layer_major() {
        let g0 = vec![1.0, 2.0, 10.0, 11.0, 20.0, 21.0];
        let g1 = vec![3.0, 4.0, 12.0, 13.0, 22.0, 23.0];
        assert_eq!(
            join_groups(vec![g0.clone(), g1], 2, 2).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]
        );
        // No DeepStack (Qwen2.5-VL): the rows alone.
        assert_eq!(
            join_groups(vec![vec![1.0], vec![2.0]], 1, 0).unwrap(),
            vec![1.0, 2.0]
        );
        assert!(join_groups(vec![g0], 2, 1).is_err());
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
