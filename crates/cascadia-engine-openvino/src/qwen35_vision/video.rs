//! Native video for the Qwen VL families: what transformers 5.14.1's video
//! processors and processors do to a video's sampled frames, so rank 0 can
//! feed the vision IRs one video item instead of one image per frame.
//!
//! The vision IRs need nothing new. Their patch embedding is a Conv3d over
//! `temporal_patch_size` (2) frames, which an image fills by duplicating its
//! one frame; a video fills it with two consecutive frames (an odd count
//! pads with the last frame, which is what GenAI's encoder does by encoding
//! the trailing frame as an image). Each pair is one temporal group of
//! `h x w` patches, so a video costs about half the tokens of its frames as
//! images.
//!
//! What differs per family is around the IRs:
//!
//! - **Size.** Qwen3-VL and Qwen3.5 (`Qwen3VLVideoProcessor`) resize with a
//!   `smart_resize` whose pixel budget covers every frame
//!   ([`temporal_smart_resize`]) from their own video config (4096 ..
//!   25165824 pixels); Qwen2.5-VL (`Qwen2VLVideoProcessor`) resizes each
//!   frame as an image with its image budget ([`VideoResize`]).
//! - **Markup.** Qwen3-VL and Qwen3.5 split a video per temporal group:
//!   `<t seconds><|vision_start|>` + the group's pads + `<|vision_end|>`, `t`
//!   the mean of the group's first and last frame times ([`group_times`],
//!   [`frame_markup`]); it replaces the template's one `<|video_pad|>`,
//!   which keeps its own `<|vision_start|>` / `<|vision_end|>` around it.
//!   Qwen2.5-VL writes one run of `t * h * w / 4` pads.
//! - **Positions** ([`super::rope::VideoLayout`]): one `t = 1` grid per
//!   group (Qwen3-VL, Qwen3.5), or one grid whose temporal index steps by
//!   `tokens_per_second * int(second_per_grid_t)` (Qwen2.5-VL,
//!   [`time_interval`]).

use super::{preprocess_at, smart_resize, Patches, PreprocessConfig};

pub const VISION_START: &str = "<|vision_start|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";
pub const VISION_END: &str = "<|vision_end|>";

/// Which video processor's resize a family uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoResize {
    /// `Qwen2VLVideoProcessor` (Qwen2.5-VL): each frame as an image.
    PerFrame,
    /// `Qwen3VLVideoProcessor` (Qwen3-VL, Qwen3.5): one budget over all
    /// frames, [`temporal_smart_resize`].
    Temporal,
}

/// transformers' `video_processing_qwen3_vl.smart_resize`: like the image
/// one, but the pixel budget counts `ceil(frames / temporal) * temporal`
/// frames, and a side under `factor` is an error rather than scaled up.
pub fn temporal_smart_resize(
    num_frames: u32,
    height: u32,
    width: u32,
    temporal: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32), String> {
    if height < factor || width < factor {
        return Err(format!(
            "height:{height} or width:{width} must be larger than factor:{factor}"
        ));
    }
    let (h, w, f) = (f64::from(height), f64::from(width), f64::from(factor));
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let n = f64::from(num_frames);
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let t_bar = f64::from(num_frames.div_ceil(temporal.max(1)) * temporal.max(1));
    if t_bar * h_bar * w_bar > max_pixels as f64 {
        let beta = ((n * h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if t_bar * h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (n * h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// The `(height, width)` a video's frames are resized to.
pub fn video_size(
    num_frames: u32,
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
    resize: VideoResize,
) -> Result<(u32, u32), String> {
    match resize {
        VideoResize::PerFrame => {
            smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)
        }
        VideoResize::Temporal => temporal_smart_resize(
            num_frames,
            height,
            width,
            cfg.temporal_patch_size,
            cfg.factor(),
            cfg.min_pixels,
            cfg.max_pixels,
        ),
    }
}

/// `video_grid_thw` from the frame count and size alone: one temporal step
/// per frame pair (the last padded), patches over the resized frame.
pub fn video_grid(
    num_frames: u32,
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
    resize: VideoResize,
) -> Result<[u32; 3], String> {
    if num_frames == 0 {
        return Err("a video with no frames".into());
    }
    let (h, w) = video_size(num_frames, width, height, cfg, resize)?;
    Ok([
        num_frames.div_ceil(cfg.temporal_patch_size.max(1)),
        h / cfg.patch_size,
        w / cfg.patch_size,
    ])
}

/// A video's frame stack (`[frames, height, width, 3]` RGB8) as the vision
/// embeddings IR's rows, grid `[t, h, w]` ([`video_grid`]).
pub fn preprocess_video(
    pixels: &[u8],
    num_frames: u32,
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
    resize: VideoResize,
) -> Result<Patches, String> {
    let frame = width as usize * height as usize * 3;
    if num_frames == 0 || pixels.len() != frame * num_frames as usize {
        return Err(format!(
            "{} bytes is not {num_frames} frames of {width}x{height} RGB8",
            pixels.len()
        ));
    }
    let (rh, rw) = video_size(num_frames, width, height, cfg, resize)?;
    let frames: Vec<&[u8]> = pixels.chunks_exact(frame).collect();
    preprocess_at(&frames, width, height, rh, rw, cfg)
}

/// Temporal group `t` of a video's patches as an item of its own (grid
/// `[1, h, w]`). The vision tower attends within one temporal group only
/// (`cu_seqlens` has one sequence per group, and Qwen2.5-VL's windows never
/// cross one), and every other op is per patch, so encoding the groups one
/// by one gives the rows of encoding the video whole while the attention
/// mask stays one group's size.
pub fn temporal_group(p: &Patches, t: usize) -> Patches {
    let [_, h, w] = p.grid_thw;
    let rows = (h * w) as usize;
    Patches {
        pixel_values: p.pixel_values[t * rows * p.cols..(t + 1) * rows * p.cols].to_vec(),
        rows,
        cols: p.cols,
        grid_thw: [1, h, w],
    }
}

/// `Qwen3VLProcessor._calculate_timestamps`: each temporal group's time in
/// seconds, the mean of its first and last frame's (frames padded with the
/// last one to a multiple of `temporal`).
pub fn group_times(timestamps_ms: &[u32], temporal: usize) -> Vec<f64> {
    let tp = temporal.max(1);
    let secs: Vec<f64> = timestamps_ms
        .iter()
        .map(|&ms| f64::from(ms) / 1000.0)
        .collect();
    let at = |i: usize| secs[i.min(secs.len() - 1)];
    (0..secs.len().div_ceil(tp))
        .map(|g| (at(g * tp) + at(g * tp + tp - 1)) / 2.0)
        .collect()
}

/// What replaces a Qwen3-VL / Qwen3.5 video's `<|video_pad|>`: per temporal
/// group, `<{t:.1f} seconds>` and one pad framed by `<|vision_start|>` and
/// `<|vision_end|>`. Rank 0 widens each pad to its group's tokens on ids.
/// Rust's `{:.1}` rounds the exact binary value half to even, as Python's.
pub fn frame_markup(group_times: &[f64]) -> String {
    group_times
        .iter()
        .map(|t| format!("<{t:.1} seconds>{VISION_START}{VIDEO_PAD}{VISION_END}"))
        .collect()
}

/// `prompt` with its `k`-th `<|video_pad|>` replaced by `markups[k]`. A
/// count mismatch leaves the prompt as it is, for the pad expansion to
/// refuse with the counts.
pub fn expand_video_text(prompt: &str, markups: &[String]) -> String {
    if prompt.matches(VIDEO_PAD).count() != markups.len() {
        return prompt.to_owned();
    }
    let mut out =
        String::with_capacity(prompt.len() + markups.iter().map(String::len).sum::<usize>());
    let mut rest = prompt;
    for m in markups {
        let at = rest.find(VIDEO_PAD).expect("counted above");
        out.push_str(&rest[..at]);
        out.push_str(m);
        rest = &rest[at + VIDEO_PAD.len()..];
    }
    out.push_str(rest);
    out
}

/// Qwen2.5-VL's `second_per_grid_t`: `temporal_patch_size / sampled_fps`,
/// the sampled rate read off the evenly spaced frame times (as GenAI's
/// `sampled_fps`). `None` with fewer than two frames, where the video has
/// one temporal step and the spacing never applies.
pub fn second_per_grid(timestamps_ms: &[u32], temporal: u32) -> Option<f64> {
    match timestamps_ms {
        [first, .., last] if last > first => {
            let step = f64::from(last - first) / (timestamps_ms.len() - 1) as f64 / 1000.0;
            Some(f64::from(temporal) * step)
        }
        _ => None,
    }
}

/// Qwen2.5-VL's temporal position step between a video's groups, as
/// transformers 5.14.1 `Qwen2_5_VLModel.get_rope_index` computes it:
/// `tokens_per_second * int(second_per_grid_t)`. The truncation is HF's
/// (Qwen's original code floored the product instead, and GenAI 2026.4.1
/// steps by `tokens_per_second`); the millisecond times are rounded to a
/// microsecond first so a whole number of seconds never truncates down.
pub fn time_interval(second_per_grid: Option<f64>, tokens_per_second: u32) -> i64 {
    let spg = second_per_grid.unwrap_or(1.0);
    let whole = ((spg * 1e6).round() / 1e6).floor() as i64;
    i64::from(tokens_per_second) * whole
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_resize_budgets_all_frames() {
        // Within budget: like the image resize.
        assert_eq!(
            temporal_smart_resize(4, 120, 200, 2, 32, 4096, 25_165_824),
            Ok((128, 192))
        );
        // Over budget: 8 frames of 448x448 under 4 frames' worth of pixels.
        let (h, w) = temporal_smart_resize(8, 448, 448, 2, 32, 1, 4 * 448 * 448).unwrap();
        assert!(8 * u64::from(h) * u64::from(w) <= 4 * 448 * 448 * 2);
        assert!(h < 448 && h % 32 == 0 && w % 32 == 0);
        // An odd count budgets the padded frame.
        assert_eq!(
            temporal_smart_resize(3, 64, 64, 2, 32, 4 * 64 * 64 + 1, u64::MAX),
            Ok((96, 96))
        );
        assert!(temporal_smart_resize(2, 20, 200, 2, 32, 0, u64::MAX).is_err());
    }

    #[test]
    fn grid_counts_frame_pairs() {
        let mut c = PreprocessConfig::qwen35();
        c.min_pixels = 4096;
        c.max_pixels = 25_165_824;
        assert_eq!(
            video_grid(5, 200, 120, &c, VideoResize::Temporal),
            Ok([3, 8, 12])
        );
        assert_eq!(
            video_grid(4, 200, 120, &c, VideoResize::Temporal),
            Ok([2, 8, 12])
        );
        assert!(video_grid(0, 200, 120, &c, VideoResize::Temporal).is_err());
    }

    #[test]
    fn odd_videos_pad_with_the_last_frame() {
        let c = PreprocessConfig::qwen35();
        let (w, h) = (256u32, 256u32);
        let frame = |v: u8| vec![v; (w * h * 3) as usize];
        let stack: Vec<u8> = [frame(0), frame(255), frame(255)].concat();
        let p = preprocess_video(&stack, 3, w, h, &c, VideoResize::PerFrame).unwrap();
        assert_eq!(p.grid_thw, [2, 16, 16]);
        // Group 1 is frame 2 twice: every value is 255's.
        let g1 = temporal_group(&p, 1);
        assert_eq!((g1.rows, g1.grid_thw), (256, [1, 16, 16]));
        assert!(g1.pixel_values.iter().all(|&v| v == 1.0));
        // Group 0 pairs frames 0 and 1: each row holds both.
        let g0 = temporal_group(&p, 0);
        let half = 16 * 16;
        assert!(g0.pixel_values[..half].iter().all(|&v| v == -1.0));
        assert!(g0.pixel_values[half..2 * half].iter().all(|&v| v == 1.0));
        assert!(preprocess_video(&stack[1..], 3, w, h, &c, VideoResize::PerFrame).is_err());
    }

    #[test]
    fn group_times_average_each_pair() {
        assert_eq!(
            group_times(&[0, 500, 1000, 1500, 2000], 2),
            vec![0.25, 1.25, 2.0]
        );
        assert_eq!(group_times(&[0], 2), vec![0.0]);
        // HF prints "<0.2 seconds>" for 0.25: half to even on the exact value.
        assert_eq!(
            frame_markup(&[0.25, 1.25]),
            "<0.2 seconds><|vision_start|><|video_pad|><|vision_end|>\
             <1.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
        );
    }

    #[test]
    fn video_text_expands_in_order() {
        let p = "a<|video_pad|>b<|video_pad|>c";
        assert_eq!(expand_video_text(p, &["X".into(), "Y".into()]), "aXbYc");
        assert_eq!(expand_video_text(p, &["X".into()]), p);
    }

    #[test]
    fn qwen25_interval_truncates_seconds_per_group() {
        // 1 fps: 2 s per group, 2 tokens per second.
        let s = second_per_grid(&[0, 1000, 2000, 3000], 2);
        assert_eq!(s, Some(2.0));
        assert_eq!(time_interval(s, 2), 4);
        // 1.5 fps: 1.33 s per group truncates to 1.
        let s = second_per_grid(&[0, 667, 1333, 2000], 2);
        assert_eq!(time_interval(s, 2), 2);
        // 0.5 fps: 4 s.
        assert_eq!(time_interval(second_per_grid(&[0, 2000], 2), 2), 8);
        assert_eq!(second_per_grid(&[0], 2), None);
    }
}
