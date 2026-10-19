//! Qwen3.5 M-RoPE positions for prompts carrying image or video tokens: a
//! port of transformers' `Qwen3_5MoeModel.get_rope_index` (5.14.1) and the
//! 4-row layout GenAI's Qwen3.5 embedder feeds the language model
//! (`qwen3_5/classes.cpp`).
//!
//! Text tokens advance one position on all three axes. A run of vision
//! tokens takes `t = start`, `h = start + row`, `w = start + col` over its
//! merged grid and then advances the position by `max(h, w) / merge`, not by
//! its token count. `rope_delta = max_position + 1 - len` is what decode adds
//! to the token index from then on.
//!
//! Videos are split into one-frame grids, because Qwen3.5 (and Qwen3-VL)
//! separates frames with timestamp text (`<t> <vision_start> frame
//! <vision_end>`), so each frame is its own run of video tokens. Qwen2.5-VL
//! keeps a video as one run whose temporal index steps by a per-video
//! interval ([`VideoLayout::Temporal`], `Qwen2_5_VLModel.get_rope_index`).

/// Position ids for one prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MropePositions {
    /// HF's three rows (temporal, height, width), one entry per token.
    pub thw: [Vec<i64>; 3],
    pub rope_delta: i64,
}

impl MropePositions {
    pub fn len(&self) -> usize {
        self.thw[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `[4, T]` layout the Qwen3.5 language model IR takes: row 0 is the
    /// plain token index, rows 1-3 the M-RoPE rows.
    pub fn four_rows(&self) -> [Vec<i64>; 4] {
        let [t, h, w] = self.thw.clone();
        [(0..self.len() as i64).collect(), t, h, w]
    }
}

/// Positions for the token at absolute index `index` during decode (every
/// token after the prompt): row 0 the index, rows 1-3 the index plus
/// `rope_delta`, as GenAI's `get_generation_phase_position_ids`.
pub fn decode_position(index: i64, rope_delta: i64) -> [i64; 4] {
    let p = index + rope_delta;
    [index, p, p, p]
}

/// Which modality a token carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Text,
    Image,
    Video,
}

/// How a video's tokens sit in the prompt and take positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoLayout {
    /// Qwen3-VL / Qwen3.5: every temporal group is its own run (timestamp
    /// text separates them) with a `t = 1` grid.
    PerFrame,
    /// Qwen2.5-VL: one run per video; group `k`'s temporal index is
    /// `start + k * interval`, one interval per video in prompt order
    /// ([`super::video::time_interval`]). The position after the run still
    /// advances by `max(h, w) / merge` only, as transformers 5.14.1 does.
    Temporal(Vec<i64>),
}

/// `get_rope_index` for one sequence with no padding.
///
/// `image_grids` / `video_grids` are `[t, h, w]` in patches (pre-merge), in
/// prompt order, as the processor returns them. Unlike HF, a run whose token
/// count does not match its grid, or grids left over or missing, is an error
/// rather than a silent misalignment. Videos take the Qwen3.5 layout
/// ([`VideoLayout::PerFrame`]); [`mrope_positions_with`] picks another.
pub fn mrope_positions(
    input_ids: &[i64],
    image_token_id: i64,
    video_token_id: i64,
    image_grids: &[[u32; 3]],
    video_grids: &[[u32; 3]],
    spatial_merge_size: u32,
) -> Result<MropePositions, String> {
    mrope_positions_with(
        input_ids,
        image_token_id,
        video_token_id,
        image_grids,
        video_grids,
        spatial_merge_size,
        &VideoLayout::PerFrame,
    )
}

/// [`mrope_positions`] with the family's video layout.
pub fn mrope_positions_with(
    input_ids: &[i64],
    image_token_id: i64,
    video_token_id: i64,
    image_grids: &[[u32; 3]],
    video_grids: &[[u32; 3]],
    spatial_merge_size: u32,
    layout: &VideoLayout,
) -> Result<MropePositions, String> {
    let kind = |id: i64| {
        if id == image_token_id {
            Kind::Image
        } else if id == video_token_id {
            Kind::Video
        } else {
            Kind::Text
        }
    };
    // Each video run's grid and temporal step: one `t = 1` grid per frame,
    // or the whole video with its interval.
    let video_runs: Vec<([u32; 3], i64)> = match layout {
        VideoLayout::PerFrame => video_grids
            .iter()
            .flat_map(|&[t, h, w]| std::iter::repeat_n(([1, h, w], 1), t as usize))
            .collect(),
        VideoLayout::Temporal(intervals) => {
            if intervals.len() != video_grids.len() {
                return Err(format!(
                    "{} video intervals for {} videos",
                    intervals.len(),
                    video_grids.len()
                ));
            }
            video_grids
                .iter()
                .copied()
                .zip(intervals.iter().copied())
                .collect()
        }
    };
    let mut images = image_grids.iter().map(|&g| (g, 1i64));
    let mut videos = video_runs.iter().copied();
    let m = i64::from(spatial_merge_size);

    let mut rows: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(input_ids.len()));
    let mut pos: i64 = 0;
    let mut i = 0;
    while i < input_ids.len() {
        let k = kind(input_ids[i]);
        let mut j = i + 1;
        while j < input_ids.len() && kind(input_ids[j]) == k {
            j += 1;
        }
        let run = j - i;
        match k {
            Kind::Text => {
                for r in &mut rows {
                    r.extend(pos..pos + run as i64);
                }
                pos += run as i64;
            }
            Kind::Image | Kind::Video => {
                let what = match (k, layout) {
                    (Kind::Image, _) => "image",
                    (_, VideoLayout::PerFrame) => "video frame",
                    _ => "video",
                };
                let (grid, interval) = if k == Kind::Image {
                    images.next()
                } else {
                    videos.next()
                }
                .ok_or_else(|| format!("more {what} token runs than {what} grids"))?;
                let (t, h, w) = (
                    i64::from(grid[0]),
                    i64::from(grid[1]) / m,
                    i64::from(grid[2]) / m,
                );
                if (t * h * w) as usize != run {
                    return Err(format!(
                        "{what} run of {run} tokens at {i} does not match grid {grid:?} \
                         ({} tokens after merge)",
                        t * h * w
                    ));
                }
                for tt in 0..t {
                    for hh in 0..h {
                        for ww in 0..w {
                            rows[0].push(pos + tt * interval);
                            rows[1].push(pos + hh);
                            rows[2].push(pos + ww);
                        }
                    }
                }
                pos += h.max(w);
            }
        }
        i = j;
    }
    if images.next().is_some() {
        return Err("image grids left over after the last image run".into());
    }
    if videos.next().is_some() {
        return Err("video grids left over after the last video run".into());
    }
    let max = rows
        .iter()
        .flat_map(|r| r.iter())
        .copied()
        .max()
        .unwrap_or(-1);
    Ok(MropePositions {
        rope_delta: max + 1 - input_ids.len() as i64,
        thw: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: i64 = 9;
    const VID: i64 = 8;

    #[test]
    fn text_only_is_the_token_index() {
        let p = mrope_positions(&[1, 2, 3], IMG, VID, &[], &[], 2).unwrap();
        assert_eq!(p.thw, [vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]]);
        assert_eq!(p.rope_delta, 0);
        assert_eq!(p.four_rows()[0], vec![0, 1, 2]);
        assert_eq!(decode_position(3, p.rope_delta), [3, 3, 3, 3]);
    }

    #[test]
    fn an_image_takes_a_grid_and_advances_by_its_longer_side() {
        // text, 2x3 merged image (grid 1x4x6), text
        let ids = [1, IMG, IMG, IMG, IMG, IMG, IMG, 2];
        let p = mrope_positions(&ids, IMG, VID, &[[1, 4, 6]], &[], 2).unwrap();
        assert_eq!(p.thw[0], vec![0, 1, 1, 1, 1, 1, 1, 4]);
        assert_eq!(p.thw[1], vec![0, 1, 1, 1, 2, 2, 2, 4]);
        assert_eq!(p.thw[2], vec![0, 1, 2, 3, 1, 2, 3, 4]);
        // max 4, len 8.
        assert_eq!(p.rope_delta, -3);
        assert_eq!(decode_position(8, p.rope_delta), [8, 5, 5, 5]);
    }

    /// Qwen2.5-VL: one run per video, temporal index stepping by the
    /// interval, the next text only `max(h, w) / merge` further on.
    #[test]
    fn a_temporal_video_steps_its_groups_by_the_interval() {
        // text, video of 2 groups of 1x2 merged (grid 2x2x4), text
        let ids = [1, VID, VID, VID, VID, 2];
        let layout = VideoLayout::Temporal(vec![8]);
        let p = mrope_positions_with(&ids, IMG, VID, &[], &[[2, 2, 4]], 2, &layout).unwrap();
        assert_eq!(p.thw[0], vec![0, 1, 1, 9, 9, 3]);
        assert_eq!(p.thw[1], vec![0, 1, 1, 1, 1, 3]);
        assert_eq!(p.thw[2], vec![0, 1, 2, 1, 2, 3]);
        // max 9, len 6.
        assert_eq!(p.rope_delta, 4);
        // The Qwen3.5 layout wants one run per group instead.
        assert!(mrope_positions(&ids, IMG, VID, &[], &[[2, 2, 4]], 2).is_err());
        assert!(mrope_positions_with(
            &ids,
            IMG,
            VID,
            &[],
            &[[2, 2, 4]],
            2,
            &VideoLayout::Temporal(vec![])
        )
        .is_err());
    }

    #[test]
    fn mismatches_are_errors() {
        let ids = [IMG, IMG, IMG];
        assert!(mrope_positions(&ids, IMG, VID, &[[1, 4, 4]], &[], 2).is_err());
        assert!(mrope_positions(&ids, IMG, VID, &[], &[], 2).is_err());
        assert!(mrope_positions(&[1], IMG, VID, &[[1, 2, 2]], &[], 2).is_err());
    }
}
