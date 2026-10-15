//! A prompt with images, as rank 0 of the staged engine prefills it.
//!
//! The API renders one `<|vision_start|><|image_pad|><|vision_end|>` per
//! image. Admission expands each single `<|image_pad|>` into as many pad
//! tokens as the image has embedding rows, keeps those rows, and precomputes
//! the M-RoPE positions for the whole prompt. Prefill then overwrites the pad
//! rows of each span's text embeddings with image rows ([`VisionPrompt::splice`])
//! and feeds [`VisionPrompt::positions`] instead of the plain token index.

use super::rope::{decode_position, mrope_positions, MropePositions};

/// Rank-0 state for one prompt carrying images.
#[derive(Clone, Debug)]
pub struct VisionPrompt {
    /// Prompt ids with every image's pad token expanded.
    pub ids: Vec<u32>,
    /// Absolute index of every pad token, ascending; image row `k` belongs
    /// at `pad_positions[k]`.
    pub pad_positions: Vec<usize>,
    /// Image rows `[pad_positions.len(), hidden]`, in prompt order.
    pub rows: Vec<f32>,
    pub hidden: usize,
    pub mrope: MropePositions,
}

/// Expand the `i`-th single `pad` token in `ids` into `counts[i]` copies.
/// The number of pad tokens must equal `counts.len()`: a mismatch means the
/// prompt and the attached media disagree, which must fail rather than shift
/// every image onto the wrong tokens.
pub fn expand_pads(ids: &[u32], pad: u32, counts: &[usize]) -> Result<Vec<u32>, String> {
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
            let n = *next.next().expect("counted above");
            out.extend(std::iter::repeat_n(pad, n));
        } else {
            out.push(t);
        }
    }
    Ok(out)
}

impl VisionPrompt {
    /// Build from the unexpanded prompt ids and each image's encoder output
    /// (`[tokens_i, hidden]`) and grid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ids: &[u32],
        image_pad: u32,
        video_pad: u32,
        images: &[(Vec<f32>, [u32; 3])],
        hidden: usize,
        merge: u32,
    ) -> Result<Self, String> {
        let counts: Vec<usize> = images.iter().map(|(rows, _)| rows.len() / hidden).collect();
        for (i, (rows, [t, h, w])) in images.iter().enumerate() {
            let want = (t * (h / merge) * (w / merge)) as usize;
            if rows.len() != want * hidden {
                return Err(format!(
                    "image {i}: {} values for a grid of {want} tokens x {hidden}",
                    rows.len()
                ));
            }
        }
        let ids = expand_pads(ids, image_pad, &counts)?;
        let pad_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == image_pad).then_some(i))
            .collect();
        let ids_i64: Vec<i64> = ids.iter().map(|&t| i64::from(t)).collect();
        let grids: Vec<[u32; 3]> = images.iter().map(|(_, g)| *g).collect();
        let mrope = mrope_positions(
            &ids_i64,
            i64::from(image_pad),
            i64::from(video_pad),
            &grids,
            &[],
            merge,
        )?;
        let rows = images.iter().flat_map(|(r, _)| r.iter().copied()).collect();
        Ok(Self {
            ids,
            pad_positions,
            rows,
            hidden,
            mrope,
        })
    }

    /// Overwrite the image rows of `embeds`, the text embeddings of the span
    /// starting at absolute index `t0` (`[n, hidden]`).
    pub fn splice(&self, t0: usize, embeds: &mut [f32]) {
        let n = embeds.len() / self.hidden;
        let start = self.pad_positions.partition_point(|&p| p < t0);
        let end = self.pad_positions.partition_point(|&p| p < t0 + n);
        for k in start..end {
            let row = self.pad_positions[k] - t0;
            embeds[row * self.hidden..(row + 1) * self.hidden]
                .copy_from_slice(&self.rows[k * self.hidden..(k + 1) * self.hidden]);
        }
    }

    /// The `[4, t1 - t0]` position block for absolute indices `[t0, t1)`,
    /// row-major: prompt positions inside the prompt, decode positions
    /// (index + rope_delta) after it.
    pub fn positions(&self, t0: usize, t1: usize) -> Vec<i64> {
        let n = t1 - t0;
        let len = self.mrope.len();
        let mut out = vec![0i64; 4 * n];
        for (j, t) in (t0..t1).enumerate() {
            let p = if t < len {
                [
                    t as i64,
                    self.mrope.thw[0][t],
                    self.mrope.thw[1][t],
                    self.mrope.thw[2][t],
                ]
            } else {
                decode_position(t as i64, self.mrope.rope_delta)
            };
            for (r, v) in p.into_iter().enumerate() {
                out[r * n + j] = v;
            }
        }
        out
    }
}

/// Positions for a text-only span: the token index on every row.
pub fn text_positions(t0: usize, t1: usize) -> Vec<i64> {
    (0..4).flat_map(|_| (t0 as i64)..(t1 as i64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: u32 = 9;

    #[test]
    fn pads_expand_per_image_and_counts_must_match() {
        assert_eq!(
            expand_pads(&[1, PAD, 2, PAD], PAD, &[2, 1]).unwrap(),
            vec![1, PAD, PAD, 2, PAD]
        );
        assert!(expand_pads(&[1, PAD], PAD, &[]).is_err());
        assert!(expand_pads(&[1], PAD, &[3]).is_err());
    }

    fn two_by_two_image(hidden: usize, base: f32) -> (Vec<f32>, [u32; 3]) {
        // grid 1x4x4 -> 2x2 merged = 4 tokens.
        (
            (0..4 * hidden).map(|i| base + i as f32).collect(),
            [1, 4, 4],
        )
    }

    #[test]
    fn splice_fills_pad_rows_even_across_span_boundaries() {
        let hidden = 2;
        let vp = VisionPrompt::new(
            &[1, PAD, 2],
            PAD,
            8,
            &[two_by_two_image(hidden, 100.0)],
            hidden,
            2,
        )
        .unwrap();
        assert_eq!(vp.ids, vec![1, PAD, PAD, PAD, PAD, 2]);
        assert_eq!(vp.pad_positions, vec![1, 2, 3, 4]);
        // Prefill in spans [0,3) and [3,6): the image straddles them.
        let mut a = vec![0f32; 3 * hidden];
        vp.splice(0, &mut a);
        assert_eq!(a, vec![0.0, 0.0, 100.0, 101.0, 102.0, 103.0]);
        let mut b = vec![-1f32; 3 * hidden];
        vp.splice(3, &mut b);
        assert_eq!(b, vec![104.0, 105.0, 106.0, 107.0, -1.0, -1.0]);
    }

    #[test]
    fn positions_cover_prompt_then_decode() {
        let vp =
            VisionPrompt::new(&[1, PAD, 2], PAD, 8, &[two_by_two_image(1, 0.0)], 1, 2).unwrap();
        // text 0; image rows 1..: t=1 h=1,1,2,2 w=1,2,1,2; advance 2 -> text 3.
        let all = vp.positions(0, 6);
        assert_eq!(&all[0..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(&all[6..12], &[0, 1, 1, 1, 1, 3]);
        assert_eq!(&all[12..18], &[0, 1, 1, 2, 2, 3]);
        assert_eq!(&all[18..24], &[0, 1, 2, 1, 2, 3]);
        // rope_delta = 3 + 1 - 6 = -2; decode index 6 -> 4 on rows 1-3.
        assert_eq!(vp.mrope.rope_delta, -2);
        assert_eq!(vp.positions(6, 7), vec![6, 4, 4, 4]);
        // A span crossing the prompt end mixes both.
        assert_eq!(vp.positions(5, 7), vec![5, 6, 3, 4, 3, 4, 3, 4]);
    }

    #[test]
    fn text_positions_repeat_the_index() {
        assert_eq!(text_positions(3, 5), vec![3, 4, 3, 4, 3, 4, 3, 4]);
    }

    #[test]
    fn wrong_row_count_is_refused() {
        let bad = (vec![0f32; 3], [1, 4, 4]);
        assert!(VisionPrompt::new(&[PAD], PAD, 8, &[bad], 1, 2).is_err());
    }
}
