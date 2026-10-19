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
    /// `ids` with every pad token replaced by a pseudo-id derived from its
    /// image's pixel hash and its row in that image ([`pad_cache_id`]). Prefix
    /// caches key on this, not on `ids`: pad tokens are identical for every
    /// image, so keying on `ids` would serve one image's state for another.
    pub cache_key: Vec<u32>,
}

/// The cache-key stand-in for row `row` of the image hashed `sha256`, in the
/// cache namespace `ns` (the turn's tenant). The high bit is set, so it can
/// never equal a real token id (vocabularies are far below 2^31); 31 hash
/// bits per row over a run of hundreds of rows make two different images'
/// runs agreeing vanishingly unlikely.
///
/// `ns` keeps image spans from matching across tenants: otherwise a caller
/// could time a request to learn whether another tenant recently sent the
/// same image. Tenants are `""` until the plane asserts them (design H.1b),
/// so today every caller shares one namespace, as the text prefix cache does.
pub fn pad_cache_id(ns: &str, sha256: &[u8; 32], row: u32) -> u32 {
    // FNV-1a, 64-bit, over the namespace (length-prefixed), hash and row.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let ns_len = (ns.len() as u32).to_le_bytes();
    for &b in ns_len
        .iter()
        .chain(ns.as_bytes())
        .chain(sha256.iter())
        .chain(row.to_le_bytes().iter())
    {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    0x8000_0000 | ((h ^ (h >> 32)) as u32 & 0x7fff_ffff)
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
    /// (`[tokens_i, hidden]`), grid and pixel hash. `cache_ns` namespaces the
    /// image spans of [`Self::cache_key`] (the turn's tenant).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ids: &[u32],
        image_pad: u32,
        video_pad: u32,
        images: &[(Vec<f32>, [u32; 3], [u8; 32])],
        hidden: usize,
        merge: u32,
        cache_ns: &str,
    ) -> Result<Self, String> {
        let kinds = vec![false; images.len()];
        Self::with_videos(
            ids, image_pad, video_pad, images, &kinds, hidden, merge, cache_ns,
        )
    }

    /// [`Self::new`] over images and videos in prompt order (`is_video[k]`
    /// for item `k`, its grid `[t, h, w]` with `t` temporal groups). The
    /// prompt carries one `<|image_pad|>` per image and, per video, one
    /// `<|video_pad|>` per temporal group behind its timestamp text
    /// ([`super::video::with_frame_markup`]). Positions come from the real
    /// ids; then every video pad is fed as an image pad, so the splice and
    /// the cache key cover video rows the same way (the model never reads a
    /// pad's own embedding).
    #[allow(clippy::too_many_arguments)]
    pub fn with_videos(
        ids: &[u32],
        image_pad: u32,
        video_pad: u32,
        items: &[(Vec<f32>, [u32; 3], [u8; 32])],
        is_video: &[bool],
        hidden: usize,
        merge: u32,
        cache_ns: &str,
    ) -> Result<Self, String> {
        if is_video.len() != items.len() {
            return Err(format!(
                "{} kinds for {} media items",
                is_video.len(),
                items.len()
            ));
        }
        let counts: Vec<usize> = items
            .iter()
            .map(|(rows, _, _)| rows.len() / hidden)
            .collect();
        let (mut image_counts, mut group_counts) = (Vec::new(), Vec::new());
        let (mut image_grids, mut video_grids) = (Vec::new(), Vec::new());
        for (i, ((rows, [t, h, w], _), &video)) in items.iter().zip(is_video).enumerate() {
            let per = ((h / merge) * (w / merge)) as usize;
            let want = *t as usize * per;
            if rows.len() != want * hidden {
                let what = if video { "video" } else { "image" };
                return Err(format!(
                    "{what} {i}: {} values for a grid of {want} tokens x {hidden}",
                    rows.len()
                ));
            }
            if video {
                group_counts.extend(std::iter::repeat_n(per, *t as usize));
                video_grids.push([*t, *h, *w]);
            } else {
                image_counts.push(want);
                image_grids.push([*t, *h, *w]);
            }
        }
        let ids = expand_pads(ids, image_pad, &image_counts)?;
        let found = ids.iter().filter(|&&t| t == video_pad).count();
        if found != group_counts.len() {
            return Err(format!(
                "prompt has {found} video frame placeholders but the videos have {} frame groups",
                group_counts.len()
            ));
        }
        let mut ids = expand_pads(&ids, video_pad, &group_counts)?;
        let ids_i64: Vec<i64> = ids.iter().map(|&t| i64::from(t)).collect();
        let mrope = mrope_positions(
            &ids_i64,
            i64::from(image_pad),
            i64::from(video_pad),
            &image_grids,
            &video_grids,
            merge,
        )?;
        for t in &mut ids {
            if *t == video_pad {
                *t = image_pad;
            }
        }
        let pad_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == image_pad).then_some(i))
            .collect();
        let rows = items
            .iter()
            .flat_map(|(r, _, _)| r.iter().copied())
            .collect();
        let mut cache_key = ids.clone();
        let mut at = 0;
        for ((_, _, sha), &count) in items.iter().zip(&counts) {
            for row in 0..count {
                cache_key[pad_positions[at]] = pad_cache_id(cache_ns, sha, row as u32);
                at += 1;
            }
        }
        Ok(Self {
            ids,
            pad_positions,
            rows,
            hidden,
            mrope,
            cache_key,
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

    fn two_by_two_image(hidden: usize, base: f32) -> (Vec<f32>, [u32; 3], [u8; 32]) {
        // grid 1x4x4 -> 2x2 merged = 4 tokens; the pixel hash follows `base`.
        (
            (0..4 * hidden).map(|i| base + i as f32).collect(),
            [1, 4, 4],
            [base as u8; 32],
        )
    }

    /// The cache key keeps text ids, gives each image row a high-bit
    /// pseudo-id, matches for the same image and differs for another one.
    #[test]
    fn cache_key_follows_image_content() {
        let mk = |base| {
            VisionPrompt::new(&[1, PAD, 2], PAD, 8, &[two_by_two_image(1, base)], 1, 2, "").unwrap()
        };
        let (a, same, other) = (mk(10.0), mk(10.0), mk(20.0));
        assert_eq!(a.cache_key, same.cache_key);
        assert_eq!(
            (a.cache_key[0], a.cache_key[5]),
            (1, 2),
            "text ids unchanged"
        );
        assert!(a.cache_key[1..5].iter().all(|&k| k & 0x8000_0000 != 0));
        assert_ne!(
            a.cache_key[1], a.cache_key[2],
            "rows differ within an image"
        );
        assert!((1..5).all(|i| a.cache_key[i] != other.cache_key[i]));
        assert_eq!(a.ids, other.ids, "the model still sees the real pad ids");
        // Another tenant sending the same image gets different image ids.
        let ns = VisionPrompt::new(
            &[1, PAD, 2],
            PAD,
            8,
            &[two_by_two_image(1, 10.0)],
            1,
            2,
            "t2",
        )
        .unwrap();
        assert_eq!(ns.cache_key[0], a.cache_key[0]);
        assert!((1..5).all(|i| ns.cache_key[i] != a.cache_key[i]));
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
            "",
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
            VisionPrompt::new(&[1, PAD, 2], PAD, 8, &[two_by_two_image(1, 0.0)], 1, 2, "").unwrap();
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

    /// A video's groups each sit behind their own pad (timestamp text
    /// between them), take one `t = 1` grid each, and are fed as image pads
    /// with the video's rows in order.
    #[test]
    fn video_groups_expand_and_feed_as_image_pads() {
        const VID: u32 = 8;
        // image (1x4x4 -> 4 tokens), then a video of 2 groups of 2x2 tokens
        // (grid 2x4x4) behind "<t0>" (5) and "<t1>" (6).
        let image = (vec![1.0; 4], [1, 4, 4], [1u8; 32]);
        let video = (
            (0..8).map(|v| 10.0 + v as f32).collect(),
            [2, 4, 4],
            [2u8; 32],
        );
        let ids = [1, PAD, 5, VID, 6, VID, 2];
        let vp =
            VisionPrompt::with_videos(&ids, PAD, VID, &[image, video], &[false, true], 1, 2, "")
                .unwrap();
        assert_eq!(vp.ids.len(), 1 + 4 + 1 + 4 + 1 + 4 + 1);
        assert!(!vp.ids.contains(&VID));
        assert_eq!(vp.pad_positions.len(), 12);
        let mut e = vec![0f32; vp.ids.len()];
        vp.splice(0, &mut e);
        assert_eq!(&e[6..10], &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(&e[11..15], &[14.0, 15.0, 16.0, 17.0]);
        // Positions: image at 1 (advances 2), text 3, group 0 at 4 (h/w
        // 4..5, advances 2), text 6, group 1 at 7, text 9.
        assert_eq!(
            vp.mrope.thw[0][..],
            [0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6, 7, 7, 7, 7, 9]
        );
        assert_eq!(vp.mrope.thw[2][6..10], [4, 5, 4, 5]);
        // A video's group count must match its pads.
        let one = (vec![0.0; 4], [1, 4, 4], [3u8; 32]);
        assert!(
            VisionPrompt::with_videos(&[VID, VID], PAD, VID, &[one], &[true], 1, 2, "").is_err()
        );
    }

    #[test]
    fn wrong_row_count_is_refused() {
        let bad = (vec![0f32; 3], [1, 4, 4], [0u8; 32]);
        assert!(VisionPrompt::new(&[PAD], PAD, 8, &[bad], 1, 2, "").is_err());
    }
}
