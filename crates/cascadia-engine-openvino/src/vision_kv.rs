//! Warm-resume keys for prompts with images (the `kv_coord` KV cache).
//!
//! The KV cache keys a captured turn on its token ids, but every image of
//! one size expands to the same placeholder ids, so keying on them would
//! resume one image's KV for another. The key here replaces each image's
//! soft-token positions with stand-ins derived from the image's pixel hash
//! ([`pad_cache_id`], the same ids the `qwen35` prefix cache uses), so a
//! later turn resumes only past images it really repeats.
//!
//! [`pad_cache_id`]: crate::qwen35_vision::prompt::pad_cache_id

use crate::qwen35_vision::prompt::pad_cache_id;

/// Where each image sits in an expanded prompt: from its first soft token to
/// one past its last (row breaks inside the run belong to it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSpans(Vec<(usize, usize)>);

impl ImageSpans {
    /// Spans of the images in `ids`, given each image's soft-token count in
    /// prompt order. Fails when the counts do not add up to the placeholders.
    pub fn new(ids: &[i64], image_token: i64, per_image: &[usize]) -> Result<Self, String> {
        let at: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == image_token).collect();
        if at.len() != per_image.iter().sum::<usize>() {
            return Err(format!(
                "{} image positions in the prompt, the images carry {}",
                at.len(),
                per_image.iter().sum::<usize>()
            ));
        }
        let mut spans = Vec::with_capacity(per_image.len());
        let mut next = 0;
        for &n in per_image {
            if n == 0 {
                return Err("an image with no soft tokens".into());
            }
            spans.push((at[next], at[next + n - 1] + 1));
            next += n;
        }
        Ok(Self(spans))
    }

    /// True when position `p` falls strictly inside an image, so a prefix of
    /// `p` tokens would hold part of it.
    pub fn splits(&self, p: usize) -> bool {
        self.0.iter().any(|&(a, b)| a < p && p < b)
    }

    /// The rows of the images that start at or after `p`, the ones a prefill
    /// of `ids[p..]` still splices. Call only when `p` splits no image.
    pub fn rows_from(&self, p: usize, rows: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        self.0
            .iter()
            .zip(rows)
            .filter(|((a, _), _)| *a >= p)
            .map(|(_, r)| r)
            .collect()
    }
}

/// The cache key of an expanded prompt: its ids, with the `k`-th image's
/// soft tokens replaced by stand-ins for that image's hash `hashes[k]` in
/// namespace `ns` (the turn's tenant).
pub fn cache_key(
    ids: &[i64],
    image_token: i64,
    hashes: &[[u8; 32]],
    per_image: &[usize],
    ns: &str,
) -> Vec<i32> {
    let mut slots = hashes
        .iter()
        .zip(per_image)
        .flat_map(|(h, &n)| (0..n as u32).map(move |row| (h, row)));
    ids.iter()
        .map(|&t| {
            if t == image_token {
                match slots.next() {
                    Some((h, row)) => pad_cache_id(ns, h, row) as i32,
                    None => t as i32,
                }
            } else {
                t as i32
            }
        })
        .collect()
}

/// A turn's cache key and image spans, from its expanded `ids` and each
/// image's encoded rows (`rows[k]` holds image `k`'s soft tokens times the
/// hidden width), for the images `media` in prompt order.
pub fn for_turn(
    ids: &[i64],
    image_token: i64,
    rows: &[Vec<f32>],
    media: &[cascadia_types::MediaInput],
    ns: &str,
) -> Result<(Vec<i32>, ImageSpans), String> {
    if rows.len() != media.len() {
        return Err(format!(
            "{} images encoded for {} attached",
            rows.len(),
            media.len()
        ));
    }
    let positions = ids.iter().filter(|&&t| t == image_token).count();
    let floats: usize = rows.iter().map(Vec::len).sum();
    if positions == 0 || !floats.is_multiple_of(positions) {
        return Err(format!(
            "{floats} image values over {positions} image positions"
        ));
    }
    let hidden = floats / positions;
    let per_image: Vec<usize> = rows.iter().map(|r| r.len() / hidden).collect();
    let spans = ImageSpans::new(ids, image_token, &per_image)?;
    let hashes: Vec<[u8; 32]> = media.iter().map(|m| m.sha256).collect();
    Ok((cache_key(ids, image_token, &hashes, &per_image, ns), spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: i64 = 10;
    const BRK: i64 = 11;

    // text, image A (2 rows of 2 with a break), text, image B (1 token), text
    fn prompt() -> Vec<i64> {
        vec![1, IMG, IMG, BRK, IMG, IMG, 2, IMG, 3]
    }

    #[test]
    fn spans_cover_each_image_including_its_breaks() {
        let s = ImageSpans::new(&prompt(), IMG, &[4, 1]).unwrap();
        assert_eq!(s, ImageSpans(vec![(1, 6), (7, 8)]));
        assert!(!s.splits(1) && s.splits(2) && s.splits(5) && !s.splits(6));
        assert!(!s.splits(7) && !s.splits(8));
        assert!(ImageSpans::new(&prompt(), IMG, &[4, 2]).is_err());
    }

    #[test]
    fn rows_from_drops_images_inside_the_prefix() {
        let s = ImageSpans::new(&prompt(), IMG, &[4, 1]).unwrap();
        let rows = || vec![vec![1.0], vec![2.0]];
        assert_eq!(s.rows_from(0, rows()).len(), 2);
        assert_eq!(s.rows_from(6, rows()), vec![vec![2.0]]);
        assert!(s.rows_from(8, rows()).is_empty());
    }

    #[test]
    fn key_tells_images_apart_and_keeps_text() {
        let (a, b) = ([1u8; 32], [2u8; 32]);
        let ka = cache_key(&prompt(), IMG, &[a, b], &[4, 1], "");
        let kb = cache_key(&prompt(), IMG, &[b, b], &[4, 1], "");
        assert_eq!(ka.len(), prompt().len());
        assert_eq!((ka[0], ka[3], ka[6], ka[8]), (1, BRK as i32, 2, 3));
        assert!(
            ka[1] < 0 && ka[1] != ka[2],
            "stand-ins are per row, never a real id"
        );
        assert_ne!(ka[1..6], kb[1..6], "a different image keys differently");
        assert_eq!(ka[7], kb[7], "the same image at the same row keys the same");
        assert_ne!(ka, cache_key(&prompt(), IMG, &[a, b], &[4, 1], "tenant"));
    }
}
