//! Native Gemma 4 video: the work Hugging Face's `Gemma4VideoProcessor` and
//! `Gemma4Processor.replace_video_token` do (GenAI 2026.4.1's
//! `InputsEmbedderGemma4` does the same).
//!
//! Gemma 4 has no temporal patch: each sampled frame goes through the image
//! encoder on its own, at the video processor's smaller budget (70 soft
//! tokens on E2B / E4B against 280 for an image), and the prompt's one
//! `<|video|>` becomes, per frame, its time as `mm:ss`, a space, then
//! begin-of-image, `<|video|>` once per soft token and end-of-image; frames
//! are joined by a space:
//!
//! `00:00 <|image><|video|>...<image|> 00:01 <|image><|video|>...<image|>`
//!
//! The language model embeds `<|video|>` soft tokens exactly as image ones
//! (the pad id's embedding, then the encoder rows scattered over them), so
//! rank 0 hands stage 0 the `<|image|>` id in their place ([`stage_ids`]).
//! E2B / E4B attend causally (`use_bidirectional_attention` is unset), so a
//! frame needs no attention block of its own.

use std::borrow::Cow;
use std::path::Path;

use cascadia_types::{MediaInput, MediaKind};

use super::{ImageTokens, PreprocessConfig};

/// The video processor's settings (`processor_config.json`
/// `video_processor`, or a standalone `video_preprocessor_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct VideoConfig {
    /// How each frame is resized and patchified.
    pub frame: PreprocessConfig,
    /// Frames sampled from a longer video (`num_frames`; 0 keeps them all).
    pub num_frames: u32,
    /// `do_sample_frames`: false keeps every frame given.
    pub do_sample_frames: bool,
}

impl VideoConfig {
    /// `Gemma4VideoProcessor`'s class defaults: 70 soft tokens a frame,
    /// 32 frames, otherwise the image processor's settings.
    pub fn gemma4() -> Self {
        Self {
            frame: PreprocessConfig {
                max_soft_tokens: 70,
                ..PreprocessConfig::gemma4()
            },
            num_frames: 32,
            do_sample_frames: true,
        }
    }

    /// Read the video processor settings from an export directory; the
    /// class defaults when it ships none (missing keys keep them too).
    pub fn from_model_dir(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| -> Result<Option<serde_json::Value>, String> {
            let path = dir.join(name);
            if !path.exists() {
                return Ok(None);
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            serde_json::from_str(&text)
                .map(Some)
                .map_err(|e| format!("parsing {}: {e}", path.display()))
        };
        let v = match read("processor_config.json")?.and_then(|v| v.get("video_processor").cloned())
        {
            Some(v) => v,
            None => match read("video_preprocessor_config.json")? {
                Some(v) => v,
                None => return Ok(Self::gemma4()),
            },
        };
        Self::from_json(&v)
    }

    pub(crate) fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let mut base = serde_json::json!({ "max_soft_tokens": 70 });
        if let (Some(b), Some(o)) = (base.as_object_mut(), v.as_object()) {
            for (k, x) in o {
                b.insert(k.clone(), x.clone());
            }
        }
        let frame = PreprocessConfig::from_json(&base)?;
        let d = Self::gemma4();
        Ok(Self {
            frame,
            num_frames: v
                .get("num_frames")
                .and_then(serde_json::Value::as_u64)
                .map_or(d.num_frames, |n| n as u32),
            do_sample_frames: v
                .get("do_sample_frames")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(d.do_sample_frames),
        })
    }

    /// The frames of a `total`-frame video the processor keeps: all of them
    /// unless sampling is on and there are more than `num_frames`, then
    /// `i * total / num_frames` in integers (HF's `default_sample_indices_fn`;
    /// GenAI's `fill_video_metadata` steps in floating point to the same
    /// frames wherever the step is exact).
    pub fn sample(&self, total: usize) -> Vec<usize> {
        let n = if self.do_sample_frames && self.num_frames > 0 {
            (self.num_frames as usize).min(total)
        } else {
            total
        };
        (0..n).map(|i| i * total / n).collect()
    }
}

/// `mm:ss` for a frame `ms` milliseconds in, as `Gemma4Processor` formats
/// `seconds`: whole minutes (not wrapped at an hour), then whole seconds.
pub fn timestamp(ms: u32) -> String {
    let s = f64::from(ms) / 1000.0;
    format!(
        "{:02}:{:02}",
        (s / 60.0).floor() as u64,
        (s % 60.0).floor() as u64
    )
}

/// How the tokens a video expands to are spelled in the prompt text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoMarkup {
    /// `<|video|>`: the chat template's one-per-video tag, and each soft
    /// token after expansion.
    pub video: String,
    /// `<|image>`, begin of image (also of each video frame).
    pub boi: String,
    /// `<image|>`, end of image.
    pub eoi: String,
}

impl VideoMarkup {
    /// Gemma 4's spellings.
    pub fn gemma4() -> Self {
        Self {
            video: "<|video|>".into(),
            boi: "<|image>".into(),
            eoi: "<image|>".into(),
        }
    }

    /// One video's expansion: per frame `mm:ss <boi><video>*n<eoi>`, joined
    /// by a space.
    pub fn expand(&self, timestamps_ms: &[u32], soft_tokens: usize) -> String {
        timestamps_ms
            .iter()
            .map(|&t| {
                format!(
                    "{} {}{}{}",
                    timestamp(t),
                    self.boi,
                    self.video.repeat(soft_tokens),
                    self.eoi
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A video as the processor sees it: the frames it keeps, their times, and
/// the soft tokens each frame takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoLayout {
    pub frames: Vec<usize>,
    pub timestamps_ms: Vec<u32>,
    pub soft_tokens: usize,
}

/// Lay out one video input: its sampled frames and their times, and the
/// soft tokens per frame from the frame size.
pub fn layout(m: &MediaInput, cfg: &VideoConfig) -> Result<VideoLayout, String> {
    if m.kind != MediaKind::Video {
        return Err("not a video".into());
    }
    if m.num_frames == 0 || m.pixels.len() != m.expected_len() {
        return Err("malformed frame stack".into());
    }
    let ts = m
        .timestamps_ms
        .as_deref()
        .filter(|t| t.len() == m.num_frames as usize)
        .ok_or("a video needs one timestamp per frame")?;
    let frames = cfg.sample(m.num_frames as usize);
    let timestamps_ms = frames.iter().map(|&i| ts[i]).collect();
    let soft_tokens = super::soft_tokens_for_size(m.width, m.height, &cfg.frame)?;
    Ok(VideoLayout {
        frames,
        timestamps_ms,
        soft_tokens,
    })
}

/// The text rank 0 tokenizes: the `k`-th `<|video|>` of the rendered prompt
/// replaced by video `k`'s expansion, as the processor does before it
/// tokenizes. Unchanged for a turn without videos; a video it cannot lay out,
/// or a tag count that does not match, leaves the prompt as it is for
/// [`expand_media_tokens`] to refuse.
pub fn prompt_text<'a>(
    prompt: &'a str,
    media: &[MediaInput],
    cfg: &VideoConfig,
    markup: &VideoMarkup,
) -> Cow<'a, str> {
    let videos: Vec<&MediaInput> = media
        .iter()
        .filter(|m| m.kind == MediaKind::Video)
        .collect();
    if videos.is_empty() || prompt.matches(markup.video.as_str()).count() != videos.len() {
        return Cow::Borrowed(prompt);
    }
    let mut expansions = Vec::with_capacity(videos.len());
    for m in videos {
        match layout(m, cfg) {
            Ok(l) => expansions.push(markup.expand(&l.timestamps_ms, l.soft_tokens)),
            Err(_) => return Cow::Borrowed(prompt),
        }
    }
    let mut out =
        String::with_capacity(prompt.len() + expansions.iter().map(String::len).sum::<usize>());
    let mut rest = prompt;
    for e in &expansions {
        let at = rest.find(markup.video.as_str()).expect("counted above");
        out.push_str(&rest[..at]);
        out.push_str(e);
        rest = &rest[at + markup.video.len()..];
    }
    out.push_str(rest);
    Cow::Owned(out)
}

/// One media item's soft tokens, in prompt order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    /// An image behind one `<|image|>` placeholder: its soft-token count.
    Image(usize),
    /// A video already expanded in the text ([`prompt_text`]): its frames
    /// and soft tokens per frame.
    Video { frames: usize, soft_tokens: usize },
}

/// The processor's expansion on token ids of a prompt whose videos
/// [`prompt_text`] expanded: each `<|image|>` becomes `boi, image * n, eoi`
/// and each video's frames must already be there, `boi, video * n, eoi`, in
/// order. Begin/end-of-image tokens beyond the videos' frames (caller text)
/// are refused, as are counts that do not match the media.
pub fn expand_media_tokens(
    ids: &[i64],
    t: &ImageTokens,
    video_token: i64,
    items: &[Item],
) -> Result<Vec<i64>, String> {
    let images: Vec<usize> = items
        .iter()
        .filter_map(|i| match *i {
            Item::Image(n) => Some(n),
            Item::Video { .. } => None,
        })
        .collect();
    let runs: Vec<usize> = items
        .iter()
        .flat_map(|i| match *i {
            Item::Image(_) => Vec::new(),
            Item::Video {
                frames,
                soft_tokens,
            } => vec![soft_tokens; frames],
        })
        .collect();
    let found = ids.iter().filter(|&&x| x == t.image).count();
    if found != images.len() {
        return Err(format!(
            "prompt has {found} image placeholders for {} images",
            images.len()
        ));
    }
    let (boi, eoi) = (
        ids.iter().filter(|&&x| x == t.boi).count(),
        ids.iter().filter(|&&x| x == t.eoi).count(),
    );
    if boi != runs.len() || eoi != runs.len() {
        return Err(if runs.is_empty() {
            "prompt already contains begin/end-of-image tokens".into()
        } else {
            format!(
                "prompt has {boi} begin- and {eoi} end-of-image tokens for {} video frames",
                runs.len()
            )
        });
    }
    // Each video frame: begin-of-image, exactly its soft tokens, end.
    let mut want = runs.iter();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == video_token {
            return Err("a <|video|> token outside a video frame".into());
        }
        if ids[i] == t.boi {
            let n = *want.next().expect("counted above");
            let run = ids[i + 1..]
                .iter()
                .take_while(|&&x| x == video_token)
                .count();
            if run != n || ids.get(i + 1 + run) != Some(&t.eoi) {
                return Err(format!(
                    "a video frame holds {run} soft tokens, expected {n}"
                ));
            }
            i += run + 2;
            continue;
        }
        i += 1;
    }
    let extra: usize = images.iter().map(|n| n + 1).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut next = images.iter();
    for &x in ids {
        if x == t.image {
            let n = *next.next().expect("counted above");
            out.push(t.boi);
            out.extend(std::iter::repeat_n(t.image, n));
            out.push(t.eoi);
        } else {
            out.push(x);
        }
    }
    Ok(out)
}

/// The ids stage 0 takes: video soft tokens as `<|image|>`, the one id its
/// IR splices encoder rows at (and embeds as the pad id, as HF embeds both).
pub fn stage_ids(ids: &[i64], image_token: i64, video_token: i64) -> Vec<i64> {
    ids.iter()
        .map(|&x| if x == video_token { image_token } else { x })
        .collect()
}

/// `Gemma4Model`'s attention blocks (`get_block_sequence_ids_for_mask`):
/// each maximal run of image or video soft tokens is one block, numbered
/// from 0 in order; every other token is -1. Only models with
/// `use_bidirectional_attention == "vision"` attend inside a block both
/// ways; E2B / E4B are causal throughout.
pub fn block_ids(ids: &[i64], image_token: i64, video_token: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(ids.len());
    let mut block = -1i64;
    let mut prev = false;
    for &x in ids {
        let v = x == image_token || x == video_token;
        if v && !prev {
            block += 1;
        }
        out.push(if v { block } else { -1 });
        prev = v;
    }
    out
}

/// The `[1, height, width, 3]` RGB8 frame `i` of a video's frame stack.
pub fn frame(m: &MediaInput, i: usize) -> &[u8] {
    let n = m.width as usize * m.height as usize * 3;
    &m.pixels[i * n..(i + 1) * n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const T: ImageTokens = ImageTokens {
        image: 9,
        boi: 7,
        eoi: 8,
        pad: 0,
    };
    const V: i64 = 6;

    fn video(frames: u32, w: u32, h: u32, ts: Vec<u32>) -> MediaInput {
        MediaInput {
            kind: MediaKind::Video,
            pixels: Arc::from(vec![0u8; (frames * w * h * 3) as usize]),
            num_frames: frames,
            width: w,
            height: h,
            timestamps_ms: Some(ts),
            sha256: [0; 32],
        }
    }

    #[test]
    fn timestamps_format_as_the_processor() {
        assert_eq!(timestamp(0), "00:00");
        assert_eq!(timestamp(999), "00:00");
        assert_eq!(timestamp(59_999), "00:59");
        assert_eq!(timestamp(60_000), "01:00");
        // Minutes do not wrap at the hour.
        assert_eq!(timestamp(3_725_500), "62:05");
    }

    #[test]
    fn sampling_keeps_short_videos_and_strides_long_ones() {
        let c = VideoConfig::gemma4();
        assert_eq!(c.sample(4), vec![0, 1, 2, 3]);
        let s = c.sample(100);
        assert_eq!(s.len(), 32);
        assert_eq!(&s[..4], &[0, 3, 6, 9]);
        let off = VideoConfig {
            do_sample_frames: false,
            ..VideoConfig::gemma4()
        };
        assert_eq!(off.sample(40).len(), 40);
    }

    #[test]
    fn config_reads_the_nested_video_processor() {
        let v = serde_json::json!({"max_soft_tokens": 140, "num_frames": 8,
            "do_sample_frames": false, "patch_size": 16, "pooling_kernel_size": 3});
        let c = VideoConfig::from_json(&v).unwrap();
        assert_eq!(
            (c.frame.max_soft_tokens, c.num_frames, c.do_sample_frames),
            (140, 8, false)
        );
        // Missing budget: the video class default, not the image one.
        let c = VideoConfig::from_json(&serde_json::json!({})).unwrap();
        assert_eq!(c.frame.max_soft_tokens, 70);
    }

    #[test]
    fn prompt_text_expands_each_video_in_order() {
        let m = VideoMarkup {
            video: "<V>".into(),
            boi: "<B>".into(),
            eoi: "<E>".into(),
        };
        let cfg = VideoConfig::gemma4();
        let n = super::super::soft_tokens_for_size(48, 48, &cfg.frame).unwrap();
        let media = [
            video(2, 48, 48, vec![0, 61_000]),
            video(1, 48, 48, vec![500]),
        ];
        let out = prompt_text("a<V>b<V>c", &media, &cfg, &m);
        let f = |t: &str| format!("{t} <B>{}<E>", "<V>".repeat(n));
        assert_eq!(
            out,
            format!("a{} {}b{}c", f("00:00"), f("01:01"), f("00:00"))
        );
        // A tag count that does not match leaves the prompt alone.
        assert_eq!(prompt_text("a<V>", &media, &cfg, &m), "a<V>");
    }

    #[test]
    fn expansion_checks_frames_and_expands_images() {
        // image, then a two-frame video of 2 tokens a frame.
        let ids = [1, 9, 2, 3, 7, V, V, 8, 4, 7, V, V, 8, 5];
        let items = [
            Item::Image(3),
            Item::Video {
                frames: 2,
                soft_tokens: 2,
            },
        ];
        let e = expand_media_tokens(&ids, &T, V, &items).unwrap();
        assert_eq!(
            e,
            vec![1, 7, 9, 9, 9, 8, 2, 3, 7, V, V, 8, 4, 7, V, V, 8, 5]
        );
        assert_eq!(
            stage_ids(&e, T.image, V),
            vec![1, 7, 9, 9, 9, 8, 2, 3, 7, 9, 9, 8, 4, 7, 9, 9, 8, 5]
        );
        // Wrong per-frame count, a stray boi, a stray video token.
        let bad = [
            Item::Image(3),
            Item::Video {
                frames: 2,
                soft_tokens: 3,
            },
        ];
        assert!(expand_media_tokens(&ids, &T, V, &bad).is_err());
        assert!(expand_media_tokens(&[7, 9], &T, V, &[Item::Image(1)]).is_err());
        assert!(expand_media_tokens(&[1, V], &T, V, &[]).is_err());
    }

    #[test]
    fn blocks_are_runs_of_soft_tokens() {
        let ids = [1, 7, V, V, 8, 2, 7, V, 8, 9, 9];
        assert_eq!(
            block_ids(&ids, T.image, V),
            vec![-1, -1, 0, 0, -1, -1, -1, 1, -1, 2, 2]
        );
    }

    #[test]
    fn layout_needs_times_and_samples() {
        let cfg = VideoConfig {
            num_frames: 2,
            ..VideoConfig::gemma4()
        };
        let l = layout(&video(4, 48, 48, vec![0, 1000, 2000, 3000]), &cfg).unwrap();
        assert_eq!(l.frames, vec![0, 2]);
        assert_eq!(l.timestamps_ms, vec![0, 2000]);
        let mut m = video(2, 48, 48, vec![0]);
        assert!(layout(&m, &cfg).is_err());
        m.timestamps_ms = None;
        assert!(layout(&m, &cfg).is_err());
    }
}
