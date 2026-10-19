//! Image input for `/v1/chat/completions`: OpenAI `image_url` content parts
//! (and `/v1/responses`, whose `input_image` parts become them).
//!
//! Every image is decoded, bounded and hashed here, once, so engines only see
//! RGB pixels ([`MediaInput`]). `data:` URLs are always accepted. Remote
//! `https` URLs are fetched only when the operator opts in
//! ([`MediaConfig::allow_remote_urls`]), and only from public addresses: see
//! [`fetch_image`]. Videos (`video_url` parts) become frames here too, by
//! way of [`crate::video`], or one native video item where the engine takes
//! one ([`MediaConfig::native_video`]).

use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use base64::Engine as _;
use cascadia_types::{MediaInput, MediaKind};
use image::{imageops::FilterType, ImageFormat, ImageReader, Limits};
use sha2::{Digest, Sha256};

/// Per-engine image settings. Present on the router only when the served
/// model accepts images; `None` means every `image_url` part gets a 400.
#[derive(Clone, Debug)]
pub struct MediaConfig {
    /// Text that marks image `i` in the prompt, with `{i}` replaced by the
    /// image's index in the request (`{n}`: the index plus one, for
    /// processors that number images from 1). ov-genai uses its universal tag, which
    /// GenAI maps to each model's native one; the staged Qwen3.5 engine takes
    /// the native tag, the same for every image (no `{i}`).
    pub image_placeholder: String,
    /// Text a caller's message must not contain: the engine would read it as
    /// an image tag and fail the request or misplace an image.
    pub reserved_text: Vec<String>,
    /// A message with images reaches the chat template as one string: its
    /// text parts and image tags back to back, as HF's templates (Qwen,
    /// Gemma 4, Pixtral) and GenAI's own tag insertion write a content array.
    /// Joining with "\n" instead put one extra token per join into the
    /// prompt. Gemma 4's template also trims each text part; this does the
    /// same.
    pub trim_text_parts: bool,
    /// Write a message's image tags before its text, whatever the parts'
    /// order, as LLaVA-NeXT's processor template does (`<image>\n` per image,
    /// then the text parts back to back).
    pub images_first: bool,
    /// Images allowed in one request. A video's frames do not count
    /// against it ([`MediaConfig::max_video_frames`] bounds those).
    pub max_images: usize,
    /// Images larger than this many pixels are downscaled to fit, keeping
    /// the aspect ratio. It bounds the image's token cost (about one token
    /// per 1000 pixels on Qwen3.5) and the vision encoder's quadratic
    /// attention.
    pub max_image_pixels: u64,
    /// The same bound for `detail: "low"`.
    pub low_detail_pixels: u64,
    /// Images whose header declares more pixels than this are refused
    /// before decoding (decompression bombs).
    pub max_source_pixels: u64,
    /// Source pixels (each header's width x height) one request's images may
    /// declare together, checked from the headers before any decode. Decode
    /// time grows with source pixels, not encoded bytes: eight highly
    /// compressible 50 MP PNGs fit the byte budget and would hold the decode
    /// slots for seconds.
    pub max_request_source_pixels: u64,
    /// How long a request waits for a decode slot before it is refused with
    /// a 503, as a request over the in-flight cap is.
    pub decode_wait: Duration,
    /// Encoded image bytes one request may carry, on top of the text body
    /// limit. Also the cap on bytes fetched for one request.
    pub max_request_image_bytes: usize,
    /// Fetch `https` image URLs (public addresses only). Off by default: the
    /// server then makes outbound requests on a caller's behalf.
    pub allow_remote_urls: bool,
    /// Who a fetch may reach. Always [`FetchPolicy::PUBLIC_HTTPS`] outside
    /// tests; not exposed as a flag.
    #[doc(hidden)]
    pub fetch_policy: FetchPolicy,
    /// The ffmpeg and ffprobe that decode `video_url` parts. `None` (no
    /// ffmpeg found) refuses every video with a 400 naming `--api-ffmpeg`.
    pub ffmpeg: Option<crate::video::Ffmpeg>,
    /// Videos allowed in one request.
    pub max_videos: usize,
    /// Frames sampled from one video at most, whatever its length.
    pub max_video_frames: usize,
    /// Frames sampled per second of video, before the
    /// [`MediaConfig::max_video_frames`] cap spreads them out.
    pub video_fps: f64,
    /// Longest video accepted, in seconds, read from its container before
    /// any frame is decoded.
    pub max_video_seconds: f64,
    /// Each frame is scaled to fit this many pixels, keeping the aspect
    /// ratio (`detail: "low"` takes the smaller of this and
    /// [`MediaConfig::low_detail_pixels`], `"high"` the larger of this and
    /// [`MediaConfig::max_image_pixels`]). Every frame costs one image's
    /// tokens.
    pub max_video_frame_pixels: u64,
    /// Videos whose frames are larger than this are refused before
    /// decoding: a decoder holds several source frames at once.
    pub max_video_source_pixels: u64,
    /// Source pixels one video may make the decoder produce (its frame
    /// count, from the container, times its frame size), refused before
    /// decoding. The default is 600 s of 1080p30.
    pub max_video_decode_pixels: u64,
    /// Encoded video bytes one request may carry, on top of the image and
    /// text allowances. Also the cap on video bytes fetched for one request.
    pub max_request_video_bytes: usize,
    /// One request's video probes and decodes together; past it ffmpeg is
    /// killed and the request gets a 400.
    pub video_decode_timeout: Duration,
    /// Where video temp files go (the system temp dir when `None`). Tests
    /// only; not exposed as a flag.
    #[doc(hidden)]
    pub video_temp_dir: Option<std::path::PathBuf>,
    /// The engine takes a video whole ([`MediaKind::Video`], its sampled
    /// frames as one item behind [`MediaConfig::video_placeholder`]) instead
    /// of as labelled frames through the image path: ov-genai on the
    /// families GenAI preprocesses video for ([`GENAI_NATIVE_VIDEO`]), and a
    /// staged engine on the families its rank 0 encodes video for
    /// ([`STAGED_NATIVE_VIDEO`], [`MediaConfig::staged_native_video`]).
    pub native_video: bool,
    /// Text that marks native video `i` in the prompt, with `{i}` replaced by
    /// the video's index in the request (its own sequence, apart from the
    /// images'), or the same tag for every video (a staged engine's native
    /// one). Unused unless [`MediaConfig::native_video`].
    pub video_placeholder: String,
    /// A native video's sampled frames are rounded up to a multiple of this
    /// (within [`MediaConfig::max_video_frames`]): Qwen's vision encoder takes
    /// frames in pairs and pads an odd count with a copy of the last frame,
    /// so the extra frame costs nothing and shows the model one more moment.
    pub video_frame_multiple: usize,
    /// A native video's frames must be at least this many pixels a side
    /// (0: no bound). Qwen3-VL / Qwen3.5's video processor refuses a frame
    /// under its 32 px resize factor; the API answers that with a 400 before
    /// the engine is reached.
    pub video_min_side: u32,
}

/// Bytes a `data:` URL needs to carry `raw` bytes base64-encoded, plus room
/// for its `data:image/...;base64,` prefix and JSON quoting.
pub fn base64_len(raw: usize) -> usize {
    raw.div_ceil(3).saturating_mul(4).saturating_add(256)
}

/// GenAI's universal tags: `<ov_genai_image_{i}>` and `<ov_genai_video_{i}>`
/// mark a slot in every model's prompt.
const GENAI_UNIVERSAL_TAGS: [&str; 2] = ["<ov_genai_image_", "<ov_genai_video_"];

/// Each family's native image and video tags, which GenAI's VLMPipeline also
/// reads as media slots ("Use Image or Video Tags in Prompt" in its docs), by
/// the export's `config.json` `model_type`. A tag in caller text would
/// either take an image's place or fail the request inside GenAI, which
/// surfaces as a 5xx and marks the node degraded; refusing it is a 400.
const GENAI_NATIVE_TAGS: &[(&[&str], &[&str])] = &[
    // InternVL2, LLaVA-1.5, LLaVA-NeXT, nanoLLaVA, DeepSeek-OCR-2: `<image>`
    // (some with a newline); MiniCPM-V/-o: `<image>./</image>`.
    (
        &[
            "internvl_chat",
            "llava",
            "llava_next",
            "llava-qwen2",
            "deepseek_vl_v2",
            "deepseekocr",
        ],
        &["<image>"],
    ),
    (&["llava_next_video"], &["<image>", "<video>"]),
    (&["minicpmv", "minicpmo"], &["<image>", "</image>"]),
    // Phi-3/3.5-vision, Phi-4-multimodal: `<|image_{n}|>` from 1;
    // VideoChat-Flash uses the same tag for its videos.
    (&["phi3_v", "phi4mm", "videochat_flash_qwen"], &["<|image_"]),
    (
        &[
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ],
        &[
            "<|vision_start|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_end|>",
        ],
    ),
    // Gemma 3: GenAI 2026.4.1 takes `<start_of_image>` as its native tag
    // (a stray one fails the request), and `<image_soft_token>` is a
    // special token that corrupts the answer. `<img>` is plain text there.
    (&["gemma3"], &["<start_of_image>", "<image_soft_token>"]),
    (&["gemma3n"], &["<image_soft_token>"]),
    (&["gemma4"], &["<|image|>", "<|video|>"]),
];

/// The export `model_type`s GenAI 2026.4.1 preprocesses video for, so a video
/// goes to it whole, and the frame multiple each needs (Qwen's temporal
/// patch is two frames). Verified live (correct answers, a fraction of the
/// frames-as-images tokens): qwen2_5_vl, qwen3_vl, qwen3_5_moe, gemma4. The
/// rest share a verified type's preprocessing in GenAI (it has no
/// `qwen3_vl_moe` type, so that one is not listed). `llava_next_video`
/// is in GenAI's video list but untested here. Every other type
/// (InternVL2 and LLaVA-NeXT throw "Video preprocessing isn't implemented
/// for this model") gets a video as labelled frames.
const GENAI_NATIVE_VIDEO: &[(&str, usize)] = &[
    ("qwen2_vl", 2),
    ("qwen2_5_vl", 2),
    ("qwen3_vl", 2),
    ("qwen3_5", 2),
    ("qwen3_5_moe", 2),
    ("gemma4", 1),
    ("llava_next_video", 1),
];

/// The families whose rank 0 encodes a video whole on a staged engine
/// (`ov-runtime` Qwen VL trees, `crates/cascadia-engine-openvino/src/qwen_vl_vision.rs`;
/// the `qwen35` engine's Qwen3.5 / 3.6 trees, `qwen36.rs`; Gemma 4 trees
/// whose tokenizer has `<|video|>`, the `gemma4` engine's E2B / E4B and
/// `ov-runtime`'s 26B / 31B, `gemma4_vision/video.rs`: the caller opts in
/// per tree),
/// with the native video tag the API writes (the one HF's chat template
/// renders for a video, which rank 0 expands as HF's processor does) and the
/// frame multiple (Qwen's temporal patch is two frames). Every other family
/// on a staged engine gets a video as labelled frames.
/// The last field is the smallest frame side the family's video processor
/// takes (Qwen3-VL / Qwen3.5: their 32 px factor; Qwen2.5-VL scales any frame
/// up).
const STAGED_NATIVE_VIDEO: &[(&str, &str, usize, u32)] = &[
    ("qwen2_5_vl", QWEN_VIDEO_TAG, 2, 0),
    ("qwen3_vl", QWEN_VIDEO_TAG, 2, 32),
    ("qwen3_5", QWEN_VIDEO_TAG, 2, 32),
    ("qwen3_5_moe", QWEN_VIDEO_TAG, 2, 32),
    // Gemma 4 encodes each frame as an image (no temporal patch), and its
    // resize takes any frame size.
    ("gemma4", "<|video|>", 1, 0),
];

/// The video tag HF's Qwen chat templates render.
const QWEN_VIDEO_TAG: &str = "<|vision_start|><|video_pad|><|vision_end|>";

/// The text an ov-genai export must refuse in caller messages: the universal
/// tags plus `model_type`'s native ones, or every known native tag when the
/// type is missing or not listed (a tag GenAI reads that this server does
/// not know is a 5xx; an over-cautious 400 is the cheaper mistake).
fn genai_reserved_text(model_type: Option<&str>) -> Vec<String> {
    let native: Vec<&str> = match model_type.and_then(|t| {
        GENAI_NATIVE_TAGS
            .iter()
            .find(|(types, _)| types.contains(&t))
    }) {
        Some((_, tags)) => tags.to_vec(),
        None => GENAI_NATIVE_TAGS
            .iter()
            .flat_map(|(_, tags)| tags.iter().copied())
            .collect(),
    };
    let mut out: Vec<String> = Vec::new();
    for tag in GENAI_UNIVERSAL_TAGS.into_iter().chain(native) {
        if !out.iter().any(|t| t == tag) {
            out.push(tag.to_owned());
        }
    }
    out
}

impl MediaConfig {
    /// Settings for the ov-genai engine on a VLM-layout export whose
    /// `config.json` names `model_type` (see [`GENAI_NATIVE_TAGS`]).
    pub fn ov_genai(model_type: Option<&str>) -> Self {
        let native = model_type.and_then(|t| GENAI_NATIVE_VIDEO.iter().find(|(n, _)| *n == t));
        Self {
            image_placeholder: "<ov_genai_image_{i}>".into(),
            reserved_text: genai_reserved_text(model_type),
            trim_text_parts: false,
            images_first: false,
            max_images: 8,
            // 1024x1024: about 1000 image tokens on Qwen3.5.
            max_image_pixels: 1 << 20,
            // OpenAI's low detail is a 512x512 budget.
            low_detail_pixels: 512 * 512,
            max_source_pixels: 50_000_000,
            max_request_source_pixels: 100_000_000,
            decode_wait: Duration::from_secs(10),
            max_request_image_bytes: 20 << 20,
            allow_remote_urls: false,
            fetch_policy: FetchPolicy::PUBLIC_HTTPS,
            ffmpeg: None,
            max_videos: 1,
            max_video_frames: 16,
            video_fps: 1.0,
            max_video_seconds: 600.0,
            max_video_frame_pixels: 448 * 448,
            max_video_source_pixels: 3840 * 2160,
            max_video_decode_pixels: 600 * 30 * 1920 * 1080,
            max_request_video_bytes: 32 << 20,
            video_decode_timeout: Duration::from_secs(30),
            video_temp_dir: None,
            native_video: native.is_some(),
            video_placeholder: "<ov_genai_video_{i}>".into(),
            video_frame_multiple: native.map_or(1, |&(_, m)| m),
            video_min_side: 0,
        }
    }

    /// Opt a staged engine's settings into native video for `family` (a
    /// vision tree's family) when its rank 0 takes video whole
    /// ([`STAGED_NATIVE_VIDEO`]); any other family keeps video as frames.
    pub fn staged_native_video(mut self, family: &str) -> Self {
        if let Some(&(_, tag, multiple, min_side)) =
            STAGED_NATIVE_VIDEO.iter().find(|(f, ..)| *f == family)
        {
            self.native_video = true;
            self.video_placeholder = tag.into();
            self.video_frame_multiple = multiple;
            self.video_min_side = min_side;
        }
        self
    }

    /// Whether `video_url` parts are served: images are, ffmpeg was found,
    /// and a request may carry a video (`max_videos` 0 turns video off).
    pub fn accepts_video(&self) -> bool {
        self.ffmpeg.is_some() && self.max_videos > 0
    }

    /// Settings for the staged `qwen35` engine (rank 0 encodes the images):
    /// Qwen's native tag, which rank 0 expands to the image's token count.
    pub fn qwen35_native() -> Self {
        Self {
            image_placeholder: "<|vision_start|><|image_pad|><|vision_end|>".into(),
            reserved_text: [
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
                "<|video_pad|>",
            ]
            .map(String::from)
            .to_vec(),
            ..Self::ov_genai(None)
        }
    }

    /// Settings for a Gemma 4 vision tree on the `ov-runtime` engine (rank 0
    /// encodes the images): the chat template's own `<|image|>`, which rank 0
    /// expands to begin-of-image, the image's soft tokens and end-of-image.
    /// Gemma 4 spends at most 280 tokens on an image whatever its size, so
    /// the pixel budget only bounds decode and resize work.
    pub fn gemma4_native() -> Self {
        Self {
            image_placeholder: "<|image|>".into(),
            reserved_text: [
                "<|image|>",
                "<|image>",
                "<image|>",
                "<|video|>",
                "<|audio|>",
                "<|audio>",
                "<audio|>",
            ]
            .map(String::from)
            .to_vec(),
            trim_text_parts: true,
            ..Self::ov_genai(None)
        }
    }

    /// Settings for a Gemma 3 vision tree on `ov-runtime`: the template's own
    /// `<start_of_image>`, which rank 0 expands on the rendered text as HF's
    /// processor does (blank lines, the soft tokens, `<end_of_image>`).
    /// The template trims each text part, as Gemma 4's does. Every image
    /// costs 256 tokens (it is resized to 896x896), so the pixel budget only
    /// bounds decode and resize work.
    pub fn gemma3_native() -> Self {
        Self {
            image_placeholder: "<start_of_image>".into(),
            reserved_text: ["<start_of_image>", "<image_soft_token>", "<end_of_image>"]
                .map(String::from)
                .to_vec(),
            trim_text_parts: true,
            ..Self::ov_genai(None)
        }
    }

    /// Settings for a Pixtral (LLaVA-layout) vision tree on `ov-runtime`:
    /// the template's own `[IMG]`, which rank 0 expands to the image's rows
    /// of `[IMG]` closed by `[IMG_BREAK]` and, last, `[IMG_END]`. Pixtral
    /// spends a token per 16x16 patch (up to 64x64 at its 1024 px edge), so
    /// the pixel budget bounds prompt length too.
    pub fn pixtral_native() -> Self {
        Self {
            image_placeholder: "[IMG]".into(),
            reserved_text: ["[IMG]", "[IMG_BREAK]", "[IMG_END]"]
                .map(String::from)
                .to_vec(),
            ..Self::ov_genai(None)
        }
    }

    /// Settings for a Phi-3.5-vision (`phi3_v`) vision tree on `ov-runtime`:
    /// the processor's numbered `<|image_N|>` tag (from 1) and the newline
    /// the model card and GenAI put after it. Rank 0 cuts the prompt at each
    /// tag, as `Phi3VProcessor` does, and expands it to the image's soft
    /// tokens (757 for a 448x448 image with 4 crops).
    pub fn phi3v_native() -> Self {
        Self {
            image_placeholder: "<|image_{n}|>\n".into(),
            reserved_text: ["<|image_", "<|image|>"].map(String::from).to_vec(),
            ..Self::ov_genai(None)
        }
    }

    /// Settings for a LLaVA-NeXT (llava-v1.6) vision tree on `ov-runtime`:
    /// `<image>` followed by a newline, every image of a message before its
    /// text, which is what the processor's chat template renders. Rank 0
    /// expands each `<image>` to the image's tiles and newline rows (at most
    /// 2928 tokens on llava-v1.6's pinpoints).
    pub fn llava_next_native() -> Self {
        Self {
            image_placeholder: "<image>\n".into(),
            reserved_text: vec!["<image>".into()],
            images_first: true,
            ..Self::ov_genai(None)
        }
    }

    /// Settings for an InternVL2 (`internvl_chat`) vision tree on
    /// `ov-runtime`: `<img><IMG_CONTEXT></img>` and a newline, the string
    /// `chat` writes for `<image>\n` with one context token; rank 0 widens
    /// it to 256 per tile.
    pub fn internvl_native() -> Self {
        Self {
            image_placeholder: "<img><IMG_CONTEXT></img>\n".into(),
            reserved_text: ["<img>", "</img>", "<IMG_CONTEXT>"]
                .map(String::from)
                .to_vec(),
            ..Self::ov_genai(None)
        }
    }

    /// A message's content once its parts are joined: back to back in order,
    /// or (`images_first`) every image tag first, then the text parts. Each
    /// part is `(is_image, text)`.
    pub fn join_parts(&self, parts: Vec<(bool, String)>) -> String {
        if !self.images_first {
            return parts.into_iter().map(|(_, t)| t).collect();
        }
        let (images, texts): (Vec<_>, Vec<_>) = parts.into_iter().partition(|(img, _)| *img);
        images.into_iter().chain(texts).map(|(_, t)| t).collect()
    }

    /// One text part of a message with images, as it enters the prompt.
    pub fn text_part(&self, t: String) -> String {
        if self.trim_text_parts {
            t.trim().to_string()
        } else {
            t
        }
    }

    pub fn placeholder(&self, i: usize) -> String {
        self.image_placeholder
            .replace("{i}", &i.to_string())
            .replace("{n}", &(i + 1).to_string())
    }

    /// The placeholder of native video `i` (see [`MediaConfig::native_video`]).
    pub fn video_tag(&self, i: usize) -> String {
        self.video_placeholder.replace("{i}", &i.to_string())
    }

    /// The placeholder's text before its index, when it is indexed.
    fn indexed_prefix(&self) -> Option<&str> {
        let at = ["{i}", "{n}"]
            .iter()
            .filter_map(|k| self.image_placeholder.find(k))
            .min()?;
        Some(&self.image_placeholder[..at])
    }

    /// A reserved string `prompt` carries more often than its `n` image
    /// placeholders and `videos` native video placeholders do, if any.
    pub fn reserved_beyond_placeholders<'a>(
        &'a self,
        prompt: &str,
        n: usize,
        videos: usize,
    ) -> Option<&'a str> {
        self.reserved_text.iter().map(String::as_str).find(|r| {
            let allowed: usize = (0..n)
                .map(|i| self.placeholder(i).matches(r).count())
                .chain((0..videos).map(|j| self.video_tag(j).matches(r).count()))
                .sum();
            prompt.matches(r).count() > allowed
        })
    }

    /// The reserved text `text` contains, if any.
    pub fn reserved_in<'a>(&'a self, text: &str) -> Option<&'a str> {
        self.reserved_text
            .iter()
            .map(String::as_str)
            .find(|r| text.contains(r))
    }

    /// Whether the rendered `prompt` carries `n` native videos' placeholders,
    /// each exactly once (indexed tags), or exactly `n` of the one shared
    /// tag; `Err` holds how many video tags it carries.
    pub fn video_placeholders_complete(&self, prompt: &str, n: usize) -> Result<(), usize> {
        let Some(at) = self.video_placeholder.find("{i}") else {
            let found = prompt.matches(self.video_placeholder.as_str()).count();
            return if found == n { Ok(()) } else { Err(found) };
        };
        let found = prompt.matches(&self.video_placeholder[..at]).count();
        if found == n && (0..n).all(|j| prompt.matches(&self.video_tag(j)).count() == 1) {
            Ok(())
        } else {
            Err(found)
        }
    }

    /// Whether the rendered `prompt` carries `n` images' placeholders, each
    /// exactly once (indexed tags), or exactly `n` of the one shared tag.
    pub fn placeholders_complete(&self, prompt: &str, n: usize) -> Result<(), usize> {
        if let Some(prefix) = self.indexed_prefix() {
            let found = prompt.matches(prefix).count();
            let each_once = (0..n).all(|i| prompt.matches(&self.placeholder(i)).count() == 1);
            if found == n && each_once {
                Ok(())
            } else {
                Err(found)
            }
        } else {
            let found = prompt.matches(self.image_placeholder.as_str()).count();
            if found == n {
                Ok(())
            } else {
                Err(found)
            }
        }
    }
}

/// Hard ceiling on either side, whatever the pixel budget.
pub(crate) const MAX_SIDE: u32 = 16_384;
/// Qwen's processor refuses aspect ratios past 200:1; refusing here gives the
/// caller a 400 instead of an engine failure.
pub(crate) const MAX_ASPECT: u32 = 200;

/// Decode one `data:` image URL into pixels the engine can take as is.
#[cfg(test)]
pub fn decode_image_url(
    url: &str,
    detail: Option<&str>,
    cfg: &MediaConfig,
) -> Result<MediaInput, String> {
    let bytes = image_url_bytes(url, detail, cfg)?;
    decode_image_bytes(&bytes, detail, cfg)
}

/// The encoded image a `data:` URL carries, its `detail` checked.
pub fn image_url_bytes(
    url: &str,
    detail: Option<&str>,
    cfg: &MediaConfig,
) -> Result<Vec<u8>, String> {
    check_detail(detail)?;
    data_url_bytes(url, cfg.allow_remote_urls)
}

/// The size an encoded image's header declares, refused as
/// [`decode_image_bytes`] would refuse it, without decoding a pixel.
pub fn source_size(bytes: &[u8], cfg: &MediaConfig) -> Result<(u32, u32), String> {
    let format = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("could not read the image: {e}"))?
        .format();
    let format = match format {
        Some(f @ (ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif)) => {
            f
        }
        _ => return Err("unsupported image format; send PNG, JPEG, WEBP or GIF".into()),
    };
    let (w, h) = ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|e| format!("could not read the image header: {e}"))?;
    check_dimensions(w, h, cfg)?;
    Ok((w, h))
}

/// Refuse a request whose images declare more source pixels together than
/// [`MediaConfig::max_request_source_pixels`].
pub fn check_request_pixels(sizes: &[(u32, u32)], cfg: &MediaConfig) -> Result<(), String> {
    let total: u64 = sizes
        .iter()
        .map(|&(w, h)| u64::from(w) * u64::from(h))
        .sum();
    if total > cfg.max_request_source_pixels {
        return Err(format!(
            "this request's images are {total} pixels together; the limit is {} \
             (send fewer or smaller images)",
            cfg.max_request_source_pixels
        ));
    }
    Ok(())
}

/// OpenAI's `detail` values. `original` (full resolution, in the current
/// SDKs' types) is served as `high`: the pixel budget still bounds the image.
fn check_detail(detail: Option<&str>) -> Result<(), String> {
    match detail {
        None | Some("auto" | "high" | "original" | "low") => Ok(()),
        Some(other) => Err(format!(
            "image detail `{other}` is not supported; use `low`, `high`, `original` or `auto`"
        )),
    }
}

/// Decode encoded image bytes (from a `data:` URL or a fetch).
pub fn decode_image_bytes(
    bytes: &[u8],
    detail: Option<&str>,
    cfg: &MediaConfig,
) -> Result<MediaInput, String> {
    check_detail(detail)?;
    let budget = match detail {
        Some("low") => cfg.low_detail_pixels,
        _ => cfg.max_image_pixels,
    };

    // Check the declared size before decoding a single pixel.
    source_size(bytes, cfg)?;
    let format = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("could not read the image: {e}"))?
        .format()
        .expect("source_size checked the format");

    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(cfg.max_source_pixels * 4 + (64 << 20));
    reader.limits(limits);
    let mut rgb = reader
        .decode()
        .map_err(|e| format!("could not decode the image: {e}"))?
        .to_rgb8();

    let (w, h) = rgb.dimensions();
    let pixels = u64::from(w) * u64::from(h);
    if pixels > budget {
        let (nw, nh) = downscaled_size(w, h, budget);
        rgb = image::imageops::resize(&rgb, nw, nh, FilterType::CatmullRom);
    }

    let (width, height) = rgb.dimensions();
    Ok(rgb_image(width, height, rgb.into_raw()))
}

/// One RGB8 image as the engine takes it, hashed (size and pixels) so the
/// encode cache keys on what the model sees. A video frame is one too.
pub(crate) fn rgb_image(width: u32, height: u32, raw: Vec<u8>) -> MediaInput {
    let mut hasher = Sha256::new();
    hasher.update(width.to_le_bytes());
    hasher.update(height.to_le_bytes());
    hasher.update(&raw);
    MediaInput {
        kind: MediaKind::Image,
        pixels: raw.into(),
        num_frames: 1,
        width,
        height,
        timestamps_ms: None,
        sha256: hasher.finalize().into(),
    }
}

/// The size a `w x h` image over `budget` pixels is resized to: the short
/// side rounds up and the long side down (and to what the budget leaves), so
/// the aspect ratio can only shrink. Flooring both sides let a 16384x82 image
/// (199.8:1, accepted by `check_dimensions`) become 14474x72 (201:1), which
/// Qwen's processor refuses after the request has been accepted.
pub(crate) fn downscaled_size(w: u32, h: u32, budget: u64) -> (u32, u32) {
    let scale = (budget as f64 / (f64::from(w) * f64::from(h))).sqrt();
    let (long, short) = (w.max(h), w.min(h));
    // The epsilon keeps an exact product (200 * 0.05) from rounding up.
    let ns = ((f64::from(short) * scale - 1e-9).ceil() as u32).clamp(1, short);
    let nl = ((f64::from(long) * scale).floor() as u32)
        .min((budget / u64::from(ns)).min(u64::from(u32::MAX)) as u32)
        .max(ns);
    if w >= h {
        (nl, ns)
    } else {
        (ns, nl)
    }
}

fn check_dimensions(w: u32, h: u32, cfg: &MediaConfig) -> Result<(), String> {
    if w == 0 || h == 0 {
        return Err("image has no pixels".into());
    }
    if w > MAX_SIDE || h > MAX_SIDE || u64::from(w) * u64::from(h) > cfg.max_source_pixels {
        return Err(format!(
            "image is {w}x{h}; the limit is {MAX_SIDE} pixels a side and {} pixels in total",
            cfg.max_source_pixels
        ));
    }
    if w.max(h) > MAX_ASPECT * w.min(h) {
        return Err(format!(
            "image is {w}x{h}; the aspect ratio must be at most {MAX_ASPECT}:1"
        ));
    }
    Ok(())
}

/// The payload of a base64 `data:image/...` URL.
fn data_url_bytes(url: &str, remote_allowed: bool) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(if remote_allowed {
            "image URLs must be `https://` or `data:` (base64-encoded)".into()
        } else {
            "only `data:` image URLs are supported on this server; send the image \
             base64-encoded (`data:image/png;base64,...`)"
                .into()
        });
    };
    data_url_payload(rest, |mime| {
        if mime.starts_with("image/") {
            Ok(())
        } else {
            Err(format!("data URL media type `{mime}` is not an image type"))
        }
    })
}

/// The payload of a `data:` URL (`rest` follows `data:`) whose media type
/// `check_mime` accepts. Only base64 payloads are taken.
pub(crate) fn data_url_payload(
    rest: &str,
    check_mime: impl FnOnce(&str) -> Result<(), String>,
) -> Result<Vec<u8>, String> {
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err("malformed data URL: no `,` before the payload".into());
    };
    let mut fields = meta.split(';');
    check_mime(fields.next().unwrap_or(""))?;
    if !fields.any(|f| f == "base64") {
        return Err("data URL must be base64-encoded (`;base64,`)".into());
    }
    let payload = payload.trim();
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload))
        .map_err(|e| format!("data URL payload is not valid base64: {e}"))
}

/// Whether `url` should be fetched rather than decoded inline.
pub fn is_remote(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("http://")
}

/// Redirects followed per image, each re-validated like the original URL.
const MAX_REDIRECTS: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Who a fetch may reach. Production is [`FetchPolicy::PUBLIC_HTTPS`]; tests
/// use a policy that also allows plain HTTP to loopback so they can run a
/// local server.
#[derive(Clone, Copy, Debug)]
pub struct FetchPolicy {
    pub allow_http: bool,
    pub allow_non_public: bool,
    /// Whole fetch of one image: DNS, every redirect hop, headers and body.
    /// One deadline, not one per hop, so redirects cannot stretch it.
    pub per_image: Duration,
    /// All remote images of one request together. The request holds an
    /// in-flight permit while it fetches, so this bounds how long a caller
    /// can occupy one.
    pub per_request: Duration,
}

impl FetchPolicy {
    pub const PUBLIC_HTTPS: Self = Self {
        allow_http: false,
        allow_non_public: false,
        per_image: Duration::from_secs(10),
        per_request: Duration::from_secs(20),
    };
}

/// Fetch an image over HTTPS for a caller, without letting the caller point
/// the server at anything it should not reach (SSRF):
///
/// - `https` on port 443 only, no credentials in the URL, no proxy;
/// - every address the host resolves to must be public (no loopback,
///   private, link-local, CGNAT/Tailscale, multicast, reserved, or v4 hidden
///   inside v6), and the connection is pinned to the checked address, so a
///   second DNS answer cannot redirect it (rebinding);
/// - at most [`MAX_REDIRECTS`] redirects, each checked the same way;
/// - `image/*` content type, and at most `max_bytes`, enforced while reading;
/// - one deadline ([`FetchPolicy::per_image`]) over the whole fetch.
///
/// Errors a caller could use to map the inside (what a host resolves to, DNS
/// or connection failures) are reported generically and logged in detail.
pub async fn fetch_image(
    url: &str,
    max_bytes: usize,
    policy: FetchPolicy,
) -> Result<Vec<u8>, String> {
    fetch_remote(url, max_bytes, policy, Remote::Image).await
}

/// What a remote fetch is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Remote {
    /// `image/*` only.
    Image,
    /// `video/*`, or `application/octet-stream` or no type at all, which
    /// CDNs often send for video files: the bytes are sniffed before ffmpeg
    /// sees them.
    Video,
}

impl Remote {
    fn name(self) -> &'static str {
        match self {
            Remote::Image => "image",
            Remote::Video => "video",
        }
    }

    fn accepts(self, ctype: &str) -> bool {
        let essence = ctype.split(';').next().unwrap_or("").trim();
        match self {
            Remote::Image => ctype.starts_with("image/"),
            Remote::Video => {
                essence.starts_with("video/")
                    || essence.is_empty()
                    || essence.eq_ignore_ascii_case("application/octet-stream")
            }
        }
    }
}

/// [`fetch_image`] for any media kind, named in errors by `kind`.
pub async fn fetch_remote(
    url: &str,
    max_bytes: usize,
    policy: FetchPolicy,
    kind: Remote,
) -> Result<Vec<u8>, String> {
    let what = kind.name();
    match tokio::time::timeout(policy.per_image, fetch_inner(url, max_bytes, policy, kind)).await {
        Ok(r) => r,
        Err(_) => Err(format!(
            "fetching the {what} took longer than {} s",
            policy.per_image.as_secs_f32()
        )),
    }
}

/// Fetch failures whose cause stays in the server log.
fn unreachable(what: &str) -> String {
    format!("could not fetch the {what} from this server")
}

async fn fetch_inner(
    url: &str,
    max_bytes: usize,
    policy: FetchPolicy,
    kind: Remote,
) -> Result<Vec<u8>, String> {
    let what = kind.name();
    let mut current = url::Url::parse(url).map_err(|e| format!("invalid {what} URL: {e}"))?;
    for _ in 0..=MAX_REDIRECTS {
        let addr = check_target(&current, policy, what).await?;
        let host = current.host_str().unwrap_or_default().to_owned();
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .https_only(!policy.allow_http)
            .resolve(&host, addr)
            .user_agent("cascadia-image-fetch")
            .build()
            .map_err(|e| format!("could not set up the {what} fetch: {e}"))?;
        let mut resp = client.get(current.clone()).send().await.map_err(|e| {
            tracing::warn!(host = %host, error = %without_url(e), "{what} fetch failed");
            unreachable(what)
        })?;
        if resp.status().is_redirection() {
            let next = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("{what} URL redirected without a Location"))?;
            current = current
                .join(next)
                .map_err(|e| format!("{what} URL redirected to an invalid URL: {e}"))?;
            continue;
        }
        if !resp.status().is_success() {
            return Err(format!(
                "fetching the {what} returned HTTP {}",
                resp.status().as_u16()
            ));
        }
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !kind.accepts(ctype) {
            let article = if what.starts_with(['a', 'e', 'i', 'o', 'u']) {
                "an"
            } else {
                "a"
            };
            return Err(format!(
                "{what} URL returned content type `{ctype}`, not {article} {what}"
            ));
        }
        if resp.content_length().is_some_and(|n| n > max_bytes as u64) {
            return Err(format!("{what} is larger than the {max_bytes}-byte limit"));
        }
        let mut body = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| {
            tracing::warn!(host = %host, error = %without_url(e), "{what} fetch failed mid-body");
            unreachable(what)
        })? {
            if body.len() + chunk.len() > max_bytes {
                return Err(format!("{what} is larger than the {max_bytes}-byte limit"));
            }
            body.extend_from_slice(&chunk);
        }
        return Ok(body);
    }
    Err(format!(
        "{what} URL redirected more than {MAX_REDIRECTS} times"
    ))
}

/// reqwest errors embed the URL, which may carry the caller's tokens in its
/// query; the message the caller sees does not need it.
fn without_url(e: reqwest::Error) -> String {
    let e = e.without_url();
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(&e);
    while let Some(s) = src {
        msg = format!("{msg}: {s}");
        src = s.source();
    }
    msg
}

/// Validate a URL and resolve it to the one address the fetch will use.
async fn check_target(
    url: &url::Url,
    policy: FetchPolicy,
    what: &str,
) -> Result<SocketAddr, String> {
    match url.scheme() {
        "https" => {}
        "http" if policy.allow_http => {}
        other => {
            return Err(format!(
                "{what} URL scheme `{other}` is not allowed; use https"
            ))
        }
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(format!("{what} URLs must not carry credentials"));
    }
    let port = url.port_or_known_default().unwrap_or(0);
    if !policy.allow_http && port != 443 {
        return Err(format!("{what} URLs must use port 443, not {port}"));
    }
    let host = match url.host() {
        Some(url::Host::Domain(d)) => d.to_owned(),
        Some(url::Host::Ipv4(ip)) => ip.to_string(),
        Some(url::Host::Ipv6(ip)) => ip.to_string(),
        None => return Err(format!("{what} URL has no host")),
    };
    let addrs: Vec<SocketAddr> = match tokio::net::lookup_host((host.as_str(), port)).await {
        Ok(a) => a.collect(),
        Err(e) => {
            tracing::warn!(host = %host, error = %e, "{what} host did not resolve");
            return Err(unreachable(what));
        }
    };
    let Some(first) = addrs.first().copied() else {
        tracing::warn!(host = %host, "{what} host resolved to no addresses");
        return Err(unreachable(what));
    };
    // Every answer must be public: a host that resolves to one public and one
    // internal address would otherwise be a coin flip away from the inside.
    // The caller learns only that the host is refused, never what it
    // resolved to: that would turn the endpoint into an internal DNS oracle.
    if !policy.allow_non_public {
        if let Some(bad) = addrs.iter().find(|a| !is_public(a.ip())) {
            tracing::warn!(host = %host, ip = %bad.ip(), "{what} host refused: not a public address");
            return Err(format!("{what} host is not a public address"));
        }
    }
    Ok(first)
}

/// Globally routable unicast only.
pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || o[0] == 0 // 0.0.0.0/8 "this network"
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10 CGNAT (Tailscale)
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24 IETF assignments
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18.0.0/15 benchmarking
        || o[0] >= 240) // 240.0.0.0/4 reserved
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    // v4 carried inside v6 is judged as that v4 address.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_public_v4(v4);
    }
    if s[0] == 0x0064 && s[1] == 0xff9b {
        // 64:ff9b::/96 NAT64
        return is_public_v4(Ipv4Addr::new(
            (s[6] >> 8) as u8,
            s[6] as u8,
            (s[7] >> 8) as u8,
            s[7] as u8,
        ));
    }
    if s[0] == 0x2002 {
        // 2002::/16 6to4 embeds the v4 address in segments 1-2
        return is_public_v4(Ipv4Addr::new(
            (s[1] >> 8) as u8,
            s[1] as u8,
            (s[2] >> 8) as u8,
            s[2] as u8,
        ));
    }
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
        || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
        || (s[0] & 0xffc0) == 0xfec0 // fec0::/10 site-local (deprecated)
        || (s[0] == 0x2001 && s[1] == 0x0db8) // 2001:db8::/32 documentation
        || (s[0] == 0x2001 && s[1] == 0) // 2001::/32 Teredo
        || (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0) // 100::/64 discard
        || s[..6] == [0, 0, 0, 0, 0, 0]) // ::/96 v4-compatible (deprecated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_data_url(w: u32, h: u32) -> String {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 7])
        });
        let mut buf = Vec::new();
        img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(buf)
        )
    }

    #[test]
    fn decodes_a_png_data_url_to_rgb() {
        let cfg = MediaConfig::ov_genai(None);
        let m = decode_image_url(&png_data_url(40, 30), None, &cfg).unwrap();
        assert_eq!((m.width, m.height, m.num_frames), (40, 30, 1));
        assert_eq!(m.pixels.len(), m.expected_len());
        assert_eq!(&m.pixels[..3], &[0, 0, 7]);
        // Same pixels, same hash; it is what caches will key on.
        let again = decode_image_url(&png_data_url(40, 30), Some("auto"), &cfg).unwrap();
        assert_eq!(m.sha256, again.sha256);
        let other = decode_image_url(&png_data_url(30, 40), None, &cfg).unwrap();
        assert_ne!(m.sha256, other.sha256);
    }

    #[test]
    fn downscales_to_the_pixel_budget_keeping_aspect() {
        let mut cfg = MediaConfig::ov_genai(None);
        cfg.max_image_pixels = 100 * 50;
        let m = decode_image_url(&png_data_url(400, 200), None, &cfg).unwrap();
        assert_eq!((m.width, m.height), (100, 50));
        cfg.low_detail_pixels = 20 * 10;
        let low = decode_image_url(&png_data_url(400, 200), Some("low"), &cfg).unwrap();
        assert_eq!((low.width, low.height), (20, 10));
    }

    /// A ratio just under the cap must stay under it through the downscale:
    /// 16384x82 used to become 14474x72 (201:1), which the Qwen processor
    /// refuses after the request was accepted.
    #[test]
    fn downscale_never_grows_the_aspect_ratio() {
        let cfg = MediaConfig::qwen35_native();
        let m = decode_image_url(&png_data_url(16384, 82), None, &cfg).unwrap();
        let (w, h) = (m.width, m.height);
        assert!(w <= MAX_ASPECT * h, "{w}x{h} exceeds {MAX_ASPECT}:1");
        assert!(
            u64::from(w) * u64::from(h) <= cfg.max_image_pixels,
            "{w}x{h}"
        );
        assert_eq!((w, h), (14364, 73));
        // Portrait the same way, and exact scales keep exact sizes.
        assert_eq!(downscaled_size(82, 16384, 1 << 20), (73, 14364));
        assert_eq!(downscaled_size(400, 200, 20 * 10), (20, 10));
        for (w, h, b) in [
            (16384u32, 82u32, 1u64 << 20),
            (5000, 25, 4096),
            (1001, 777, 1000),
        ] {
            let (nw, nh) = downscaled_size(w, h, b);
            assert!(u64::from(nw) * u64::from(nh) <= b, "{w}x{h} -> {nw}x{nh}");
            assert!(nw.max(nh) as f64 / nw.min(nh) as f64 <= w.max(h) as f64 / w.min(h) as f64);
        }
    }

    #[test]
    fn refuses_what_it_cannot_take() {
        let cfg = MediaConfig::ov_genai(None);
        for (url, detail, needle) in [
            ("https://example.com/a.png", None, "only `data:`"),
            ("ftp://example.com/a.png", None, "only `data:`"),
            ("data:text/plain;base64,aGk=", None, "not an image type"),
            ("data:image/png,rawbytes", None, "base64-encoded"),
            ("data:image/png;base64,@@@", None, "not valid base64"),
            (
                "data:image/png;base64,aGVsbG8=",
                None,
                "unsupported image format",
            ),
        ] {
            let err = decode_image_url(url, detail, &cfg).unwrap_err();
            assert!(err.contains(needle), "{url}: {err}");
        }
        let err = decode_image_url(&png_data_url(4, 4), Some("ultra"), &cfg).unwrap_err();
        assert!(err.contains("detail `ultra`"), "{err}");
        let err = decode_image_url(&png_data_url(402, 2), None, &cfg).unwrap_err();
        assert!(err.contains("aspect ratio"), "{err}");
        let mut small = MediaConfig::ov_genai(None);
        small.max_source_pixels = 100;
        let err = decode_image_url(&png_data_url(20, 20), None, &small).unwrap_err();
        assert!(err.contains("pixels in total"), "{err}");
    }

    /// A remote video may come as `application/octet-stream` or untyped
    /// (its bytes are sniffed later); an image must say it is one.
    #[test]
    fn remote_content_types_by_kind() {
        for (ctype, image, video) in [
            ("image/png", true, false),
            ("video/mp4", false, true),
            ("video/webm; codecs=vp9", false, true),
            ("application/octet-stream", false, true),
            ("Application/Octet-Stream; charset=binary", false, true),
            ("", false, true),
            ("text/html", false, false),
        ] {
            assert_eq!(Remote::Image.accepts(ctype), image, "{ctype}");
            assert_eq!(Remote::Video.accepts(ctype), video, "{ctype}");
        }
    }

    #[test]
    fn public_address_rules() {
        let public = [
            "8.8.8.8",
            "1.1.1.1",
            "2606:4700:4700::1111",
            "100.63.255.255",
            "100.128.0.1",
        ];
        let internal = [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.72.95.18",
            "100.64.0.1",
            "0.0.0.0",
            "0.1.2.3",
            "255.255.255.255",
            "224.0.0.1",
            "240.0.0.1",
            "192.0.2.1",
            "198.18.0.1",
            "192.0.0.8",
            "::",
            "::1",
            "fd00::1",
            "fe80::1",
            "fec0::1",
            "ff02::1",
            "2001:db8::1",
            "2001::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "64:ff9b::a00:1",
            "2002:7f00:1::1",
            "::127.0.0.1",
            "100::1",
        ];
        for ip in public {
            assert!(is_public(ip.parse().unwrap()), "{ip} should be public");
        }
        for ip in internal {
            assert!(!is_public(ip.parse().unwrap()), "{ip} should be refused");
        }
        // v4 inside v6 is judged as the v4 address.
        assert!(is_public("::ffff:8.8.8.8".parse().unwrap()));
        assert!(is_public("2002:0808:0808::1".parse().unwrap()));
    }

    #[tokio::test]
    async fn remote_urls_are_refused_before_any_connection() {
        for (url, needle) in [
            ("http://example.com/a.png", "scheme `http`"),
            ("https://user:pw@example.com/a.png", "credentials"),
            ("https://example.com:8443/a.png", "port 443"),
            ("https://127.0.0.1/a.png", "not a public address"),
            ("https://localhost/a.png", "not a public address"),
            ("https://10.0.0.5/a.png", "not a public address"),
            ("https://100.72.95.18/a.png", "not a public address"),
            (
                "https://[::ffff:169.254.169.254]/a.png",
                "not a public address",
            ),
            ("https://[fd00::1]/a.png", "not a public address"),
        ] {
            let err = fetch_image(url, 1 << 20, FetchPolicy::PUBLIC_HTTPS)
                .await
                .unwrap_err();
            assert!(err.contains(needle), "{url}: {err}");
        }
        // What a refused name resolved to stays in the server log.
        let err = fetch_image(
            "https://localhost/a.png",
            1 << 20,
            FetchPolicy::PUBLIC_HTTPS,
        )
        .await
        .unwrap_err();
        assert_eq!(err, "image host is not a public address");
    }

    /// One deadline covers DNS, redirects and the body; a slow host cannot
    /// hold the request's in-flight permit past it.
    #[tokio::test]
    async fn fetch_has_one_deadline() {
        let base = image_server().await;
        let quick = FetchPolicy {
            per_image: Duration::from_millis(300),
            ..LOCAL
        };
        let t0 = std::time::Instant::now();
        let err = fetch_image(&format!("{base}/slow"), 1 << 20, quick)
            .await
            .unwrap_err();
        assert!(err.contains("took longer than"), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    }

    /// A local server stands in for a remote host; the test policy allows
    /// plain HTTP to loopback, which production never does.
    async fn image_server() -> String {
        use axum::http::header;
        use axum::response::IntoResponse;
        use axum::routing::get;
        let png = {
            let img = image::RgbImage::from_pixel(6, 4, image::Rgb([1, 2, 3]));
            let mut buf = Vec::new();
            img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
                .unwrap();
            buf
        };
        let png2 = png.clone();
        let app = axum::Router::new()
            .route(
                "/img.png",
                get(move || async move { ([(header::CONTENT_TYPE, "image/png")], png.clone()) }),
            )
            .route(
                "/hop",
                get(|| async {
                    (
                        axum::http::StatusCode::FOUND,
                        [(header::LOCATION, "/img.png")],
                    )
                        .into_response()
                }),
            )
            .route(
                "/loop",
                get(|| async {
                    (axum::http::StatusCode::FOUND, [(header::LOCATION, "/loop")]).into_response()
                }),
            )
            .route(
                "/page",
                get(|| async { ([(header::CONTENT_TYPE, "text/html")], "<html/>") }),
            )
            .route(
                "/big",
                get(
                    move || async move { ([(header::CONTENT_TYPE, "image/png")], png2.repeat(50)) },
                ),
            )
            .route("/gone", get(|| async { axum::http::StatusCode::NOT_FOUND }))
            .route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    "late"
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    const LOCAL: FetchPolicy = FetchPolicy {
        allow_http: true,
        allow_non_public: true,
        ..FetchPolicy::PUBLIC_HTTPS
    };

    #[tokio::test]
    async fn fetch_follows_checked_redirects_and_enforces_limits() {
        let base = image_server().await;
        let cfg = MediaConfig::ov_genai(None);
        let bytes = fetch_image(&format!("{base}/img.png"), 1 << 20, LOCAL)
            .await
            .unwrap();
        let m = decode_image_bytes(&bytes, None, &cfg).unwrap();
        assert_eq!((m.width, m.height), (6, 4));
        assert_eq!(
            fetch_image(&format!("{base}/hop"), 1 << 20, LOCAL)
                .await
                .unwrap(),
            bytes
        );
        for (path, max, needle) in [
            ("/loop", 1 << 20, "more than 3 times"),
            ("/page", 1 << 20, "content type `text/html`"),
            ("/big", bytes.len() * 10, "byte limit"),
            ("/gone", 1 << 20, "HTTP 404"),
        ] {
            let err = fetch_image(&format!("{base}{path}"), max, LOCAL)
                .await
                .unwrap_err();
            assert!(err.contains(needle), "{path}: {err}");
        }
        // The same local server is out of reach under the production policy.
        let err = fetch_image(
            &format!("{base}/img.png"),
            1 << 20,
            FetchPolicy::PUBLIC_HTTPS,
        )
        .await
        .unwrap_err();
        assert!(err.contains("scheme `http`"), "{err}");
    }

    #[test]
    fn native_video_routes_by_model_type() {
        for t in [
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
        ] {
            let cfg = MediaConfig::ov_genai(Some(t));
            assert!(cfg.native_video, "{t}");
            assert_eq!(cfg.video_frame_multiple, 2, "{t}");
        }
        for t in ["gemma4", "llava_next_video"] {
            let cfg = MediaConfig::ov_genai(Some(t));
            assert!(cfg.native_video, "{t}");
            assert_eq!(cfg.video_frame_multiple, 1, "{t}");
        }
        for t in [
            Some("internvl_chat"),
            Some("llava_next"),
            Some("gemma3"),
            Some("phi3_v"),
            Some("qwen3_vl_moe"),
            Some("some_new_vlm"),
            None,
        ] {
            let cfg = MediaConfig::ov_genai(t);
            assert!(!cfg.native_video, "{t:?}");
            assert_eq!(cfg.video_frame_multiple, 1, "{t:?}");
        }
        // A staged engine takes video whole only where its rank 0 encodes
        // it, behind Qwen's own tag.
        for f in ["qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_moe"] {
            let cfg = MediaConfig::qwen35_native().staged_native_video(f);
            assert!(cfg.native_video, "{f}");
            assert_eq!(cfg.video_frame_multiple, 2, "{f}");
            assert_eq!(
                cfg.video_tag(3),
                "<|vision_start|><|video_pad|><|vision_end|>"
            );
            let min = if f == "qwen2_5_vl" { 0 } else { 32 };
            assert_eq!(cfg.video_min_side, min, "{f}");
        }
        // Gemma 4 (the gemma4 engine): its own tag, frames one by one.
        let cfg = MediaConfig::gemma4_native().staged_native_video("gemma4");
        assert!(cfg.native_video);
        assert_eq!(cfg.video_tag(1), "<|video|>");
        assert_eq!((cfg.video_frame_multiple, cfg.video_min_side), (1, 0));
        for f in ["qwen3_vl_moe", "gemma3", "pixtral", ""] {
            let cfg = MediaConfig::qwen35_native().staged_native_video(f);
            assert!(!cfg.native_video, "{f}");
            assert_eq!(cfg.video_frame_multiple, 1, "{f}");
        }
        // The engines that encode images themselves take video as frames
        // unless they opt in.
        for cfg in [
            MediaConfig::qwen35_native(),
            MediaConfig::gemma4_native(),
            MediaConfig::gemma3_native(),
            MediaConfig::pixtral_native(),
            MediaConfig::phi3v_native(),
            MediaConfig::llava_next_native(),
            MediaConfig::internvl_native(),
        ] {
            assert!(!cfg.native_video, "{}", cfg.image_placeholder);
        }
    }

    /// A staged engine's video tag is the same for every video, and shares
    /// Qwen's framing tokens with the image tag.
    #[test]
    fn shared_video_tags_are_counted() {
        let cfg = MediaConfig::qwen35_native().staged_native_video("qwen3_vl");
        let (img, vid) = (cfg.placeholder(0), cfg.video_tag(0));
        let prompt = format!("{vid}a{img}b{vid}");
        assert_eq!(cfg.placeholders_complete(&prompt, 1), Ok(()));
        assert_eq!(cfg.video_placeholders_complete(&prompt, 2), Ok(()));
        assert_eq!(cfg.video_placeholders_complete(&prompt, 1), Err(2));
        assert_eq!(cfg.reserved_beyond_placeholders(&prompt, 1, 2), None);
        assert_eq!(
            cfg.reserved_beyond_placeholders(&format!("{prompt}<|video_pad|>"), 1, 2),
            Some("<|video_pad|>")
        );
        assert_eq!(
            cfg.reserved_beyond_placeholders(&prompt, 1, 1),
            Some("<|vision_start|>")
        );
    }

    #[test]
    fn video_tags_are_counted_apart_from_image_tags() {
        let cfg = MediaConfig::ov_genai(Some("qwen3_vl"));
        assert_eq!(cfg.video_tag(1), "<ov_genai_video_1>");
        let prompt = "<ov_genai_image_0><ov_genai_video_0>x<ov_genai_image_1><ov_genai_video_1>";
        assert_eq!(cfg.placeholders_complete(prompt, 2), Ok(()));
        assert_eq!(cfg.video_placeholders_complete(prompt, 2), Ok(()));
        assert_eq!(cfg.video_placeholders_complete(prompt, 1), Err(2));
        assert_eq!(
            cfg.video_placeholders_complete("<ov_genai_image_0>", 0),
            Ok(())
        );
        assert_eq!(
            cfg.video_placeholders_complete("<ov_genai_video_1>", 1),
            Err(1)
        );
        assert_eq!(cfg.reserved_beyond_placeholders(prompt, 2, 2), None);
        assert_eq!(
            cfg.reserved_beyond_placeholders(prompt, 2, 1),
            Some("<ov_genai_video_")
        );
    }

    #[test]
    fn placeholders_and_reserved_text() {
        let cfg = MediaConfig::ov_genai(None);
        assert_eq!(cfg.placeholder(3), "<ov_genai_image_3>");
        assert_eq!(
            cfg.reserved_in("see <ov_genai_image_0>"),
            Some("<ov_genai_image_")
        );
        assert_eq!(
            cfg.placeholders_complete("<ov_genai_image_0> a <ov_genai_image_1>", 2),
            Ok(())
        );
        assert_eq!(cfg.placeholders_complete("<ov_genai_image_1>", 2), Err(1));
        assert!(cfg
            .placeholders_complete("<ov_genai_image_0><ov_genai_image_0>", 2)
            .is_err());

        let q = MediaConfig::qwen35_native();
        let tag = q.placeholder(0);
        assert_eq!(tag, q.placeholder(5), "one shared tag for every image");
        assert_eq!(q.placeholders_complete(&format!("{tag}x{tag}"), 2), Ok(()));
        assert_eq!(q.placeholders_complete(&tag, 2), Err(1));
        assert_eq!(
            q.reserved_in("raw <|image_pad|> text"),
            Some("<|image_pad|>")
        );
        assert_eq!(q.reserved_in("plain text"), None);

        let g = MediaConfig::gemma4_native();
        let tag = g.placeholder(0);
        assert_eq!(tag, "<|image|>");
        assert_eq!(
            g.placeholders_complete(&format!("A:{tag}B:{tag}"), 2),
            Ok(())
        );
        assert_eq!(g.placeholders_complete(&tag, 2), Err(1));
        assert_eq!(g.reserved_in("raw <image|> text"), Some("<image|>"));
        assert!(g.reserved_in("raw <|image> text").is_some());
        assert_eq!(g.reserved_in("plain text"), None);

        assert_eq!(g.text_part("  a b \n".into()), "a b");

        // Phi-3.5-vision numbers its tags from 1.
        let p = MediaConfig::phi3v_native();
        assert_eq!(p.placeholder(0), "<|image_1|>\n");
        assert_eq!(p.placeholder(1), "<|image_2|>\n");
        let two = format!("{}a{}", p.placeholder(0), p.placeholder(1));
        assert_eq!(p.placeholders_complete(&two, 2), Ok(()));
        assert_eq!(p.placeholders_complete(&p.placeholder(1), 1), Err(1));
        assert_eq!(p.reserved_beyond_placeholders(&two, 2, 0), None);
        assert_eq!(
            p.reserved_beyond_placeholders(&format!("{two}<|image_9|>"), 2, 0),
            Some("<|image_")
        );
        assert_eq!(p.reserved_in("a <|image|> b"), Some("<|image|>"));

        let iv = MediaConfig::internvl_native();
        let tag = iv.placeholder(3);
        assert_eq!(tag, "<img><IMG_CONTEXT></img>\n");
        assert_eq!(iv.placeholders_complete(&format!("{tag}{tag}"), 2), Ok(()));
        assert_eq!(
            iv.reserved_beyond_placeholders(&format!("{tag}</img>"), 1, 0),
            Some("</img>")
        );
        assert_eq!(iv.reserved_in("<image> is fine"), None);

        let g3 = MediaConfig::gemma3_native();
        let tag = g3.placeholder(1);
        assert_eq!(tag, "<start_of_image>");
        assert_eq!(g3.placeholders_complete(&format!("{tag}a{tag}"), 2), Ok(()));
        assert_eq!(
            g3.reserved_in("raw <image_soft_token> text"),
            Some("<image_soft_token>")
        );
        assert!(g3.reserved_in("x<end_of_image>").is_some());
        assert_eq!(
            g3.reserved_beyond_placeholders(&format!("{tag}a{tag}"), 1, 0),
            Some("<start_of_image>")
        );
        assert_eq!(g3.text_part(" a \n".into()), "a");

        let x = MediaConfig::pixtral_native();
        assert_eq!(x.text_part(" a ".into()), " a ");
        // A tag split across two text parts only appears once they are joined.
        assert_eq!(
            x.reserved_beyond_placeholders("[IMG]a[IMG_BREAK]", 1, 0),
            Some("[IMG_BREAK]")
        );
        assert_eq!(
            x.reserved_beyond_placeholders("[IMG]a[IMG]", 1, 0),
            Some("[IMG]")
        );
        assert_eq!(x.reserved_beyond_placeholders("[IMG]a[IMG]", 2, 0), None);
        let q = MediaConfig::qwen35_native();
        let tag = q.placeholder(0);
        assert_eq!(
            q.reserved_beyond_placeholders(&format!("a{tag}b"), 1, 0),
            None
        );
        assert_eq!(
            q.reserved_beyond_placeholders(&format!("a{tag}<|video_pad|>"), 1, 0),
            Some("<|video_pad|>")
        );
        let o = MediaConfig::ov_genai(None);
        assert_eq!(
            o.reserved_beyond_placeholders("<ov_genai_image_0><ov_genai_image_1>", 2, 0),
            None
        );
        assert_eq!(base64_len(3), 4 + 256);
        assert_eq!(x.placeholder(1), "[IMG]");
        assert_eq!(x.placeholders_complete("[IMG]a[IMG]", 2), Ok(()));
        assert_eq!(x.reserved_in("raw [IMG_END] text"), Some("[IMG_END]"));
        assert!(x.reserved_in("raw [IMG_BREAK] text").is_some());
        assert_eq!(x.reserved_in("plain text"), None);
    }

    /// LLaVA-NeXT's processor template writes `<image>\n` per image, then the
    /// text parts back to back, whatever the parts' order; HF renders
    /// `[INST] <image>\n<image>\nA:B:Compare. [/INST]` for this message
    /// (tests/fixtures/llava_next_vision in the OpenVINO engine crate).
    #[test]
    fn llava_next_writes_images_first() {
        let l = MediaConfig::llava_next_native();
        let parts = vec![
            (false, "A:".to_string()),
            (true, l.placeholder(0)),
            (false, "B:".to_string()),
            (true, l.placeholder(1)),
            (false, "Compare.".to_string()),
        ];
        assert_eq!(
            l.join_parts(parts.clone()),
            "<image>\n<image>\nA:B:Compare."
        );
        assert_eq!(l.placeholders_complete("<image>\n<image>\nA:B:", 2), Ok(()));
        assert_eq!(l.reserved_in("say <image> please"), Some("<image>"));
        assert_eq!(
            l.reserved_beyond_placeholders("<image>\n<image>\nA:B:", 2, 0),
            None
        );
        assert_eq!(
            l.reserved_beyond_placeholders("<image>\nA <image>", 1, 0),
            Some("<image>")
        );
        // Every other family keeps the parts in order.
        let x = MediaConfig::pixtral_native();
        assert_eq!(
            x.join_parts(vec![(false, "A".into()), (true, "[IMG]".into())]),
            "A[IMG]"
        );
    }

    /// `original` is served as `high`: the full pixel budget, not low's.
    #[test]
    fn original_detail_takes_the_high_budget() {
        let mut cfg = MediaConfig::ov_genai(None);
        cfg.max_image_pixels = 100 * 50;
        cfg.low_detail_pixels = 20 * 10;
        let m = decode_image_url(&png_data_url(400, 200), Some("original"), &cfg).unwrap();
        assert_eq!((m.width, m.height), (100, 50));
        let err = decode_image_url(&png_data_url(4, 4), Some("ultra"), &cfg).unwrap_err();
        assert!(err.contains("`original`"), "{err}");
    }

    /// ov-genai reserves the universal image and video tags on every export,
    /// plus the native tags GenAI reads for the export's family; an unknown
    /// family reserves every known native tag.
    #[test]
    fn ov_genai_reserves_each_familys_native_tags() {
        for (model_type, native, not_reserved) in [
            (Some("llava_next"), "say <image> now", Some("<|image|>")),
            (
                Some("internvl_chat"),
                "<image>\nhi",
                Some("<|vision_start|>"),
            ),
            (Some("llava_next_video"), "a <video> b", None),
            (Some("minicpmv"), "<image>./</image>", None),
            (Some("phi3_v"), "<|image_1|>\nhi", Some("<image>")),
            (Some("phi4mm"), "<|image_2|>", None),
            (
                Some("qwen2_vl"),
                "<|vision_start|><|image_pad|><|vision_end|>",
                Some("<image>"),
            ),
            (Some("qwen2_5_vl"), "x<|image_pad|>", None),
            (Some("qwen3_vl"), "x<|video_pad|>", None),
            (Some("qwen3_5"), "<|vision_start|>", None),
            (Some("gemma3"), "<start_of_image>", Some("<|image|>")),
            (Some("gemma3"), "<image_soft_token>", Some("<img>")),
            (Some("gemma3n"), "<image_soft_token>", None),
            (Some("gemma4"), "<|image|>", Some("<image>")),
            (Some("gemma4"), "<|video|>", None),
        ] {
            let cfg = MediaConfig::ov_genai(model_type);
            assert!(
                cfg.reserved_in(native).is_some(),
                "{model_type:?}: {native:?}"
            );
            assert!(cfg.reserved_in("<ov_genai_image_3>").is_some());
            assert!(cfg.reserved_in("<ov_genai_video_0>").is_some());
            assert_eq!(cfg.reserved_in("plain text"), None);
            if let Some(other) = not_reserved {
                assert_eq!(cfg.reserved_in(other), None, "{model_type:?}: {other:?}");
            }
            // The universal placeholder never trips its own family's tags.
            let tags = format!("{}{}", cfg.placeholder(0), cfg.placeholder(1));
            assert_eq!(
                cfg.reserved_beyond_placeholders(&tags, 2, 0),
                None,
                "{model_type:?}"
            );
        }
        for unknown in [None, Some("some_new_vlm")] {
            let cfg = MediaConfig::ov_genai(unknown);
            for tag in [
                "<image>",
                "<video>",
                "<|image_1|>",
                "<|vision_start|>",
                "<|video_pad|>",
                "<start_of_image>",
                "<image_soft_token>",
                "<|image|>",
                "<|video|>",
            ] {
                assert!(cfg.reserved_in(tag).is_some(), "{unknown:?}: {tag}");
            }
        }
    }

    /// The request-wide source-pixel budget, from headers alone.
    #[test]
    fn request_source_pixels_are_summed_from_headers() {
        let mut cfg = MediaConfig::ov_genai(None);
        assert_eq!(
            cfg.max_request_source_pixels,
            2 * cfg.max_source_pixels,
            "default is two full-size images"
        );
        let url = png_data_url(30, 20);
        let bytes = image_url_bytes(&url, None, &cfg).unwrap();
        assert_eq!(source_size(&bytes, &cfg).unwrap(), (30, 20));
        cfg.max_request_source_pixels = 1200;
        assert!(check_request_pixels(&[(30, 20), (30, 20)], &cfg).is_ok());
        let err = check_request_pixels(&[(30, 20), (30, 20), (1, 1)], &cfg).unwrap_err();
        assert!(err.contains("1201 pixels"), "{err}");
        assert!(source_size(b"not an image", &cfg).is_err());
    }
}
