//! Image input for `/v1/chat/completions`: OpenAI `image_url` content parts.
//!
//! Every image is decoded, bounded and hashed here, once, so engines only see
//! RGB pixels ([`MediaInput`]). Only `data:` URLs are accepted; fetching
//! remote URLs needs SSRF guards and is a separate, opt-in feature.

use std::io::Cursor;

use base64::Engine as _;
use cascadia_types::{MediaInput, MediaKind};
use image::{imageops::FilterType, ImageFormat, ImageReader, Limits};
use sha2::{Digest, Sha256};

/// Per-engine image settings. Present on the router only when the served
/// model accepts images; `None` means every `image_url` part gets a 400.
#[derive(Clone, Debug)]
pub struct MediaConfig {
    /// Text that marks image `i` in the prompt, with `{i}` replaced by the
    /// image's index in the request. ov-genai uses its universal tag, which
    /// GenAI maps to each model's native one.
    pub image_placeholder: String,
    /// Images allowed in one request.
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
    /// Encoded image bytes one request may carry, on top of the text body
    /// limit.
    pub max_request_image_bytes: usize,
}

impl MediaConfig {
    /// Settings for the ov-genai engine on a VLM-layout export.
    pub fn ov_genai() -> Self {
        Self {
            image_placeholder: "<ov_genai_image_{i}>".into(),
            max_images: 8,
            // 1024x1024: about 1000 image tokens on Qwen3.5.
            max_image_pixels: 1 << 20,
            // OpenAI's low detail is a 512x512 budget.
            low_detail_pixels: 512 * 512,
            max_source_pixels: 50_000_000,
            max_request_image_bytes: 20 << 20,
        }
    }

    pub fn placeholder(&self, i: usize) -> String {
        self.image_placeholder.replace("{i}", &i.to_string())
    }

    /// The part of the placeholder before the index. User text containing it
    /// is refused: GenAI would read it as an image tag and fail the request
    /// or misplace an image.
    pub fn placeholder_prefix(&self) -> &str {
        self.image_placeholder
            .split("{i}")
            .next()
            .unwrap_or(&self.image_placeholder)
    }
}

/// Hard ceiling on either side, whatever the pixel budget.
const MAX_SIDE: u32 = 16_384;
/// Qwen's processor refuses aspect ratios past 200:1; refusing here gives the
/// caller a 400 instead of an engine failure.
const MAX_ASPECT: u32 = 200;

/// Decode one `image_url` into pixels the engine can take as is.
pub fn decode_image_url(
    url: &str,
    detail: Option<&str>,
    cfg: &MediaConfig,
) -> Result<MediaInput, String> {
    let budget = match detail {
        None | Some("auto") | Some("high") => cfg.max_image_pixels,
        Some("low") => cfg.low_detail_pixels,
        Some(other) => {
            return Err(format!(
                "image detail `{other}` is not supported; use `low`, `high` or `auto`"
            ))
        }
    };
    let bytes = data_url_bytes(url)?;

    let format = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| format!("could not read the image: {e}"))?
        .format();
    match format {
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif) => {}
        _ => return Err("unsupported image format; send PNG, JPEG, WEBP or GIF".into()),
    }

    // Check the declared size before decoding a single pixel.
    let (w, h) = ImageReader::with_format(Cursor::new(&bytes), format.unwrap())
        .into_dimensions()
        .map_err(|e| format!("could not read the image header: {e}"))?;
    check_dimensions(w, h, cfg)?;

    let mut reader = ImageReader::with_format(Cursor::new(&bytes), format.unwrap());
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
        let scale = (budget as f64 / pixels as f64).sqrt();
        let nw = ((f64::from(w) * scale).floor() as u32).max(1);
        let nh = ((f64::from(h) * scale).floor() as u32).max(1);
        rgb = image::imageops::resize(&rgb, nw, nh, FilterType::CatmullRom);
    }

    let (width, height) = rgb.dimensions();
    let raw = rgb.into_raw();
    let mut hasher = Sha256::new();
    hasher.update(width.to_le_bytes());
    hasher.update(height.to_le_bytes());
    hasher.update(&raw);
    Ok(MediaInput {
        kind: MediaKind::Image,
        pixels: raw.into(),
        num_frames: 1,
        width,
        height,
        timestamps_ms: None,
        sha256: hasher.finalize().into(),
    })
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
fn data_url_bytes(url: &str) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(
            "only `data:` image URLs are supported; send the image base64-encoded \
             (`data:image/png;base64,...`)"
                .into(),
        );
    };
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err("malformed data URL: no `,` before the payload".into());
    };
    let mut fields = meta.split(';');
    let mime = fields.next().unwrap_or("");
    if !mime.starts_with("image/") {
        return Err(format!("data URL media type `{mime}` is not an image type"));
    }
    if !fields.any(|f| f == "base64") {
        return Err("data URL must be base64-encoded (`;base64,`)".into());
    }
    let payload = payload.trim();
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload))
        .map_err(|e| format!("data URL payload is not valid base64: {e}"))
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
        let cfg = MediaConfig::ov_genai();
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
        let mut cfg = MediaConfig::ov_genai();
        cfg.max_image_pixels = 100 * 50;
        let m = decode_image_url(&png_data_url(400, 200), None, &cfg).unwrap();
        assert_eq!((m.width, m.height), (100, 50));
        cfg.low_detail_pixels = 20 * 10;
        let low = decode_image_url(&png_data_url(400, 200), Some("low"), &cfg).unwrap();
        assert_eq!((low.width, low.height), (20, 10));
    }

    #[test]
    fn refuses_what_it_cannot_take() {
        let cfg = MediaConfig::ov_genai();
        for (url, detail, needle) in [
            ("https://example.com/a.png", None, "only `data:`"),
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
        let mut small = MediaConfig::ov_genai();
        small.max_source_pixels = 100;
        let err = decode_image_url(&png_data_url(20, 20), None, &small).unwrap_err();
        assert!(err.contains("pixels in total"), "{err}");
    }

    #[test]
    fn placeholder_and_prefix() {
        let cfg = MediaConfig::ov_genai();
        assert_eq!(cfg.placeholder(3), "<ov_genai_image_3>");
        assert_eq!(cfg.placeholder_prefix(), "<ov_genai_image_");
    }
}
