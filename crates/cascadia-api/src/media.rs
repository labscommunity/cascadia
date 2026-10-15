//! Image input for `/v1/chat/completions`: OpenAI `image_url` content parts.
//!
//! Every image is decoded, bounded and hashed here, once, so engines only see
//! RGB pixels ([`MediaInput`]). `data:` URLs are always accepted. Remote
//! `https` URLs are fetched only when the operator opts in
//! ([`MediaConfig::allow_remote_urls`]), and only from public addresses: see
//! [`fetch_image`].

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
    /// limit. Also the cap on bytes fetched for one request.
    pub max_request_image_bytes: usize,
    /// Fetch `https` image URLs (public addresses only). Off by default: the
    /// server then makes outbound requests on a caller's behalf.
    pub allow_remote_urls: bool,
    /// Who a fetch may reach. Always [`FetchPolicy::PUBLIC_HTTPS`] outside
    /// tests; not exposed as a flag.
    #[doc(hidden)]
    pub fetch_policy: FetchPolicy,
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
            allow_remote_urls: false,
            fetch_policy: FetchPolicy::PUBLIC_HTTPS,
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

/// Decode one `data:` image URL into pixels the engine can take as is.
pub fn decode_image_url(
    url: &str,
    detail: Option<&str>,
    cfg: &MediaConfig,
) -> Result<MediaInput, String> {
    check_detail(detail)?;
    let bytes = data_url_bytes(url, cfg.allow_remote_urls)?;
    decode_image_bytes(&bytes, detail, cfg)
}

fn check_detail(detail: Option<&str>) -> Result<(), String> {
    match detail {
        None | Some("auto" | "high" | "low") => Ok(()),
        Some(other) => Err(format!(
            "image detail `{other}` is not supported; use `low`, `high` or `auto`"
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
        None | Some("auto") | Some("high") => cfg.max_image_pixels,
        _ => cfg.low_detail_pixels,
    };

    let format = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("could not read the image: {e}"))?
        .format();
    match format {
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif) => {}
        _ => return Err("unsupported image format; send PNG, JPEG, WEBP or GIF".into()),
    }

    // Check the declared size before decoding a single pixel.
    let (w, h) = ImageReader::with_format(Cursor::new(bytes), format.unwrap())
        .into_dimensions()
        .map_err(|e| format!("could not read the image header: {e}"))?;
    check_dimensions(w, h, cfg)?;

    let mut reader = ImageReader::with_format(Cursor::new(bytes), format.unwrap());
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
    match tokio::time::timeout(policy.per_image, fetch_image_inner(url, max_bytes, policy)).await {
        Ok(r) => r,
        Err(_) => Err(format!(
            "fetching the image took longer than {} s",
            policy.per_image.as_secs_f32()
        )),
    }
}

/// Fetch failures whose cause stays in the server log.
const UNREACHABLE: &str = "could not fetch the image from this server";

async fn fetch_image_inner(
    url: &str,
    max_bytes: usize,
    policy: FetchPolicy,
) -> Result<Vec<u8>, String> {
    let mut current = url::Url::parse(url).map_err(|e| format!("invalid image URL: {e}"))?;
    for _ in 0..=MAX_REDIRECTS {
        let addr = check_target(&current, policy).await?;
        let host = current.host_str().unwrap_or_default().to_owned();
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .https_only(!policy.allow_http)
            .resolve(&host, addr)
            .user_agent("cascadia-image-fetch")
            .build()
            .map_err(|e| format!("could not set up the image fetch: {e}"))?;
        let mut resp = client.get(current.clone()).send().await.map_err(|e| {
            tracing::warn!(host = %host, error = %without_url(e), "image fetch failed");
            UNREACHABLE.to_string()
        })?;
        if resp.status().is_redirection() {
            let next = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or("image URL redirected without a Location")?;
            current = current
                .join(next)
                .map_err(|e| format!("image URL redirected to an invalid URL: {e}"))?;
            continue;
        }
        if !resp.status().is_success() {
            return Err(format!(
                "fetching the image returned HTTP {}",
                resp.status().as_u16()
            ));
        }
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !ctype.starts_with("image/") {
            return Err(format!(
                "image URL returned content type `{ctype}`, not an image"
            ));
        }
        if resp.content_length().is_some_and(|n| n > max_bytes as u64) {
            return Err(format!("image is larger than the {max_bytes}-byte limit"));
        }
        let mut body = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| {
            tracing::warn!(host = %host, error = %without_url(e), "image fetch failed mid-body");
            UNREACHABLE.to_string()
        })? {
            if body.len() + chunk.len() > max_bytes {
                return Err(format!("image is larger than the {max_bytes}-byte limit"));
            }
            body.extend_from_slice(&chunk);
        }
        return Ok(body);
    }
    Err(format!(
        "image URL redirected more than {MAX_REDIRECTS} times"
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
async fn check_target(url: &url::Url, policy: FetchPolicy) -> Result<SocketAddr, String> {
    match url.scheme() {
        "https" => {}
        "http" if policy.allow_http => {}
        other => {
            return Err(format!(
                "image URL scheme `{other}` is not allowed; use https"
            ))
        }
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("image URLs must not carry credentials".into());
    }
    let port = url.port_or_known_default().unwrap_or(0);
    if !policy.allow_http && port != 443 {
        return Err(format!("image URLs must use port 443, not {port}"));
    }
    let host = match url.host() {
        Some(url::Host::Domain(d)) => d.to_owned(),
        Some(url::Host::Ipv4(ip)) => ip.to_string(),
        Some(url::Host::Ipv6(ip)) => ip.to_string(),
        None => return Err("image URL has no host".into()),
    };
    let addrs: Vec<SocketAddr> = match tokio::net::lookup_host((host.as_str(), port)).await {
        Ok(a) => a.collect(),
        Err(e) => {
            tracing::warn!(host = %host, error = %e, "image host did not resolve");
            return Err(UNREACHABLE.into());
        }
    };
    let Some(first) = addrs.first().copied() else {
        tracing::warn!(host = %host, "image host resolved to no addresses");
        return Err(UNREACHABLE.into());
    };
    // Every answer must be public: a host that resolves to one public and one
    // internal address would otherwise be a coin flip away from the inside.
    // The caller learns only that the host is refused, never what it
    // resolved to: that would turn the endpoint into an internal DNS oracle.
    if !policy.allow_non_public {
        if let Some(bad) = addrs.iter().find(|a| !is_public(a.ip())) {
            tracing::warn!(host = %host, ip = %bad.ip(), "image host refused: not a public address");
            return Err("image host is not a public address".into());
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
        let mut small = MediaConfig::ov_genai();
        small.max_source_pixels = 100;
        let err = decode_image_url(&png_data_url(20, 20), None, &small).unwrap_err();
        assert!(err.contains("pixels in total"), "{err}");
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
        let cfg = MediaConfig::ov_genai();
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
    fn placeholder_and_prefix() {
        let cfg = MediaConfig::ov_genai();
        assert_eq!(cfg.placeholder(3), "<ov_genai_image_3>");
        assert_eq!(cfg.placeholder_prefix(), "<ov_genai_image_");
    }
}
