//! Video input: `video_url` content parts in `/v1/chat/completions` (and
//! `/v1/responses`, whose `input_video` parts become them), the extension
//! vLLM, SGLang and OpenRouter accept. OpenAI has no video input; its own
//! advice is to send sampled frames as `image_url` parts.
//!
//! This does the same server-side: a video is probed and decoded by an
//! ffmpeg subprocess (never an in-process binding: the bytes are a caller's)
//! into uniformly sampled RGB frames, and each frame enters the image path as
//! one image ([`MediaInput`] with one frame), so every engine that takes
//! images takes video with no change of its own.
//!
//! ffmpeg reads only a private temp file (`-protocol_whitelist file`), with
//! the container forced to the one the bytes start with (MP4/MOV or
//! Matroska/WebM), under one deadline per request, with its output capped
//! and the child killed when the request goes away (`kill_on_drop`). The
//! probe bounds the work before any decode: source frame size, duration, and
//! frames x pixels; long clips are read with one seek per sampled frame.

use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};

use cascadia_types::{MediaInput, MediaKind};
use serde::Deserialize;
use tokio::io::AsyncReadExt as _;
use tokio::time::Instant;

use crate::media::{self, MediaConfig, MAX_ASPECT, MAX_SIDE};

/// The ffmpeg and ffprobe binaries a server decodes videos with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ffmpeg {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Ffmpeg {
    /// `explicit` (`--api-ffmpeg`), or `ffmpeg` on `PATH`; `ffprobe` next to
    /// it, else on `PATH`. The error says what is missing.
    pub fn locate(explicit: Option<&Path>) -> Result<Self, String> {
        let exe = |name: &str| format!("{name}{}", std::env::consts::EXE_SUFFIX);
        let ffmpeg = match explicit {
            Some(p) if is_executable(p) => p.to_path_buf(),
            // `C:\ffmpeg\bin\ffmpeg` names `ffmpeg.exe` on Windows.
            Some(p)
                if cfg!(windows)
                    && p.extension().is_none()
                    && is_executable(&p.with_extension("exe")) =>
            {
                p.with_extension("exe")
            }
            Some(p) => return Err(format!("`{}` is not an executable file", p.display())),
            None => find_on_path(&exe("ffmpeg")).ok_or("no `ffmpeg` on PATH")?,
        };
        let sibling = ffmpeg.with_file_name(exe("ffprobe"));
        let ffprobe = if is_executable(&sibling) {
            sibling
        } else {
            find_on_path(&exe("ffprobe"))
                .ok_or_else(|| format!("no `ffprobe` next to `{}` or on PATH", ffmpeg.display()))?
        };
        Ok(Self { ffmpeg, ffprobe })
    }
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| is_executable(p))
}

fn is_executable(p: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(p) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        meta.is_file() && meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        meta.is_file()
    }
}

/// The 400 a video gets on a server without ffmpeg.
pub const NEEDS_FFMPEG: &str = "video input needs ffmpeg on the server, which was not found \
     (the operator can point `--api-ffmpeg` at it); send the video's frames as `image_url` \
     parts instead";

/// The `data:` media types taken for a video.
const VIDEO_MIME_TYPES: [&str; 4] = [
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",
];

/// What a refused video's 400 says it may be.
const SUPPORTED: &str = "send MP4, MOV, WebM or MKV";

/// The `detail` values a video takes. `auto` (the default) is
/// [`MediaConfig::max_video_frame_pixels`] per frame.
fn check_detail(detail: Option<&str>) -> Result<(), String> {
    match detail {
        None | Some("auto" | "low" | "high") => Ok(()),
        Some(other) => Err(format!(
            "video detail `{other}` is not supported; use `low`, `high` or `auto`"
        )),
    }
}

/// The encoded video a `data:` URL carries, its `detail` checked.
pub fn video_url_bytes(
    url: &str,
    detail: Option<&str>,
    cfg: &MediaConfig,
) -> Result<Vec<u8>, String> {
    check_detail(detail)?;
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(if cfg.allow_remote_urls {
            "video URLs must be `https://` or `data:` (base64-encoded)".into()
        } else {
            "only `data:` video URLs are supported on this server; send the video \
             base64-encoded (`data:video/mp4;base64,...`)"
                .into()
        });
    };
    media::data_url_payload(rest, |mime| {
        if VIDEO_MIME_TYPES.contains(&mime) {
            Ok(())
        } else {
            Err(format!(
                "data URL media type `{mime}` is not a supported video type; {SUPPORTED}"
            ))
        }
    })
}

/// The demuxer a video's first bytes name. ffmpeg is told to use it, so it
/// never picks one by probing (HLS, concat and image2 would open other
/// files).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Container {
    /// ISO BMFF: MP4, MOV, M4V, 3GP.
    Mov,
    /// Matroska and WebM.
    Matroska,
}

impl Container {
    /// The container `bytes` start with, if it is one this server decodes.
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
            return Some(Self::Matroska);
        }
        // A top-level box: a size, then its type. QuickTime files may open
        // with any of these before `moov`.
        let kind = bytes.get(4..8)?;
        [
            b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip", b"pnot",
        ]
        .iter()
        .any(|k| kind == *k)
        .then_some(Self::Mov)
    }

    fn demuxer(self) -> &'static str {
        match self {
            Self::Mov => "mov",
            Self::Matroska => "matroska",
        }
    }
}

/// The names in ffprobe's `format_name` this server accepts: the MP4/MOV
/// demuxer's and Matroska's.
const CONTAINER_NAMES: [&str; 8] = ["mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm"];

/// What the probe found: the duration when the container states one, the
/// frame size as displayed (a rotated phone video's sides swapped, as
/// ffmpeg's autorotate outputs it), and the frame rate and count when known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub duration: Option<f64>,
    pub width: u32,
    pub height: u32,
    pub frame_rate: Option<f64>,
    pub nb_frames: Option<u64>,
}

/// A probed video within every bound, its duration known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Video {
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    /// Frames the source holds: its packets when they were counted, else
    /// what the container states.
    pub source_frames: f64,
    /// Keyframes the source holds, when its packets were counted.
    pub keyframes: Option<u64>,
}

#[derive(Deserialize)]
struct ProbeJson {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<String>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    nb_frames: Option<String>,
    #[serde(default)]
    side_data_list: Vec<ProbeSideData>,
}

#[derive(Deserialize)]
struct ProbeSideData {
    rotation: Option<f64>,
}

#[derive(Deserialize)]
struct ProbeFormat {
    format_name: Option<String>,
    duration: Option<String>,
}

/// The frame rate the work bound assumes when the container states none.
const ASSUMED_FPS: f64 = 60.0;
/// The highest frame rate believed: a variable-rate stream's `r_frame_rate`
/// can be its timebase (`1000/1`).
const MAX_FPS: f64 = 120.0;

/// ffprobe's `num/den` rate (`30000/1001`), at most [`MAX_FPS`]; `0/0` is
/// unknown.
fn parse_rate(r: &str) -> Option<f64> {
    let (n, d) = r.split_once('/').unwrap_or((r, "1"));
    let rate = n.trim().parse::<f64>().ok()? / d.trim().parse::<f64>().ok()?;
    (rate.is_finite() && rate > 0.0).then_some(rate.min(MAX_FPS))
}

/// Read ffprobe's JSON and refuse, before any frame is decoded, a video
/// this server will not decode: no video stream, a container outside
/// [`CONTAINER_NAMES`], or frames past [`MediaConfig::max_video_source_pixels`].
/// Its length is checked once it is known ([`Probe::bounded`]).
pub fn parse_probe(json: &[u8], cfg: &MediaConfig) -> Result<Probe, String> {
    let p: ProbeJson =
        serde_json::from_slice(json).map_err(|_| "could not read the video".to_string())?;
    let format = p.format.ok_or("could not read the video's container")?;
    let name = format.format_name.unwrap_or_default();
    if name.is_empty() || !name.split(',').all(|n| CONTAINER_NAMES.contains(&n)) {
        return Err(format!(
            "video container `{name}` is not supported; {SUPPORTED}"
        ));
    }
    let stream = p
        .streams
        .into_iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or("the video has no video stream")?;
    let secs = |d: Option<String>| {
        d.and_then(|d| d.parse::<f64>().ok())
            .filter(|d| d.is_finite() && *d > 0.0)
    };
    let duration = secs(format.duration).or_else(|| secs(stream.duration));
    let (mut w, mut h) = (stream.width.unwrap_or(0), stream.height.unwrap_or(0));
    if w == 0 || h == 0 {
        return Err("the video's frames have no pixels".into());
    }
    if w > MAX_SIDE || h > MAX_SIDE || u64::from(w) * u64::from(h) > cfg.max_video_source_pixels {
        return Err(format!(
            "video is {w}x{h}; the limit is {} pixels a frame; send a lower-resolution clip",
            cfg.max_video_source_pixels
        ));
    }
    if w.max(h) > MAX_ASPECT * w.min(h) {
        return Err(format!(
            "video is {w}x{h}; the aspect ratio must be at most {MAX_ASPECT}:1"
        ));
    }
    let quarter_turn = stream
        .side_data_list
        .iter()
        .filter_map(|d| d.rotation)
        .any(|r| (r.rem_euclid(180.0) - 90.0).abs() < 1.0);
    if quarter_turn {
        std::mem::swap(&mut w, &mut h);
    }
    let frame_rate = [stream.avg_frame_rate, stream.r_frame_rate]
        .into_iter()
        .flatten()
        .find_map(|r| parse_rate(&r));
    Ok(Probe {
        duration,
        width: w,
        height: h,
        frame_rate,
        nb_frames: stream.nb_frames.and_then(|n| n.parse().ok()),
    })
}

impl Probe {
    /// Refuse a video longer than [`MediaConfig::max_video_seconds`], or one
    /// whose frames would make the decoder produce more than
    /// [`MediaConfig::max_video_decode_pixels`] source pixels. Its frames are
    /// the scanned packets when there was a scan, else the larger of the
    /// stated frame count and duration x frame rate.
    pub fn bounded(
        &self,
        duration: f64,
        packets: Option<&Packets>,
        cfg: &MediaConfig,
    ) -> Result<Video, String> {
        if duration > cfg.max_video_seconds {
            return Err(format!(
                "video is {duration:.1} s long; the limit is {} s",
                cfg.max_video_seconds
            ));
        }
        let source_frames = match packets {
            Some(p) => p.count as f64,
            None => (duration * self.frame_rate.unwrap_or(ASSUMED_FPS))
                .max(self.nb_frames.unwrap_or(0) as f64),
        };
        let (w, h) = (self.width, self.height);
        let work = source_frames * f64::from(w) * f64::from(h);
        if work > cfg.max_video_decode_pixels as f64 {
            return Err(format!(
                "video is {w}x{h} with about {source_frames:.0} frames; this server decodes at \
                 most {} source pixels a video (600 s of 1080p30 by default); send a shorter or \
                 lower-resolution clip",
                cfg.max_video_decode_pixels
            ));
        }
        Ok(Video {
            duration,
            width: w,
            height: h,
            source_frames,
            keyframes: packets.map(|p| p.keyframes),
        })
    }
}

/// What a demux-only packet scan ([`packet_args`]) found: how many video
/// packets (frames) and keyframes there are, and the time they span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Packets {
    pub count: u64,
    pub keyframes: u64,
    /// From the first packet's time to the last's, plus one frame: the
    /// duration of a container that states none (WebM written to a pipe, as
    /// browsers' MediaRecorder does), from its start whatever its first
    /// timestamp.
    pub span: Option<f64>,
}

/// Read a packet scan's `pts_time,flags` lines.
pub fn parse_packets(csv: &[u8], frame_rate: Option<f64>) -> Packets {
    let (mut count, mut keyframes) = (0, 0);
    let (mut first, mut last) = (f64::INFINITY, f64::NEG_INFINITY);
    for line in String::from_utf8_lossy(csv).lines() {
        let mut fields = line.trim().split(',');
        let pts = fields.next().and_then(|t| t.parse::<f64>().ok());
        let flags = fields.next().unwrap_or("");
        if pts.is_none() && flags.is_empty() {
            continue;
        }
        count += 1;
        if flags.contains('K') {
            keyframes += 1;
        }
        if let Some(t) = pts.filter(|t| t.is_finite()) {
            first = first.min(t);
            last = last.max(t);
        }
    }
    let span = last - first + frame_rate.map_or(0.0, |r| 1.0 / r);
    Packets {
        count,
        keyframes,
        span: (span.is_finite() && span > 0.0).then_some(span),
    }
}

/// Sampled frames whose sources hold more than this many frames each are
/// read with one seek per frame instead of one pass over the whole video,
/// when keyframes are dense enough (see [`Plan::new`]): a long or high-rate
/// clip then costs about half a GOP per frame, not every frame it has (60 s
/// of 4K30 at a 2 s GOP on an M-series Mac: 9 s in one pass, 3.6 s with 16
/// seeks).
const SEEK_RATIO: f64 = 32.0;

/// How a probed video is sampled: `frames` frames, uniform over its
/// duration from 0, each `width x height` RGB, in one pass or by `seek`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub frames: usize,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub seek: bool,
}

impl Plan {
    /// `ceil(duration * video_fps)` frames, at least 1 and at most
    /// [`MediaConfig::max_video_frames`] (for a native video, rounded up to
    /// [`MediaConfig::video_frame_multiple`] when that fits), each fit to the `detail` budget
    /// keeping the aspect ratio, with even sides. `low` is the smallest
    /// budget (a quarter of the frame budget, at most the image low-detail
    /// one), `high` the largest.
    ///
    /// A seek decodes from the keyframe before its time, about half a GOP,
    /// so `frames` seeks cost about `frames / (2 * keyframes)` of one pass:
    /// seeking needs at least one keyframe per sampled frame. A clip with
    /// rare keyframes (a browser recording, `keyint=infinite`) is read in
    /// one pass: seeking it would decode most of it once per frame.
    pub fn new(video: &Video, detail: Option<&str>, cfg: &MediaConfig) -> Self {
        let wanted = (video.duration * cfg.video_fps).ceil();
        let max = cfg.max_video_frames.max(1);
        let frames = if wanted.is_finite() && wanted >= 1.0 {
            (wanted as usize).min(max)
        } else {
            1
        };
        // A native video's encoder takes frames in groups (Qwen: pairs) and
        // pads a short group with copies: sample the frames it would pad.
        let multiple = cfg.video_frame_multiple.max(1);
        let frames = match frames.next_multiple_of(multiple) {
            up if cfg.native_video && up <= max => up,
            _ => frames,
        };
        let auto = cfg.max_video_frame_pixels;
        let budget = match detail {
            Some("low") => cfg.low_detail_pixels.min(auto / 4).max(4),
            Some("high") => cfg.max_image_pixels.max(auto),
            _ => auto,
        };
        let (w, h) = (video.width, video.height);
        let (w, h) = if u64::from(w) * u64::from(h) > budget {
            media::downscaled_size(w, h, budget)
        } else {
            (w, h)
        };
        let even = |x: u32| (x & !1).max(2);
        Self {
            frames,
            width: even(w),
            height: even(h),
            duration: video.duration,
            seek: video.source_frames > SEEK_RATIO * frames as f64
                && video.keyframes.is_some_and(|k| k as usize >= frames),
        }
    }

    /// The sampling rate that spreads `frames` over the duration.
    pub fn fps(&self) -> f64 {
        self.frames as f64 / self.duration
    }

    /// Frame `k`'s time in seconds (from 0).
    pub fn timestamp(&self, k: usize) -> f64 {
        k as f64 * self.duration / self.frames as f64
    }

    fn frame_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }
}

/// A video's sampled frames as one native [`MediaInput`]: the frames
/// stacked `[T, H, W, 3]`, their times in milliseconds, and one hash over
/// the frame count, size and every pixel (an encode cache keys on what the
/// model sees). `None` when there are no frames or they differ in size.
pub fn video_input(frames: Vec<Frame>) -> Option<MediaInput> {
    use sha2::{Digest, Sha256};
    let first = frames.first()?;
    let (width, height) = (first.input.width, first.input.height);
    if frames
        .iter()
        .any(|f| (f.input.width, f.input.height, f.input.num_frames) != (width, height, 1))
    {
        return None;
    }
    let num_frames = u32::try_from(frames.len()).ok()?;
    let mut pixels = Vec::with_capacity(frames.iter().map(|f| f.input.pixels.len()).sum());
    let mut timestamps_ms = Vec::with_capacity(frames.len());
    for f in &frames {
        pixels.extend_from_slice(&f.input.pixels);
        timestamps_ms.push((f.secs * 1000.0).round() as u32);
    }
    let mut hasher = Sha256::new();
    hasher.update(num_frames.to_le_bytes());
    hasher.update(width.to_le_bytes());
    hasher.update(height.to_le_bytes());
    // The frame times shape the prompt (Qwen3-VL timestamps) and the
    // positions (Qwen2.5-VL's spacing), so the same frames at another rate
    // are another video to every cache.
    for t in &timestamps_ms {
        hasher.update(t.to_le_bytes());
    }
    hasher.update(&pixels);
    Some(MediaInput {
        kind: MediaKind::Video,
        pixels: pixels.into(),
        num_frames,
        width,
        height,
        timestamps_ms: Some(timestamps_ms),
        sha256: hasher.finalize().into(),
    })
}

/// The text a frame's image placeholder follows in the prompt; `k` counts
/// from 1, and so does `video` (named only when a request has several).
pub fn frame_label(video: Option<usize>, k: usize, secs: f64) -> String {
    match video {
        Some(j) => format!("Video {j} frame {k} ({secs:.1}s): "),
        None => format!("Frame {k} ({secs:.1}s): "),
    }
}

/// `file:` and the path: ffmpeg reads it as a local file whatever it looks
/// like (a `C:` drive is not a protocol).
fn file_url(file: &Path) -> OsString {
    let mut url = OsString::from("file:");
    url.push(file);
    url
}

fn args(list: &[&str]) -> Vec<OsString> {
    list.iter().map(OsString::from).collect()
}

/// ffprobe's arguments: the container forced, local files only, and only the
/// fields [`parse_probe`] reads (no tags, so a caller cannot bloat the
/// output). `V` skips cover art.
pub fn probe_args(container: Container, file: &Path) -> Vec<OsString> {
    let mut a = args(&[
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-f",
        container.demuxer(),
        "-print_format",
        "json",
        "-select_streams",
        "V:0",
        "-show_entries",
        "format=format_name,duration\
         :stream=codec_type,width,height,duration,avg_frame_rate,r_frame_rate,nb_frames\
         :stream_side_data=rotation",
    ]);
    a.push(file_url(file));
    a
}

/// ffprobe's arguments for a packet scan: each video packet's time and
/// flags (`K` for a keyframe), one per line (demuxing only, no decode).
pub fn packet_args(container: Container, file: &Path) -> Vec<OsString> {
    let mut a = args(&[
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-f",
        container.demuxer(),
        "-select_streams",
        "V:0",
        "-show_entries",
        "packet=pts_time,flags",
        "-of",
        "csv=p=0",
    ]);
    a.push(file_url(file));
    a
}

/// The input half of a decode: quiet, local files only, two threads, the
/// container forced, and (`seek`) a fast seek to `seek` seconds first.
fn input_args(container: Container, file: &Path, seek: Option<f64>) -> Vec<OsString> {
    let mut a = args(&[
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-threads",
        "2",
    ]);
    if let Some(t) = seek {
        a.extend(args(&["-ss", &format!("{t:.6}")]));
    }
    a.extend(args(&["-f", container.demuxer(), "-i"]));
    a.push(file_url(file));
    a
}

/// The output half: the first non-cover-art video stream, `filter`, at most
/// `frames` frames of raw RGB to stdout. No audio, subtitle or data streams.
fn output_args(filter: &str, frames: usize) -> Vec<OsString> {
    args(&[
        "-map",
        "0:V:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filter,
        "-frames:v",
        &frames.to_string(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ])
}

/// ffmpeg's arguments for one pass that samples `plan.frames` frames.
pub fn decode_args(container: Container, file: &Path, plan: &Plan) -> Vec<OsString> {
    let mut a = input_args(container, file, None);
    let filter = format!("fps={:.9},scale={}:{}", plan.fps(), plan.width, plan.height);
    a.extend(output_args(&filter, plan.frames));
    a
}

/// ffmpeg's arguments for frame `k` alone, sought to its time (accurate:
/// ffmpeg decodes from the keyframe before it up to the time).
pub fn seek_args(container: Container, file: &Path, plan: &Plan, k: usize) -> Vec<OsString> {
    let mut a = input_args(container, file, Some(plan.timestamp(k)));
    a.extend(output_args(
        &format!("scale={}:{}", plan.width, plan.height),
        1,
    ));
    a
}

/// A video refused for the caller's reasons (400), or one this server failed
/// to run its decoder on (500).
#[derive(Debug, PartialEq, Eq)]
pub enum VideoError {
    Invalid(String),
    Internal(String),
}

/// One sampled frame and its time in the video.
pub struct Frame {
    pub input: MediaInput,
    pub secs: f64,
}

/// Bytes of ffprobe JSON read at most; the requested fields fit in far less.
const MAX_PROBE_OUTPUT: usize = 64 << 10;
/// Bytes of packet lines read at most: about 250 000 packets, past 600 s at
/// [`MAX_FPS`] three times over.
const MAX_PACKET_SCAN: usize = 4 << 20;
/// ffmpeg's stderr kept for the log.
const STDERR_KEPT: usize = 2048;

/// The caller's video on disk for ffmpeg, removed on drop. Removal is
/// retried by one reaper thread when it fails: on Windows a decoder killed
/// as the request is dropped may still hold the file open for a moment.
struct TempVideo(PathBuf);

impl TempVideo {
    fn write(bytes: &[u8], dir: Option<&Path>) -> std::io::Result<Self> {
        let mut b = tempfile::Builder::new();
        b.prefix("cascadia-video-");
        let mut f = match dir {
            Some(d) => b.tempfile_in(d)?,
            None => b.tempfile()?,
        };
        f.write_all(bytes)?;
        f.flush()?;
        // Our handle closes here; ffmpeg's is then the only one.
        let path = f.into_temp_path().keep().map_err(|e| e.error)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempVideo {
    fn drop(&mut self) {
        let path = std::mem::take(&mut self.0);
        if !removed(&path) {
            reap_later(path);
        }
    }
}

/// Whether `path` is gone now (removed, or never there).
fn removed(path: &Path) -> bool {
    match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
    }
}

/// Hand a temp file that could not be removed to the one reaper thread,
/// which retries every 100 ms for 5 s.
fn reap_later(path: PathBuf) {
    use std::sync::mpsc;
    static REAPER: std::sync::OnceLock<Option<mpsc::Sender<PathBuf>>> = std::sync::OnceLock::new();
    let tx = REAPER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<PathBuf>();
        std::thread::Builder::new()
            .name("video-temp-reaper".into())
            .spawn(move || {
                let mut pending: Vec<(PathBuf, u32)> = Vec::new();
                loop {
                    let next = if pending.is_empty() {
                        rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected)
                    } else {
                        rx.recv_timeout(std::time::Duration::from_millis(100))
                    };
                    match next {
                        Ok(p) => pending.push((p, 0)),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                    pending.retain_mut(|(p, tries)| {
                        *tries += 1;
                        if removed(p) {
                            return false;
                        }
                        if *tries >= 50 {
                            tracing::warn!(path = %p.display(), "could not remove a video temp file");
                            return false;
                        }
                        true
                    });
                }
            })
            .ok()
            .map(|_| tx)
    });
    match tx {
        Some(tx) => {
            let _ = tx.send(path);
        }
        None => tracing::warn!(path = %path.display(), "could not remove a video temp file"),
    }
}

/// Probe, bound and decode one encoded video into its sampled frames, by
/// `deadline` (one per request, shared by its videos).
pub async fn decode_video(
    bytes: Vec<u8>,
    detail: Option<&str>,
    cfg: &MediaConfig,
    deadline: Instant,
) -> Result<Vec<Frame>, VideoError> {
    let tools = cfg
        .ffmpeg
        .as_ref()
        .ok_or_else(|| VideoError::Invalid(NEEDS_FFMPEG.into()))?;
    check_detail(detail).map_err(VideoError::Invalid)?;
    let container = Container::sniff(&bytes).ok_or_else(|| {
        VideoError::Invalid(format!(
            "the data is not a video this server decodes; {SUPPORTED}"
        ))
    })?;
    let timed_out = || {
        VideoError::Invalid(format!(
            "decoding this request's videos took longer than {} s; send a shorter or \
             lower-resolution clip",
            cfg.video_decode_timeout.as_secs_f32()
        ))
    };
    let run = |program: &Path, args: Vec<OsString>, cap: usize| {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);
        async move {
            match run_bounded(cmd, cap, deadline).await {
                Ok(o) => Ok(o),
                Err(RunError::TimedOut) => Err(timed_out()),
                Err(e) => Err(e.into_video_error()),
            }
        }
    };
    // mp4 often needs seeking (its index may sit at the end), so ffmpeg reads
    // a file, not a pipe. Private to this user; removed on drop.
    let dir = cfg.video_temp_dir.clone();
    let file = tokio::task::spawn_blocking(move || TempVideo::write(&bytes, dir.as_deref()))
        .await
        .map_err(|e| VideoError::Internal(format!("video temp file task failed: {e}")))?
        .map_err(|e| VideoError::Internal(format!("could not write the video temp file: {e}")))?;

    let out = run(
        &tools.ffprobe,
        probe_args(container, file.path()),
        MAX_PROBE_OUTPUT,
    )
    .await?;
    if !out.status.success() {
        tracing::warn!(status = %out.status, stderr = %out.stderr, "ffprobe refused a video");
        return Err(VideoError::Invalid(format!(
            "could not read the video; {SUPPORTED}"
        )));
    }
    let probed = parse_probe(&out.stdout, cfg).map_err(VideoError::Invalid)?;
    // Demux only: the frame and keyframe counts that bound the work and pick
    // the decode, and the duration when the container states none.
    let mut scan = tokio::process::Command::new(&tools.ffprobe);
    scan.args(packet_args(container, file.path()));
    let out = match run_bounded(scan, MAX_PACKET_SCAN, deadline).await {
        Ok(o) => o,
        Err(RunError::TimedOut) => return Err(timed_out()),
        Err(RunError::TooMuchOutput) => {
            return Err(VideoError::Invalid(
                "the video has too many frames; send a shorter clip".into(),
            ))
        }
        Err(e) => return Err(e.into_video_error()),
    };
    if !out.status.success() {
        tracing::warn!(status = %out.status, stderr = %out.stderr, "ffprobe could not scan a video");
        return Err(VideoError::Invalid(format!(
            "could not read the video; {SUPPORTED}"
        )));
    }
    let packets = parse_packets(&out.stdout, probed.frame_rate);
    let duration = probed
        .duration
        .or(packets.span)
        .ok_or_else(|| VideoError::Invalid("the video's duration is unknown".into()))?;
    let video = probed
        .bounded(duration, Some(&packets), cfg)
        .map_err(VideoError::Invalid)?;
    let plan = Plan::new(&video, detail, cfg);

    let fb = plan.frame_bytes();
    let mut raw: Vec<(Vec<u8>, f64)> = Vec::with_capacity(plan.frames);
    if plan.seek {
        for k in 0..plan.frames {
            let out = run(
                &tools.ffmpeg,
                seek_args(container, file.path(), &plan, k),
                fb,
            )
            .await?;
            if !out.status.success() {
                tracing::warn!(status = %out.status, stderr = %out.stderr, "ffmpeg failed to decode a video");
                return Err(VideoError::Invalid("could not decode the video".into()));
            }
            // A time past the last frame gives nothing; the rest still count.
            if out.stdout.len() == fb {
                raw.push((out.stdout, plan.timestamp(k)));
            }
        }
    } else {
        let out = run(
            &tools.ffmpeg,
            decode_args(container, file.path(), &plan),
            plan.frames * fb,
        )
        .await?;
        if !out.status.success() {
            tracing::warn!(status = %out.status, stderr = %out.stderr, "ffmpeg failed to decode a video");
            return Err(VideoError::Invalid("could not decode the video".into()));
        }
        raw.extend(
            out.stdout
                .chunks_exact(fb)
                .enumerate()
                .map(|(k, f)| (f.to_vec(), plan.timestamp(k))),
        );
    }
    drop(file);
    // Hashing a few MB of pixels is blocking-pool work, as an image decode is.
    let frames = tokio::task::spawn_blocking(move || {
        raw.into_iter()
            .map(|(f, secs)| Frame {
                input: media::rgb_image(plan.width, plan.height, f),
                secs,
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| VideoError::Internal(format!("video frame task failed: {e}")))?;
    if frames.is_empty() {
        return Err(VideoError::Invalid("the video yielded no frames".into()));
    }
    Ok(frames)
}

#[derive(Debug)]
struct RunOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    /// The first [`STDERR_KEPT`] bytes, for the log only.
    stderr: String,
}

#[derive(Debug)]
enum RunError {
    TimedOut,
    TooMuchOutput,
    Io(std::io::Error),
}

impl RunError {
    fn into_video_error(self) -> VideoError {
        match self {
            RunError::TimedOut => unreachable!("handled by the caller"),
            RunError::TooMuchOutput => {
                tracing::warn!("video decoder wrote more than its output cap; killed");
                VideoError::Invalid("could not decode the video".into())
            }
            RunError::Io(e) => {
                tracing::error!(error = %e, "could not run the video decoder");
                VideoError::Internal("video decoding failed".into())
            }
        }
    }
}

/// Run `cmd` to completion by `deadline`, reading at most `max_stdout`
/// bytes of its stdout. Past either bound the child is killed and reaped, as
/// it is when this future is dropped (`kill_on_drop`).
async fn run_bounded(
    mut cmd: tokio::process::Command,
    max_stdout: usize,
    deadline: Instant,
) -> Result<RunOutput, RunError> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().map_err(RunError::Io)?;
    #[cfg(test)]
    tests::SPAWNED.with(|s| s.borrow_mut().extend(child.id()));
    let mut stdout = child.stdout.take().expect("stdout is piped");
    let mut stderr = child.stderr.take().expect("stderr is piped");
    let run = async {
        let read_out = async {
            let mut buf = Vec::new();
            let mut chunk = vec![0u8; 64 << 10];
            loop {
                let n = stdout.read(&mut chunk).await.map_err(RunError::Io)?;
                if n == 0 {
                    return Ok(buf);
                }
                if buf.len() + n > max_stdout {
                    // Closing the pipes ends the stderr read too.
                    let _ = child.start_kill();
                    return Err(RunError::TooMuchOutput);
                }
                buf.extend_from_slice(&chunk[..n]);
            }
        };
        // Drained to the end, so a chatty decoder never blocks on a full
        // pipe; only the head is kept.
        let read_err = async {
            let mut kept = Vec::new();
            let mut chunk = [0u8; 4096];
            while let Ok(n @ 1..) = stderr.read(&mut chunk).await {
                let room = STDERR_KEPT.saturating_sub(kept.len());
                kept.extend_from_slice(&chunk[..n.min(room)]);
            }
            String::from_utf8_lossy(&kept).into_owned()
        };
        let (out, err) = tokio::join!(read_out, read_err);
        let status = child.wait().await.map_err(RunError::Io)?;
        Ok(RunOutput {
            status,
            stdout: out?,
            stderr: err,
        })
    };
    match tokio::time::timeout_at(deadline, run).await {
        Ok(r) => r,
        Err(_) => {
            // Kill and reap now, not whenever the runtime gets to it.
            let _ = child.kill().await;
            Err(RunError::TimedOut)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn probe_json(format_name: &str, duration: Option<&str>, w: u32, h: u32) -> Vec<u8> {
        let mut v = serde_json::json!({
            "streams": [{"codec_type": "video", "width": w, "height": h,
                "avg_frame_rate": "30/1"}],
            "format": {"format_name": format_name},
        });
        if let Some(d) = duration {
            v["format"]["duration"] = d.into();
        }
        v.to_string().into_bytes()
    }

    /// A bounded video of `duration` s at 30 fps, a keyframe a second.
    fn probe(duration: f64, width: u32, height: u32) -> Video {
        Video {
            duration,
            width,
            height,
            source_frames: duration * 30.0,
            keyframes: Some(duration.ceil() as u64),
        }
    }

    /// `ceil(duration * fps)` frames, at least 1 and at most the cap, spread
    /// uniformly from 0.
    #[test]
    fn sampling_spreads_frames_uniformly() {
        let cfg = MediaConfig::ov_genai(None);
        let p = Plan::new(&probe(3.0, 320, 240), None, &cfg);
        assert_eq!((p.frames, p.width, p.height), (3, 320, 240));
        assert_eq!(p.fps(), 1.0);
        assert_eq!(
            (0..3).map(|k| p.timestamp(k)).collect::<Vec<_>>(),
            [0.0, 1.0, 2.0]
        );
        // A long video keeps the cap, spread over its whole length.
        let p = Plan::new(&probe(600.0, 320, 240), None, &cfg);
        assert_eq!(p.frames, 16);
        assert_eq!(p.timestamp(1), 37.5);
        assert!((p.fps() - 16.0 / 600.0).abs() < 1e-12);
        // A clip shorter than one sampling period still gives a frame, and a
        // fractional second rounds up.
        assert_eq!(Plan::new(&probe(0.2, 320, 240), None, &cfg).frames, 1);
        assert_eq!(Plan::new(&probe(3.03, 320, 240), None, &cfg).frames, 4);
        let mut fast = MediaConfig::ov_genai(None);
        fast.video_fps = 2.0;
        assert_eq!(Plan::new(&probe(3.0, 320, 240), None, &fast).frames, 6);
        assert_eq!(frame_label(None, 2, 1.04), "Frame 2 (1.0s): ");
        assert_eq!(frame_label(Some(2), 1, 0.0), "Video 2 frame 1 (0.0s): ");
    }

    /// A native Qwen video rounds an odd frame count up to the next pair
    /// when the cap allows (GenAI would pad it with a copy of the last
    /// frame); at an odd cap it keeps the cap. Frames-as-images and Gemma 4
    /// keep the count as sampled.
    #[test]
    fn native_qwen_video_samples_whole_pairs() {
        let qwen = MediaConfig::ov_genai(Some("qwen3_vl"));
        assert_eq!(Plan::new(&probe(3.0, 320, 240), None, &qwen).frames, 4);
        assert_eq!(Plan::new(&probe(0.2, 320, 240), None, &qwen).frames, 2);
        assert_eq!(Plan::new(&probe(4.0, 320, 240), None, &qwen).frames, 4);
        assert_eq!(Plan::new(&probe(600.0, 320, 240), None, &qwen).frames, 16);
        let mut odd_cap = qwen.clone();
        odd_cap.max_video_frames = 5;
        assert_eq!(Plan::new(&probe(600.0, 320, 240), None, &odd_cap).frames, 5);
        assert_eq!(Plan::new(&probe(3.0, 320, 240), None, &odd_cap).frames, 4);
        let p = Plan::new(&probe(3.0, 320, 240), None, &qwen);
        assert_eq!(
            (0..4).map(|k| p.timestamp(k)).collect::<Vec<_>>(),
            [0.0, 0.75, 1.5, 2.25]
        );
        for other in [Some("gemma4"), Some("gemma3"), Some("internvl_chat"), None] {
            let cfg = MediaConfig::ov_genai(other);
            assert_eq!(
                Plan::new(&probe(3.0, 320, 240), None, &cfg).frames,
                3,
                "{other:?}"
            );
        }
    }

    /// A native video is one item: the frames stacked in order, their times
    /// in milliseconds, one hash over the stack that changes with any frame.
    #[test]
    fn video_input_stacks_frames_with_their_times() {
        let frame = |v: u8, secs: f64| Frame {
            input: media::rgb_image(2, 1, vec![v; 6]),
            secs,
        };
        let v = video_input(vec![frame(1, 0.0), frame(2, 0.75), frame(3, 1.5)]).unwrap();
        assert_eq!(v.kind, MediaKind::Video);
        assert_eq!((v.num_frames, v.width, v.height), (3, 2, 1));
        assert_eq!(v.pixels.len(), v.expected_len());
        assert_eq!(&v.pixels[..], &[[1u8; 6], [2; 6], [3; 6]].concat()[..]);
        assert_eq!(v.timestamps_ms.as_deref(), Some(&[0, 750, 1500][..]));
        let w = video_input(vec![frame(1, 0.0), frame(2, 0.75), frame(4, 1.5)]).unwrap();
        assert_ne!(v.sha256, w.sha256);
        // The same frames at another rate are another video.
        let slow = video_input(vec![frame(1, 0.0), frame(2, 1.5), frame(3, 3.0)]).unwrap();
        assert_eq!(slow.pixels, v.pixels);
        assert_ne!(slow.sha256, v.sha256);
        assert!(video_input(Vec::new()).is_none());
        let odd = Frame {
            input: media::rgb_image(1, 1, vec![0; 3]),
            secs: 1.0,
        };
        assert!(video_input(vec![frame(1, 0.0), odd]).is_none());
    }

    /// Short clips decode in one pass; one whose source holds many frames
    /// per sampled frame is read by seeking, if it has a keyframe per
    /// sampled frame.
    #[test]
    fn long_clips_are_read_by_seeking() {
        let cfg = MediaConfig::ov_genai(None);
        // 3 s at 30 fps: 90 source frames for 3 samples.
        assert!(!Plan::new(&probe(3.0, 320, 240), None, &cfg).seek);
        // 16 s: 480 for 16, still one pass.
        assert!(!Plan::new(&probe(16.0, 320, 240), None, &cfg).seek);
        // 60 s: 1800 for 16, with 60 keyframes.
        assert!(Plan::new(&probe(60.0, 320, 240), None, &cfg).seek);
        // The same clip with one keyframe (`keyint=infinite`, a browser
        // recording), with fewer keyframes than samples, or uncounted.
        for keyframes in [Some(1), Some(15), None] {
            let rare = Video {
                keyframes,
                ..probe(60.0, 320, 240)
            };
            assert!(!Plan::new(&rare, None, &cfg).seek, "{keyframes:?}");
        }
    }

    /// Frames fit the detail's pixel budget, keep the aspect ratio and have
    /// even sides; `low` is the smallest budget and `high` the largest.
    #[test]
    fn frame_size_follows_detail_with_even_sides() {
        let mut cfg = MediaConfig::ov_genai(None);
        let hd = probe(1.0, 1920, 1080);
        let mut sizes = Vec::new();
        for detail in [Some("low"), None, Some("auto"), Some("high")] {
            let p = Plan::new(&hd, detail, &cfg);
            assert_eq!((p.width % 2, p.height % 2), (0, 0), "{p:?}");
            let ratio = f64::from(p.width) / f64::from(p.height);
            assert!((ratio - 16.0 / 9.0).abs() < 0.02, "{detail:?}: {p:?}");
            sizes.push(u64::from(p.width) * u64::from(p.height));
        }
        assert!(sizes[0] <= cfg.max_video_frame_pixels / 4, "{sizes:?}");
        assert!(
            sizes[0] < sizes[1] && sizes[1] == sizes[2] && sizes[2] < sizes[3],
            "{sizes:?}"
        );
        assert!(sizes[1] <= cfg.max_video_frame_pixels && sizes[3] <= cfg.max_image_pixels);
        let p = Plan::new(&hd, None, &cfg);
        assert_eq!((p.width, p.height), (596, 336));
        // The order holds whatever the operator sets: a frame budget past
        // the image one makes `high` that budget, and `low` stays under it.
        cfg.max_video_frame_pixels = 2 << 20;
        let [low, auto, high] =
            [Some("low"), None, Some("high")].map(|d| Plan::new(&hd, d, &cfg).width);
        assert!(low <= auto && auto <= high, "{low} {auto} {high}");
        // Under the budget the size is kept, rounded down to even.
        let cfg = MediaConfig::ov_genai(None);
        let p = Plan::new(&probe(1.0, 321, 241), None, &cfg);
        assert_eq!((p.width, p.height), (320, 240));
        let mut tiny = MediaConfig::ov_genai(None);
        tiny.max_video_frame_pixels = 4;
        let p = Plan::new(&probe(1.0, 3, 3), None, &tiny);
        assert_eq!((p.width, p.height), (2, 2));
    }

    #[test]
    fn probe_json_is_read_and_bounded() {
        let cfg = MediaConfig::ov_genai(None);
        let mp4 = "mov,mp4,m4a,3gp,3g2,mj2";
        let p = parse_probe(&probe_json(mp4, Some("3.000000"), 320, 240), &cfg).unwrap();
        assert_eq!(
            (p.duration, p.width, p.height, p.frame_rate),
            (Some(3.0), 320, 240, Some(30.0))
        );
        assert_eq!(
            p.bounded(3.0, None, &cfg).unwrap(),
            Video {
                duration: 3.0,
                width: 320,
                height: 240,
                source_frames: 90.0,
                keyframes: None,
            }
        );
        // A packet scan's count stands in for the stated rate.
        let scanned = Packets {
            count: 31,
            keyframes: 2,
            span: Some(3.1),
        };
        let v = p.bounded(3.0, Some(&scanned), &cfg).unwrap();
        assert_eq!((v.source_frames, v.keyframes), (31.0, Some(2)));
        let p = parse_probe(&probe_json("matroska,webm", None, 64, 48), &cfg).unwrap();
        assert_eq!(p.duration, None, "a streamed WebM states none");
        // A quarter turn swaps the sides, as ffmpeg's autorotate does; the
        // stream's duration stands in for the container's; `0/0` is no rate.
        let rotated = serde_json::json!({
            "streams": [{"codec_type": "video", "width": 320, "height": 240,
                "duration": "3.000000", "avg_frame_rate": "0/0", "r_frame_rate": "30000/1001",
                "nb_frames": "120", "side_data_list": [{"rotation": -90}]}],
            "format": {"format_name": mp4},
        });
        let p = parse_probe(rotated.to_string().as_bytes(), &cfg).unwrap();
        assert_eq!((p.duration, p.width, p.height), (Some(3.0), 240, 320));
        assert!((p.frame_rate.unwrap() - 29.97).abs() < 0.01);
        // A variable-rate stream's timebase is no frame rate past 120 fps.
        let vfr = serde_json::json!({
            "streams": [{"codec_type": "video", "width": 320, "height": 240,
                "avg_frame_rate": "0/0", "r_frame_rate": "1000/1"}],
            "format": {"format_name": "matroska,webm", "duration": "3"},
        });
        let p2 = parse_probe(vfr.to_string().as_bytes(), &cfg).unwrap();
        assert_eq!(p2.frame_rate, Some(120.0));
        // The larger of the stated count and duration x rate counts.
        assert_eq!(p.bounded(3.0, None, &cfg).unwrap().source_frames, 120.0);

        let none = br#"{"streams": [], "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1"}}"#;
        for (json, needle) in [
            (none.to_vec(), "no video stream"),
            (probe_json("hls", Some("3"), 320, 240), "container `hls`"),
            (
                probe_json("concat", Some("3"), 320, 240),
                "container `concat`",
            ),
            (probe_json("image2", Some("3"), 320, 240), "not supported"),
            (
                probe_json("mov,mp4,lavfi", Some("3"), 320, 240),
                "not supported",
            ),
            // 7000x7000 is under the image cap (50 MP) but over the video one.
            (
                probe_json(mp4, Some("3"), 7000, 7000),
                "8294400 pixels a frame",
            ),
            (probe_json(mp4, Some("3"), 0, 240), "no pixels"),
            (probe_json(mp4, Some("3"), 4000, 10), "aspect ratio"),
            (b"not json".to_vec(), "could not read the video"),
        ] {
            let err = parse_probe(&json, &cfg).unwrap_err();
            assert!(err.contains(needle), "{needle}: {err}");
        }
        for (d, needle) in [("0.000000", None), ("N/A", None), ("3", Some(3.0))] {
            let p = parse_probe(&probe_json(mp4, Some(d), 320, 240), &cfg).unwrap();
            assert_eq!(p.duration, needle, "{d}");
        }

        let mut short = MediaConfig::ov_genai(None);
        short.max_video_seconds = 2.0;
        let p = parse_probe(&probe_json(mp4, Some("3"), 320, 240), &cfg).unwrap();
        let err = p.bounded(3.0, None, &short).unwrap_err();
        assert!(err.contains("3.0 s long; the limit is 2 s"), "{err}");
        // The work bound: 600 s of 1080p30 passes, 601 s does not, and 4K
        // gets a quarter of that.
        let hd = Probe {
            duration: None,
            width: 1920,
            height: 1080,
            frame_rate: Some(30.0),
            nb_frames: None,
        };
        let mut long = MediaConfig::ov_genai(None);
        long.max_video_seconds = 10_000.0;
        assert!(hd.bounded(600.0, None, &long).is_ok());
        let err = hd.bounded(601.0, None, &long).unwrap_err();
        assert!(err.contains("shorter or lower-resolution"), "{err}");
        let uhd = Probe {
            width: 3840,
            height: 2160,
            ..hd
        };
        assert!(uhd.bounded(150.0, None, &long).is_ok());
        assert!(uhd.bounded(151.0, None, &long).is_err());
        // No stated rate: the bound assumes 60 fps.
        let unknown = Probe {
            frame_rate: None,
            ..hd
        };
        assert!(unknown.bounded(300.0, None, &long).is_ok());
        assert!(unknown.bounded(301.0, None, &long).is_err());
        // A stated frame count past duration x rate counts.
        let dense = Probe {
            nb_frames: Some(100_000),
            ..hd
        };
        assert!(dense.bounded(10.0, None, &long).is_err());
    }

    /// A packet scan counts frames and keyframes, and spans first to last
    /// packet plus one frame, whatever the first timestamp.
    #[test]
    fn packets_are_counted_and_spanned() {
        let p = parse_packets(
            b"0.000000,K__\n0.100000,___\n2.900000,___\n2.800000,K__\n",
            Some(10.0),
        );
        assert_eq!((p.count, p.keyframes), (4, 2));
        assert!((p.span.unwrap() - 3.0).abs() < 1e-9, "{p:?}");
        // A stream that starts at an hour spans its own length.
        let p = parse_packets(b"3600.000000,K__\n3602.900000,___\n", Some(10.0));
        assert!((p.span.unwrap() - 3.0).abs() < 1e-9, "{p:?}");
        assert_eq!(parse_packets(b"0.5,K_\n", None).span, None);
        assert_eq!(
            parse_packets(b"0.000000,K__\n", Some(25.0)).span,
            Some(0.04)
        );
        let empty = parse_packets(b"", Some(25.0));
        assert_eq!((empty.count, empty.span), (0, None));
        let untimed = parse_packets(b"N/A,K__\nN/A,___\n", None);
        assert_eq!(
            (untimed.count, untimed.keyframes, untimed.span),
            (2, 1, None)
        );
    }

    #[test]
    fn ffmpeg_reads_only_the_local_file_with_the_container_forced() {
        let file = Path::new("/tmp/v.bin");
        let strings = |a: Vec<OsString>| -> Vec<String> {
            a.into_iter().map(|a| a.into_string().unwrap()).collect()
        };
        let has = |args: &[String], pair: [&str; 2]| args.windows(2).any(|w| w == pair);
        for args in [
            strings(probe_args(Container::Mov, file)),
            strings(packet_args(Container::Mov, file)),
        ] {
            assert!(has(&args, ["-protocol_whitelist", "file"]), "{args:?}");
            assert!(has(&args, ["-f", "mov"]), "{args:?}");
            assert_eq!(args.last().unwrap(), "file:/tmp/v.bin");
        }
        let plan = Plan {
            frames: 3,
            width: 320,
            height: 240,
            duration: 3.0,
            seek: false,
        };
        let args = strings(decode_args(Container::Matroska, file, &plan));
        for pair in [
            ["-protocol_whitelist", "file"],
            ["-f", "matroska"],
            ["-i", "file:/tmp/v.bin"],
            ["-vf", "fps=1.000000000,scale=320:240"],
            ["-frames:v", "3"],
            ["-pix_fmt", "rgb24"],
            ["-threads", "2"],
        ] {
            assert!(has(&args, pair), "{pair:?} in {args:?}");
        }
        // Input options precede the input.
        let at = |args: &[String], a: &str| args.iter().position(|x| x == a).unwrap();
        assert!(at(&args, "-protocol_whitelist") < at(&args, "-i"));
        assert!(args.contains(&"-nostdin".to_string()));
        assert_eq!(args.last().unwrap(), "pipe:1");
        // A seek reads one frame from its time, the seek before the input.
        let args = strings(seek_args(Container::Mov, file, &plan, 2));
        for pair in [
            ["-ss", "2.000000"],
            ["-vf", "scale=320:240"],
            ["-frames:v", "1"],
            ["-protocol_whitelist", "file"],
        ] {
            assert!(has(&args, pair), "{pair:?} in {args:?}");
        }
        assert!(at(&args, "-ss") < at(&args, "-i"));
    }

    #[test]
    fn containers_are_sniffed_from_their_first_bytes() {
        assert_eq!(
            Container::sniff(b"\x00\x00\x00\x20ftypisom...."),
            Some(Container::Mov)
        );
        assert_eq!(
            Container::sniff(b"\x00\x00\x00\x08wide\x00\x00"),
            Some(Container::Mov)
        );
        assert_eq!(
            Container::sniff(&[0x1a, 0x45, 0xdf, 0xa3, 0x01]),
            Some(Container::Matroska)
        );
        for other in [
            &b"#EXTM3U\n#EXT-X-VERSION:3\n"[..],
            b"ffconcat version 1.0\n",
            b"\x89PNG\r\n\x1a\n",
            b"abc",
        ] {
            assert_eq!(Container::sniff(other), None, "{other:?}");
        }
    }

    #[test]
    fn video_urls_are_data_urls_of_a_video_type() {
        use base64::Engine as _;
        let mut cfg = MediaConfig::ov_genai(None);
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"\x00\x00\x00\x08ftyp");
        for mime in VIDEO_MIME_TYPES {
            let url = format!("data:{mime};base64,{b64}");
            assert_eq!(video_url_bytes(&url, None, &cfg).unwrap().len(), 8);
        }
        for (url, detail, needle) in [
            ("https://example.com/a.mp4", None, "only `data:` video URLs"),
            (
                "data:image/png;base64,AAAA",
                None,
                "not a supported video type",
            ),
            (
                "data:video/x-msvideo;base64,AAAA",
                None,
                "not a supported video type",
            ),
            ("data:video/mp4,raw", None, "base64-encoded"),
            ("data:video/mp4;base64,@@", None, "not valid base64"),
            ("data:video/mp4;base64,AAAA", Some("original"), "detail"),
        ] {
            let err = video_url_bytes(url, detail, &cfg).unwrap_err();
            assert!(err.contains(needle), "{url}: {err}");
        }
        cfg.allow_remote_urls = true;
        let err = video_url_bytes("ftp://example.com/a.mp4", None, &cfg).unwrap_err();
        assert!(err.contains("`https://` or `data:`"), "{err}");
    }

    thread_local! {
        /// The pids [`run_bounded`] spawned on this thread. `#[tokio::test]`
        /// runs on one thread, so each test sees only its own.
        pub(super) static SPAWNED: std::cell::RefCell<Vec<u32>> = const {
            std::cell::RefCell::new(Vec::new())
        };
    }

    pub(super) fn take_spawned() -> Vec<u32> {
        SPAWNED.with(|s| std::mem::take(&mut *s.borrow_mut()))
    }

    /// Whether `pid` is gone: no process and no zombie (a killed child that
    /// was never waited for still shows, as `Z`).
    pub(super) fn reaped(pid: u32) -> bool {
        let out = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().is_empty()
    }

    /// Past the deadline, or past the output cap, the child is killed and
    /// reaped: no process (or zombie) is left behind.
    #[cfg(unix)]
    #[tokio::test]
    async fn bounded_runs_kill_and_reap_their_child() {
        let mut sleep = tokio::process::Command::new("sleep");
        sleep.arg("30");
        let t0 = std::time::Instant::now();
        let r = run_bounded(sleep, 1024, Instant::now() + Duration::from_millis(200)).await;
        assert!(matches!(r, Err(RunError::TimedOut)), "{r:?}");
        assert!(t0.elapsed() < Duration::from_secs(5));
        let pids = take_spawned();
        assert_eq!(pids.len(), 1);
        assert!(pids.iter().all(|&p| reaped(p)), "{pids:?}");

        let yes = tokio::process::Command::new("yes");
        let r = run_bounded(yes, 1 << 20, Instant::now() + Duration::from_secs(10)).await;
        assert!(matches!(r, Err(RunError::TooMuchOutput)), "{r:?}");
        let pids = take_spawned();
        assert!(pids.iter().all(|&p| reaped(p)), "{pids:?}");

        let echo = {
            let mut c = tokio::process::Command::new("sh");
            c.args(["-c", "printf abc; printf oops >&2; exit 3"]);
            c
        };
        let out = run_bounded(echo, 1024, Instant::now() + Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(out.stdout, b"abc");
        assert_eq!(out.stderr, "oops");
        assert_eq!(out.status.code(), Some(3));
    }

    #[test]
    fn ffmpeg_is_located_or_named_as_missing() {
        let err = Ffmpeg::locate(Some(Path::new("/nonexistent/ffmpeg"))).unwrap_err();
        assert!(err.contains("not an executable"), "{err}");
        if let Ok(found) = Ffmpeg::locate(None) {
            assert!(found.ffprobe.ends_with("ffprobe"), "{found:?}");
            assert_eq!(
                Ffmpeg::locate(Some(&found.ffmpeg)).unwrap().ffmpeg,
                found.ffmpeg
            );
        }
    }

    /// Through the API, against the mock engine, with a real ffmpeg. Each
    /// test skips (and says so) where ffmpeg is not on PATH.
    mod api {
        use super::super::*;
        use crate::tests::{make_app, png_url, post_chat};
        use crate::{make_router_with_config, ChatTemplateConfig, Config};
        use axum::body::{to_bytes, Body};
        use axum::http::{Request, StatusCode};
        use axum::Router;
        use base64::Engine as _;
        use cascadia_engine_mock::MockEngine;
        use cascadia_runner::Runner;
        use cascadia_types::{Chunk, GenerationTask, PeerLayout, ShardSpec, TaskId};
        use serde_json::{json, Value};
        use std::sync::{Arc, Mutex};
        use std::time::Duration;
        use tower::ServiceExt;

        /// What the engine was handed: each task's prompt and its images'
        /// sizes.
        type Seen = Arc<Mutex<Vec<(String, Vec<(u32, u32, u32)>)>>>;

        /// The mock, recording each task; a task with more than
        /// `max_media` images fails as over the context window, as a real
        /// engine counting each image's tokens would.
        struct RecordingEngine {
            mock: MockEngine,
            seen: Seen,
            max_media: usize,
        }

        impl cascadia_engine::Engine for RecordingEngine {
            fn warmup(&mut self) {}
            fn cancel(&mut self, task_id: &TaskId) {
                self.mock.cancel(task_id)
            }
            fn submit(&mut self, task: GenerationTask) -> cascadia_engine::EngineResult<()> {
                let sizes = task
                    .media
                    .iter()
                    .map(|m| (m.width, m.height, m.num_frames))
                    .collect();
                self.seen.lock().unwrap().push((task.prompt.clone(), sizes));
                if task.media.len() > self.max_media {
                    return Err(cascadia_engine::EngineError::PromptTooLong(format!(
                        "{} images do not fit the context",
                        task.media.len()
                    )));
                }
                // Recorded above; the mock (which refuses a native video)
                // only echoes the prompt.
                let mut task = task;
                task.media.clear();
                self.mock.submit(task)
            }
            fn step(&mut self) -> cascadia_engine::EngineResult<Vec<(TaskId, Chunk)>> {
                self.mock.step()
            }
        }

        struct RecordingBuilder(Seen, usize);

        #[::async_trait::async_trait]
        impl cascadia_engine::Builder for RecordingBuilder {
            async fn connect(&mut self, _: PeerLayout) -> cascadia_engine::EngineResult<()> {
                Ok(())
            }
            async fn load(
                &mut self,
                _: ShardSpec,
            ) -> cascadia_engine::EngineResult<cascadia_engine::LoadStream> {
                Ok(Box::pin(futures::stream::empty()))
            }
            fn build(
                self: Box<Self>,
            ) -> cascadia_engine::EngineResult<Box<dyn cascadia_engine::Engine>> {
                Ok(Box::new(RecordingEngine {
                    mock: MockEngine::new(),
                    seen: self.0,
                    max_media: self.1,
                }))
            }
        }

        fn ffmpeg() -> Option<Ffmpeg> {
            match Ffmpeg::locate(None) {
                Ok(f) => Some(f),
                Err(why) => {
                    eprintln!("skipping: video tests need ffmpeg ({why})");
                    None
                }
            }
        }

        /// An image server that decodes videos with `ffmpeg`, `tweak`ed.
        async fn video_app(ffmpeg: Ffmpeg, tweak: impl FnOnce(&mut MediaConfig)) -> (Router, Seen) {
            video_app_with_window(ffmpeg, usize::MAX, tweak).await
        }

        /// [`video_app`] on an engine whose context fits `max_media` images.
        async fn video_app_with_window(
            ffmpeg: Ffmpeg,
            max_media: usize,
            tweak: impl FnOnce(&mut MediaConfig),
        ) -> (Router, Seen) {
            let seen = Seen::default();
            let runner = Runner::new(Box::new(RecordingBuilder(Arc::clone(&seen), max_media)));
            runner
                .start(
                    PeerLayout::single_stage(),
                    ShardSpec::single_stage("mock-model", "CPU"),
                )
                .await
                .unwrap();
            let mut media = MediaConfig::ov_genai(None);
            media.ffmpeg = Some(ffmpeg);
            tweak(&mut media);
            let cfg = Config {
                chat_template: ChatTemplateConfig {
                    template: Some(
                        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}".into(),
                    ),
                    bos_token: None,
                    eos_token: None,
                },
                media: Some(media),
                ..Config::default()
            };
            (
                make_router_with_config(Arc::new(runner), "mock-model", cfg),
                seen,
            )
        }

        /// A `testsrc` clip made by ffmpeg: `secs` long at 10 fps, 320x240,
        /// as a `data:` URL of `mime`.
        fn clip(ffmpeg: &Ffmpeg, ext: &str, mime: &str, secs: u32) -> String {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(format!("clip.{ext}"));
            let mut cmd = std::process::Command::new(&ffmpeg.ffmpeg);
            cmd.args(["-nostdin", "-loglevel", "error", "-f", "lavfi", "-i"])
                .arg(format!("testsrc=size=320x240:rate=10:duration={secs}"));
            if ext == "webm" {
                cmd.args(["-c:v", "libvpx", "-b:v", "200k"]);
            } else {
                cmd.args(["-pix_fmt", "yuv420p"]);
            }
            let status = cmd.arg(&path).status().unwrap();
            assert!(status.success(), "ffmpeg could not make the {ext} clip");
            let bytes = std::fs::read(&path).unwrap();
            format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            )
        }

        fn video_msg(text: &str, url: &str) -> Value {
            json!({"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "video_url", "video_url": {"url": url}}]})
        }

        fn error_message(v: &Value) -> &str {
            v["error"]["message"].as_str().unwrap_or_default()
        }

        /// A 3 s clip at 1 fps reaches the engine as three 320x240 images,
        /// each placeholder behind its frame's label, where the caller put
        /// the video; MP4 and WebM alike.
        #[tokio::test]
        async fn a_video_reaches_the_engine_as_labelled_frames() {
            let Some(ff) = ffmpeg() else { return };
            for (ext, mime) in [("mp4", "video/mp4"), ("webm", "video/webm")] {
                let url = clip(&ff, ext, mime, 3);
                let (app, seen) = video_app(ff.clone(), |_| {}).await;
                let (status, v) = post_chat(
                    app,
                    json!({"model": "mock-model", "max_tokens": 1,
                        "messages": [video_msg("what happens?", &url)]}),
                )
                .await;
                assert_eq!(status, StatusCode::OK, "{ext}: {v}");
                let seen = seen.lock().unwrap();
                let (prompt, sizes) = &seen[0];
                assert_eq!(sizes, &vec![(320, 240, 1); 3], "{ext}");
                assert!(
                    prompt.contains(
                        "user: what happens?Frame 1 (0.0s): <ov_genai_image_0>\
                         Frame 2 (1.0s): <ov_genai_image_1>Frame 3 (2.0s): <ov_genai_image_2>\n"
                    ),
                    "{ext}: {prompt:?}"
                );
            }
        }

        /// Frames take the image indices after the images before them, and
        /// the request's images and frames are counted apart: eight images
        /// and a video's frames fit together.
        #[tokio::test]
        async fn frames_and_images_share_the_placeholder_sequence() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 2);
            let (app, seen) = video_app(ff, |m| m.max_video_frame_pixels = 160 * 120).await;
            let mut content =
                vec![json!({"type": "image_url", "image_url": {"url": png_url(8, 8)}})];
            content.push(json!({"type": "video_url", "video_url": url}));
            for _ in 0..7 {
                content.push(json!({"type": "image_url", "image_url": {"url": png_url(4, 4)}}));
            }
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [{"role": "user", "content": content}]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let seen = seen.lock().unwrap();
            let (prompt, sizes) = &seen[0];
            assert_eq!(sizes.len(), 10);
            assert_eq!(sizes[0], (8, 8, 1));
            assert_eq!(&sizes[1..3], &[(160, 120, 1); 2]);
            assert!(
                prompt.contains(
                    "<ov_genai_image_0>Frame 1 (0.0s): <ov_genai_image_1>\
                     Frame 2 (1.0s): <ov_genai_image_2><ov_genai_image_3>"
                ),
                "{prompt:?}"
            );
        }

        /// An ov-genai server on a family GenAI takes video for natively.
        fn native(model_type: &str) -> impl FnOnce(&mut MediaConfig) + '_ {
            move |m| {
                let n = MediaConfig::ov_genai(Some(model_type));
                m.native_video = n.native_video;
                m.video_frame_multiple = n.video_frame_multiple;
                m.reserved_text = n.reserved_text;
            }
        }

        /// On ov-genai/qwen3_vl a video is ONE item of all its frames (a 3 s
        /// clip at 1 fps rounded up to two pairs), behind `<ov_genai_video_0>`
        /// where the caller put it, with no per-frame labels.
        #[tokio::test]
        async fn a_native_video_reaches_the_engine_as_one_item() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 3);
            let (app, seen) = video_app(ff, native("qwen3_vl")).await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("what happens?", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let seen = seen.lock().unwrap();
            let (prompt, sizes) = &seen[0];
            assert_eq!(sizes, &vec![(320, 240, 4)]);
            assert!(
                prompt.contains("user: what happens?<ov_genai_video_0>\n"),
                "{prompt:?}"
            );
            assert!(!prompt.contains("Frame"), "{prompt:?}");
        }

        /// The same request on an ov-genai/gemma3 export (no native video in
        /// GenAI) arrives as labelled images.
        #[tokio::test]
        async fn a_video_on_a_family_without_native_video_stays_frames() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 3);
            let (app, seen) = video_app(ff, native("gemma3")).await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("what happens?", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let seen = seen.lock().unwrap();
            let (prompt, sizes) = &seen[0];
            assert_eq!(sizes, &vec![(320, 240, 1); 3]);
            assert!(
                prompt.contains("Frame 1 (0.0s): <ov_genai_image_0>"),
                "{prompt:?}"
            );
            assert!(!prompt.contains("<ov_genai_video_"), "{prompt:?}");
        }

        /// Images and native videos number their tags apart, each from 0,
        /// in the order the caller sent them, and reach the engine in that
        /// order.
        #[tokio::test]
        async fn native_videos_and_images_number_their_tags_apart() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 2);
            let (app, seen) = video_app(ff, |m| {
                native("qwen2_5_vl")(m);
                m.max_videos = 2;
            })
            .await;
            let content = json!([
                {"type": "image_url", "image_url": {"url": png_url(8, 8)}},
                {"type": "video_url", "video_url": url},
                {"type": "text", "text": "and"},
                {"type": "image_url", "image_url": {"url": png_url(4, 4)}},
                {"type": "video_url", "video_url": url},
            ]);
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [{"role": "user", "content": content}]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let seen = seen.lock().unwrap();
            let (prompt, sizes) = &seen[0];
            assert_eq!(
                sizes,
                &vec![(8, 8, 1), (320, 240, 2), (4, 4, 1), (320, 240, 2)]
            );
            assert!(
                prompt.contains(
                    "<ov_genai_image_0><ov_genai_video_0>and<ov_genai_image_1><ov_genai_video_1>"
                ),
                "{prompt:?}"
            );
        }

        /// A staged Qwen3-VL tree (`ov-runtime`) takes each video whole
        /// behind Qwen's own video tag, the same for every video, next to
        /// its image tag; rank 0 expands both.
        #[tokio::test]
        async fn a_staged_qwen_tree_gets_each_video_behind_qwens_tag() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 3);
            let (app, seen) = video_app(ff, |m| {
                let q = MediaConfig::qwen35_native().staged_native_video("qwen3_vl");
                m.image_placeholder = q.image_placeholder;
                m.reserved_text = q.reserved_text;
                m.native_video = q.native_video;
                m.video_placeholder = q.video_placeholder;
                m.video_frame_multiple = q.video_frame_multiple;
                m.max_videos = 2;
            })
            .await;
            let content = json!([
                {"type": "video_url", "video_url": url},
                {"type": "text", "text": "and"},
                {"type": "image_url", "image_url": {"url": png_url(4, 4)}},
                {"type": "video_url", "video_url": url},
            ]);
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [{"role": "user", "content": content}]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let seen = seen.lock().unwrap();
            let (prompt, sizes) = &seen[0];
            // 3 s at 1 fps, rounded up to two frame pairs.
            assert_eq!(sizes, &vec![(320, 240, 4), (4, 4, 1), (320, 240, 4)]);
            let vid = "<|vision_start|><|video_pad|><|vision_end|>";
            let img = "<|vision_start|><|image_pad|><|vision_end|>";
            assert!(
                prompt.contains(&format!("{vid}and{img}{vid}")),
                "{prompt:?}"
            );
            assert!(!prompt.contains("Frame"), "{prompt:?}");
        }

        /// A Qwen3-VL tree's video encoder refuses a frame under 32 px a
        /// side; the API says so with a 400 before the engine sees it.
        #[tokio::test]
        async fn a_staged_qwen3_video_too_small_to_encode_is_a_400() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 2);
            let (app, seen) = video_app(ff, |m| {
                let q = MediaConfig::qwen35_native().staged_native_video("qwen3_vl");
                m.image_placeholder = q.image_placeholder;
                m.reserved_text = q.reserved_text;
                m.native_video = q.native_video;
                m.video_placeholder = q.video_placeholder;
                m.video_frame_multiple = q.video_frame_multiple;
                m.video_min_side = q.video_min_side;
                // 320x240 scaled to fit 40x30.
                m.max_video_frame_pixels = 40 * 30;
            })
            .await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("what happens?", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            assert!(v.to_string().contains("at least 32 pixels a side"), "{v}");
            assert!(seen.lock().unwrap().is_empty());
        }

        /// Caller text still may not carry the video tag a native video uses.
        #[tokio::test]
        async fn caller_text_cannot_forge_a_video_tag() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 1);
            let (app, seen) = video_app(ff, native("qwen3_vl")).await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("see <ov_genai_video_0>", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            assert!(error_message(&v).contains("<ov_genai_video_"), "{v}");
            assert!(
                error_message(&v).contains("reserved for video placeholders"),
                "{v}"
            );
            assert!(seen.lock().unwrap().is_empty());
        }

        /// Frames past the context window get the engine's own
        /// `context_length_exceeded`, as too many images do.
        #[tokio::test]
        async fn frames_past_the_context_are_context_length_exceeded() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 3);
            let (app, _) = video_app_with_window(ff, 2, |_| {}).await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "messages": [video_msg("x", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
            assert_eq!(v["error"]["code"], "context_length_exceeded", "{v}");
        }

        /// Gemma 3 trims each text part; a frame's label is trimmed as the
        /// caller's own label would be.
        #[tokio::test]
        async fn labels_follow_the_familys_text_part_rules() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 1);
            let (app, seen) = video_app(ff, |m| {
                let g = MediaConfig::gemma3_native();
                m.image_placeholder = g.image_placeholder;
                m.reserved_text = g.reserved_text;
                m.trim_text_parts = true;
            })
            .await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg(" what? ", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let prompt = &seen.lock().unwrap()[0].0;
            assert!(
                prompt.contains("user: what?Frame 1 (0.0s):<start_of_image>\n"),
                "{prompt:?}"
            );
        }

        /// What the video path refuses comes back as an OpenAI 400, and a
        /// broken file never takes the server down with it.
        #[tokio::test]
        async fn bad_videos_are_400s_and_the_server_keeps_serving() {
            let Some(ff) = ffmpeg() else { return };
            let good = clip(&ff, "mp4", "video/mp4", 3);
            let (app, _) = video_app(ff.clone(), |m| m.max_video_seconds = 2.0).await;
            let raw = base64::engine::general_purpose::STANDARD
                .decode(good.split_once(',').unwrap().1)
                .unwrap();
            let truncated = format!(
                "data:video/mp4;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&raw[..raw.len() / 3])
            );
            let bogus = format!(
                "data:video/mp4;base64,{}",
                base64::engine::general_purpose::STANDARD
                    .encode(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00junkjunkjunk")
            );
            let not_video = format!(
                "data:video/mp4;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(b"#EXTM3U\nfile:///etc/hosts\n")
            );
            for (url, needle) in [
                (good.as_str(), "3.0 s long; the limit is 2 s"),
                (bogus.as_str(), "video 0: could not read the video"),
                (truncated.as_str(), "video 0: "),
                (not_video.as_str(), "not a video this server decodes"),
                ("https://example.com/a.mp4", "only `data:` video URLs"),
            ] {
                let (status, v) = post_chat(
                    app.clone(),
                    json!({"model": "mock-model", "messages": [video_msg("x", url)]}),
                )
                .await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{needle}: {v}");
                assert_eq!(v["error"]["type"], "invalid_request_error");
                assert!(error_message(&v).contains(needle), "{needle}: {v}");
            }
            let (status, v) = post_chat(
                app.clone(),
                json!({"model": "mock-model",
                    "messages": [video_msg("x", &good), video_msg("y", &good)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(error_message(&v).contains("2 videos in one request; the limit is 1"));
            let (status, v) = post_chat(
                app.clone(),
                json!({"model": "mock-model", "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [
                        {"type": "video_url", "video_url": {"url": good}}]}]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(error_message(&v).contains("`video_url` parts are only accepted in `user`"));
            // Still serving.
            let short = clip(&ff, "mp4", "video/mp4", 1);
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("x", &short)]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
        }

        /// An ffmpeg that probes fine but decodes forever (`sleep` behind
        /// the real ffprobe): past the request's deadline the decode is
        /// killed and reaped, the request is a 400 that says what to send,
        /// the temp file is gone, and the same server serves the next
        /// request.
        #[cfg(unix)]
        #[tokio::test]
        async fn a_decode_past_its_deadline_is_killed() {
            use std::os::unix::fs::PermissionsExt as _;
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 2);
            let bin = tempfile::tempdir().unwrap();
            let slow = bin.path().join("ffmpeg");
            std::fs::write(&slow, "#!/bin/sh\nexec sleep 29\n").unwrap();
            std::fs::set_permissions(&slow, std::fs::Permissions::from_mode(0o755)).unwrap();
            let temp = tempfile::tempdir().unwrap();
            let slow_ff = Ffmpeg {
                ffmpeg: slow,
                ffprobe: ff.ffprobe.clone(),
            };
            let (app, _) = video_app(slow_ff, |m| {
                m.video_decode_timeout = Duration::from_millis(1500);
                m.decode_wait = Duration::from_millis(500);
                m.video_temp_dir = Some(temp.path().to_path_buf());
            })
            .await;
            super::take_spawned();
            // Twice: a request that kept its decode slot would leave none
            // for the image below (two slots, a short wait).
            for _ in 0..2 {
                let t0 = std::time::Instant::now();
                let (status, v) = post_chat(
                    app.clone(),
                    json!({"model": "mock-model", "messages": [video_msg("x", &url)]}),
                )
                .await;
                assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", t0.elapsed());
                assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
                assert!(
                    error_message(&v).contains(
                        "decoding this request's videos took longer than 1.5 s; send a shorter \
                         or lower-resolution clip"
                    ),
                    "{v}"
                );
                let pids = super::take_spawned();
                assert_eq!(pids.len(), 3, "probe, packet scan, decode: {pids:?}");
                assert!(pids.iter().all(|&p| super::reaped(p)), "{pids:?}");
                assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
            }
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1, "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "still there?"},
                        {"type": "image_url", "image_url": {"url": png_url(8, 8)}}]}]}),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::OK,
                "the decode slots were released: {v}"
            );
        }

        /// The temp file is removed after a decode that works and after one
        /// that fails.
        #[tokio::test]
        async fn temp_files_are_removed() {
            let Some(ff) = ffmpeg() else { return };
            let good = clip(&ff, "mp4", "video/mp4", 1);
            let bogus = format!(
                "data:video/mp4;base64,{}",
                base64::engine::general_purpose::STANDARD
                    .encode(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00junkjunkjunk")
            );
            let temp = tempfile::tempdir().unwrap();
            let (app, _) =
                video_app(ff, |m| m.video_temp_dir = Some(temp.path().to_path_buf())).await;
            for (url, want) in [(&good, StatusCode::OK), (&bogus, StatusCode::BAD_REQUEST)] {
                let (status, v) = post_chat(
                    app.clone(),
                    json!({"model": "mock-model", "max_tokens": 1,
                        "messages": [video_msg("x", url)]}),
                )
                .await;
                assert_eq!(status, want, "{v}");
                assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
            }
        }

        /// ffmpeg's output for `args`, as a `data:` URL of `mime`; `pipe`
        /// writes it to stdout, as a browser's MediaRecorder streams WebM
        /// (no duration, no cues).
        fn made(ffmpeg: &Ffmpeg, args: &[&str], mime: &str) -> String {
            let out = std::process::Command::new(&ffmpeg.ffmpeg)
                .args(["-nostdin", "-loglevel", "error"])
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(out.stdout)
            )
        }

        async fn frames_of(app: Router, seen: &Seen, url: &str) -> (String, Vec<(u32, u32, u32)>) {
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("x", url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            seen.lock().unwrap().pop().unwrap()
        }

        /// A WebM muxed to a pipe states no duration; it is read from the
        /// packets, and the clip is sampled as any other.
        #[tokio::test]
        async fn a_streamed_webm_without_a_duration_is_served() {
            let Some(ff) = ffmpeg() else { return };
            let url = made(
                &ff,
                &[
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=320x240:rate=10:duration=3",
                    "-c:v",
                    "libvpx",
                    "-b:v",
                    "200k",
                    "-f",
                    "webm",
                    "pipe:1",
                ],
                "video/webm",
            );
            let (app, seen) = video_app(ff, |_| {}).await;
            let (prompt, sizes) = frames_of(app, &seen, &url).await;
            assert_eq!(sizes, vec![(320, 240, 1); 3]);
            assert!(prompt.contains("Frame 3 (2.0s): "), "{prompt:?}");
        }

        /// A phone video's rotation reaches the model upright: a 320x240
        /// clip rotated a quarter turn gives 240x320 frames.
        #[tokio::test]
        async fn a_rotated_clip_gives_upright_frames() {
            let Some(ff) = ffmpeg() else { return };
            let dir = tempfile::tempdir().unwrap();
            let plain = dir.path().join("plain.mp4");
            let status = std::process::Command::new(&ff.ffmpeg)
                .args(["-nostdin", "-loglevel", "error", "-f", "lavfi", "-i"])
                .arg("testsrc=size=320x240:rate=10:duration=2")
                .args(["-pix_fmt", "yuv420p"])
                .arg(&plain)
                .status()
                .unwrap();
            assert!(status.success());
            let plain = plain.to_str().unwrap();
            let url = made(
                &ff,
                &[
                    "-display_rotation",
                    "90",
                    "-i",
                    plain,
                    "-c",
                    "copy",
                    "-f",
                    "mp4",
                    "-movflags",
                    "frag_keyframe+empty_moov",
                    "pipe:1",
                ],
                "video/mp4",
            );
            let (app, seen) = video_app(ff, |_| {}).await;
            let (_, sizes) = frames_of(app, &seen, &url).await;
            assert_eq!(sizes, vec![(240, 320, 1); 2]);
        }

        /// A `testsrc` clip of `secs` s at 160x120 and 30 fps, its timestamps
        /// starting at an hour, with `x264` options (its GOP), as streamed
        /// MP4.
        fn long_clip(ff: &Ffmpeg, secs: u32, x264: &str) -> String {
            let src = format!("testsrc=size=160x120:rate=30:duration={secs}");
            made(
                ff,
                &[
                    "-f",
                    "lavfi",
                    "-i",
                    &src,
                    "-pix_fmt",
                    "yuv420p",
                    "-x264-params",
                    x264,
                    "-output_ts_offset",
                    "3600",
                    "-f",
                    "mp4",
                    "-movflags",
                    "frag_keyframe+empty_moov",
                    "pipe:1",
                ],
                "video/mp4",
            )
        }

        /// A clip with many source frames per sample and a keyframe a second
        /// is read by seeking, one frame per sampled time; the labels count
        /// from the clip's start, not its first timestamp (an hour).
        #[tokio::test]
        async fn a_long_clip_is_sampled_by_seeking() {
            let Some(ff) = ffmpeg() else { return };
            let url = long_clip(&ff, 60, "keyint=30:min-keyint=30:scenecut=0");
            let (app, seen) = video_app(ff, |_| {}).await;
            super::take_spawned();
            let (prompt, sizes) = frames_of(app, &seen, &url).await;
            assert_eq!(sizes, vec![(160, 120, 1); 16]);
            assert!(prompt.contains("Frame 1 (0.0s): "), "{prompt:?}");
            assert!(prompt.contains("Frame 2 (3.8s): "), "{prompt:?}");
            assert!(prompt.contains("Frame 16 (56.2s): "), "{prompt:?}");
            assert_eq!(
                super::take_spawned().len(),
                2 + 16,
                "probe, packet scan, one run per frame"
            );
        }

        /// The same length with one keyframe (`keyint=infinite`, as a
        /// browser recording nearly has) is read in one pass: seeking would
        /// decode most of it once per frame.
        #[tokio::test]
        async fn a_clip_with_rare_keyframes_is_read_in_one_pass() {
            let Some(ff) = ffmpeg() else { return };
            let url = long_clip(&ff, 60, "keyint=infinite:scenecut=0");
            let (app, seen) = video_app(ff, |_| {}).await;
            super::take_spawned();
            let (prompt, sizes) = frames_of(app, &seen, &url).await;
            assert_eq!(sizes, vec![(160, 120, 1); 16]);
            assert!(prompt.contains("Frame 16 (56.2s): "), "{prompt:?}");
            assert_eq!(
                super::take_spawned().len(),
                3,
                "probe, packet scan, one pass"
            );
        }

        /// A short clip starting at an hour: one pass, labels from 0 s.
        #[tokio::test]
        async fn timestamps_count_from_the_clips_start() {
            let Some(ff) = ffmpeg() else { return };
            let url = long_clip(&ff, 3, "keyint=30");
            let (app, seen) = video_app(ff, |_| {}).await;
            let (prompt, sizes) = frames_of(app, &seen, &url).await;
            assert_eq!(sizes.len(), 3);
            assert!(
                prompt.contains("Frame 1 (0.0s): ") && prompt.contains("Frame 3 (2.0s): "),
                "{prompt:?}"
            );
        }

        /// Several videos in one request name each video in their labels.
        #[tokio::test]
        async fn several_videos_are_told_apart() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 1);
            let (app, seen) = video_app(ff, |m| m.max_videos = 2).await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1, "messages": [
                    {"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url": url}},
                        {"type": "video_url", "video_url": {"url": url}}]}]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let prompt = &seen.lock().unwrap()[0].0;
            assert!(
                prompt.contains(
                    "Video 1 frame 1 (0.0s): <ov_genai_image_0>Video 2 frame 1 (0.0s): \
                     <ov_genai_image_1>"
                ),
                "{prompt:?}"
            );
        }

        /// Without ffmpeg a video is a 400 naming the flag, and `/v1/models`
        /// does not list video; a server without image input refuses it as
        /// it refuses images.
        #[tokio::test]
        async fn video_needs_images_and_ffmpeg() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 1);
            let modalities = |app: Router| async move {
                let r = app
                    .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                let v: Value =
                    serde_json::from_slice(&to_bytes(r.into_body(), 4096).await.unwrap()).unwrap();
                v["data"][0]["input_modalities"].clone()
            };
            let (with, _) = video_app(ff.clone(), |_| {}).await;
            assert_eq!(modalities(with).await, json!(["text", "image", "video"]));
            let (off, _) = video_app(ff.clone(), |m| m.max_videos = 0).await;
            assert_eq!(modalities(off.clone()).await, json!(["text", "image"]));
            let (status, v) = post_chat(
                off,
                json!({"model": "mock-model", "messages": [video_msg("x", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(
                error_message(&v).contains("video input is turned off"),
                "{v}"
            );
            let (without, _) = video_app(ff, |m| m.ffmpeg = None).await;
            assert_eq!(modalities(without.clone()).await, json!(["text", "image"]));
            let (status, v) = post_chat(
                without,
                json!({"model": "mock-model", "messages": [video_msg("x", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(error_message(&v).contains("`--api-ffmpeg`"), "{v}");
            let (status, v) = post_chat(
                make_app().await,
                json!({"model": "mock-model", "messages": [video_msg("x", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(
                error_message(&v).contains("does not accept image input"),
                "{v}"
            );
        }

        /// The video allowance raises the chat body limit only on a server
        /// that decodes videos, and the videos' bytes are held to it.
        #[tokio::test]
        async fn the_video_allowance_is_for_video_servers_only() {
            let Some(ff) = ffmpeg() else { return };
            let mut raw = b"\x00\x00\x00\x18ftypisom".to_vec();
            raw.resize(30 << 20, 0);
            let url = format!(
                "data:video/mp4;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&raw)
            );
            let body = json!({"model": "mock-model", "messages": [video_msg("x", &url)]});
            let (with, _) = video_app(ff.clone(), |_| {}).await;
            let (status, v) = post_chat(with, body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            assert!(
                error_message(&v).contains("could not read the video"),
                "{v}"
            );
            let (without, _) = video_app(ff.clone(), |m| m.ffmpeg = None).await;
            let (status, v) = post_chat(without, body.clone()).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
            assert!(!error_message(&v).contains("--api-max-video-mb"), "{v}");
            // Over every allowance on a video server, the 413 names the flag.
            let (tight, _) = video_app(ff.clone(), |m| m.max_request_video_bytes = 1 << 20).await;
            let (status, v) = post_chat(tight, body.clone()).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
            assert!(
                error_message(&v).contains("images or videos")
                    && error_message(&v).contains("`--api-max-video-mb`"),
                "{v}"
            );
            // The image allowance still covers the base64 overhead here; the
            // decoded bytes are what the video budget holds.
            let (small, _) = video_app(ff, |m| m.max_request_video_bytes = 29 << 20).await;
            let (status, v) = post_chat(small, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            assert!(
                error_message(&v).contains("videos are over the 30408704-byte limit"),
                "{v}"
            );
        }

        async fn post_responses(app: Router, body: Value) -> (StatusCode, Value) {
            let r = app
                .oneshot(
                    Request::post("/v1/responses")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = r.status();
            let body = to_bytes(r.into_body(), 1 << 20).await.unwrap();
            (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
        }

        /// `/v1/responses` takes `input_video` with the URL as a string (as
        /// OpenRouter documents it) or as `{url}`; `file_id` is refused.
        #[tokio::test]
        async fn responses_take_input_video() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 2);
            for video_url in [json!(url), json!({"url": url})] {
                let (app, seen) = video_app(ff.clone(), |_| {}).await;
                let (status, v) = post_responses(
                    app,
                    json!({"model": "mock-model", "max_output_tokens": 1, "input": [
                        {"role": "user", "content": [
                            {"type": "input_text", "text": "describe"},
                            {"type": "input_video", "video_url": video_url}]}]}),
                )
                .await;
                assert_eq!(status, StatusCode::OK, "{v}");
                let seen = seen.lock().unwrap();
                assert_eq!(seen[0].1.len(), 2);
                assert!(seen[0].0.contains("Frame 2 (1.0s): <ov_genai_image_1>"));
            }
            let (app, _) = video_app(ff, |_| {}).await;
            let (status, v) = post_responses(
                app,
                json!({"model": "mock-model", "input": [{"role": "user", "content": [
                    {"type": "input_video", "file_id": "file-1"}]}]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(v["error"]["param"], "input[0].content[0].file_id");
            assert_eq!(v["error"]["code"], "video_file_id_not_supported");
        }
    }
}
