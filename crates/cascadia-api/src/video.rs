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
//! Matroska/WebM), under one deadline, with its output capped and the child
//! killed when the request goes away (`kill_on_drop`).

use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};

use cascadia_types::MediaInput;
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

/// What the probe found: the duration and the frame size as displayed (a
/// rotated phone video's sides swapped, as ffmpeg's autorotate outputs it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub duration: f64,
    pub width: u32,
    pub height: u32,
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

/// Read ffprobe's JSON and refuse, before any frame is decoded, a video
/// this server will not decode: no video stream, a container outside
/// [`CONTAINER_NAMES`], no known duration or one past
/// [`MediaConfig::max_video_seconds`], or frames past the image size caps.
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
    let duration = secs(format.duration)
        .or_else(|| secs(stream.duration))
        .ok_or("the video's duration is unknown")?;
    if duration > cfg.max_video_seconds {
        return Err(format!(
            "video is {duration:.1} s long; the limit is {} s",
            cfg.max_video_seconds
        ));
    }
    let (mut w, mut h) = (stream.width.unwrap_or(0), stream.height.unwrap_or(0));
    if w == 0 || h == 0 {
        return Err("the video's frames have no pixels".into());
    }
    if w > MAX_SIDE || h > MAX_SIDE || u64::from(w) * u64::from(h) > cfg.max_source_pixels {
        return Err(format!(
            "video is {w}x{h}; the limit is {MAX_SIDE} pixels a side and {} pixels a frame",
            cfg.max_source_pixels
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
    Ok(Probe {
        duration,
        width: w,
        height: h,
    })
}

/// How a probed video is sampled: `frames` frames, uniform over its
/// duration from 0, each `width x height` RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub frames: usize,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
}

impl Plan {
    /// `ceil(duration * video_fps)` frames, at least 1 and at most
    /// [`MediaConfig::max_video_frames`], each fit to the `detail` budget
    /// keeping the aspect ratio, with even sides.
    pub fn new(probe: &Probe, detail: Option<&str>, cfg: &MediaConfig) -> Self {
        let wanted = (probe.duration * cfg.video_fps).ceil();
        let frames = if wanted.is_finite() && wanted >= 1.0 {
            (wanted as usize).min(cfg.max_video_frames.max(1))
        } else {
            1
        };
        let budget = match detail {
            Some("low") => cfg.low_detail_pixels,
            Some("high") => cfg.max_image_pixels,
            _ => cfg.max_video_frame_pixels,
        };
        let (w, h) = (probe.width, probe.height);
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
            duration: probe.duration,
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

/// The text a frame's image placeholder follows in the prompt; `k` counts
/// from 1.
pub fn frame_label(k: usize, secs: f64) -> String {
    format!("Frame {k} ({secs:.1}s): ")
}

/// ffprobe's arguments: the container forced, local files only, and only the
/// fields [`parse_probe`] reads (no tags, so a caller cannot bloat the
/// output). `V` skips cover art.
pub fn probe_args(container: Container, file: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
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
        "format=format_name,duration:stream=codec_type,width,height,duration\
         :stream_side_data=rotation",
    ]
    .map(OsString::from)
    .to_vec();
    args.push(file.into());
    args
}

/// ffmpeg's arguments: one run that samples `plan.frames` frames, scales
/// them and writes raw RGB to stdout. No audio, subtitle or data streams.
pub fn decode_args(container: Container, file: &Path, plan: &Plan) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-threads",
        "2",
        "-f",
        container.demuxer(),
        "-i",
    ]
    .map(OsString::from)
    .to_vec();
    args.push(file.into());
    let filter = format!("fps={:.9},scale={}:{}", plan.fps(), plan.width, plan.height);
    let frames = plan.frames.to_string();
    args.extend(
        [
            "-map",
            "0:V:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &filter,
            "-frames:v",
            &frames,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        .map(OsString::from),
    );
    args
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
/// ffmpeg's stderr kept for the log.
const STDERR_KEPT: usize = 2048;

/// Probe, bound and decode one encoded video into its sampled frames. The
/// probe and the decode share one deadline
/// ([`MediaConfig::video_decode_timeout`]).
pub async fn decode_video(
    bytes: Vec<u8>,
    detail: Option<&str>,
    cfg: &MediaConfig,
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
    let deadline = Instant::now() + cfg.video_decode_timeout;
    let timed_out = || {
        VideoError::Invalid(format!(
            "decoding the video took longer than {} s",
            cfg.video_decode_timeout.as_secs_f32()
        ))
    };
    // mp4 often needs seeking (its index may sit at the end), so ffmpeg reads
    // a file, not a pipe. Private to this user; removed on drop.
    let file = tokio::task::spawn_blocking(move || {
        let mut f = tempfile::Builder::new()
            .prefix("cascadia-video-")
            .tempfile()?;
        f.write_all(&bytes)?;
        f.flush()?;
        Ok::<_, std::io::Error>(f)
    })
    .await
    .map_err(|e| VideoError::Internal(format!("video temp file task failed: {e}")))?
    .map_err(|e| VideoError::Internal(format!("could not write the video temp file: {e}")))?;

    let mut probe = tokio::process::Command::new(&tools.ffprobe);
    probe.args(probe_args(container, file.path()));
    let out = match run_bounded(probe, MAX_PROBE_OUTPUT, deadline).await {
        Ok(o) => o,
        Err(RunError::TimedOut) => return Err(timed_out()),
        Err(e) => return Err(e.into_video_error("ffprobe")),
    };
    if !out.status.success() {
        tracing::warn!(status = %out.status, stderr = %out.stderr, "ffprobe refused a video");
        return Err(VideoError::Invalid(format!(
            "could not read the video; {SUPPORTED}"
        )));
    }
    let probed = parse_probe(&out.stdout, cfg).map_err(VideoError::Invalid)?;
    let plan = Plan::new(&probed, detail, cfg);

    let mut decode = tokio::process::Command::new(&tools.ffmpeg);
    decode.args(decode_args(container, file.path(), &plan));
    let out = match run_bounded(decode, plan.frames * plan.frame_bytes(), deadline).await {
        Ok(o) => o,
        Err(RunError::TimedOut) => return Err(timed_out()),
        Err(e) => return Err(e.into_video_error("ffmpeg")),
    };
    drop(file);
    if !out.status.success() {
        tracing::warn!(status = %out.status, stderr = %out.stderr, "ffmpeg failed to decode a video");
        return Err(VideoError::Invalid("could not decode the video".into()));
    }
    // Hashing a few MB of pixels is blocking-pool work, as an image decode is.
    let frames = tokio::task::spawn_blocking(move || {
        out.stdout
            .chunks_exact(plan.frame_bytes())
            .enumerate()
            .map(|(k, raw)| Frame {
                input: media::rgb_image(plan.width, plan.height, raw.to_vec()),
                secs: plan.timestamp(k),
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
    fn into_video_error(self, tool: &str) -> VideoError {
        match self {
            RunError::TimedOut => unreachable!("handled by the caller"),
            RunError::TooMuchOutput => {
                tracing::warn!(tool, "video decoder wrote more than its output cap; killed");
                VideoError::Invalid("could not decode the video".into())
            }
            RunError::Io(e) => {
                tracing::error!(tool, error = %e, "could not run the video decoder");
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
            "streams": [{"codec_type": "video", "width": w, "height": h}],
            "format": {"format_name": format_name},
        });
        if let Some(d) = duration {
            v["format"]["duration"] = d.into();
        }
        v.to_string().into_bytes()
    }

    fn probe(duration: f64, width: u32, height: u32) -> Probe {
        Probe {
            duration,
            width,
            height,
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
        assert_eq!(frame_label(2, 1.04), "Frame 2 (1.0s): ");
    }

    /// Frames fit the detail's pixel budget, keep the aspect ratio and have
    /// even sides.
    #[test]
    fn frame_size_follows_detail_with_even_sides() {
        let mut cfg = MediaConfig::ov_genai(None);
        let hd = probe(1.0, 1920, 1080);
        for (detail, budget) in [
            (None, cfg.max_video_frame_pixels),
            (Some("auto"), cfg.max_video_frame_pixels),
            (Some("low"), cfg.low_detail_pixels),
            (Some("high"), cfg.max_image_pixels),
        ] {
            let p = Plan::new(&hd, detail, &cfg);
            assert!(u64::from(p.width) * u64::from(p.height) <= budget, "{p:?}");
            assert_eq!((p.width % 2, p.height % 2), (0, 0), "{p:?}");
            let ratio = f64::from(p.width) / f64::from(p.height);
            assert!((ratio - 16.0 / 9.0).abs() < 0.02, "{detail:?}: {p:?}");
        }
        let p = Plan::new(&hd, None, &cfg);
        assert_eq!((p.width, p.height), (596, 336));
        // Under the budget the size is kept, rounded down to even.
        let p = Plan::new(&probe(1.0, 321, 241), None, &cfg);
        assert_eq!((p.width, p.height), (320, 240));
        cfg.max_video_frame_pixels = 4;
        let p = Plan::new(&probe(1.0, 3, 3), None, &cfg);
        assert_eq!((p.width, p.height), (2, 2));
    }

    #[test]
    fn probe_json_is_read_and_bounded() {
        let cfg = MediaConfig::ov_genai(None);
        let mp4 = "mov,mp4,m4a,3gp,3g2,mj2";
        assert_eq!(
            parse_probe(&probe_json(mp4, Some("3.000000"), 320, 240), &cfg).unwrap(),
            probe(3.0, 320, 240)
        );
        assert_eq!(
            parse_probe(&probe_json("matroska,webm", Some("2.5"), 64, 48), &cfg).unwrap(),
            probe(2.5, 64, 48)
        );
        // A quarter turn swaps the sides, as ffmpeg's autorotate does.
        let rotated = serde_json::json!({
            "streams": [{"codec_type": "video", "width": 320, "height": 240,
                "duration": "3.000000", "side_data_list": [{"rotation": -90}]}],
            "format": {"format_name": mp4},
        });
        assert_eq!(
            parse_probe(rotated.to_string().as_bytes(), &cfg).unwrap(),
            probe(3.0, 240, 320),
            "the stream's duration stands in for the container's"
        );
        let mut short = MediaConfig::ov_genai(None);
        short.max_video_seconds = 2.0;
        let none = br#"{"streams": [], "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1"}}"#;
        for (json, cfg, needle) in [
            (none.to_vec(), &cfg, "no video stream"),
            (
                probe_json("hls", Some("3"), 320, 240),
                &cfg,
                "container `hls`",
            ),
            (
                probe_json("concat", Some("3"), 320, 240),
                &cfg,
                "container `concat`",
            ),
            (
                probe_json("image2", Some("3"), 320, 240),
                &cfg,
                "not supported",
            ),
            (
                probe_json("mov,mp4,lavfi", Some("3"), 320, 240),
                &cfg,
                "not supported",
            ),
            (probe_json(mp4, None, 320, 240), &cfg, "duration is unknown"),
            (
                probe_json(mp4, Some("0.000000"), 320, 240),
                &cfg,
                "duration is unknown",
            ),
            (
                probe_json(mp4, Some("N/A"), 320, 240),
                &cfg,
                "duration is unknown",
            ),
            (
                probe_json(mp4, Some("3"), 320, 240),
                &short,
                "3.0 s long; the limit is 2 s",
            ),
            (
                probe_json(mp4, Some("3"), 10_000, 10_000),
                &cfg,
                "pixels a frame",
            ),
            (probe_json(mp4, Some("3"), 0, 240), &cfg, "no pixels"),
            (probe_json(mp4, Some("3"), 4000, 10), &cfg, "aspect ratio"),
            (b"not json".to_vec(), &cfg, "could not read the video"),
        ] {
            let err = parse_probe(&json, cfg).unwrap_err();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn ffmpeg_reads_only_the_local_file_with_the_container_forced() {
        let file = Path::new("/tmp/v.bin");
        let args: Vec<String> = probe_args(Container::Mov, file)
            .into_iter()
            .map(|a| a.into_string().unwrap())
            .collect();
        let has = |args: &[String], pair: [&str; 2]| args.windows(2).any(|w| w == pair);
        assert!(has(&args, ["-protocol_whitelist", "file"]), "{args:?}");
        assert!(has(&args, ["-f", "mov"]), "{args:?}");
        assert_eq!(args.last().unwrap(), "/tmp/v.bin");
        let plan = Plan {
            frames: 3,
            width: 320,
            height: 240,
            duration: 3.0,
        };
        let args: Vec<String> = decode_args(Container::Matroska, file, &plan)
            .into_iter()
            .map(|a| a.into_string().unwrap())
            .collect();
        for pair in [
            ["-protocol_whitelist", "file"],
            ["-f", "matroska"],
            ["-i", "/tmp/v.bin"],
            ["-vf", "fps=1.000000000,scale=320:240"],
            ["-frames:v", "3"],
            ["-pix_fmt", "rgb24"],
            ["-threads", "2"],
        ] {
            assert!(has(&args, pair), "{pair:?} in {args:?}");
        }
        // Input options precede the input.
        let at = |a: &str| args.iter().position(|x| x == a).unwrap();
        assert!(at("-protocol_whitelist") < at("-i"));
        assert!(args.contains(&"-nostdin".to_string()));
        assert_eq!(args.last().unwrap(), "pipe:1");
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

    /// The children of this process whose command is `name`, zombies
    /// included: a killed child that was never waited for still shows.
    fn children_named(name: &str) -> Vec<String> {
        let out = std::process::Command::new("ps")
            .args(["-A", "-o", "ppid=,pid=,stat=,comm="])
            .output()
            .unwrap();
        let me = std::process::id().to_string();
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                f.first() == Some(&me.as_str()) && f.last().is_some_and(|c| c.ends_with(name))
            })
            .map(str::to_owned)
            .collect()
    }

    /// Past the deadline, or past the output cap, the child is killed and
    /// reaped: no process (or zombie) is left behind.
    #[tokio::test]
    async fn bounded_runs_kill_and_reap_their_child() {
        let mut sleep = tokio::process::Command::new("sleep");
        sleep.arg("30");
        let t0 = std::time::Instant::now();
        let r = run_bounded(sleep, 1024, Instant::now() + Duration::from_millis(200)).await;
        assert!(matches!(r, Err(RunError::TimedOut)), "{r:?}");
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(
            children_named("sleep").is_empty(),
            "{:?}",
            children_named("sleep")
        );

        let yes = tokio::process::Command::new("yes");
        let r = run_bounded(yes, 1 << 20, Instant::now() + Duration::from_secs(10)).await;
        assert!(matches!(r, Err(RunError::TooMuchOutput)), "{r:?}");
        assert!(
            children_named("yes").is_empty(),
            "{:?}",
            children_named("yes")
        );

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

        /// A decode past its deadline is refused like a slow fetch (a 400),
        /// and the next request is served. That the child is killed and
        /// reaped is `bounded_runs_kill_and_reap_their_child`.
        #[tokio::test]
        async fn a_decode_past_its_deadline_is_killed() {
            let Some(ff) = ffmpeg() else { return };
            let url = clip(&ff, "mp4", "video/mp4", 2);
            let (app, _) = video_app(ff.clone(), |m| {
                m.video_decode_timeout = Duration::from_millis(1)
            })
            .await;
            let (status, v) = post_chat(
                app,
                json!({"model": "mock-model", "messages": [video_msg("x", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
            assert!(
                error_message(&v).contains("decoding the video took longer than"),
                "{v}"
            );
            let (app, _) = video_app(ff, |_| {}).await;
            let (status, _) = post_chat(
                app,
                json!({"model": "mock-model", "max_tokens": 1,
                    "messages": [video_msg("x", &url)]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
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
            let (status, _) = post_chat(without, body.clone()).await;
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
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
