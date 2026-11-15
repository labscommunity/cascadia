//! physicalai's remote-inference protocol, version 1: the messages Intel's
//! `physicalai.inference.RemoteInferenceModel` exchanges with its
//! `InferenceServer`, so that client can drive a Cascadia policy unchanged.
//!
//! Every message is one msgpack map with string keys. Numeric arrays are the
//! `{"__np__": true, "dtype", "shape", "data"}` maps of `physicalai.transport`;
//! images travel in a separate `images` map as JPEG or raw RGB. The checks
//! follow `physicalai/inference/remote/_protocol.py` in order, then the
//! inputs are mapped onto an [`Observation`] and checked like a WebSocket
//! request. The transport (Zenoh) lives in `zenoh_server`.

use cascadia_engine::{check_observation, EngineError};
use cascadia_types::{
    ActionChunk, Image, NoiseMode, Observation, PolicySpec, ResizeConvention, ServerResize,
};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use super::msgpack::{self, Val, Writer};
use super::RequestError;

pub const PROTOCOL_VERSION: u64 = 1;

/// Largest `budget_ms` a predict request may carry.
pub const MAX_REQUEST_BUDGET_MS: u64 = 60_000;

pub const IMAGE_CODECS: [&str; 2] = ["jpeg", "raw"];

/// physicalai's default `max_request_bytes`: room for frames at a camera's
/// native resolution, which its client sends unless told to downscale.
pub const MAX_REQUEST_BYTES: u64 = 32 * 1024 * 1024;

/// Largest frame side accepted before resizing.
pub const MAX_IMAGE_SIDE: u64 = 4096;

/// The `seq` an error carries when the request had none.
pub const NO_SEQ: i128 = -1;

/// The `code` of an error reply; the client maps each to an exception.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    InvalidRequest,
    UnsupportedVersion,
    ShuttingDown,
    Superseded,
    Expired,
    ModelError,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::UnsupportedVersion => "unsupported_version",
            Self::ShuttingDown => "shutting_down",
            Self::Superseded => "superseded",
            Self::Expired => "expired",
            Self::ModelError => "model_error",
        }
    }
}

/// What the handshake advertises. The client compares `name` with the name it
/// was given and, when configured, `policy_name` and `manifest_sha256` with
/// the values it expects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerIdentity {
    /// A fresh id per server start; the client treats a change as a restart.
    pub server_id: String,
    /// The deployment-slot name, the last segment of the key prefix.
    pub name: String,
    /// `policy.name` from the package manifest.
    pub policy_name: String,
    /// SHA-256 (lowercase hex) of the package's `manifest.json` bytes.
    pub manifest_sha256: String,
    pub backend: String,
    pub device: String,
}

/// A refused metadata or reset request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlError {
    UnsupportedVersion,
    Invalid(RequestError),
}

/// A decoded predict request, ready to queue.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictRequest {
    pub seq: u64,
    pub budget_ms: u64,
    pub observation: Observation,
}

fn refuse<T>(msg: impl Into<String>) -> Result<T, RequestError> {
    Err(RequestError::new(msg))
}

/// Whether `name` is a valid slot name: physicalai's `model_key_prefix` rule.
pub fn check_name(name: &str) -> Result<(), RequestError> {
    let mut chars = name.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_alphanumeric());
    if first_ok && chars.all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c)) {
        Ok(())
    } else {
        refuse("model_name must be a non-empty key segment containing only letters, digits, '.', '_' or '-'")
    }
}

/// `physicalai/inference/<name>`, under which the three keys live.
pub fn key_prefix(name: &str) -> String {
    format!("physicalai/inference/{name}")
}

// ---- decoding ----

type Map<'a> = Vec<(&'a str, Val<'a>)>;

/// A string-keyed map, or `None`.
fn str_map(v: Val<'_>) -> Option<Map<'_>> {
    let Val::Map(entries) = v else { return None };
    entries
        .into_iter()
        .map(|(k, v)| match k {
            Val::Str(s) => Some((s, v)),
            _ => None,
        })
        .collect()
}

/// A tagged numpy array, validated as `physicalai.transport` does.
struct NpArray<'a> {
    dtype: &'a str,
    shape: Vec<u64>,
    data: &'a [u8],
}

/// `(kind, itemsize)` for the numpy dtype strings a client sends: the names
/// `str(np.dtype)` prints and the `<f4` style.
fn np_dtype(s: &str) -> Option<(char, u64)> {
    let named = match s {
        "bool" => Some(('b', 1)),
        "int8" => Some(('i', 1)),
        "int16" => Some(('i', 2)),
        "int32" => Some(('i', 4)),
        "int64" => Some(('i', 8)),
        "uint8" => Some(('u', 1)),
        "uint16" => Some(('u', 2)),
        "uint32" => Some(('u', 4)),
        "uint64" => Some(('u', 8)),
        "float16" => Some(('f', 2)),
        "float32" => Some(('f', 4)),
        "float64" => Some(('f', 8)),
        "complex64" => Some(('c', 8)),
        "complex128" => Some(('c', 16)),
        _ => None,
    };
    if named.is_some() {
        return named;
    }
    let s = s.strip_prefix(['<', '>', '=', '|']).unwrap_or(s);
    let mut chars = s.chars();
    let kind = chars.next()?;
    let size: u64 = chars.as_str().parse().ok()?;
    let ok = match kind {
        'b' => size == 1,
        'i' | 'u' => [1, 2, 4, 8].contains(&size),
        'f' => [2, 4, 8].contains(&size),
        'c' => [8, 16].contains(&size),
        _ => false,
    };
    ok.then_some((kind, size))
}

/// Is `v` a tagged array? `Ok(None)` for any other map.
fn np_array<'a>(v: &Val<'a>, budget: &mut u64) -> Result<Option<NpArray<'a>>, RequestError> {
    let Val::Map(entries) = v else {
        return Ok(None);
    };
    let get = |key: &str| {
        entries
            .iter()
            .rev()
            .find(|(k, _)| *k == Val::Str(key))
            .map(|(_, v)| v)
    };
    if get("__np__") != Some(&Val::Bool(true)) {
        return Ok(None);
    }
    let (Some(Val::Str(dtype)), Some(Val::Array(dims)), Some(Val::Bin(data))) =
        (get("dtype"), get("shape"), get("data"))
    else {
        return refuse("Malformed __np__ array payload");
    };
    let Some((_, itemsize)) = np_dtype(dtype) else {
        return refuse("Invalid __np__ array dtype");
    };
    let Some(shape) = dims.iter().map(Val::as_u64).collect::<Option<Vec<_>>>() else {
        return refuse("Invalid __np__ array shape");
    };
    let size = shape.iter().try_fold(itemsize, |a, &d| a.checked_mul(d));
    if size != Some(data.len() as u64) {
        return refuse("__np__ array shape and dtype do not match its data length");
    }
    let size = data.len() as u64;
    if size > *budget {
        return refuse("Decoded NumPy arrays exceed the byte limit");
    }
    *budget -= size;
    Ok(Some(NpArray { dtype, shape, data }))
}

/// Validate every tagged array under `v`, as `unpack_msgpack` does.
fn walk_arrays(v: &Val<'_>, budget: &mut u64) -> Result<(), RequestError> {
    if np_array(v, budget)?.is_some() {
        return Ok(());
    }
    match v {
        Val::Map(entries) => entries.iter().try_for_each(|(_, x)| walk_arrays(x, budget)),
        Val::Array(items) => items.iter().try_for_each(|x| walk_arrays(x, budget)),
        _ => Ok(()),
    }
}

/// `decode_message`: one string-keyed map, at most `max_bytes`, whose tagged
/// arrays are well formed and fit `max_bytes` together.
fn decode_message(bytes: &[u8], max_bytes: u64) -> Result<Map<'_>, RequestError> {
    if bytes.len() as u64 > max_bytes {
        return refuse(format!(
            "Invalid MessagePack protocol map: payload of {} bytes exceeds the {max_bytes}-byte limit",
            bytes.len()
        ));
    }
    let v = msgpack::parse(bytes)
        .map_err(|e| RequestError::new(format!("Invalid MessagePack protocol map: {}", e.0)))?;
    let mut budget = max_bytes;
    walk_arrays(&v, &mut budget)
        .map_err(|e| RequestError::new(format!("Invalid MessagePack protocol map: {}", e.0)))?;
    let Some(map) = str_map(v) else {
        return refuse("Protocol message must be a string-keyed MessagePack map");
    };
    for (i, (k, _)) in map.iter().enumerate() {
        if map[..i].iter().any(|(seen, _)| seen == k) {
            return refuse(format!("key {k:?} sent twice"));
        }
    }
    Ok(map)
}

fn get<'m, 'a>(map: &'m Map<'a>, key: &str) -> Option<&'m Val<'a>> {
    map.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
}

fn is_version(v: Option<&Val<'_>>) -> bool {
    v.and_then(Val::as_u64) == Some(PROTOCOL_VERSION)
}

/// A metadata or reset request: exactly `{"protocol_version": 1}`.
pub fn decode_control(bytes: &[u8], max_bytes: u64) -> Result<(), ControlError> {
    let map = decode_message(bytes, max_bytes).map_err(ControlError::Invalid)?;
    if !is_version(get(&map, "protocol_version")) {
        return Err(ControlError::UnsupportedVersion);
    }
    if map.len() != 1 {
        return Err(ControlError::Invalid(RequestError::new(
            "Request must contain only protocol_version",
        )));
    }
    Ok(())
}

/// The `seq` of a refused request, for its error reply: the top-level `seq`
/// when it is an integer, else -1.
pub fn read_seq(bytes: &[u8]) -> i128 {
    let Ok(Val::Map(entries)) = msgpack::parse(bytes) else {
        return NO_SEQ;
    };
    match entries.iter().rev().find(|(k, _)| *k == Val::Str("seq")) {
        Some((_, Val::UInt(u))) => i128::from(*u),
        Some((_, Val::Int(i))) => i128::from(*i),
        _ => NO_SEQ,
    }
}

fn is_image_key(name: &str) -> bool {
    name == "images" || name.starts_with("images.")
}

/// `_jpeg_dimensions`: `(height, width)` from the first start-of-frame marker.
fn jpeg_dimensions(data: &[u8]) -> Result<(u64, u64), RequestError> {
    if data.len() < 4 || data[..2] != [0xff, 0xd8] {
        return refuse("Malformed JPEG image data");
    }
    const SOF: [u8; 13] = [
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    ];
    let mut off = 2;
    while off + 4 <= data.len() {
        if data[off] != 0xff {
            return refuse("Malformed JPEG marker");
        }
        while off < data.len() && data[off] == 0xff {
            off += 1;
        }
        if off >= data.len() {
            break;
        }
        let marker = data[off];
        off += 1;
        if matches!(marker, 0xD8 | 0xD9 | 0x01 | 0xD0..=0xD7) {
            continue;
        }
        if off + 2 > data.len() {
            break;
        }
        let len = u16::from_be_bytes([data[off], data[off + 1]]) as usize;
        if len < 2 || off + len > data.len() {
            return refuse("Malformed JPEG segment");
        }
        if SOF.contains(&marker) {
            if len < 7 {
                return refuse("Malformed JPEG frame header");
            }
            let h = u16::from_be_bytes([data[off + 3], data[off + 4]]);
            let w = u16::from_be_bytes([data[off + 5], data[off + 6]]);
            if h == 0 || w == 0 {
                return refuse("Invalid JPEG dimensions");
            }
            return Ok((h.into(), w.into()));
        }
        off += len;
    }
    refuse("JPEG image has no dimensions")
}

/// RGB pixels of one JPEG frame, `height * width * 3` bytes.
fn decode_jpeg(data: &[u8], height: u64, width: u64) -> Result<Vec<u8>, RequestError> {
    if jpeg_dimensions(data)? != (height, width) {
        return refuse("JPEG dimensions do not match the declared shape");
    }
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(width as usize)
        .set_max_height(height as usize);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(data), opts);
    let pixels = dec
        .decode()
        .map_err(|e| RequestError::new(format!("Failed to decode JPEG image: {e:?}")))?;
    let info = dec.info();
    let dims = info.map(|i| (u64::from(i.height), u64::from(i.width)));
    if dims != Some((height, width)) || Some(pixels.len() as u64) != height.checked_mul(width * 3) {
        return refuse("Decoded JPEG image does not match its declared shape");
    }
    Ok(pixels)
}

/// `decode_images`: each entry's codec, dtype, shape and bytes, then pixels.
fn decode_images(v: Val<'_>, budget: &mut u64) -> Result<Vec<(String, Image)>, RequestError> {
    let Some(entries) = str_map(v) else {
        return refuse("Request images must be a string-keyed map");
    };
    let mut out = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        if !is_image_key(name) {
            return refuse(format!(
                "Image key {name:?} must use the 'images' namespace"
            ));
        }
        let Some(entry) = str_map(entry) else {
            return refuse("Image entry must be a map");
        };
        let field = |k: &str| entry.iter().rev().find(|(n, _)| *n == k).map(|(_, v)| v);
        let (Some(Val::Str(codec)), Some(Val::Str("uint8")), Some(Val::Bin(data))) =
            (field("codec"), field("dtype"), field("data"))
        else {
            return refuse("Malformed image codec, dtype, or data");
        };
        if !IMAGE_CODECS.contains(codec) {
            return refuse("Malformed image codec, dtype, or data");
        }
        let shape = match field("shape") {
            Some(Val::Array(dims)) => dims
                .iter()
                .map(|d| d.as_u64().filter(|&d| d >= 1))
                .collect::<Option<Vec<_>>>(),
            _ => None,
        };
        let Some(shape) = shape else {
            return refuse("Image shape dimensions must be positive integers");
        };
        if !matches!(shape.len(), 3 | 4) || shape[shape.len() - 1] != 3 {
            return refuse("Images must declare HxWx3 or BxHxWx3 shapes");
        }
        if shape.len() == 4 && shape[0] != 1 {
            return refuse("Remote inference image batch dimension must be one");
        }
        let size = shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d));
        let Some(size) = size.filter(|&s| s <= *budget) else {
            return refuse("Decoded image arrays exceed max_request_bytes");
        };
        let (h, w) = (shape[shape.len() - 3], shape[shape.len() - 2]);
        if h > MAX_IMAGE_SIDE || w > MAX_IMAGE_SIDE {
            return refuse(format!(
                "Image is {h}x{w}; at most {MAX_IMAGE_SIDE} pixels a side"
            ));
        }
        let pixels = if *codec == "raw" {
            if data.len() as u64 != size {
                return refuse("Raw image data length does not match its shape");
            }
            data.to_vec()
        } else {
            decode_jpeg(data, h, w)?
        };
        *budget -= size;
        let (Ok(height), Ok(width)) = (u32::try_from(h), u32::try_from(w)) else {
            return refuse("Images must declare HxWx3 or BxHxWx3 shapes");
        };
        out.push((
            name.to_string(),
            Image {
                height,
                width,
                data: pixels,
            },
        ));
    }
    Ok(out)
}

fn floats(what: &str, a: &NpArray<'_>) -> Result<Vec<f32>, RequestError> {
    match a.dtype {
        "float32" | "<f4" => Ok(a
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "float64" | "<f8" => Ok(a
            .data
            .chunks_exact(8)
            .map(|c| {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                f64::from_le_bytes(b) as f32
            })
            .collect()),
        other => refuse(format!(
            "{what} has dtype {other}; expected float32 or float64"
        )),
    }
}

const INPUT_KEYS: &str = "state, task, images and optional noise";

/// Map physicalai's inputs onto an [`Observation`] for `spec`.
///
/// `state` is `[D]` or `[1, D]`; `task` a one-string list (the prompt);
/// cameras `images.<name>`, or a bare `images` for a one-camera package;
/// `noise` `float32 [1, *noise_shape]`, input-mode packages only.
fn observation(
    inputs: Map<'_>,
    images: Vec<(String, Image)>,
    spec: &PolicySpec,
) -> Result<Observation, RequestError> {
    let (mut state, mut prompt, mut noise) = (None, None, None);
    let mut budget = u64::MAX;
    for (name, v) in inputs {
        let arr = np_array(&v, &mut budget)?;
        match (name, arr) {
            ("state", Some(a)) => {
                let d = spec.state_len as u64;
                if a.shape != [d] && a.shape != [1, d] {
                    return refuse(format!(
                        "state has shape {:?}; expected [{d}] or [1, {d}]",
                        a.shape
                    ));
                }
                state = Some(floats("state", &a)?);
            }
            ("task", None) => match v {
                Val::Array(items) => match items.as_slice() {
                    [Val::Str(s)] => prompt = Some(s.to_string()),
                    _ => return refuse("task must be a list of exactly one string"),
                },
                _ => return refuse("task must be a list of exactly one string"),
            },
            ("noise", Some(a)) => {
                if spec.noise != NoiseMode::Input {
                    return refuse("this package takes no noise input");
                }
                if !matches!(a.dtype, "float32" | "<f4") {
                    return refuse(format!("noise has dtype {}; expected float32", a.dtype));
                }
                let mut want = vec![1u64];
                want.extend(spec.noise_shape.iter().flatten().map(|&d| d as u64));
                if a.shape != want {
                    return refuse(format!("noise has shape {:?}; expected {want:?}", a.shape));
                }
                noise = Some(floats("noise", &a)?);
            }
            ("state" | "noise", None) => return refuse(format!("{name} must be a numeric array")),
            (other, _) => {
                return refuse(format!(
                    "unknown input {other:?}; a request has {INPUT_KEYS}"
                ))
            }
        }
    }
    let cameras = images
        .into_iter()
        .map(|(key, img)| {
            let cam = match key.strip_prefix("images.") {
                Some(cam) => cam.to_string(),
                None if spec.cameras.len() == 1 => spec.cameras[0].clone(),
                None => {
                    return refuse(format!(
                    "a bare \"images\" key needs a one-camera package; send images.<name> for {:?}",
                    spec.cameras
                ))
                }
            };
            let img = if (img.height, img.width) == (spec.height, spec.width) {
                img
            } else {
                server_resize(&cam, &img, spec)?
            };
            Ok((cam, img))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some(state) = state else {
        return refuse(format!(
            "missing input \"state\"; a request has {INPUT_KEYS}"
        ));
    };
    let Some(prompt) = prompt else {
        return refuse(format!(
            "missing input \"task\"; a request has {INPUT_KEYS}"
        ));
    };
    let obs = Observation {
        images: cameras,
        state,
        prompt,
        seed: None,
        noise,
    };
    check_observation(spec, &obs).map_err(|e| match e {
        EngineError::InvalidConfig(m) => RequestError(m),
        other => RequestError(other.to_string()),
    })?;
    Ok(obs)
}

/// `decode_predict_request`, then the mapping onto `spec`. Every refusal is
/// an `invalid_request` reply.
pub fn decode_predict(
    bytes: &[u8],
    spec: &PolicySpec,
    max_bytes: u64,
) -> Result<PredictRequest, RequestError> {
    let map = decode_message(bytes, max_bytes)?;
    const FIELDS: [&str; 5] = ["protocol_version", "seq", "budget_ms", "inputs", "images"];
    if map.len() != FIELDS.len() || FIELDS.iter().any(|f| get(&map, f).is_none()) {
        return refuse("Predict request has unexpected or missing fields");
    }
    let (mut seq, mut budget_ms, mut inputs, mut images) = (None, None, None, None);
    for (k, v) in map {
        match k {
            "protocol_version" if !is_version(Some(&v)) => {
                return refuse("Unsupported predict protocol version")
            }
            "seq" => seq = Some(v),
            "budget_ms" => budget_ms = Some(v),
            "inputs" => inputs = Some(v),
            "images" => images = Some(v),
            _ => {}
        }
    }
    let Some(seq) = seq.as_ref().and_then(Val::as_u64) else {
        return refuse("Invalid prediction sequence");
    };
    let budget_ms = match budget_ms.as_ref().and_then(Val::as_u64) {
        Some(b) if b > MAX_REQUEST_BUDGET_MS => {
            return refuse("Prediction budget exceeds the server maximum")
        }
        Some(b) if b >= 1 => b,
        _ => return refuse("Invalid prediction budget"),
    };
    let Some(inputs) = inputs.and_then(str_map) else {
        return refuse("Predict inputs must be a string-keyed map");
    };
    let mut budget = max_bytes;
    for (name, v) in &inputs {
        if is_image_key(name) {
            return refuse("Image inputs must use the separate images map");
        }
        let mut unbounded = u64::MAX;
        match np_array(v, &mut unbounded)? {
            Some(a) => {
                let size = a.data.len() as u64;
                if size > budget {
                    return refuse("Decoded input arrays exceed max_request_bytes");
                }
                budget -= size;
            }
            None => {
                let strings = matches!(v, Val::Array(items)
                    if items.iter().all(|x| matches!(x, Val::Str(_))));
                if !strings {
                    return refuse("Predict inputs must contain numeric arrays or list[str]");
                }
            }
        }
    }
    let images = decode_images(images.unwrap_or(Val::Nil), &mut budget)?;
    if let Some((dup, _)) = images
        .iter()
        .find(|(k, _)| inputs.iter().any(|(n, _)| n == k))
    {
        return refuse(format!("Duplicate input and image keys: [{dup:?}]"));
    }
    let observation = observation(inputs, images, spec)?;
    Ok(PredictRequest {
        seq,
        budget_ms,
        observation,
    })
}

// ---- encoding ----

fn write_seq(w: &mut Writer, seq: i128) {
    if seq < 0 {
        w.int(seq.max(i128::from(i64::MIN)) as i64);
    } else {
        w.uint(seq.min(i128::from(u64::MAX)) as u64);
    }
}

/// The handshake reply. `chunk_size` is the chunk's T; RTC is not offered.
pub fn encode_metadata(id: &ServerIdentity, spec: &PolicySpec, max_request_bytes: u64) -> Vec<u8> {
    let mut w = Writer::default();
    w.map(8);
    w.str("protocol_version");
    w.uint(PROTOCOL_VERSION);
    w.str("server_id");
    w.str(&id.server_id);
    w.str("name");
    w.str(&id.name);
    w.str("policy");
    w.map(4);
    w.str("name");
    w.str(&id.policy_name);
    w.str("manifest_sha256");
    w.str(&id.manifest_sha256);
    w.str("backend");
    w.str(&id.backend);
    w.str("device");
    w.str(&id.device);
    w.str("chunk_size");
    w.uint(spec.chunk[0] as u64);
    w.str("rtc");
    w.map(2);
    w.str("supported");
    w.bool(false);
    w.str("chunk_size");
    w.nil();
    w.str("image_codecs");
    w.array(IMAGE_CODECS.len());
    IMAGE_CODECS.iter().for_each(|c| w.str(c));
    w.str("max_request_bytes");
    w.uint(max_request_bytes);
    w.buf
}

/// Server-side timings of one call, in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub queue_s: f64,
    pub compute_s: f64,
    pub decode_s: f64,
}

/// The predict reply: `outputs.action` as `float32 [1, T, D]`, the shape
/// physicalai's own models return.
///
/// The caller refuses non-finite actions first ([`super::check_reply`]).
pub fn encode_predict_reply(chunk: &ActionChunk, seq: u64, server_id: &str, t: Timings) -> Vec<u8> {
    let data: Vec<u8> = chunk.actions.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut w = Writer::default();
    w.map(7);
    w.str("protocol_version");
    w.uint(PROTOCOL_VERSION);
    w.str("seq");
    w.uint(seq);
    w.str("server_id");
    w.str(server_id);
    w.str("outputs");
    w.map(1);
    w.str("action");
    w.map(4);
    w.str("__np__");
    w.bool(true);
    w.str("dtype");
    w.str("float32");
    w.str("shape");
    w.array(3);
    for d in [1, chunk.steps, chunk.dim] {
        w.uint(d as u64);
    }
    w.str("data");
    w.bin(&data);
    w.str("server_queue_s");
    w.f64(t.queue_s);
    w.str("server_compute_s");
    w.f64(t.compute_s);
    w.str("server_decode_s");
    w.f64(t.decode_s);
    w.buf
}

/// An error reply (`query.reply_err`): `{protocol_version, code, seq}`.
pub fn encode_error(code: ErrorCode, request_seq: i128) -> Vec<u8> {
    let mut w = Writer::default();
    w.map(3);
    w.str("protocol_version");
    w.uint(PROTOCOL_VERSION);
    w.str("code");
    w.str(code.as_str());
    w.str("seq");
    write_seq(&mut w, request_seq);
    w.buf
}

/// The reset reply, `{protocol_version: 1, ok: true}`.
pub fn encode_reset_ok() -> Vec<u8> {
    let mut w = Writer::default();
    w.map(2);
    w.str("protocol_version");
    w.uint(PROTOCOL_VERSION);
    w.str("ok");
    w.bool(true);
    w.buf
}

// ---- resizing ----

/// Why a frame of another size is refused for a package whose own
/// preprocessor resize is not ported: guessing at it would feed the robot
/// frames its policy was not trained on.
fn not_resizable(cam: &str, img: &Image, spec: &PolicySpec) -> String {
    let (h, w) = (u64::from(spec.height), u64::from(spec.width));
    let same_aspect = u64::from(img.height) * w == u64::from(img.width) * h;
    let hint = if same_aspect {
        format!(", or set max_image_side={}", h.max(w))
    } else {
        String::new()
    };
    format!(
        "camera {cam:?} is {}x{}; this package's image preprocessing is not done on the \
         server, so send frames at {h}x{w}{hint}",
        img.height, img.width
    )
}

/// One axis of OpenCV's `INTER_LINEAR`: for each output index, the source
/// index and the weight of the next source sample.
fn linear_taps(src: usize, dst: usize) -> Vec<(usize, f32)> {
    let scale = 1.0 / (dst as f64 / src as f64);
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let mut s = f.floor() as i64;
            let mut f = f - s as f32;
            if s < 0 {
                (s, f) = (0, 0.0);
            }
            if s >= src as i64 - 1 {
                (s, f) = (src as i64 - 1, 0.0);
            }
            (s as usize, f)
        })
        .collect()
}

/// `cv2.resize(..., INTER_LINEAR)` of an RGB frame held as `f32`: rows
/// first, then columns, as OpenCV does. Values stay in the input's scale.
fn bilinear(src: &[f32], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<f32> {
    let xs = linear_taps(sw, dw);
    let ys = linear_taps(sh, dh);
    let row = |y: usize| -> Vec<f32> {
        let line = &src[y * sw * 3..(y + 1) * sw * 3];
        let mut out = Vec::with_capacity(dw * 3);
        for &(x, f) in &xs {
            let x1 = (x + 1).min(sw - 1);
            for c in 0..3 {
                let a = line[x * 3 + c];
                let b = if f == 0.0 { 0.0 } else { line[x1 * 3 + c] };
                out.push(a * (1.0 - f) + b * f);
            }
        }
        out
    };
    let mut out = Vec::with_capacity(dh * dw * 3);
    for &(y, f) in &ys {
        let r0 = row(y);
        if f == 0.0 {
            out.extend_from_slice(&r0);
        } else {
            let r1 = row((y + 1).min(sh - 1));
            out.extend(r0.iter().zip(&r1).map(|(a, b)| a * (1.0 - f) + b * f));
        }
    }
    out
}

/// The frame resized to `height x width` as physicalai's server preprocessor
/// for `conv` does it, in `f32` on `[0, 255]`: bilinear (`INTER_LINEAR`, no
/// antialiasing) to the aspect-kept size with pad 0 on the top and left
/// (`smolvla_resize`) or centred (`pi05`), or a plain bilinear stretch.
/// Exposed for tests; [`decode_predict`] rounds it to `u8`.
pub fn resize_frame_f32(img: &Image, height: u32, width: u32, conv: ResizeConvention) -> Vec<f32> {
    let (sh, sw) = (img.height as usize, img.width as usize);
    let (h, w) = (height as usize, width as usize);
    let src: Vec<f32> = img.data.iter().map(|&b| f32::from(b)).collect();
    if conv == ResizeConvention::Stretch {
        return bilinear(&src, sh, sw, h, w);
    }
    let ratio = f64::max(sw as f64 / w as f64, sh as f64 / h as f64);
    let rh = ((sh as f64 / ratio) as usize).clamp(1, h);
    let rw = ((sw as f64 / ratio) as usize).clamp(1, w);
    let inner = if (rh, rw) == (sh, sw) {
        src
    } else {
        bilinear(&src, sh, sw, rh, rw)
    };
    let (top, left) = match conv {
        ResizeConvention::TopLeft => (h - rh, w - rw),
        _ => ((h - rh) / 2, (w - rw) / 2),
    };
    let mut out = vec![0.0f32; h * w * 3];
    for y in 0..rh {
        let at = ((top + y) * w + left) * 3;
        out[at..at + rw * 3].copy_from_slice(&inner[y * rw * 3..(y + 1) * rw * 3]);
    }
    out
}

/// [`resize_frame_f32`] rounded to the `u8` frame the engine takes.
pub fn resize_frame(img: &Image, height: u32, width: u32, conv: ResizeConvention) -> Image {
    Image {
        height,
        width,
        data: resize_frame_f32(img, height, width, conv)
            .into_iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    }
}

/// A frame of another size through the package's own first resize, or the
/// reason it cannot be.
fn server_resize(cam: &str, img: &Image, spec: &PolicySpec) -> Result<Image, RequestError> {
    match spec.server_resize {
        Some(ServerResize::Letterbox) => {
            Ok(resize_frame(img, spec.height, spec.width, spec.resize))
        }
        Some(ServerResize::Area) => Ok(area_frame(img, spec.height, spec.width)),
        Some(ServerResize::SmartResizeCubic { factor, max_pixels }) => {
            match smart_resize(img.height, img.width, factor, max_pixels) {
                Some(hw) if hw == (spec.height, spec.width) => {
                    Ok(cubic_frame(img, spec.height, spec.width))
                }
                Some((th, tw)) => refuse(format!(
                    "camera {cam:?} is {}x{}, which this package's resize turns into {th}x{tw}; \
                     its export takes {}x{}, so send frames at {}x{}",
                    img.height, img.width, spec.height, spec.width, spec.height, spec.width
                )),
                None => refuse(format!(
                    "camera {cam:?} is {}x{}: an aspect ratio over 200",
                    img.height, img.width
                )),
            }
        }
        None => refuse(not_resizable(cam, img, spec)),
    }
}

// ---- VLA-JEPA: torch `interpolate(mode="area")` ----

/// physicalai's `vla_jepa._area_weights`: adaptive average pooling, each
/// output averaging `[i*in/out, ceil((i+1)*in/out))`, weights stored `f32`.
fn area_taps(size_in: usize, size_out: usize) -> Vec<(usize, Vec<f32>)> {
    (0..size_out)
        .map(|i| {
            let start = i * size_in / size_out;
            let end = ((i + 1) * size_in).div_ceil(size_out);
            let w = (1.0f64 / (end - start) as f64) as f32;
            (start, vec![w; end - start])
        })
        .collect()
}

/// `rows @ image @ cols.T` on `[0, 1]` `f32` values, each product summed in
/// `f64` and rounded to `f32` once, the row pass rounded before the column pass.
fn separable(
    src: &[f32],
    (sh, sw): (usize, usize),
    rows: &[(usize, Vec<f32>)],
    cols: &[(usize, Vec<f32>)],
) -> Vec<f32> {
    let dh = rows.len();
    let dw = cols.len();
    let mut tmp = vec![0f32; dh * sw * 3];
    for (i, (start, ws)) in rows.iter().enumerate() {
        for x in 0..sw * 3 {
            let acc: f64 = ws
                .iter()
                .enumerate()
                .map(|(k, &w)| f64::from(w) * f64::from(src[(start + k) * sw * 3 + x]))
                .sum();
            tmp[i * sw * 3 + x] = acc as f32;
        }
    }
    let mut out = vec![0f32; dh * dw * 3];
    for y in 0..dh {
        for (j, (start, ws)) in cols.iter().enumerate() {
            for c in 0..3 {
                let acc: f64 = ws
                    .iter()
                    .enumerate()
                    .map(|(k, &w)| f64::from(w) * f64::from(tmp[(y * sw + start + k) * 3 + c]))
                    .sum();
                out[(y * dw + j) * 3 + c] = acc as f32;
            }
        }
    }
    out
}

/// VLA-JEPA's first step, in `f32` on `[0, 255]`; exposed for tests.
pub fn area_frame_f32(img: &Image, height: u32, width: u32) -> Vec<f32> {
    let (sh, sw) = (img.height as usize, img.width as usize);
    let src: Vec<f32> = img.data.iter().map(|&b| f32::from(b) / 255.0).collect();
    let rows = area_taps(sh, height as usize);
    let cols = area_taps(sw, width as usize);
    separable(&src, (sh, sw), &rows, &cols)
        .into_iter()
        .map(|v| v * 255.0)
        .collect()
}

/// [`area_frame_f32`] rounded to the `u8` frame the engine takes.
pub fn area_frame(img: &Image, height: u32, width: u32) -> Image {
    Image {
        height,
        width,
        data: area_frame_f32(img, height, width)
            .into_iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    }
}

// ---- XR0: Qwen `smart_resize`, OpenCV `INTER_CUBIC` on u8 ----

/// Python's `round()`: half to even.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// physicalai's `xr0._resize_image` target `(height, width)`; `None` past its
/// aspect-ratio limit of 200.
pub fn smart_resize(height: u32, width: u32, factor: u32, max_pixels: u32) -> Option<(u32, u32)> {
    let (h, w, f) = (f64::from(height), f64::from(width), f64::from(factor));
    if h.max(w) / h.min(w) > 200.0 {
        return None;
    }
    let min_pixels = f * f;
    let mut nh = f.max(round_half_even(h / f) * f);
    let mut nw = f.max(round_half_even(w / f) * f);
    if nh * nw > f64::from(max_pixels) {
        let scale = (h * w / f64::from(max_pixels)).sqrt();
        nh = f.max((h / scale / f).floor() * f);
        nw = f.max((w / scale / f).floor() * f);
    } else if nh * nw < min_pixels {
        let scale = (min_pixels / (h * w)).sqrt();
        nh = f.max((h * scale / f).ceil() * f);
        nw = f.max((w * scale / f).ceil() * f);
    }
    Some((nh as u32, nw as u32))
}

/// Bytes OpenCV's 128-bit vector loops process at once.
const VLANES: usize = 16;

/// OpenCV's `v_muladd`: fused on AArch64 (NEON `vfmaq`), a product then a sum
/// on x86 builds without FMA in the baseline. Only the AArch64 path is checked
/// against OpenCV here.
#[inline]
fn muladd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

const COEF_BITS: u32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;

/// OpenCV's `interpolateCubic` (A = -0.75), in `f32` as it computes it.
fn cubic_coeffs(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// One axis: the first source index (before the `-1` tap) and the fixed-point
/// weights, `saturate_cast<short>(c * 2048)` (round half to even).
fn cubic_taps(src: usize, dst: usize) -> Vec<(i64, [i32; 4])> {
    let scale = 1.0 / (dst as f64 / src as f64);
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            let c = cubic_coeffs(f - s);
            (
                s as i64,
                c.map(|v| (v * COEF_SCALE).round_ties_even() as i32),
            )
        })
        .collect()
}

/// `cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)` on a `u8`
/// RGB frame: OpenCV's fixed-point path, edges replicated. Exposed for tests.
pub fn cubic_frame(img: &Image, height: u32, width: u32) -> Image {
    let (sh, sw) = (img.height as usize, img.width as usize);
    let (h, w) = (height as usize, width as usize);
    if (sh, sw) == (h, w) {
        return img.clone();
    }
    let xs = cubic_taps(sw, w);
    let ys = cubic_taps(sh, h);
    let clamp = |i: i64, n: usize| i.clamp(0, n as i64 - 1) as usize;
    // Horizontal pass per source row, into ints scaled by 2048.
    let hrow = |y: usize| -> Vec<i64> {
        let line = &img.data[y * sw * 3..(y + 1) * sw * 3];
        let mut out = Vec::with_capacity(w * 3);
        for &(sx, a) in &xs {
            for c in 0..3 {
                let v: i64 = (0..4)
                    .map(|k| {
                        i64::from(line[clamp(sx - 1 + k as i64, sw) * 3 + c]) * i64::from(a[k])
                    })
                    .sum();
                out.push(v);
            }
        }
        out
    };
    let rows: Vec<Vec<i64>> = (0..sh).map(hrow).collect();
    let mut data = Vec::with_capacity(h * w * 3);
    // OpenCV's vertical pass runs 16 bytes at a time in f32 (`VResizeCubicVec_32s8u`)
    // and finishes the row in fixed point.
    let vector_end = (w * 3) / VLANES * VLANES;
    for &(sy, b) in &ys {
        let r: [&Vec<i64>; 4] = std::array::from_fn(|k| &rows[clamp(sy - 1 + k as i64, sh)]);
        let bf = b.map(|v| v as f32 * (1.0 / (COEF_SCALE * COEF_SCALE)));
        for x in 0..w * 3 {
            let px = if x < vector_end {
                let s = |k: usize| r[k][x] as f32;
                let f = muladd(
                    s(0),
                    bf[0],
                    muladd(s(1), bf[1], muladd(s(2), bf[2], s(3) * bf[3])),
                );
                (f.round_ties_even() as i64).clamp(0, 255)
            } else {
                let v: i64 = (0..4).map(|k| r[k][x] * i64::from(b[k])).sum();
                let shift = 2 * COEF_BITS;
                ((v + (1 << (shift - 1))) >> shift).clamp(0, 255)
            };
            data.push(px as u8);
        }
    }
    Image {
        height,
        width,
        data,
    }
}
