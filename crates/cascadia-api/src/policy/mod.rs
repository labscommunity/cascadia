//! The robot-policy (VLA) wire format: openpi's `msgpack_numpy` framing, so
//! `openpi_client.WebsocketClientPolicy` can drive the server.
//!
//! Public because other serving front ends reuse the codec without the router.

#[cfg(feature = "zenoh")]
pub mod bridge;
mod msgpack;
pub mod physicalai;
pub mod server;
#[cfg(feature = "zenoh")]
pub mod zenoh_server;

use cascadia_engine::{check_observation, payload_cap, EngineError, PAYLOAD_MAX_BYTES};
use cascadia_types::{ActionChunk, Image, NoiseMode, Observation, PolicySpec, RtcInputs};

use msgpack::{Val, Writer};

/// Whether `host` (an IP, `[v6]`, or `localhost`) is a loopback address: the
/// robot front ends have no authentication, so they bind nothing else
/// without an explicit opt-in.
pub fn is_loopback_host(host: &str) -> bool {
    match host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<std::net::IpAddr>()
    {
        Ok(ip) => ip.is_loopback(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

/// A request the server refuses; the message goes back to the client as-is.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct RequestError(pub String);

impl RequestError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

fn refuse<T>(msg: impl Into<String>) -> Result<T, RequestError> {
    Err(RequestError::new(msg))
}

const REQUEST_KEYS: &str = "images, state, prompt, and optional seed, noise and the RTC fields";

/// The real-time chunking fields: a request sends all four or none.
const RTC_KEYS: [&str; 4] = [
    "prev_chunk_left_over",
    "inference_delay",
    "max_guidance_weight",
    "execution_horizon",
];

/// One `msgpack_numpy` array, borrowed from the message.
struct NdArray<'a> {
    dtype: &'a str,
    shape: Vec<u64>,
    data: &'a [u8],
}

impl<'a> NdArray<'a> {
    fn parse(what: &str, v: Val<'a>) -> Result<Self, RequestError> {
        let Val::Map(entries) = v else {
            return refuse(format!("{what} must be a numpy array (msgpack_numpy)"));
        };
        let (mut marked, mut dtype, mut shape, mut data) = (false, None, None, None);
        for (k, v) in entries {
            match (k.key(), v) {
                (Some("__ndarray__"), Val::Bool(b)) => marked = b,
                (Some("dtype"), v) => dtype = v.key(),
                (Some("shape"), Val::Array(dims)) => {
                    shape = dims.iter().map(Val::as_u64).collect::<Option<Vec<_>>>()
                }
                (Some("data"), Val::Bin(b)) => data = Some(b),
                _ => {}
            }
        }
        match (marked, dtype, shape, data) {
            (true, Some(dtype), Some(shape), Some(data)) => Ok(Self { dtype, shape, data }),
            _ => refuse(format!(
                "{what} must be a numpy array (msgpack_numpy: __ndarray__, data, dtype, shape)"
            )),
        }
    }

    /// The data, once its length is `itemsize * prod(shape)`; checked before
    /// anything is allocated for the values.
    fn bytes(&self, what: &str, itemsize: u64) -> Result<&'a [u8], RequestError> {
        let want = self
            .shape
            .iter()
            .try_fold(itemsize, |a, &d| a.checked_mul(d));
        if want != Some(self.data.len() as u64) {
            return refuse(format!(
                "{what} has {} data bytes; expected {} for dtype {} and shape {:?}",
                self.data.len(),
                want.map_or_else(|| "an overflowing count".into(), |w| w.to_string()),
                self.dtype,
                self.shape
            ));
        }
        Ok(self.data)
    }

    fn expect_shape(&self, what: &str, want: &[u64]) -> Result<(), RequestError> {
        if self.shape != want {
            return refuse(format!(
                "{what} has shape {:?}; expected {want:?}",
                self.shape
            ));
        }
        Ok(())
    }

    fn floats(&self, what: &str) -> Result<Vec<f32>, RequestError> {
        match self.dtype {
            "<f4" => Ok(self
                .bytes(what, 4)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            "<f8" => Ok(self
                .bytes(what, 8)?
                .chunks_exact(8)
                .map(|c| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(c);
                    f64::from_le_bytes(a) as f32
                })
                .collect()),
            other => refuse(format!("{what} has dtype {other}; expected <f4 or <f8")),
        }
    }
}

fn seed(v: Val<'_>) -> Result<Option<u64>, RequestError> {
    let bad = || refuse("seed must be a non-negative integer");
    match v {
        Val::Nil => Ok(None),
        Val::UInt(_) | Val::Int(_) => v.as_u64().map(Some).map_or_else(bad, Ok),
        // A numpy scalar (`np.int64(7)`) arrives as `__npgeneric__`.
        Val::Map(entries) => entries
            .into_iter()
            .find(|(k, _)| k.key() == Some("data"))
            .and_then(|(_, d)| d.as_u64())
            .map(Some)
            .map_or_else(bad, Ok),
        _ => bad(),
    }
}

/// An integer, or a numpy scalar (`__npgeneric__`) holding one.
fn uint(what: &str, v: Val<'_>) -> Result<u32, RequestError> {
    let n = match v {
        Val::Map(entries) => entries
            .into_iter()
            .find(|(k, _)| k.key() == Some("data"))
            .and_then(|(_, d)| d.as_u64()),
        v => v.as_u64(),
    };
    n.and_then(|n| u32::try_from(n).ok()).map_or_else(
        || refuse(format!("{what} must be a non-negative integer")),
        Ok,
    )
}

/// A number, or a numpy scalar holding one.
fn number(what: &str, v: Val<'_>) -> Result<f32, RequestError> {
    let x = match v {
        Val::Map(entries) => entries
            .into_iter()
            .find(|(k, _)| k.key() == Some("data"))
            .map(|(_, d)| d),
        v => Some(v),
    };
    match x {
        Some(Val::Float(f)) => Ok(f as f32),
        Some(Val::UInt(u)) => Ok(u as f32),
        Some(Val::Int(i)) => Ok(i as f32),
        _ => refuse(format!("{what} must be a number")),
    }
}

/// `prev_chunk_left_over`: `<f4` `[T, D]`, or `[1, T, D]` as physicalai sends it.
fn leftover(v: Val<'_>, spec: &PolicySpec) -> Result<Vec<f32>, RequestError> {
    const WHAT: &str = "prev_chunk_left_over";
    let Some(t) = spec.rtc_chunk_size else {
        return refuse("this model takes no RTC inputs; export it with RTC enabled");
    };
    let a = NdArray::parse(WHAT, v)?;
    if a.dtype != "<f4" {
        return refuse(format!("{WHAT} has dtype {}; expected <f4", a.dtype));
    }
    let want = [t as u64, spec.chunk[1] as u64];
    if a.shape.len() == 3 && a.shape[0] == 1 {
        a.expect_shape(WHAT, &[1, want[0], want[1]])?;
    } else {
        a.expect_shape(WHAT, &want)?;
    }
    a.floats(WHAT)
}

fn images(v: Val<'_>, spec: &PolicySpec) -> Result<Vec<(String, Image)>, RequestError> {
    let Val::Map(entries) = v else {
        return refuse("images must be a map of camera name to array");
    };
    let hw3 = [spec.height as u64, spec.width as u64, 3];
    let shape = match spec.history {
        1 => hw3.to_vec(),
        h => [&[h as u64][..], &hw3].concat(),
    };
    // `max(1)`: an empty frame yields no chunks, and the camera is then missing.
    let frame = (spec.height as usize * spec.width as usize * 3).max(1);
    let mut out = Vec::with_capacity(entries.len().min(spec.cameras.len()) * spec.history);
    for (k, v) in entries {
        let Some(name) = k.key() else {
            return refuse("camera names must be strings");
        };
        if !spec.cameras.iter().any(|c| c == name) {
            return refuse(format!(
                "unknown camera {name:?}; expected {:?}",
                spec.cameras
            ));
        }
        let what = format!("camera {name:?}");
        let a = NdArray::parse(&what, v)?;
        if a.dtype != "|u1" {
            return refuse(format!("{what} has dtype {}; expected |u1", a.dtype));
        }
        a.expect_shape(&what, &shape)?;
        for data in a.bytes(&what, 1)?.chunks_exact(frame) {
            out.push((
                name.to_string(),
                Image {
                    height: spec.height,
                    width: spec.width,
                    data: data.to_vec(),
                },
            ));
        }
    }
    Ok(out)
}

/// Decode one request message and check it against `spec`.
///
/// The map holds `images` (camera name to `|u1` `[H, W, 3]`), `state` (`<f4`
/// or `<f8`, `[state_len]`), `prompt`, and optional `seed` and `noise` (`<f4`
/// `[1, *noise_shape]`, input-mode packages only). A package with a
/// [`PolicySpec::history`] `h` above 1 takes `[h, H, W, 3]` per camera and a
/// `[h, state_len]` state instead, oldest first. An RTC package also takes
/// `prev_chunk_left_over` (`<f4` `[rtc_chunk_size, D]`), `inference_delay`,
/// `max_guidance_weight` and `execution_horizon`, all four or none. Everything
/// [`check_observation`] covers is checked too, so a returned observation is
/// ready to queue.
pub fn decode_request(bytes: &[u8], spec: &PolicySpec) -> Result<Observation, RequestError> {
    let Val::Map(entries) = msgpack::parse(bytes)? else {
        return refuse(format!("a request is one msgpack map of {REQUEST_KEYS}"));
    };
    let (mut imgs, mut state, mut prompt, mut sd, mut noise) = (None, None, None, None, None);
    let (mut left, mut delay, mut weight, mut horizon) = (None, None, None, None);
    let mut seen = Vec::new();
    for (k, v) in entries {
        let Some(key) = k.key() else {
            return refuse("request keys must be strings");
        };
        if seen.contains(&key) {
            return refuse(format!("key {key:?} sent twice"));
        }
        seen.push(key);
        match key {
            "images" => imgs = Some(images(v, spec)?),
            "state" => {
                let a = NdArray::parse("state", v)?;
                let row = spec.state_len as u64;
                match spec.history {
                    1 => a.expect_shape("state", &[row])?,
                    h => a.expect_shape("state", &[h as u64, row])?,
                }
                state = Some(a.floats("state")?);
            }
            "prompt" => match v {
                Val::Str(s) if s.len() <= spec.prompt_max_bytes => prompt = Some(s.to_string()),
                Val::Str(s) => {
                    return refuse(format!(
                        "prompt is {} bytes; the limit is {}",
                        s.len(),
                        spec.prompt_max_bytes
                    ))
                }
                _ => return refuse("prompt must be a string"),
            },
            "seed" => sd = seed(v)?,
            "noise" if v == Val::Nil => {}
            "noise" => {
                if spec.noise != NoiseMode::Input {
                    let mode = match spec.noise {
                        NoiseMode::Input => "input",
                        NoiseMode::InGraphRandom => "in_graph_random",
                        NoiseMode::Constant => "constant",
                    };
                    return refuse(format!(
                        "this model takes no noise input (noise mode {mode})"
                    ));
                }
                let a = NdArray::parse("noise", v)?;
                if a.dtype != "<f4" {
                    return refuse(format!("noise has dtype {}; expected <f4", a.dtype));
                }
                let mut want = vec![1u64];
                want.extend(spec.noise_shape.iter().flatten().map(|&d| d as u64));
                a.expect_shape("noise", &want)?;
                noise = Some(a.floats("noise")?);
            }
            _ if RTC_KEYS.contains(&key) && v == Val::Nil => {}
            "prev_chunk_left_over" => left = Some(leftover(v, spec)?),
            "inference_delay" => delay = Some(uint(key, v)?),
            "max_guidance_weight" => weight = Some(number(key, v)?),
            "execution_horizon" => horizon = Some(uint(key, v)?),
            other => {
                return refuse(format!(
                    "unknown key {other:?}; a request has {REQUEST_KEYS}"
                ))
            }
        }
    }
    let (Some(images), Some(state), Some(prompt)) = (imgs, state, prompt) else {
        let missing = ["images", "state", "prompt"]
            .into_iter()
            .find(|k| !seen.contains(k))
            .unwrap_or_default();
        return refuse(format!(
            "missing key {missing:?}; a request has {REQUEST_KEYS}"
        ));
    };
    let rtc = match (left, delay, weight, horizon) {
        (None, None, None, None) => None,
        (Some(prev_chunk_left_over), Some(inference_delay), Some(w), Some(execution_horizon)) => {
            Some(RtcInputs {
                prev_chunk_left_over,
                inference_delay,
                max_guidance_weight: w,
                execution_horizon,
            })
        }
        _ => return refuse(format!("RTC needs all of {RTC_KEYS:?}, or none")),
    };
    let obs = Observation {
        images,
        state,
        prompt,
        seed: sd,
        noise,
        rtc,
    };
    check_observation(spec, &obs).map_err(|e| match e {
        EngineError::InvalidConfig(m) => RequestError(m),
        other => RequestError(other.to_string()),
    })?;
    Ok(obs)
}

fn ndarray_header(w: &mut Writer, dtype: &str, shape: &[usize], data: &[u8]) {
    w.map(4);
    w.bin(b"__ndarray__");
    w.bool(true);
    w.bin(b"data");
    w.bin(data);
    w.bin(b"dtype");
    w.str(dtype);
    w.bin(b"shape");
    w.array(shape.len());
    for &d in shape {
        w.uint(d as u64);
    }
}

/// `{"actions": <f4 [T, D], "server_timing": {"infer_ms": f64}}`, byte for
/// byte what `msgpack_numpy.packb` writes for the same values.
///
/// The caller refuses non-finite actions first (see [`check_reply`]).
pub fn encode_reply(chunk: &ActionChunk) -> Vec<u8> {
    let data: Vec<u8> = chunk.actions.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut w = Writer::default();
    w.map(2);
    w.str("actions");
    ndarray_header(&mut w, "<f4", &[chunk.steps, chunk.dim], &data);
    w.str("server_timing");
    w.map(1);
    w.str("infer_ms");
    w.f64(chunk.infer_ms);
    w.buf
}

/// Refuse a reply the robot must not act on: a non-finite action (f16
/// overflow on the GPU) or a buffer that is not `[steps, dim]`.
pub fn check_reply(chunk: &ActionChunk) -> Result<(), RequestError> {
    if chunk.steps.checked_mul(chunk.dim) != Some(chunk.actions.len()) {
        return refuse(format!(
            "the model returned {} actions for a [{}, {}] chunk",
            chunk.actions.len(),
            chunk.steps,
            chunk.dim
        ));
    }
    if chunk.actions.iter().any(|x| !x.is_finite()) {
        return refuse("the model returned a non-finite action; refusing to send it");
    }
    Ok(())
}

/// What the metadata says beyond the spec (which carries `manifest_sha256`
/// and `precision`), so a bridge can answer physicalai's handshake.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackageIdentity {
    /// `policy.name` in the package manifest.
    pub policy_name: String,
    /// The slowest warmup call, in milliseconds: the latency a client should budget for.
    pub warmup_ms: Option<u32>,
}

impl PackageIdentity {
    /// Read `manifest.json` in a package directory.
    pub fn from_package_dir(dir: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(dir.join("manifest.json"))?;
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let policy_name = manifest["policy"]["name"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        Ok(Self {
            policy_name,
            warmup_ms: None,
        })
    }
}

/// The spec's fields (`rtc_chunk_size` among them) plus `policy_name` and
/// `chunk_size` (T); `warmup_ms` too when `timing`.
fn metadata_value(spec: &PolicySpec, id: &PackageIdentity, timing: bool) -> serde_json::Value {
    let mut v = serde_json::to_value(spec).expect("PolicySpec serializes to JSON");
    let o = v.as_object_mut().expect("PolicySpec is an object");
    // Keep the key order the metadata had before `rtc_chunk_size` joined the
    // spec (it followed `chunk_size`), so that with serde_json's
    // `preserve_order` a package's metadata bytes, and their hash, are
    // unchanged. It is the last field serialized while `history` is 1 (left
    // out), so removing it moves nothing; a history package has no older bytes.
    let rtc = o.remove("rtc_chunk_size").unwrap_or_default();
    o.insert("policy_name".into(), id.policy_name.clone().into());
    o.insert("chunk_size".into(), spec.chunk[0].into());
    o.insert("rtc_chunk_size".into(), rtc);
    if timing {
        o.insert("warmup_ms".into(), id.warmup_ms.into());
    }
    v
}

/// The spec as JSON, served at `GET /v1/policy/metadata`, with no identity.
pub fn metadata_json(spec: &PolicySpec) -> String {
    metadata_json_for(spec, &PackageIdentity::default())
}

/// The HTTP metadata, which also states `warmup_ms`.
pub fn metadata_json_for(spec: &PolicySpec, id: &PackageIdentity) -> String {
    metadata_value(spec, id, true).to_string()
}

/// The metadata as one msgpack map, sent first on every WebSocket, with no identity.
pub fn encode_metadata(spec: &PolicySpec) -> Vec<u8> {
    encode_metadata_for(spec, &PackageIdentity::default())
}

/// The socket metadata. It carries no timing, so two loads of one package at
/// one precision send identical bytes: a client may hash them to tell whether
/// another server serves the same policy.
pub fn encode_metadata_for(spec: &PolicySpec, id: &PackageIdentity) -> Vec<u8> {
    let mut w = Writer::default();
    w.json(&metadata_value(spec, id, false));
    w.buf
}

/// The metadata a server sent, as [`encode_metadata_for`] writes it.
pub fn decode_metadata(bytes: &[u8]) -> Result<(PolicySpec, PackageIdentity), RequestError> {
    let v = msgpack::parse(bytes)?.to_json();
    let spec: PolicySpec = serde_json::from_value(v.clone())
        .map_err(|e| RequestError::new(format!("metadata is not a policy spec: {e}")))?;
    if spec.history == 0 {
        return refuse("metadata states history 0; it must be at least 1");
    }
    let id = PackageIdentity {
        policy_name: v["policy_name"].as_str().unwrap_or_default().to_string(),
        warmup_ms: None,
    };
    Ok((spec, id))
}

/// A request for `obs` in openpi's layout, as `msgpack_numpy` packs it: what
/// [`decode_request`] reads. `spec` gives the noise and leftover shapes and
/// the history (a camera's frames packed as one array); RTC inputs for a
/// package without RTC are refused.
pub fn encode_request(obs: &Observation, spec: &PolicySpec) -> Result<Vec<u8>, RequestError> {
    if obs.rtc.is_some() && spec.rtc_chunk_size.is_none() {
        return refuse("this model takes no RTC inputs; export it with RTC enabled");
    }
    if spec.history == 0 {
        return refuse("the spec's history is 0; it must be at least 1");
    }
    let mut w = Writer::default();
    let n = 3
        + usize::from(obs.seed.is_some())
        + usize::from(obs.noise.is_some())
        + 4 * usize::from(obs.rtc.is_some());
    w.map(n);
    w.str("images");
    if spec.history == 1 {
        w.map(obs.images.len());
        for (name, img) in &obs.images {
            w.str(name);
            ndarray_header(
                &mut w,
                "|u1",
                &[img.height as usize, img.width as usize, 3],
                &img.data,
            );
        }
    } else {
        // One `[history, H, W, 3]` array per camera, in first-sent order.
        let mut names: Vec<&str> = Vec::new();
        for (name, _) in &obs.images {
            if !names.contains(&name.as_str()) {
                names.push(name);
            }
        }
        w.map(names.len());
        for name in names {
            let frames: Vec<&Image> = obs
                .images
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, img)| img)
                .collect();
            let (h, wd) = (frames[0].height as usize, frames[0].width as usize);
            if frames
                .iter()
                .any(|f| (f.height, f.width) != (frames[0].height, frames[0].width))
            {
                return refuse(format!("camera {name:?} frames differ in size"));
            }
            let data: Vec<u8> = frames.iter().flat_map(|f| f.data.iter().copied()).collect();
            w.str(name);
            ndarray_header(&mut w, "|u1", &[frames.len(), h, wd, 3], &data);
        }
    }
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    w.str("state");
    let state_shape = match spec.history {
        1 => vec![obs.state.len()],
        h => vec![h, obs.state.len() / h],
    };
    ndarray_header(&mut w, "<f4", &state_shape, &f32s(&obs.state));
    w.str("prompt");
    w.str(&obs.prompt);
    if let Some(seed) = obs.seed {
        w.str("seed");
        w.uint(seed);
    }
    if let Some(noise) = &obs.noise {
        w.str("noise");
        let mut shape = vec![1];
        shape.extend(spec.noise_shape.iter().flatten());
        ndarray_header(&mut w, "<f4", &shape, &f32s(noise));
    }
    if let (Some(rtc), Some(t)) = (&obs.rtc, spec.rtc_chunk_size) {
        w.str("prev_chunk_left_over");
        let shape = [t, spec.chunk[1]];
        ndarray_header(&mut w, "<f4", &shape, &f32s(&rtc.prev_chunk_left_over));
        w.str("inference_delay");
        w.uint(rtc.inference_delay.into());
        w.str("max_guidance_weight");
        w.f64(rtc.max_guidance_weight.into());
        w.str("execution_horizon");
        w.uint(rtc.execution_horizon.into());
    }
    Ok(w.buf)
}

/// A reply [`encode_reply`] wrote: `actions` `<f4 [T, D]` and `infer_ms`.
pub fn decode_reply(bytes: &[u8]) -> Result<ActionChunk, RequestError> {
    let Val::Map(entries) = msgpack::parse(bytes)? else {
        return refuse("a reply is one msgpack map");
    };
    let (mut actions, mut infer_ms) = (None, None);
    for (k, v) in entries {
        match (k.key(), v) {
            (Some("actions"), v) => {
                let a = NdArray::parse("actions", v)?;
                let [steps, dim] = a.shape[..] else {
                    return refuse(format!("actions has shape {:?}; expected [T, D]", a.shape));
                };
                actions = Some((a.floats("actions")?, steps as usize, dim as usize));
            }
            (Some("server_timing"), Val::Map(t)) => {
                infer_ms = t.into_iter().find_map(|(k, v)| match (k.key(), v) {
                    (Some("infer_ms"), Val::Float(f)) => Some(f),
                    _ => None,
                });
            }
            _ => {}
        }
    }
    let Some((actions, steps, dim)) = actions else {
        return refuse("the reply has no actions");
    };
    Ok(ActionChunk {
        actions,
        steps,
        dim,
        infer_ms: infer_ms.unwrap_or_default(),
    })
}

/// The WebSocket message cap for `spec` ([`cascadia_engine::payload_cap`]).
///
/// Loading refuses a package over [`PAYLOAD_MAX_BYTES`], and so does
/// [`server::policy_router`]; a spec that skipped both gets the bound itself.
pub fn max_payload_bytes(spec: &PolicySpec) -> u32 {
    payload_cap(spec).unwrap_or(PAYLOAD_MAX_BYTES)
}
