//! The robot-policy (VLA) wire format: openpi's `msgpack_numpy` framing, so
//! `openpi_client.WebsocketClientPolicy` can drive the server.
//!
//! Public because other serving front ends reuse the codec without the router.

mod msgpack;
pub mod server;

use cascadia_engine::{check_observation, payload_cap, EngineError, PAYLOAD_MAX_BYTES};
use cascadia_types::{ActionChunk, Image, NoiseMode, Observation, PolicySpec};

use msgpack::{Val, Writer};

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

const REQUEST_KEYS: &str = "images, state, prompt, and optional seed and noise";

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

fn images(v: Val<'_>, spec: &PolicySpec) -> Result<Vec<(String, Image)>, RequestError> {
    let Val::Map(entries) = v else {
        return refuse("images must be a map of camera name to array");
    };
    let hw3 = [spec.height as u64, spec.width as u64, 3];
    let mut out = Vec::with_capacity(entries.len().min(spec.cameras.len()));
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
        a.expect_shape(&what, &hw3)?;
        out.push((
            name.to_string(),
            Image {
                height: spec.height,
                width: spec.width,
                data: a.bytes(&what, 1)?.to_vec(),
            },
        ));
    }
    Ok(out)
}

/// Decode one request message and check it against `spec`.
///
/// The map holds `images` (camera name to `|u1` `[H, W, 3]`), `state` (`<f4`
/// or `<f8`, `[state_len]`), `prompt`, and optional `seed` and `noise` (`<f4`
/// `[1, *noise_shape]`, input-mode packages only). Everything
/// [`check_observation`] covers is checked too, so a returned observation is
/// ready to queue.
pub fn decode_request(bytes: &[u8], spec: &PolicySpec) -> Result<Observation, RequestError> {
    let Val::Map(entries) = msgpack::parse(bytes)? else {
        return refuse(format!("a request is one msgpack map of {REQUEST_KEYS}"));
    };
    let (mut imgs, mut state, mut prompt, mut sd, mut noise) = (None, None, None, None, None);
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
                a.expect_shape("state", &[spec.state_len as u64])?;
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
                        "this package takes no noise input (noise mode {mode})"
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
    let obs = Observation {
        images,
        state,
        prompt,
        seed: sd,
        noise,
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

/// The spec as JSON, served at `GET /v1/policy/metadata`.
pub fn metadata_json(spec: &PolicySpec) -> String {
    serde_json::to_string(spec).expect("PolicySpec serializes to JSON")
}

/// The spec as one msgpack map, sent first on every WebSocket.
pub fn encode_metadata(spec: &PolicySpec) -> Vec<u8> {
    let v = serde_json::to_value(spec).expect("PolicySpec serializes to JSON");
    let mut w = Writer::default();
    w.json(&v);
    w.buf
}

/// The WebSocket message cap for `spec` ([`cascadia_engine::payload_cap`]).
///
/// Loading refuses a package over [`PAYLOAD_MAX_BYTES`], and so does
/// [`server::policy_router`]; a spec that skipped both gets the bound itself.
pub fn max_payload_bytes(spec: &PolicySpec) -> u32 {
    payload_cap(spec).unwrap_or(PAYLOAD_MAX_BYTES)
}
