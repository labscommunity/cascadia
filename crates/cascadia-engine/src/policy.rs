//! The robot-policy (VLA) engine seam, a sibling of [`crate::Engine`].
//!
//! A policy call is tensors in, one `[T, D]` float array out, so it does not
//! fit `Engine`'s token stream.

use std::collections::HashSet;

use cascadia_types::{ActionChunk, NoiseMode, Observation, PolicySpec};

use crate::{EngineError, EngineResult};

/// Bound on `|noise|` a caller may send; `N(0,1)` never gets near it.
pub const NOISE_ABS_MAX: f32 = 10.0;

/// Calls [`PolicyEngine::warmup`] makes; the first compiles kernels, the rest
/// show the steady-state time.
pub const WARMUP_CALLS: usize = 3;

/// Bytes a WebSocket message may carry beyond the arrays and the prompt:
/// msgpack keys, ndarray headers and the frame header.
pub const PAYLOAD_FRAMING_BYTES: u64 = 64 * 1024;

/// Largest WebSocket message cap a package may need; a spec over it is
/// refused at load. 8 MiB less the framing allowance.
pub const PAYLOAD_MAX_BYTES: u32 = 8 * 1024 * 1024 - 64 * 1024;

/// The WebSocket message cap for `spec`: every camera's `H*W*3` bytes, the
/// state as f64 (numpy's default dtype, the widest a request may send), the
/// noise as f32, the prompt cap, plus [`PAYLOAD_FRAMING_BYTES`].
///
/// Refuses a spec whose cap overflows or exceeds [`PAYLOAD_MAX_BYTES`].
pub fn payload_cap(spec: &PolicySpec) -> EngineResult<u32> {
    let frame = u64::from(spec.height)
        .checked_mul(u64::from(spec.width))
        .and_then(|p| p.checked_mul(3));
    let noise = match &spec.noise_shape {
        Some(shape) => shape
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d as u64))
            .and_then(|n| n.checked_mul(4)),
        None => Some(0),
    };
    let total = frame
        .and_then(|f| f.checked_mul(spec.cameras.len() as u64))
        .zip((spec.state_len as u64).checked_mul(8))
        .and_then(|(a, b)| a.checked_add(b))
        .zip(noise)
        .and_then(|(a, b)| a.checked_add(b))
        .and_then(|a| a.checked_add(spec.prompt_max_bytes as u64))
        .and_then(|a| a.checked_add(PAYLOAD_FRAMING_BYTES));
    match total {
        Some(t) if t <= u64::from(PAYLOAD_MAX_BYTES) => Ok(t as u32),
        _ => Err(EngineError::InvalidConfig(format!(
            "a request for this package needs {} bytes; the WebSocket limit is {PAYLOAD_MAX_BYTES}",
            total.map_or_else(|| "more than 2^64".into(), |t| t.to_string())
        ))),
    }
}

/// A loaded robot policy. Held behind a mutex by the server, so it must be `Send`.
pub trait PolicyEngine: Send {
    fn spec(&self) -> &PolicySpec;

    /// [`WARMUP_CALLS`] calls on synthetic inputs to compile kernels and warm
    /// device caches.
    ///
    /// Returns each call's time in whole milliseconds, rounded up (the
    /// [`ActionChunk::infer_ms`] measurement), so a caller can size a deadline
    /// from the slowest. Any failed call fails the warmup: a package that
    /// cannot infer must fail its load rather than serve errors.
    fn warmup(&mut self) -> EngineResult<Vec<u32>>;

    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk>;

    /// Tear down the engine. Idempotent.
    fn close(&mut self) {}
}

/// Check an observation against `spec` before any work is queued.
///
/// Covers what the spec alone settles: camera names (exactly the advertised
/// set), frame sizes, state length and finiteness, prompt size, and whether
/// noise is allowed and its length.
pub fn check_observation(spec: &PolicySpec, obs: &Observation) -> EngineResult<()> {
    let bad = |msg: String| Err(EngineError::InvalidConfig(msg));
    let mut seen = HashSet::new();
    for (name, img) in &obs.images {
        if !seen.insert(name.as_str()) {
            return bad(format!("camera {name:?} sent twice"));
        }
        if !spec.cameras.iter().any(|c| c == name) {
            return bad(format!(
                "unknown camera {name:?}; expected {:?}",
                spec.cameras
            ));
        }
        if img.height != spec.height || img.width != spec.width {
            return bad(format!(
                "camera {name:?} is {}x{}; expected {}x{} (HxW)",
                img.height, img.width, spec.height, spec.width
            ));
        }
        if img.expected_len() != Some(img.data.len()) {
            return bad(format!(
                "camera {name:?} has {} bytes; expected {} (H*W*3)",
                img.data.len(),
                spec.height as u64 * spec.width as u64 * 3
            ));
        }
    }
    if let Some(missing) = spec.cameras.iter().find(|c| !seen.contains(c.as_str())) {
        return bad(format!(
            "missing camera {missing:?}; expected {:?}",
            spec.cameras
        ));
    }
    if obs.state.len() != spec.state_len {
        return bad(format!(
            "state has {} values; expected {}",
            obs.state.len(),
            spec.state_len
        ));
    }
    if obs.state.iter().any(|x| !x.is_finite()) {
        return bad("state contains a non-finite value".into());
    }
    if obs.prompt.len() > spec.prompt_max_bytes {
        return bad(format!(
            "prompt is {} bytes; the limit is {}",
            obs.prompt.len(),
            spec.prompt_max_bytes
        ));
    }
    if let Some(noise) = &obs.noise {
        if spec.noise != NoiseMode::Input {
            return bad("this package takes no noise input".into());
        }
        let Some(shape) = spec.noise_shape.as_deref() else {
            return bad("the spec has no noise shape for an input-mode package".into());
        };
        let want = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
        if want != Some(noise.len()) {
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            return bad(format!(
                "noise has {} values; expected {} for shape [1, {}]",
                noise.len(),
                want.map_or_else(|| "an overflowing count".into(), |w| w.to_string()),
                dims.join(", ")
            ));
        }
        if noise
            .iter()
            .any(|x| !x.is_finite() || x.abs() > NOISE_ABS_MAX)
        {
            return bad(format!("noise must be finite with |x| <= {NOISE_ABS_MAX}"));
        }
    }
    Ok(())
}

/// Deterministic [`PolicyEngine`] for API and CLI tests without OpenVINO.
///
/// The same observation always gives the same chunk: action `[t, d]` is
/// `state[d % state_len] + 0.01 * t`, plus the call's noise value at the same
/// flat index when it carries noise.
pub struct MockPolicyEngine {
    spec: PolicySpec,
    warmed: bool,
    warmup_ms: Vec<u32>,
    calls: u64,
    closed: bool,
}

impl MockPolicyEngine {
    pub fn new(spec: PolicySpec) -> Self {
        Self {
            spec,
            warmed: false,
            warmup_ms: vec![0; WARMUP_CALLS],
            calls: 0,
            closed: false,
        }
    }

    /// Timings `warmup` returns; [`WARMUP_CALLS`] zeros by default.
    pub fn with_warmup_ms(mut self, ms: Vec<u32>) -> Self {
        self.warmup_ms = ms;
        self
    }

    pub fn warmed(&self) -> bool {
        self.warmed
    }

    /// Successful `infer` calls so far.
    pub fn calls(&self) -> u64 {
        self.calls
    }
}

impl PolicyEngine for MockPolicyEngine {
    fn spec(&self) -> &PolicySpec {
        &self.spec
    }

    fn warmup(&mut self) -> EngineResult<Vec<u32>> {
        self.warmed = true;
        Ok(self.warmup_ms.clone())
    }

    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        if self.closed {
            return Err(EngineError::NotLoaded);
        }
        check_observation(&self.spec, obs)?;
        let [steps, dim] = self.spec.chunk;
        let noise = obs.noise.as_deref();
        let mut actions = Vec::with_capacity(steps * dim);
        for t in 0..steps {
            for d in 0..dim {
                let base = if obs.state.is_empty() {
                    0.0
                } else {
                    obs.state[d % obs.state.len()]
                };
                let n = noise
                    .and_then(|n| n.get(t * dim + d))
                    .copied()
                    .unwrap_or(0.0);
                actions.push(base + 0.01 * t as f32 + n);
            }
        }
        self.calls += 1;
        Ok(ActionChunk {
            actions,
            steps,
            dim,
            infer_ms: 0.0,
        })
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{Image, ResizeConvention};

    fn spec(noise: NoiseMode) -> PolicySpec {
        PolicySpec {
            model: "mock".into(),
            device: "CPU".into(),
            cameras: vec!["image".into(), "wrist".into()],
            height: 2,
            width: 2,
            state_len: 3,
            chunk: [2, 3],
            noise_shape: (noise == NoiseMode::Input).then(|| vec![2, 3]),
            noise,
            resize: ResizeConvention::Center,
            prompt_max_bytes: 16,
            attribution: None,
            manifest_sha256: String::new(),
            precision: String::new(),
            server_resize: false,
        }
    }

    fn frame() -> Image {
        Image {
            height: 2,
            width: 2,
            data: vec![0; 12],
        }
    }

    fn obs() -> Observation {
        Observation {
            images: vec![("wrist".into(), frame()), ("image".into(), frame())],
            state: vec![1.0, 2.0, 3.0],
            prompt: "pick".into(),
            seed: None,
            noise: None,
        }
    }

    fn err_of(spec: &PolicySpec, o: &Observation) -> String {
        check_observation(spec, o).unwrap_err().to_string()
    }

    #[test]
    fn accepts_cameras_in_any_order() {
        check_observation(&spec(NoiseMode::Constant), &obs()).unwrap();
    }

    #[test]
    fn rejects_missing_extra_and_duplicate_cameras() {
        let s = spec(NoiseMode::Constant);
        let mut o = obs();
        o.images.pop();
        assert!(err_of(&s, &o).contains("missing camera \"image\""));
        let mut o = obs();
        o.images.push(("top".into(), frame()));
        assert!(err_of(&s, &o).contains("unknown camera \"top\""));
        let mut o = obs();
        o.images.push(("wrist".into(), frame()));
        assert!(err_of(&s, &o).contains("sent twice"));
    }

    #[test]
    fn rejects_wrong_frame_size_and_byte_count() {
        let s = spec(NoiseMode::Constant);
        let mut o = obs();
        o.images[0].1.width = 3;
        assert!(err_of(&s, &o).contains("expected 2x2"));
        let mut o = obs();
        o.images[0].1.data.pop();
        assert!(err_of(&s, &o).contains("11 bytes; expected 12"));
    }

    #[test]
    fn rejects_bad_state_prompt_and_noise() {
        let s = spec(NoiseMode::Constant);
        let mut o = obs();
        o.state.push(0.0);
        assert!(err_of(&s, &o).contains("4 values; expected 3"));
        let mut o = obs();
        o.state[1] = f32::NAN;
        assert!(err_of(&s, &o).contains("non-finite"));
        let mut o = obs();
        o.prompt = "x".repeat(17);
        assert!(err_of(&s, &o).contains("17 bytes; the limit is 16"));
        let mut o = obs();
        o.noise = Some(vec![0.0; 6]);
        assert!(err_of(&s, &o).contains("takes no noise"));
        let s = spec(NoiseMode::Input);
        o.noise = Some(vec![0.0, 10.5, 0.0, 0.0, 0.0, 0.0]);
        assert!(err_of(&s, &o).contains("|x| <= 10"));
        o.noise = Some(vec![0.0; 5]);
        assert!(err_of(&s, &o).contains("5 values; expected 6 for shape [1, 2, 3]"));
        o.noise = Some(vec![0.0; 6]);
        check_observation(&s, &o).unwrap();
    }

    #[test]
    fn mock_is_deterministic_and_uses_noise() {
        let mut e = MockPolicyEngine::new(spec(NoiseMode::Input));
        assert_eq!(e.warmup().unwrap(), vec![0; WARMUP_CALLS]);
        assert!(e.warmed());
        let a = e.infer(&obs()).unwrap();
        assert_eq!((a.steps, a.dim), (2, 3));
        assert_eq!(a.actions, vec![1.0, 2.0, 3.0, 1.01, 2.01, 3.01]);
        assert_eq!(e.infer(&obs()).unwrap(), a);
        let mut o = obs();
        o.noise = Some(vec![1.0; 6]);
        let b = e.infer(&o).unwrap();
        assert_eq!(b.actions[0], 2.0);
        assert_eq!(e.calls(), 3);
        // A wrong-length noise is refused, not ignored.
        o.noise = Some(vec![1.0; 7]);
        assert!(e.infer(&o).is_err());
        assert_eq!(e.calls(), 3);
        e.close();
        assert!(matches!(e.infer(&obs()), Err(EngineError::NotLoaded)));
    }

    #[test]
    fn payload_cap_sums_the_arrays_and_refuses_over_the_bound() {
        let s = spec(NoiseMode::Input);
        // 2 cameras * 2*2*3 + 3 states * 8 + noise 6 * 4 + prompt 16 + framing.
        assert_eq!(
            payload_cap(&s).unwrap() as u64,
            24 + 24 + 24 + 16 + PAYLOAD_FRAMING_BYTES
        );
        let mut big = spec(NoiseMode::Constant);
        big.height = 1024;
        big.width = 1024;
        big.cameras = vec!["a".into(), "b".into(), "c".into()];
        let e = payload_cap(&big).unwrap_err().to_string();
        assert!(e.contains("WebSocket limit is 8323072"), "{e}");
        big.height = u32::MAX;
        big.width = u32::MAX;
        assert!(payload_cap(&big)
            .unwrap_err()
            .to_string()
            .contains("more than 2^64"));
    }

    #[test]
    fn mock_returns_its_warmup_timings() {
        let mut e = MockPolicyEngine::new(spec(NoiseMode::Constant)).with_warmup_ms(vec![40, 9, 8]);
        assert_eq!(e.warmup().unwrap(), vec![40, 9, 8]);
    }

    #[test]
    fn mock_rejects_through_check_observation() {
        let mut e = MockPolicyEngine::new(spec(NoiseMode::Constant));
        let mut o = obs();
        o.state.clear();
        assert!(e.infer(&o).is_err());
        assert_eq!(e.calls(), 0);
    }
}
