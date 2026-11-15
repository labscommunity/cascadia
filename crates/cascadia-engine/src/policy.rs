//! The robot-policy (VLA) engine seam, a sibling of [`crate::Engine`].
//!
//! A policy call is tensors in, one `[T, D]` float array out, so it does not
//! fit `Engine`'s token stream.

use std::collections::HashSet;

use cascadia_types::{ActionChunk, NoiseMode, Observation, PolicySpec};

use crate::{EngineError, EngineResult};

/// Bound on `|noise|` a caller may send; `N(0,1)` never gets near it.
pub const NOISE_ABS_MAX: f32 = 10.0;

/// A loaded robot policy. Held behind a mutex by the server, so it must be `Send`.
pub trait PolicyEngine: Send {
    fn spec(&self) -> &PolicySpec;

    /// One call on synthetic inputs to compile kernels and warm device caches.
    fn warmup(&mut self);

    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk>;

    /// Tear down the engine. Idempotent.
    fn close(&mut self) {}
}

/// Check an observation against `spec` before any work is queued.
///
/// Covers what the spec alone settles: camera names (exactly the advertised
/// set), frame sizes, state length and finiteness, prompt size, and whether
/// noise is allowed. Noise length is the engine's to check, since the spec
/// does not carry the noise shape.
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
/// `state[d % state_len] + 0.01 * t`, plus `noise[t * D + d]` when the call
/// carries noise of exactly `T * D` values.
pub struct MockPolicyEngine {
    spec: PolicySpec,
    warmed: bool,
    calls: u64,
    closed: bool,
}

impl MockPolicyEngine {
    pub fn new(spec: PolicySpec) -> Self {
        Self {
            spec,
            warmed: false,
            calls: 0,
            closed: false,
        }
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

    fn warmup(&mut self) {
        self.warmed = true;
    }

    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        if self.closed {
            return Err(EngineError::NotLoaded);
        }
        check_observation(&self.spec, obs)?;
        let [steps, dim] = self.spec.chunk;
        let noise = obs.noise.as_deref().filter(|n| n.len() == steps * dim);
        let mut actions = Vec::with_capacity(steps * dim);
        for t in 0..steps {
            for d in 0..dim {
                let base = if obs.state.is_empty() {
                    0.0
                } else {
                    obs.state[d % obs.state.len()]
                };
                let n = noise.map_or(0.0, |n| n[t * dim + d]);
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
            noise,
            resize: ResizeConvention::Center,
            prompt_max_bytes: 16,
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
    }

    #[test]
    fn mock_is_deterministic_and_uses_noise() {
        let mut e = MockPolicyEngine::new(spec(NoiseMode::Input));
        e.warmup();
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
        e.close();
        assert!(matches!(e.infer(&obs()), Err(EngineError::NotLoaded)));
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
