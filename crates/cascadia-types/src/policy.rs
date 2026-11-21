//! Robot-policy (VLA) value types: one observation in, one action chunk out.
//!
//! Engine-agnostic so the API, the CLI and every `PolicyEngine` share them.

use serde::{Deserialize, Serialize};

/// One camera frame: `height * width * 3` bytes, uint8 HWC RGB.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    pub height: u32,
    pub width: u32,
    pub data: Vec<u8>,
}

impl Image {
    /// Byte length a well-formed frame of this size has, or `None` on overflow.
    pub fn expected_len(&self) -> Option<usize> {
        (self.height as usize)
            .checked_mul(self.width as usize)?
            .checked_mul(3)
    }
}

/// One policy call's input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Camera name to frame; names must be exactly [`PolicySpec::cameras`].
    /// With [`PolicySpec::history`] above 1, each camera appears that many
    /// times, oldest frame first.
    pub images: Vec<(String, Image)>,
    /// `state_len` values, or `history` rows of them, oldest first.
    pub state: Vec<f32>,
    pub prompt: String,
    pub seed: Option<u64>,
    /// Flow-matching start noise, row-major `[1, *noise_shape]`; only for
    /// [`NoiseMode::Input`] packages.
    pub noise: Option<Vec<f32>>,
    /// Real-time chunking guidance; only for packages with
    /// [`PolicySpec::rtc_chunk_size`]. `None` serves the plain chunk.
    #[serde(default)]
    pub rtc: Option<RtcInputs>,
}

/// Real-time chunking (Black et al., arXiv 2506.07339): the new chunk is
/// denoised toward the part of the previous one the robot has not executed.
/// The client keeps the action queue, so the server stays stateless.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RtcInputs {
    /// The previous chunk's unexecuted actions as the server returned them,
    /// zero-padded at the end to `[rtc_chunk_size, D]`, row-major.
    pub prev_chunk_left_over: Vec<f32>,
    /// Actions the robot executes while this call runs.
    pub inference_delay: u32,
    /// Upper bound on the guidance weight; 0 turns guidance off.
    pub max_guidance_weight: f32,
    /// Actions of the leftover the new chunk is pulled toward, at most half the chunk.
    pub execution_horizon: u32,
}

/// One policy call's output: `actions` is row-major `[steps, dim]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionChunk {
    pub actions: Vec<f32>,
    pub steps: usize,
    pub dim: usize,
    pub infer_ms: f64,
}

/// Where a package's flow-matching start noise comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseMode {
    /// The model has a `noise` input; the engine draws it unless the call carries it.
    Input,
    /// A `RandomUniform` op inside the graph.
    InGraphRandom,
    /// Neither: the start is fixed in the graph.
    Constant,
}

/// The convention a client must resize with before sending frames.
///
/// `Center` and `TopLeft` are bilinear letterboxes with pad value 0 applied
/// before scaling to `[-1, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeConvention {
    /// Padding split evenly on both sides (`pi05`).
    Center,
    /// Padding on the top and left, content at the bottom right (`smolvla_resize`).
    TopLeft,
    /// A plain stretch to `height x width`, no padding, aspect ratio not kept
    /// (`vla_jepa_inputs`, `xr0`; `smolvla_resize` with `"resize": "stretch"`).
    Stretch,
}

/// A package's own first image resize, as the server can run it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ServerResize {
    /// Bilinear (OpenCV `INTER_LINEAR`, torch `align_corners=False`) to the
    /// aspect-kept size, padded with 0 as `resize` says (`pi05`, `pi0`,
    /// `smolvla_resize`).
    Letterbox,
    /// Average pooling, torch `interpolate(mode="area")` (VLA-JEPA's first step).
    Area,
    /// Qwen `smart_resize` to multiples of `factor` within `max_pixels`, then
    /// OpenCV `INTER_CUBIC` on `u8` (XR0); only frames it sends to `height x width`.
    SmartResizeCubic { factor: u32, max_pixels: u32 },
}

/// The policy name rule: a non-empty key segment that starts with a letter or
/// digit and holds only letters, digits, `.`, `_` or `-` (physicalai's
/// `model_key_prefix` rule). A served model id and a Zenoh slot name both follow it.
pub fn is_valid_policy_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
}

/// What a loaded policy accepts and returns; served as `GET /v1/policy/metadata`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicySpec {
    pub model: String,
    pub device: String,
    /// Camera names a request must carry, exactly.
    pub cameras: Vec<String>,
    pub height: u32,
    pub width: u32,
    pub state_len: usize,
    /// `[T, D]` of the returned action chunk.
    pub chunk: [usize; 2],
    pub noise: NoiseMode,
    /// Noise dims after the batch of 1 (a request sends `[1, *noise_shape]`);
    /// `Some` only in [`NoiseMode::Input`].
    pub noise_shape: Option<Vec<usize>>,
    pub resize: ResizeConvention,
    pub prompt_max_bytes: usize,
    /// The package's licence notice, served verbatim when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    /// SHA-256 (lowercase hex) of the package's `manifest.json` bytes, so two
    /// fine-tunes with the same shapes advertise different metadata.
    #[serde(default)]
    pub manifest_sha256: String,
    /// The precision the model was compiled at (`f16` or `f32`).
    #[serde(default)]
    pub precision: String,
    /// SHA-256 (lowercase hex) over every file in the package, the
    /// manifest's and any other (`package_sha256` in the OpenVINO engine), so
    /// packages that differ only in a file the manifest does not name (int8
    /// and f16 weights) advertise different metadata. Empty when unknown, and
    /// then left out of the metadata.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub package_sha256: String,
    /// How the server may resize a frame of another size to `height x width`:
    /// this package's own preprocessor resize, ported exactly. `None`: a client
    /// must send `height x width`.
    #[serde(default)]
    pub server_resize: Option<ServerResize>,
    /// `T` of the `prev_chunk_left_over` a request may send, for a package
    /// exported with real-time chunking; `None` otherwise.
    #[serde(default)]
    pub rtc_chunk_size: Option<usize>,
    /// Observations a request carries, oldest first: frames per camera and
    /// state rows (XR1 takes 4). Left out of the metadata when 1.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub history: usize,
}

fn one() -> usize {
    1
}

fn is_one(n: &usize) -> bool {
    *n == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_names_are_key_segments() {
        for ok in ["smolvla-libero", "pi0.5_droid", "A1"] {
            assert!(is_valid_policy_name(ok), "{ok}");
        }
        for bad in ["", "-x", ".x", "a/b", "a b", "a:b", "é"] {
            assert!(!is_valid_policy_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn spec_serializes_with_snake_case_enums() {
        let spec = PolicySpec {
            model: "m".into(),
            device: "GPU".into(),
            cameras: vec!["image".into(), "image2".into()],
            height: 224,
            width: 224,
            state_len: 8,
            chunk: [10, 7],
            noise: NoiseMode::InGraphRandom,
            noise_shape: None,
            resize: ResizeConvention::TopLeft,
            prompt_max_bytes: 1024,
            attribution: None,
            manifest_sha256: String::new(),
            precision: String::new(),
            package_sha256: String::new(),
            server_resize: None,
            rtc_chunk_size: None,
            history: 1,
        };
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["noise"], "in_graph_random");
        assert_eq!(v["resize"], "top_left");
        assert_eq!(v["chunk"], serde_json::json!([10, 7]));
        assert!(v["noise_shape"].is_null());
        assert!(v.get("attribution").is_none());
        assert!(v["rtc_chunk_size"].is_null());
        assert!(v.get("history").is_none());
        assert_eq!(serde_json::from_value::<PolicySpec>(v).unwrap(), spec);
        let video = PolicySpec { history: 4, ..spec };
        let v = serde_json::to_value(&video).unwrap();
        assert_eq!(v["history"], 4);
        assert_eq!(serde_json::from_value::<PolicySpec>(v).unwrap(), video);
    }

    #[test]
    fn image_expected_len_is_hwc_and_checked() {
        let img = Image {
            height: 2,
            width: 3,
            data: vec![],
        };
        assert_eq!(img.expected_len(), Some(18));
        let huge = Image {
            height: u32::MAX,
            width: u32::MAX,
            data: vec![],
        };
        // 64-bit usize: (2^32-1)^2 * 3 overflows.
        assert_eq!(huge.expected_len(), None);
    }
}
