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
    pub images: Vec<(String, Image)>,
    pub state: Vec<f32>,
    pub prompt: String,
    pub seed: Option<u64>,
    /// Flow-matching start noise, row-major `[1, *noise_shape]`; only for
    /// [`NoiseMode::Input`] packages.
    pub noise: Option<Vec<f32>>,
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
    /// How the server may resize a frame of another size to `height x width`:
    /// this package's own preprocessor resize, ported exactly. `None`: a client
    /// must send `height x width`.
    #[serde(default)]
    pub server_resize: Option<ServerResize>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
            server_resize: None,
        };
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["noise"], "in_graph_random");
        assert_eq!(v["resize"], "top_left");
        assert_eq!(v["chunk"], serde_json::json!([10, 7]));
        assert!(v["noise_shape"].is_null());
        assert!(v.get("attribution").is_none());
        assert_eq!(serde_json::from_value::<PolicySpec>(v).unwrap(), spec);
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
