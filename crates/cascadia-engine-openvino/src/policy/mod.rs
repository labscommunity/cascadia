//! The `ov-policy` engine: `policy_package` manifest parsing, the pre- and
//! postprocessing steps, camera slot order, spec derivation and the serving
//! pipeline, which runs against fakes in the default stub build.

pub mod cameras;
pub mod gr00t;
pub mod manifest;
pub mod molmoact2;
pub mod ov;
pub mod package;
pub mod pipeline;
pub mod qwen_vl;
pub mod spec;
pub mod special_tokens;
pub mod steps;
pub mod vla_jepa;
pub mod xr0;

pub use cameras::{images_tensor, CameraLayout, CameraPlan};
pub use manifest::{LoadOptions, PolicyKind, PolicyManifest};
pub use ov::{OvPolicyEngine, OvPolicyOptions};
pub use package::{
    check_digest_ignore, engine_build, engine_build_is_known, package_sha256, package_sha256_with,
};
pub use pipeline::{compile_config, PolicyPipeline, Precision};
pub use spec::{derive_spec, IrFacts, ManifestSpec, ServerFacts};
