//! OpenVINO engines for tahoma.
//!
//! Currently exposes [`OvGenaiBuilder`] / [`OvGenaiEngine`] — the Rust
//! port of `tahoma/worker/engines/openvino/genai_engine.py`. Wraps
//! `tahoma-ov-genai-shim` (FFI to openvino-genai) under the
//! [`tahoma_engine::Engine`] and [`tahoma_engine::Builder`] traits.
//!
//! Two further engines from the Python tree (`ov-runtime`,
//! `ov-dist-spec`) are deferred — they need the lower-level
//! `openvino` crate (Core/CompiledModel/InferRequest) and the
//! distributed spec-decode protocol; tracked separately.

pub mod dist_spec;
pub mod genai;
pub mod rotary;
pub mod runtime;

pub use dist_spec::{
    DistributedMaskedReq, FrameKind, MaskedReq, OvDistSpecBuilder, OvDistSpecEngine,
    OvDistSpecWorkerBuilder, OvDistSpecWorkerEngine, SpecDecodeStats,
};
pub use genai::{OvGenaiBuilder, OvGenaiEngine};
pub use rotary::{ModelTextConfig, RopeScalingConfig, Rotary};
pub use runtime::{OvRuntimeBuilder, OvRuntimeEngine};
