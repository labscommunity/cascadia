//! OpenVINO engines for cascadia.
//!
//! Currently exposes [`OvGenaiBuilder`] / [`OvGenaiEngine`] — the Rust
//! port of `cascadia/worker/engines/openvino/genai_engine.py`. Wraps
//! `cascadia-ov-genai-shim` (FFI to openvino-genai) under the
//! [`cascadia_engine::Engine`] and [`cascadia_engine::Builder`] traits.
//!
//! Two further engines from the Python tree (`ov-runtime`,
//! `ov-dist-spec`) are deferred — they need the lower-level
//! `openvino` crate (Core/CompiledModel/InferRequest) and the
//! distributed spec-decode protocol; tracked separately.

pub mod constrained;
pub mod dist_spec;
pub mod gemma4;
pub mod genai;
pub mod qwen36;
pub mod rotary;
pub mod runtime;
mod warn_limit;

pub use dist_spec::{
    DistributedMaskedReq, FrameKind, MaskedReq, OvDistSpecBuilder, OvDistSpecEngine,
    OvDistSpecWorkerBuilder, OvDistSpecWorkerEngine, SpecDecodeStats,
};
pub use gemma4::{Gemma4Builder, Gemma4Engine};
pub use genai::{OvGenaiBuilder, OvGenaiEngine};
pub use qwen36::{Qwen36Builder, Qwen36Engine};
pub use rotary::{ModelTextConfig, RopeScalingConfig, Rotary};
pub use runtime::{OvRuntimeBuilder, OvRuntimeEngine};
