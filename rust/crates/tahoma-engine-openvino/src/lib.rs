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

pub mod genai;

pub use genai::{OvGenaiBuilder, OvGenaiEngine};
