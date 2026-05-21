//! Per-channel FFN sparsity thresholds — file format + loader.
//!
//! Issue #38, the CHESS extension to the global-τ FFN sparsity landed
//! in PR #34. Each (layer, channel) pair gets its own threshold
//! `τ[layer, channel]`, calibrated offline:
//!
//! ```text
//! τ[layer, channel] = quantile_{1 - target_active_frac}(
//!     |silu(gate[layer, channel])| / max_j |silu(gate[layer, j])|
//! )
//! ```
//!
//! ## File format
//!
//! JSON, schema `version: 1`. One file per model (not per layer) —
//! at K2.6 dims this is 60 layers × 2048 channels × 4 B ≈ 480 KiB on
//! disk after JSON encoding (~1.5 MB pretty-printed).
//!
//! ```json
//! {
//!   "version": 1,
//!   "model_id": "kimi-k2.6-instruct",
//!   "n_intermediate": 2048,
//!   "calibration_n_tokens": 12345,
//!   "target_active_frac": 0.5,
//!   "notes": "optional free-form provenance",
//!   "layers": [
//!     { "layer_id": 0, "thresholds": [0.123, 0.456, ...] },
//!     ...
//!   ]
//! }
//! ```
//!
//! The runtime treats the file as advisory: if a layer is missing
//! from `layers`, [`PerChannelThresholds::get`] returns `None` and
//! the dispatcher falls back to the global-τ (or dense) path for
//! that layer. This is the right behaviour for partial-coverage
//! calibration runs (e.g. only the first N layers were covered).
//!
//! Schema bumps: increment `version` and add a v2-aware path here.
//! Today the loader rejects anything that isn't `version: 1`.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current threshold-file schema version. Bump on any layout change.
pub const THRESHOLDS_FILE_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ThresholdsError {
    #[error("io error reading thresholds file: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported thresholds schema version: got {got}, supported up to {supported}")]
    UnsupportedVersion { got: u32, supported: u32 },
    #[error(
        "intermediate dim mismatch on layer_id {layer_id}: file says {file}, runtime expects {expected}"
    )]
    IntermediateMismatch {
        layer_id: u32,
        file: usize,
        expected: usize,
    },
    #[error("duplicate layer_id {0} in thresholds file")]
    DuplicateLayer(u32),
}

/// One layer's per-channel threshold vector. `thresholds.len()` should
/// equal the model's intermediate dim; mismatch is checked at
/// [`PerChannelThresholds::verify_for_intermediate`] time, not at file
/// load (so calibration tools can build files iteratively).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerThresholds {
    pub layer_id: u32,
    pub thresholds: Vec<f32>,
}

/// Per-channel thresholds for every covered MoE layer.
///
/// Files load via [`Self::load`]; runtime lookup via [`Self::get`].
/// Layers absent from the file return `None` from `get`, letting the
/// dispatcher fall back to global-τ for them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerChannelThresholds {
    pub version: u32,
    pub model_id: String,
    pub n_intermediate: usize,
    /// How many tokens the calibration corpus contributed. Useful for
    /// catching "we calibrated on 200 tokens, the file might be noisy"
    /// at startup.
    #[serde(default)]
    pub calibration_n_tokens: u64,
    /// The target active fraction the calibration aimed at — i.e. the
    /// quantile at `1 - target_active_frac` of each channel's
    /// magnitude ratio distribution. Stored so a deployment can sanity-
    /// check it matches their expected sparsity budget.
    pub target_active_frac: f32,
    #[serde(default)]
    pub notes: String,
    pub layers: Vec<LayerThresholds>,

    /// Index built at load time: `layer_id -> position in self.layers`.
    /// `#[serde(skip)]` so it doesn't bloat the file.
    #[serde(skip)]
    by_layer: HashMap<u32, usize>,
}

impl PerChannelThresholds {
    /// Build a fresh in-memory thresholds object (used by the
    /// calibration tool before writing to disk). Caller must invoke
    /// [`Self::rebuild_index`] (or [`Self::save`], which does it for
    /// you) before [`Self::get`] is meaningful.
    pub fn new(
        model_id: impl Into<String>,
        n_intermediate: usize,
        target_active_frac: f32,
    ) -> Self {
        Self {
            version: THRESHOLDS_FILE_VERSION,
            model_id: model_id.into(),
            n_intermediate,
            calibration_n_tokens: 0,
            target_active_frac,
            notes: String::new(),
            layers: Vec::new(),
            by_layer: HashMap::new(),
        }
    }

    /// Add (or replace) a layer's thresholds. `thresholds.len()` must
    /// equal `self.n_intermediate`.
    pub fn upsert_layer(&mut self, layer_id: u32, thresholds: Vec<f32>) {
        assert_eq!(
            thresholds.len(),
            self.n_intermediate,
            "layer {layer_id}: thresholds.len() {} != n_intermediate {}",
            thresholds.len(),
            self.n_intermediate,
        );
        if let Some(&idx) = self.by_layer.get(&layer_id) {
            self.layers[idx].thresholds = thresholds;
        } else {
            let idx = self.layers.len();
            self.layers.push(LayerThresholds {
                layer_id,
                thresholds,
            });
            self.by_layer.insert(layer_id, idx);
        }
    }

    /// Number of layers covered by this file.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Per-channel thresholds for one layer, or `None` if not covered.
    pub fn get(&self, layer_id: u32) -> Option<&[f32]> {
        self.by_layer
            .get(&layer_id)
            .map(|&idx| self.layers[idx].thresholds.as_slice())
    }

    /// Verify every loaded layer matches the runtime's intermediate
    /// dim. Returns the first mismatch (if any) — call once at startup
    /// after loading from disk and before threading into a runner.
    pub fn verify_for_intermediate(&self, expected: usize) -> Result<(), ThresholdsError> {
        if self.n_intermediate != expected {
            return Err(ThresholdsError::IntermediateMismatch {
                layer_id: u32::MAX,
                file: self.n_intermediate,
                expected,
            });
        }
        for layer in &self.layers {
            if layer.thresholds.len() != expected {
                return Err(ThresholdsError::IntermediateMismatch {
                    layer_id: layer.layer_id,
                    file: layer.thresholds.len(),
                    expected,
                });
            }
        }
        Ok(())
    }

    /// Build the `by_layer` index after deserialisation (or after
    /// bulk-mutating `self.layers`). Rejects duplicate layer ids.
    pub fn rebuild_index(&mut self) -> Result<(), ThresholdsError> {
        self.by_layer.clear();
        self.by_layer.reserve(self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            if self.by_layer.insert(layer.layer_id, idx).is_some() {
                return Err(ThresholdsError::DuplicateLayer(layer.layer_id));
            }
        }
        Ok(())
    }

    /// Read + parse + index. Rejects unsupported `version` and
    /// duplicate layer ids early.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ThresholdsError> {
        let bytes = std::fs::read(path.as_ref())?;
        let mut t: Self = serde_json::from_slice(&bytes)?;
        if t.version > THRESHOLDS_FILE_VERSION {
            return Err(ThresholdsError::UnsupportedVersion {
                got: t.version,
                supported: THRESHOLDS_FILE_VERSION,
            });
        }
        t.rebuild_index()?;
        Ok(t)
    }

    /// Atomically write to `path`: serialise → write to a sibling
    /// `<path>.tmp` → rename. Caller-supplied path becomes the final
    /// file on success; the .tmp is removed on failure inside this
    /// function (best-effort).
    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<(), ThresholdsError> {
        self.rebuild_index()?;
        let path = path.as_ref();
        let tmp = match path.file_name() {
            Some(name) => {
                let mut t = name.to_owned();
                t.push(".tmp");
                path.with_file_name(t)
            }
            None => {
                return Err(ThresholdsError::Io(std::io::Error::other(
                    "path has no file name",
                )))
            }
        };
        let bytes = serde_json::to_vec_pretty(self)?;
        // Write + fsync the temp, then rename. fsync of the parent
        // directory isn't strictly required for the rename to be
        // visible — the rename is the durability barrier — but on
        // Linux/macOS we sync the file itself so the contents are on
        // disk before the rename publishes them.
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&tmp).map_err(|e| {
                ThresholdsError::Io(std::io::Error::new(
                    e.kind(),
                    format!("create {}: {e}", tmp.display()),
                ))
            })?;
            f.write_all(&bytes)?;
            f.sync_all().ok();
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(ThresholdsError::Io(std::io::Error::new(
                e.kind(),
                format!("rename {} -> {}: {e}", tmp.display(), path.display()),
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Round-trip: upsert two layers, save, load — get same data back.
    #[test]
    fn save_load_round_trip() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("th.json");
        let mut t = PerChannelThresholds::new("test-model", 4, 0.5);
        t.upsert_layer(0, vec![0.1, 0.2, 0.3, 0.4]);
        t.upsert_layer(5, vec![0.5, 0.6, 0.7, 0.8]);
        t.notes = "round-trip test".into();
        t.calibration_n_tokens = 1234;
        t.save(&path).expect("save");

        let loaded = PerChannelThresholds::load(&path).expect("load");
        assert_eq!(loaded.version, THRESHOLDS_FILE_VERSION);
        assert_eq!(loaded.model_id, "test-model");
        assert_eq!(loaded.n_intermediate, 4);
        assert_eq!(loaded.target_active_frac, 0.5);
        assert_eq!(loaded.notes, "round-trip test");
        assert_eq!(loaded.calibration_n_tokens, 1234);
        assert_eq!(loaded.n_layers(), 2);
        assert_eq!(loaded.get(0), Some(&[0.1f32, 0.2, 0.3, 0.4][..]));
        assert_eq!(loaded.get(5), Some(&[0.5f32, 0.6, 0.7, 0.8][..]));
        assert_eq!(loaded.get(1), None, "uncovered layer returns None");
    }

    /// upsert with the same layer_id replaces in-place rather than
    /// duplicating.
    #[test]
    fn upsert_replaces_existing_layer() {
        let mut t = PerChannelThresholds::new("m", 2, 0.5);
        t.upsert_layer(7, vec![0.1, 0.2]);
        t.upsert_layer(7, vec![0.9, 0.8]);
        assert_eq!(t.n_layers(), 1);
        assert_eq!(t.get(7), Some(&[0.9f32, 0.8][..]));
    }

    /// Loading a file with mismatched intermediate dim fails
    /// `verify_for_intermediate`. The load itself succeeds — the
    /// caller decides what mismatches mean (could be a debug build
    /// pointing at a prod calibration file).
    #[test]
    fn verify_rejects_intermediate_mismatch() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("th.json");
        let mut t = PerChannelThresholds::new("m", 4, 0.5);
        t.upsert_layer(0, vec![0.1, 0.2, 0.3, 0.4]);
        t.save(&path).expect("save");

        let loaded = PerChannelThresholds::load(&path).expect("load");
        let err = loaded.verify_for_intermediate(8).expect_err("mismatch");
        match err {
            ThresholdsError::IntermediateMismatch { file, expected, .. } => {
                assert_eq!(file, 4);
                assert_eq!(expected, 8);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    /// Future schema version is rejected on load.
    #[test]
    fn rejects_future_version() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("th.json");
        let json = serde_json::json!({
            "version": 999,
            "model_id": "m",
            "n_intermediate": 4,
            "target_active_frac": 0.5,
            "layers": [],
        });
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let err = PerChannelThresholds::load(&path).expect_err("should reject");
        match err {
            ThresholdsError::UnsupportedVersion { got, supported } => {
                assert_eq!(got, 999);
                assert_eq!(supported, THRESHOLDS_FILE_VERSION);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    /// Duplicate layer ids in the file are caught at index-build time.
    #[test]
    fn rejects_duplicate_layer_ids() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("th.json");
        let json = serde_json::json!({
            "version": 1,
            "model_id": "m",
            "n_intermediate": 2,
            "target_active_frac": 0.5,
            "layers": [
                {"layer_id": 3, "thresholds": [0.1, 0.2]},
                {"layer_id": 3, "thresholds": [0.3, 0.4]},
            ],
        });
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let err = PerChannelThresholds::load(&path).expect_err("duplicate");
        match err {
            ThresholdsError::DuplicateLayer(lid) => assert_eq!(lid, 3),
            other => panic!("unexpected error: {other}"),
        }
    }
}
