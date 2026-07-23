//! Optional OpenVINO int4/fp16 expert backend for glm5 (iGPU / NPU / CPU).
//!
//! Runs each routed / shared SwiGLU expert as a compiled OV IR instead of the
//! Rust int4 mmap kernel. This is the counterpart of [`crate::dsv4::ov_expert`]
//! for the glm shell — glm decode is NVMe-bound at 744B on a single 32 GB box
//! (the experts stream from disk), so the OV backend only pays off once the
//! model is sharded across ENOUGH nodes that a rank's experts are RAM-resident.
//! In that regime OV's int4 GEMV is markedly faster than the scalar-dequant Rust
//! path (dsv4 measured ~4.6x on CPU, ~6x on the iGPU, batch=1, RAM-resident).
//!
//! Enabled by `CASCADIA_GLM5_OV_EXPERTS=1`; device from
//! `CASCADIA_GLM5_OV_DEVICE` (default `GPU`). The per-expert IRs live at
//! `<model>/experts_ov/layer_NN/expert_EEE/openvino_model.xml` (and
//! `expert_shared/`), produced by `tools/glm5_expert_ov.py`. When the env is
//! unset OR the `experts_ov` dir is absent this is `None` and the engine keeps
//! the Rust int4 mmap path unchanged — entirely opt-in, no-op by default.
//!
//! The IR bakes the glm SwiGLU (`down(silu(gate·x) · up·x)`, no clamp). The
//! routing weight is applied to the output (down is linear, so scaling the
//! output equals scaling the intermediate) with the same final bf16 rounding as
//! the Rust `AnyExpert::forward` boundary.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use lru::LruCache;

use crate::dsv4::math::to_bf16;

/// Sentinel expert id for the always-on shared expert.
const SHARED: u32 = u32::MAX;

pub struct OvExperts {
    dir: PathBuf, // <model>/experts_ov
    device: String,
    plugin: PluginConfig,
    dim: usize,
    /// Compiled OV models, LRU-bounded ((global layer, expert) -> Runtime).
    /// `Mutex` (not `RefCell`) so a single `OvExperts` can be shared across all
    /// MoE layers behind an `Arc` and stay `Send + Sync` (the runner is moved
    /// across threads); expert calls within a rank are sequential, so the lock
    /// is uncontended.
    cache: Mutex<LruCache<(u32, u32), Runtime>>,
}

impl OvExperts {
    /// Construct from env, or `None` to keep the Rust expert path:
    /// requires `CASCADIA_GLM5_OV_EXPERTS` set and `<model>/experts_ov` present.
    pub fn from_env(model_dir: &Path, dim: usize) -> Option<Self> {
        if std::env::var("CASCADIA_GLM5_OV_EXPERTS").is_err() {
            return None;
        }
        let dir = model_dir.join("experts_ov");
        if !dir.is_dir() {
            return None;
        }
        let device = std::env::var("CASCADIA_GLM5_OV_DEVICE").unwrap_or_else(|_| "GPU".into());
        let cap = std::env::var("CASCADIA_GLM5_OV_CACHE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| NonZeroUsize::new(1024).unwrap());
        let mut plugin = PluginConfig::new();
        // Persist compiled blobs across runs so first-touch compile isn't
        // re-paid every process start (GPU kernel JIT is expensive).
        if let Ok(cd) = std::env::var("CASCADIA_GLM5_OV_CACHE_DIR") {
            plugin = plugin.with("CACHE_DIR", cd);
        }
        Some(Self {
            dir,
            device,
            plugin,
            dim,
            cache: Mutex::new(LruCache::new(cap)),
        })
    }

    fn xml(&self, lid: u32, eid: u32) -> PathBuf {
        let name = if eid == SHARED {
            "expert_shared".to_string()
        } else {
            format!("expert_{eid:03}")
        };
        self.dir
            .join(format!("layer_{lid:02}"))
            .join(name)
            .join("openvino_model.xml")
    }

    fn run(&self, lid: u32, eid: u32, x: &[f32], route_w: Option<f32>) -> Vec<f32> {
        let key = (lid, eid);
        let t0 = std::time::Instant::now();
        let mut cache = self.cache.lock().expect("OV expert cache lock");
        let miss = !cache.contains(&key);
        if miss {
            let path = self.xml(lid, eid);
            let p = path.to_str().expect("utf-8 expert path");
            let rt = Runtime::compile(p, &self.device, &self.plugin)
                .unwrap_or_else(|e| panic!("compile OV expert {p} on {}: {e}", self.device));
            cache.put(key, rt);
        }
        let rt = cache.get_mut(&key).unwrap();
        rt.set_input("x", DType::F32, &[1, 1, self.dim], f32_bytes(x))
            .expect("OV expert set_input");
        rt.infer().expect("OV expert infer");
        let (_, _, bytes) = rt.output(0).expect("OV expert output");
        let w = route_w.unwrap_or(1.0);
        let out = bytes
            .chunks_exact(4)
            .map(|c| to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * w))
            .collect();
        stats::record(key, miss, t0);
        out
    }

    /// Routed expert `eid` on `x`, scaled by its routing weight `route_w`.
    pub fn routed(&self, lid: usize, eid: usize, x: &[f32], route_w: f32) -> Vec<f32> {
        self.run(lid as u32, eid as u32, x, Some(route_w))
    }

    /// The always-on shared expert on `x` (unscaled).
    pub fn shared(&self, lid: usize, x: &[f32]) -> Vec<f32> {
        self.run(lid as u32, SHARED, x, None)
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Opt-in expert-cache instrumentation, gated by `CASCADIA_GLM5_OV_STATS`.
/// Splits hit-path vs miss-path (LRU eviction -> `Runtime::compile`) latency,
/// counts the miss rate, and tracks the per-`(layer, expert)` access histogram
/// so the working-set size vs cache capacity can be read directly. No-op when
/// unset. Mirrors `dsv4::ov_expert::stats`.
pub mod stats {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    static HITS: AtomicU64 = AtomicU64::new(0);
    static MISSES: AtomicU64 = AtomicU64::new(0);
    static HIT_NS: AtomicU64 = AtomicU64::new(0);
    static MISS_NS: AtomicU64 = AtomicU64::new(0);
    static ON: OnceLock<bool> = OnceLock::new();

    fn access() -> &'static Mutex<HashMap<(u32, u32), u64>> {
        static M: OnceLock<Mutex<HashMap<(u32, u32), u64>>> = OnceLock::new();
        M.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn enabled() -> bool {
        *ON.get_or_init(|| std::env::var("CASCADIA_GLM5_OV_STATS").is_ok())
    }

    /// Record one expert call: `miss` = the key was absent (compile path),
    /// `since` = when the call started. `since.elapsed()` is read before the
    /// histogram lock so the lock never inflates the hit/miss latency.
    pub fn record(key: (u32, u32), miss: bool, since: Instant) {
        if !enabled() {
            return;
        }
        let ns = since.elapsed().as_nanos() as u64;
        if miss {
            MISSES.fetch_add(1, Ordering::Relaxed);
            MISS_NS.fetch_add(ns, Ordering::Relaxed);
        } else {
            HITS.fetch_add(1, Ordering::Relaxed);
            HIT_NS.fetch_add(ns, Ordering::Relaxed);
        }
        *access().lock().unwrap().entry(key).or_insert(0) += 1;
    }

    /// Emit the cumulative split to stderr (no-op when disabled / no calls).
    pub fn dump() {
        if !enabled() {
            return;
        }
        let h = HITS.load(Ordering::Relaxed);
        let m = MISSES.load(Ordering::Relaxed);
        let calls = h + m;
        if calls == 0 {
            return;
        }
        let hit_ms = HIT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let miss_ms = MISS_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let map = access().lock().unwrap();
        let unique = map.len();
        let mut hot: Vec<((u32, u32), u64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        hot.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<String> = hot
            .iter()
            .take(6)
            .map(|(k, n)| format!("l{}e{}:{}", k.0, k.1, n))
            .collect();
        eprintln!(
            "GLM5_OVSTATS calls={calls} miss={m} miss_rate={:.1}% hit_avg_ms={:.3} miss_avg_ms={:.2} unique_keys={unique} hot=[{}]",
            100.0 * m as f64 / calls as f64,
            if h > 0 { hit_ms / h as f64 } else { 0.0 },
            if m > 0 { miss_ms / m as f64 } else { 0.0 },
            top.join(","),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The OV expert backend is strictly opt-in: with `CASCADIA_GLM5_OV_EXPERTS`
    /// unset, `from_env` returns `None` regardless of the dir, so the engine keeps
    /// the Rust int4 path. Guards against the backend ever defaulting on. (No test
    /// sets that env var, so this read-only check is stable under parallelism.)
    #[test]
    fn from_env_is_none_when_disabled() {
        assert!(
            std::env::var("CASCADIA_GLM5_OV_EXPERTS").is_err(),
            "test assumes the OV env is unset"
        );
        assert!(
            OvExperts::from_env(std::path::Path::new("/nonexistent/glm5"), 4096).is_none(),
            "OV experts must be off unless CASCADIA_GLM5_OV_EXPERTS is set"
        );
    }
}
