//! Optional OpenVINO int4/fp16 expert backend for glm5 (iGPU / NPU / CPU).
//!
//! Runs each routed / shared SwiGLU expert as a compiled OV IR instead of the
//! Rust int4 mmap kernel. This is the counterpart of [`crate::dsv4::ov_expert`]
//! for the glm shell — glm decode is NVMe-bound at 744B on a single 32 GB box
//! (the experts stream from disk), so the OV backend can only help once the
//! model is sharded across ENOUGH nodes that a rank's experts are RAM-resident.
//!
//! EXPERIMENTAL — measured to lose on the sibling engine. A per-expert OV int4
//! GEMV is faster than the Rust kernel in a single-expert microbenchmark
//! (~4.6x CPU / ~6x iGPU, batch=1, RAM-resident), but the dsv4 fleet measured
//! the END-TO-END distributed path at ~0.6 tok/s vs ~1.05 tok/s for the
//! optimized Rust kernel — *slower* and fragile (the per-expert compile cache
//! thrashes once the working set exceeds the LRU cap / RAM). Kept as an opt-in
//! option; the Rust int4 path is the default and the one to prefer.
//!
//! Enabled by `CASCADIA_GLM5_OV_EXPERTS=1`; device from
//! `CASCADIA_GLM5_OV_DEVICE` (default `GPU`). The per-expert IRs live at
//! `<model>/experts_ov/layer_NN/expert_EEE/openvino_model.xml` (and
//! `expert_shared/`), produced by `tools/glm5_expert_ov.py`. When the env is
//! unset OR the `experts_ov` dir is absent this is `None` and the engine keeps
//! the Rust int4 mmap path unchanged — entirely opt-in, no-op by default. A
//! missing / uncompilable expert IR falls back to the Rust kernel (see `run`)
//! rather than taking down a serving rank.
//!
//! The IR computes the glm SwiGLU (`down(silu(gate·x) · up·x)`, no clamp) and
//! returns the bf16-rounded expert output. The routing weight is NOT baked in —
//! the caller applies it in f32 (`out += wj · expert(x)`), matching the Rust
//! path so prefill (`forward_block`) and decode (`forward_token`) agree.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use lru::LruCache;
use tracing::warn;

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
    /// `(layer, expert)` keys whose IR is missing or won't compile. A partial /
    /// corrupt `experts_ov/` (e.g. an export killed mid-run) must NOT take down a
    /// serving rank: the first touch of such an expert falls back to the Rust
    /// kernel and records the key here so subsequent tokens skip the (failing)
    /// recompile + re-warn. Warned once per key.
    failed: Mutex<HashSet<(u32, u32)>>,
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
            failed: Mutex::new(HashSet::new()),
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

    /// Run expert `eid` of layer `lid` on `x`, returning the bf16-rounded output
    /// (`[dim]`), or `None` if this expert's IR is missing / won't compile / the
    /// device call fails — the caller then falls back to the Rust kernel. The
    /// routing weight is NOT baked in: the caller applies it in f32 (matching the
    /// glm Rust path `out += wj * expert(x)`), so prefill and decode agree.
    fn run(&self, lid: u32, eid: u32, x: &[f32]) -> Option<Vec<f32>> {
        let key = (lid, eid);
        if self.failed.lock().unwrap().contains(&key) {
            return None; // known-bad IR — skip straight to the Rust fallback
        }
        let t0 = std::time::Instant::now();
        let mut cache = self.cache.lock().expect("OV expert cache lock");
        let miss = !cache.contains(&key);
        if miss {
            let path = self.xml(lid, eid);
            let Some(p) = path.to_str() else {
                self.mark_failed(key, "non-utf8 IR path");
                return None;
            };
            match Runtime::compile(p, &self.device, &self.plugin) {
                Ok(rt) => {
                    cache.put(key, rt);
                }
                Err(e) => {
                    drop(cache);
                    self.mark_failed(key, &format!("compile on {}: {e}", self.device));
                    return None;
                }
            }
        }
        let rt = cache.get_mut(&key).unwrap();
        // A device-side error (set_input/infer/output) is not necessarily
        // permanent, so fall back for this call without latching the key.
        if rt
            .set_input("x", DType::F32, &[1, 1, self.dim], f32_bytes(x))
            .is_err()
        {
            return None;
        }
        if rt.infer().is_err() {
            return None;
        }
        let (_, _, bytes) = rt.output(0).ok()?;
        let out = bytes
            .chunks_exact(4)
            .map(|c| to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        stats::record(key, miss, t0);
        Some(out)
    }

    /// Latch a permanently-bad expert key (missing / uncompilable IR) and warn
    /// once, so subsequent tokens skip the failing recompile and fall back
    /// silently to the Rust kernel.
    fn mark_failed(&self, key: (u32, u32), why: &str) {
        if self.failed.lock().unwrap().insert(key) {
            warn!(
                layer = key.0,
                expert = key.1,
                "OV expert IR unusable ({why}); falling back to the Rust int4 kernel for this expert"
            );
        }
    }

    /// Routed expert `eid` on `x` (bf16 output; caller applies the gate weight),
    /// or `None` to fall back to the Rust kernel.
    pub fn routed(&self, lid: usize, eid: usize, x: &[f32]) -> Option<Vec<f32>> {
        self.run(lid as u32, eid as u32, x)
    }

    /// The always-on shared expert on `x`, or `None` to fall back to the Rust
    /// kernel.
    pub fn shared(&self, lid: usize, x: &[f32]) -> Option<Vec<f32>> {
        self.run(lid as u32, SHARED, x)
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

    /// Minimum gap between emissions, in seconds. `dump()` is called once per
    /// decode token, and its body clones + sorts the whole access map (up to
    /// `layers * experts_per_layer` entries — 3084 on a 12-layer glm5 rank)
    /// while holding the same lock `record()` takes on EVERY expert dispatch.
    /// Unthrottled that is O(n log n) plus a stderr write per token, contending
    /// with the hot path: an instrumented 9-node run produced 0 tokens on all
    /// three timed passes with mid-stream transport resets. Counters stay exact
    /// either way — only the reporting cadence is bounded.
    /// Override with `CASCADIA_GLM5_OV_STATS_EVERY_SECS`.
    fn dump_interval_secs() -> u64 {
        std::env::var("CASCADIA_GLM5_OV_STATS_EVERY_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(10)
    }

    /// Monotonic seconds of the last emission; 0 = never emitted.
    static LAST_DUMP_S: AtomicU64 = AtomicU64::new(0);

    /// Process start, so we have a monotonic clock without pulling in Instant
    /// statics. `OnceLock<Instant>` keeps this allocation-free after first use.
    fn since_start_secs() -> u64 {
        static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_secs()
    }

    /// Emit the cumulative split to stderr (no-op when disabled / no calls).
    ///
    /// Rate-limited — see [`dump_interval_secs`]. Call [`dump_now`] for the
    /// final, unconditional emission at shutdown.
    pub fn dump() {
        if !enabled() {
            return;
        }
        let now = since_start_secs();
        let last = LAST_DUMP_S.load(Ordering::Relaxed);
        // `last == 0 && now == 0` is the first token of the run: skip, so the
        // first emission carries real data rather than a single call.
        if now.saturating_sub(last) < dump_interval_secs() {
            return;
        }
        // Race here just means two threads both emit once; harmless, and far
        // cheaper than holding a lock across the check.
        LAST_DUMP_S.store(now, Ordering::Relaxed);
        dump_now();
    }

    /// Emit unconditionally, ignoring the rate limit.
    pub fn dump_now() {
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
