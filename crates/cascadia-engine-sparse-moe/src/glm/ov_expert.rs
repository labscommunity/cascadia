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

/// Process-level GPU exhaustion (`resource unavailable try again`), as opposed
/// to one bad IR. Only this latches [`OvExperts::poison`]: `CL_INVALID_EVENT`
/// can be a one-off under momentary pressure and stays per-key. String fallback
/// for the typed `last_error_resource_exhausted` check (stub builds, old shims).
fn is_fatal_resource_error(msg: &str) -> bool {
    msg.contains("resource unavailable")
}

/// Estimated iGPU device bytes one cached expert holds: weights + request
/// buffers ≈ IR bin size × 1.6 (measured ~30 MiB held per ~19 MiB int4 bin),
/// floored at 8 MiB for tiny/missing bins.
fn expert_cost_bytes(xml: &Path) -> u64 {
    let bin = xml.with_extension("bin");
    let len = std::fs::metadata(bin).map(|m| m.len()).unwrap_or(0);
    (len.saturating_mul(8) / 5).max(8 * 1024 * 1024)
}

/// Count-capped LRU plus a running estimate of held device bytes, evicted to
/// `budget` on insert. Counts alone don't bound memory: entry size follows the
/// model's expert dims, and 1024 × ~30 MiB is what exhausted the iGPU pool.
struct OvCache {
    lru: LruCache<(u32, u32), (Runtime, u64)>,
    bytes: u64,
    budget: u64,
}

impl OvCache {
    /// Insert and evict LRU entries until back under budget. The newest entry
    /// always stays, even alone over budget — the alternative is compiling it
    /// every call.
    fn insert(&mut self, key: (u32, u32), rt: Runtime, cost: u64) {
        if let Some((_, evicted)) = self.lru.push(key, (rt, cost)) {
            self.bytes = self.bytes.saturating_sub(evicted.1);
        }
        self.bytes = self.bytes.saturating_add(cost);
        while self.bytes > self.budget && self.lru.len() > 1 {
            if let Some((_, (_, c))) = self.lru.pop_lru() {
                self.bytes = self.bytes.saturating_sub(c);
            } else {
                break;
            }
        }
    }

    fn clear(&mut self) {
        self.lru.clear();
        self.bytes = 0;
    }
}

pub struct OvExperts {
    dir: PathBuf, // <model>/experts_ov
    device: String,
    plugin: PluginConfig,
    dim: usize,
    /// Compiled OV models, bounded by count AND estimated device bytes —
    /// whichever hits first. `Mutex` (not `RefCell`) so a single `OvExperts`
    /// can be shared across all MoE layers behind an `Arc` and stay
    /// `Send + Sync` (the runner is moved across threads); expert calls within
    /// a rank are sequential, so the lock is uncontended.
    cache: Mutex<OvCache>,
    /// `(layer, expert)` keys whose IR is missing or won't compile. A partial /
    /// corrupt `experts_ov/` (e.g. an export killed mid-run) must NOT take down a
    /// serving rank: the first touch of such an expert falls back to the Rust
    /// kernel and records the key here so subsequent tokens skip the (failing)
    /// recompile + re-warn. Warned once per key.
    failed: Mutex<HashSet<(u32, u32)>>,
    /// Latched on GPU resource exhaustion; `run()` then returns `None` for
    /// everything. Compiling against an exhausted driver kills the rank
    /// (observed live), so the backend stops rather than degrades per-key.
    poisoned: std::sync::atomic::AtomicBool,
}

impl OvExperts {
    /// Construct from env, or `None` to keep the Rust expert path:
    /// requires `CASCADIA_GLM5_OV_EXPERTS` set and `<model>/experts_ov` present.
    pub fn from_env(model_dir: &Path, dim: usize) -> Option<Self> {
        Self::from_env_with(model_dir, dim, None, None)
    }

    /// [`from_env`] with config-first cache overrides (entries backstop /
    /// MiB budget); `None` falls back to the env knobs. In-process hosts
    /// can't set env per engine (edition-2024 `set_var` is unsafe).
    pub fn from_env_with(
        model_dir: &Path,
        dim: usize,
        cache_entries: Option<u32>,
        cache_mb: Option<u64>,
    ) -> Option<Self> {
        if !crate::glm::env_flag("CASCADIA_GLM5_OV_EXPERTS") {
            return None;
        }
        let dir = model_dir.join("experts_ov");
        if !dir.is_dir() {
            return None;
        }
        let device = std::env::var("CASCADIA_GLM5_OV_DEVICE").unwrap_or_else(|_| "GPU".into());
        let cap = cache_entries
            .map(|n| n as usize)
            .or_else(|| {
                std::env::var("CASCADIA_GLM5_OV_CACHE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .and_then(NonZeroUsize::new)
            // 64 ≈ 2 GiB: each cached entry holds ~30 MiB of iGPU driver
            // allocations (measured: 800 held experts = 24 GiB). The old 1024
            // default (~30 GiB) crowded the shared-RAM pool until fresh builds
            // failed (CL_INVALID_EVENT) and eventually killed the rank. Hit
            // rate loses nothing: the 3084-key space never fit any cap, and
            // the always-active set (12/rank) still fits.
            .unwrap_or_else(|| NonZeroUsize::new(64).unwrap());
        let mut plugin = PluginConfig::new();
        // Persist compiled blobs across runs so first-touch compile isn't
        // re-paid every process start (GPU kernel JIT is expensive).
        let cache_dir = std::env::var("CASCADIA_GLM5_OV_CACHE_DIR").ok();
        if let Some(cd) = &cache_dir {
            plugin = plugin.with("CACHE_DIR", cd.clone());
        }
        // Byte budget is the primary bound; count is the backstop.
        let budget = cache_mb
            .or_else(|| {
                std::env::var("CASCADIA_GLM5_OV_CACHE_MB")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024);
        // The effective config, verifiable from logs rather than guessed —
        // multiple fleet sessions burned time inferring these from behaviour.
        tracing::info!(
            target: "cascadia::glm5",
            event = "ov_experts_config",
            device = %device,
            cache_entries = cap.get(),
            cache_budget_mb = budget / (1024 * 1024),
            cache_dir = cache_dir.as_deref().unwrap_or("<unset>"),
            dir = %dir.display(),
        );
        Some(Self {
            dir,
            device,
            plugin,
            dim,
            cache: Mutex::new(OvCache {
                lru: LruCache::new(cap),
                bytes: 0,
                budget,
            }),
            failed: Mutex::new(HashSet::new()),
            poisoned: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// True when experts run on a device that shares the iGPU/NPU allocation
    /// pool with host RAM. Residency pinning and accelerator offload compete
    /// for the same physical memory on these — see the pin exclusion in
    /// `stage.rs`.
    pub fn on_accelerator(&self) -> bool {
        !self.device.eq_ignore_ascii_case("CPU")
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
        if self.poisoned.load(std::sync::atomic::Ordering::Relaxed) {
            return None; // driver reported exhaustion — OV offload is off for good
        }
        if self.failed.lock().unwrap().contains(&key) {
            return None; // known-bad IR — skip straight to the Rust fallback
        }
        let t0 = std::time::Instant::now();
        let mut cache = self.cache.lock().expect("OV expert cache lock");
        let miss = !cache.lru.contains(&key);
        if miss {
            let path = self.xml(lid, eid);
            let Some(p) = path.to_str() else {
                self.mark_failed(key, "non-utf8 IR path");
                return None;
            };
            match Runtime::compile(p, &self.device, &self.plugin) {
                Ok(rt) => {
                    let cost = expert_cost_bytes(&path);
                    cache.insert(key, rt, cost);
                }
                Err(e) => {
                    // Typed check first (thread-local, set by the compile that
                    // just failed); string sniff covers stub builds.
                    let fatal = cascadia_ov_genai_shim::last_error_resource_exhausted();
                    drop(cache);
                    let msg = format!("compile on {}: {e}", self.device);
                    if fatal || is_fatal_resource_error(&msg) {
                        self.poison(&msg);
                    } else {
                        self.mark_failed(key, &msg);
                    }
                    return None;
                }
            }
        }
        let rt = &mut cache.lru.get_mut(&key).unwrap().0;
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
    /// Disable OV offload for good and free the cached compiled models. Per-key
    /// `failed` is wrong here: it would keep hammering the other 3000+ keys
    /// against a driver that just said it is out of resources — one bad expert
    /// is an IR problem, refused resources are a process problem.
    fn poison(&self, why: &str) {
        use std::sync::atomic::Ordering;
        if !self.poisoned.swap(true, Ordering::Relaxed) {
            tracing::error!(
                device = %self.device,
                "OV expert offload disabled: GPU resource exhaustion ({why}); \
                 all experts now on the Rust int4 kernel"
            );
            self.cache.lock().expect("OV expert cache lock").clear();
        }
    }

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

    /// Only the terminal exhaustion signature latches the process-wide poison;
    /// CL_INVALID_EVENT and ordinary bad-IR errors stay per-key so one crowded
    /// moment or one corrupt export doesn't disable the whole backend.
    #[test]
    fn fatal_classifier_matches_only_terminal_exhaustion() {
        assert!(is_fatal_resource_error(
            "compile on GPU: openvino-genai error: Exception from plugin.cpp:54: \
             resource unavailable try again: resource unavailable try again"
        ));
        assert!(!is_fatal_resource_error(
            "compile on GPU: [GPU] clWaitForEvents, error code: -58 CL_INVALID_EVENT"
        ));
        assert!(!is_fatal_resource_error("compile on GPU: cannot open file"));
    }

    /// Cost = bin × 1.6 with an 8 MiB floor (missing bin included, so a broken
    /// IR never counts as free).
    #[test]
    fn expert_cost_scales_with_bin_and_floors() {
        let dir = tempfile::tempdir().unwrap();
        let xml = dir.path().join("openvino_model.xml");
        std::fs::write(&xml, "<net/>").unwrap();
        assert_eq!(
            expert_cost_bytes(&xml),
            8 * 1024 * 1024,
            "missing bin -> floor"
        );
        std::fs::write(
            dir.path().join("openvino_model.bin"),
            vec![0u8; 20 * 1024 * 1024],
        )
        .unwrap();
        assert_eq!(
            expert_cost_bytes(&xml),
            32 * 1024 * 1024,
            "20 MiB bin -> 32 MiB"
        );
    }

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
