//! Optional OpenVINO expert backend for Inkling (iGPU / NPU / CPU).
//!
//! Runs each routed / shared expert — and the two dense layers' MLP — as a
//! compiled OV IR instead of the Rust int4 mmap kernel, so a box whose
//! CPU has no wide SIMD (Panther Lake: AVX2 only) can put the expert GEMVs on
//! its Xe3 iGPU. The Inkling counterpart of [`crate::glm::ov_expert`]; same
//! per-expert IR design, same fallback contract, Inkling's naming:
//!
//! * `<model>/experts_ov/layer_NN/expert_EEE/openvino_model.xml` — routed
//!   expert `EEE`;
//! * `<model>/experts_ov/layer_NN/expert_sharedS/` — shared expert `S`
//!   (dispatched with expert id `num_experts + S`, like the Rust path);
//! * `<model>/experts_ov/layer_NN/dense/` — the dense MLP of layers 0–1.
//!
//! Produced by `tools/inkling_expert_ov.py` from the int4 bins' own nibbles
//! and bf16 group scales (no re-quantisation; the IR sits on the exact grid
//! the Rust kernel reads). The IR computes `down(silu(gate·x) · up·x)` in
//! f32 and the runtime rounds the output to bf16; the Rust kernel also rounds
//! gate/up to bf16 between its GEMVs, which the graph cannot reproduce (OV's
//! `Convert` to bf16 truncates), so the two paths differ by that unbiased
//! half-ULP inner rounding plus f32 accumulation order — measured on the
//! B390: relative rms 1.5e-6 per expert at f32, 99.9% of bf16 outputs
//! identical. The routing weight (or the dense `global_scale`) is applied by
//! the caller in f32, as on the Rust path.
//!
//! Enabled by `CASCADIA_INKLING_OV_EXPERTS=1`; device from
//! `CASCADIA_INKLING_OV_DEVICE` (default `GPU`); compiled-model cache bounded
//! by `CASCADIA_INKLING_OV_CACHE` entries and `CASCADIA_INKLING_OV_CACHE_MB`
//! of estimated device bytes (default 64 entries / 2048 MiB — a resident
//! benchmark of a few layers wants far more, e.g. 600 entries / 24000 MiB);
//! `CASCADIA_INKLING_OV_CACHE_DIR` persists compiled blobs across runs;
//! `CASCADIA_INKLING_OV_PRECISION` (default `f32`) and
//! `CASCADIA_INKLING_OV_DQ_GROUP` (default `0`) pin the plugin numerics to
//! the bins' grid — `f16` / a group size is the iGPU's faster, inexact mode.
//! When the env is unset or `experts_ov/` is absent this is `None` and every
//! layer keeps the Rust kernel. A missing / uncompilable IR falls back to the
//! Rust kernel per key; GPU resource exhaustion disables the backend for the
//! process rather than taking the rank down.
//!
//! Expert calls come from rayon workers concurrently (the layer runs a
//! token's experts in parallel), so the LRU lock is held only to look an
//! entry up; each compiled model carries its own lock for the infer.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use lru::LruCache;
use tracing::warn;

use crate::dsv4::math::to_bf16;

/// Expert-id namespace: routed ids are `0..num_experts`, shared experts are
/// `num_experts + s` (the Rust path's ids), and the dense MLP is this
/// sentinel.
pub const DENSE: u32 = u32::MAX;

/// Estimated device bytes one cached expert holds: weights + request buffers
/// ≈ IR bin size × 1.6 (measured on the glm5 backend: ~30 MiB held per
/// ~19 MiB int4 bin), floored at 8 MiB.
fn expert_cost_bytes(xml: &Path) -> u64 {
    let bin = xml.with_extension("bin");
    let len = std::fs::metadata(bin).map(|m| m.len()).unwrap_or(0);
    (len.saturating_mul(8) / 5).max(8 * 1024 * 1024)
}

fn is_fatal_resource_error(msg: &str) -> bool {
    msg.contains("resource unavailable")
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

type Entry = (Arc<Mutex<Runtime>>, u64);

/// Count-capped LRU plus a running estimate of held device bytes, evicted to
/// `budget` on insert. The newest entry always stays, even alone over budget.
struct OvCache {
    lru: LruCache<(u32, u32), Entry>,
    bytes: u64,
    budget: u64,
}

impl OvCache {
    fn insert(&mut self, key: (u32, u32), rt: Runtime, cost: u64) -> Arc<Mutex<Runtime>> {
        let rt = Arc::new(Mutex::new(rt));
        if let Some((_, evicted)) = self.lru.push(key, (Arc::clone(&rt), cost)) {
            self.bytes = self.bytes.saturating_sub(evicted.1);
        }
        self.bytes = self.bytes.saturating_add(cost);
        while self.bytes > self.budget && self.lru.len() > 1 {
            if let Some((_, evicted)) = self.lru.pop_lru() {
                self.bytes = self.bytes.saturating_sub(evicted.1);
            }
        }
        rt
    }

    fn clear(&mut self) {
        self.lru.clear();
        self.bytes = 0;
    }
}

/// Per-process counters for the benchmark read-out: how many expert calls
/// hit a compiled model, how many compiled first, and the time each side
/// took. Reported by [`OvExperts::stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OvStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_ns: u64,
    pub miss_ns: u64,
    pub fallbacks: u64,
}

pub struct OvExperts {
    dir: PathBuf, // <model>/experts_ov
    device: String,
    plugin: PluginConfig,
    dim: usize,
    cache: Mutex<OvCache>,
    /// `(layer, expert)` keys whose IR is missing or won't compile: the first
    /// touch falls back to the Rust kernel and later touches skip the failing
    /// recompile. Warned once per key.
    failed: Mutex<HashSet<(u32, u32)>>,
    /// Latched on GPU resource exhaustion; every call then returns `None`.
    poisoned: AtomicBool,
    hits: AtomicU64,
    misses: AtomicU64,
    hit_ns: AtomicU64,
    miss_ns: AtomicU64,
    fallbacks: AtomicU64,
}

impl OvExperts {
    /// Construct from the environment, or `None` to keep the Rust path:
    /// requires `CASCADIA_INKLING_OV_EXPERTS` set and `<model>/experts_ov`
    /// present.
    pub fn from_env(model_dir: &Path, dim: usize) -> Option<Self> {
        if !super::env_flag("CASCADIA_INKLING_OV_EXPERTS") {
            return None;
        }
        let dir = model_dir.join("experts_ov");
        if !dir.is_dir() {
            warn!(
                dir = %dir.display(),
                "CASCADIA_INKLING_OV_EXPERTS set but the model has no experts_ov/ dir \
                 (tools/inkling_expert_ov.py); keeping the Rust int4 kernel"
            );
            return None;
        }
        let device = std::env::var("CASCADIA_INKLING_OV_DEVICE").unwrap_or_else(|_| "GPU".into());
        let entries = std::env::var("CASCADIA_INKLING_OV_CACHE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64);
        let budget_mb = std::env::var("CASCADIA_INKLING_OV_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048);
        let cache_dir = std::env::var("CASCADIA_INKLING_OV_CACHE_DIR").ok();
        // Numerics: f32 inference and no dynamic activation quantisation keep
        // the IR on the bins' grid (the plugins' defaults — f16 on the iGPU,
        // int8 activations for int4-weight matmuls — cost ~1e-2 relative
        // error per expert; measured on the B390). `f16` + a group size is
        // the fast path, to be benchmarked as a separate, non-exact mode.
        let precision =
            std::env::var("CASCADIA_INKLING_OV_PRECISION").unwrap_or_else(|_| "f32".into());
        let dq_group = std::env::var("CASCADIA_INKLING_OV_DQ_GROUP").unwrap_or_else(|_| "0".into());
        let ov = Self::from_dir(
            dir,
            device,
            dim,
            entries,
            budget_mb,
            cache_dir.as_deref(),
            &[
                ("INFERENCE_PRECISION_HINT".into(), precision.clone()),
                ("DYNAMIC_QUANTIZATION_GROUP_SIZE".into(), dq_group.clone()),
            ],
        );
        tracing::info!(
            target: "cascadia::inkling",
            event = "ov_experts_config",
            device = %ov.device,
            precision = %precision,
            dq_group = %dq_group,
            cache_entries = entries,
            cache_budget_mb = budget_mb,
            cache_dir = cache_dir.as_deref().unwrap_or("<unset>"),
            dir = %ov.dir.display(),
        );
        Some(ov)
    }

    /// Explicit constructor (in-process hosts and tests): `dir` is the
    /// `experts_ov/` directory itself.
    pub fn from_dir(
        dir: PathBuf,
        device: String,
        dim: usize,
        cache_entries: usize,
        cache_budget_mb: u64,
        cache_dir: Option<&str>,
        plugin_entries: &[(String, String)],
    ) -> Self {
        let mut plugin = PluginConfig::new();
        if let Some(cd) = cache_dir {
            plugin = plugin.with("CACHE_DIR", cd);
        }
        for (k, v) in plugin_entries {
            plugin = plugin.with(k.clone(), v.clone());
        }
        let cap = NonZeroUsize::new(cache_entries.max(1)).unwrap();
        Self {
            dir,
            device,
            plugin,
            dim,
            cache: Mutex::new(OvCache {
                lru: LruCache::new(cap),
                bytes: 0,
                budget: cache_budget_mb.saturating_mul(1024 * 1024),
            }),
            failed: Mutex::new(HashSet::new()),
            poisoned: AtomicBool::new(false),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            hit_ns: AtomicU64::new(0),
            miss_ns: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
        }
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// True when experts run on a device that shares its allocation pool with
    /// host RAM (iGPU / NPU): residency pinning and offload then compete for
    /// the same memory.
    pub fn on_accelerator(&self) -> bool {
        !self.device.eq_ignore_ascii_case("CPU")
    }

    pub fn stats(&self) -> OvStats {
        OvStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            hit_ns: self.hit_ns.load(Ordering::Relaxed),
            miss_ns: self.miss_ns.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
        }
    }

    /// Number of compiled models currently held and their estimated device
    /// bytes.
    pub fn cached(&self) -> (usize, u64) {
        let c = self.cache.lock().expect("OV expert cache lock");
        (c.lru.len(), c.bytes)
    }

    /// Whether the model ships IRs for layer `lid` (its `experts_ov/layer_NN`
    /// directory exists); the loader attaches the backend only to those.
    pub fn has_layer(&self, lid: u32) -> bool {
        self.dir.join(format!("layer_{lid:02}")).is_dir()
    }

    fn xml(&self, lid: u32, eid: u32, num_experts: u32) -> PathBuf {
        let name = if eid == DENSE {
            "dense".to_string()
        } else if eid >= num_experts {
            format!("expert_shared{}", eid - num_experts)
        } else {
            format!("expert_{eid:03}")
        };
        self.dir
            .join(format!("layer_{lid:02}"))
            .join(name)
            .join("openvino_model.xml")
    }

    /// Compiled model for `key`, compiling on a miss, or `None` when the IR is
    /// unusable (recorded) or the backend is poisoned.
    fn compiled(&self, key: (u32, u32), num_experts: u32) -> Option<(Arc<Mutex<Runtime>>, bool)> {
        if self.poisoned.load(Ordering::Relaxed) {
            return None;
        }
        if self.failed.lock().unwrap().contains(&key) {
            return None;
        }
        let mut cache = self.cache.lock().expect("OV expert cache lock");
        if let Some((rt, _)) = cache.lru.get(&key) {
            return Some((Arc::clone(rt), false));
        }
        let path = self.xml(key.0, key.1, num_experts);
        let Some(p) = path.to_str() else {
            drop(cache);
            self.mark_failed(key, "non-utf8 IR path");
            return None;
        };
        match Runtime::compile(p, &self.device, &self.plugin) {
            Ok(rt) => {
                let cost = expert_cost_bytes(&path);
                Some((cache.insert(key, rt, cost), true))
            }
            Err(e) => {
                let fatal = cascadia_ov_genai_shim::last_error_resource_exhausted();
                drop(cache);
                let msg = format!("compile on {}: {e}", self.device);
                if fatal || is_fatal_resource_error(&msg) {
                    self.poison(&msg);
                } else {
                    self.mark_failed(key, &msg);
                }
                None
            }
        }
    }

    /// Run expert `eid` of layer `lid` on `x`, returning the bf16-rounded
    /// output (`[dim]`), or `None` when this call must take the Rust kernel.
    fn run(&self, lid: u32, eid: u32, num_experts: u32, x: &[f32]) -> Option<Vec<f32>> {
        let key = (lid, eid);
        let t0 = Instant::now();
        let Some((rt, miss)) = self.compiled(key, num_experts) else {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let out = {
            let mut rt = rt.lock().expect("OV expert runtime lock");
            // A device-side error is not necessarily permanent: fall back for
            // this call without latching the key.
            if rt
                .set_input("x", DType::F32, &[1, 1, self.dim], f32_bytes(x))
                .is_err()
                || rt.infer().is_err()
            {
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let (_, _, bytes) = rt.output(0).ok()?;
            bytes
                .chunks_exact(4)
                .map(|c| to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect::<Vec<f32>>()
        };
        if out.len() != self.dim {
            // An IR built for other dims infers cleanly; a short vector would
            // silently truncate the caller's accumulation.
            self.mark_failed(
                key,
                &format!("output len {} != dim {}", out.len(), self.dim),
            );
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let ns = t0.elapsed().as_nanos() as u64;
        if miss {
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.miss_ns.fetch_add(ns, Ordering::Relaxed);
        } else {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.hit_ns.fetch_add(ns, Ordering::Relaxed);
        }
        Some(out)
    }

    /// Routed or shared expert `eid` (shared = `num_experts + s`) of layer
    /// `lid` on `x`; the caller applies the gate weight.
    pub fn expert(&self, lid: u32, eid: u32, num_experts: u32, x: &[f32]) -> Option<Vec<f32>> {
        self.run(lid, eid, num_experts, x)
    }

    /// The dense MLP of layer `lid` on `x`; the caller applies `global_scale`.
    pub fn dense(&self, lid: u32, x: &[f32]) -> Option<Vec<f32>> {
        self.run(lid, DENSE, 0, x)
    }

    /// Compile (or load from the blob cache) every listed key so a benchmark's
    /// timed region starts with the experts resident on the device. Returns
    /// the keys that could not be compiled.
    pub fn warm(&self, keys: &[(u32, u32)], num_experts: u32) -> Vec<(u32, u32)> {
        let mut bad = Vec::new();
        for &key in keys {
            if self.compiled(key, num_experts).is_none() {
                bad.push(key);
            }
        }
        bad
    }

    fn poison(&self, why: &str) {
        if !self.poisoned.swap(true, Ordering::Relaxed) {
            tracing::error!(
                device = %self.device,
                "inkling OV expert offload disabled: resource exhaustion ({why}); \
                 all experts now on the Rust int4 kernel"
            );
            self.cache.lock().expect("OV expert cache lock").clear();
        }
    }

    fn mark_failed(&self, key: (u32, u32), why: &str) {
        if self.failed.lock().unwrap().insert(key) {
            warn!(
                layer = key.0,
                expert = key.1,
                "inkling OV expert IR unusable ({why}); Rust int4 kernel for this expert"
            );
        }
    }

    /// Keys known unusable (for tests and the benchmark read-out).
    pub fn failed_keys(&self) -> HashMap<u32, Vec<u32>> {
        let mut m: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(l, e) in self.failed.lock().unwrap().iter() {
            m.entry(l).or_default().push(e);
        }
        for v in m.values_mut() {
            v.sort_unstable();
        }
        m
    }
}

impl std::fmt::Debug for OvExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OvExperts")
            .field("dir", &self.dir)
            .field("device", &self.device)
            .field("dim", &self.dim)
            .finish()
    }
}
