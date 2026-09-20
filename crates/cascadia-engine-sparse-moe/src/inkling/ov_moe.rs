//! OpenVINO fused-MoE backend for Inkling: one compiled model per MoE layer.
//!
//! The per-expert backend ([`super::ov_expert`]) compiles 258 models per layer
//! and issues eight device calls per token. OpenVINO 2026.3's GPU plugin can
//! instead fuse a whole MoE layer into its `moe_3gemm_fused_compressed`
//! kernel: all experts as one expert-major compressed constant, a token's k
//! experts computed in one launch, rows grouped per expert for prefill. This
//! backend runs those per-layer IRs — `<model>/moe_ov/layer_NN/openvino_model.xml`,
//! produced by `tools/inkling_moe_layer_ov.py` from the bins' own nibbles and
//! scales — with the routing supplied from the Rust gate, so Inkling's routing
//! stays exact and the graph carries no router.
//!
//! Inputs per call: `x [1, rows, hidden]` f32, `topk_indices [rows, K]` i32 and
//! `routing_weights [rows, K]` f32 with `K = top_k + n_shared` (the shared
//! experts are the stack's last two ids, always selected with their gammas);
//! output `y [1, rows, hidden]` — the weighted sum, which the caller adds to
//! the residual like `MoeLayer::forward`'s.
//!
//! Enabled by `CASCADIA_INKLING_OV_MOE=1`; `CASCADIA_INKLING_OV_MOE_DEVICE`
//! (default `GPU`), `CASCADIA_INKLING_OV_MOE_CACHE_DIR` (compiled-blob cache),
//! `CASCADIA_INKLING_OV_MOE_OFFLOAD` (the plugin's `OFFLOAD_RATIO`, percent of
//! experts not pre-loaded on the device but streamed from the IR .bin into
//! LRU slots on first touch; off by default — the shim then materialises the
//! IR's constants in memory before compiling, the only form 2026.3.1 builds
//! the fused op from without offload; the streaming itself measured ~1 GB/s
//! on the Arc B390, so the offload ratio is a benchmark knob, not a speed-up).
//! Layers without an IR keep whatever path they had (per-expert OV or the
//! Rust kernel), as does any call the device refuses.
//!
//! Two plugin behaviours measured on the Arc B390 (driver 32.0.101.8860,
//! OpenVINO 2026.3.1 and the 2026.5 nightly) shape this backend: the
//! batched-GEMV decode kernel crashes the process, so the backend routes decode
//! through the grouped-GEMM path (`OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=0`, set
//! in the process environment before the first compile unless the operator set
//! it), and a single-row call still crashes there, so decode is padded to two
//! rows (the second a copy of the first with zero routing weights).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use tracing::warn;

/// Set an environment variable for this process AND the C runtime's copy of
/// the environment: the plugin reads its knobs with `getenv`, and on Windows
/// `std::env::set_var` (SetEnvironmentVariable) does not update the CRT's
/// table, so the plugin would keep the default.
fn set_process_env(name: &str, value: &str) {
    std::env::set_var(name, value);
    #[cfg(windows)]
    {
        use std::ffi::CString;
        extern "C" {
            fn _putenv_s(
                name: *const std::os::raw::c_char,
                value: *const std::os::raw::c_char,
            ) -> std::os::raw::c_int;
        }
        if let (Ok(n), Ok(v)) = (CString::new(name), CString::new(value)) {
            // SAFETY: both strings are valid NUL-terminated C strings for the call.
            unsafe {
                let _ = _putenv_s(n.as_ptr(), v.as_ptr());
            }
        }
    }
}

/// Row count a device call is padded to: the plugin pays a kernel-setup
/// cost the first time it sees a row count (~55–140 ms, ~1 s at the 32-row
/// boundary on the B390), so calls use a few fixed shapes — 2 (decode), then
/// multiples of 8 up to 32, then multiples of 32 — and the padding rows are
/// ignored on the way out.
pub(crate) fn bucket_rows(rows: usize) -> usize {
    match rows {
        0..=2 => 2,
        3..=32 => rows.div_ceil(8) * 8,
        _ => rows.div_ceil(32) * 32,
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn i32_bytes(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Per-process counters for the benchmark read-out.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OvMoeStats {
    pub calls: u64,
    pub rows: u64,
    pub call_ns: u64,
    pub compiles: u64,
    pub compile_ns: u64,
    pub fallbacks: u64,
}

pub struct OvMoe {
    dir: PathBuf, // <model>/moe_ov
    device: String,
    plugin: PluginConfig,
    hidden: usize,
    /// Experts per row the IR expects: `top_k + n_shared`.
    k_total: usize,
    /// Real experts in the stack (`num_experts + n_shared`); the IR may carry
    /// dummies after them (see `tools/inkling_moe_layer_ov.py --pad-experts`).
    n_experts: usize,
    offload: Option<String>,
    require_fused: bool,
    profiles: Mutex<HashMap<u32, String>>,
    layers: Mutex<HashMap<u32, Arc<Mutex<Runtime>>>>,
    failed: Mutex<std::collections::HashSet<u32>>,
    /// Layers whose first call failure has been reported.
    noted: Mutex<std::collections::HashSet<u32>>,
    calls: AtomicU64,
    rows: AtomicU64,
    call_ns: AtomicU64,
    compiles: AtomicU64,
    compile_ns: AtomicU64,
    fallbacks: AtomicU64,
}

impl OvMoe {
    /// Construct from the environment, or `None` to keep the other paths:
    /// requires `CASCADIA_INKLING_OV_MOE` set and `<model>/moe_ov` present.
    pub fn from_env(
        model_dir: &Path,
        hidden: usize,
        k_total: usize,
        n_experts: usize,
    ) -> Option<Self> {
        if !super::env_flag("CASCADIA_INKLING_OV_MOE") {
            return None;
        }
        let dir = model_dir.join("moe_ov");
        if !dir.is_dir() {
            warn!(
                dir = %dir.display(),
                "CASCADIA_INKLING_OV_MOE set but the model has no moe_ov/ dir \
                 (tools/inkling_moe_layer_ov.py); keeping the other expert paths"
            );
            return None;
        }
        let device =
            std::env::var("CASCADIA_INKLING_OV_MOE_DEVICE").unwrap_or_else(|_| "GPU".into());
        let cache_dir = std::env::var("CASCADIA_INKLING_OV_MOE_CACHE_DIR").ok();
        // OpenVINO 2026.3.1's GPU plugin fails to compile a fused MoE layer
        // read straight from an IR file ("Node which is about to be added in
        // between two other nodes should not have any existing dependencies
        // ... postponed_decompression"). The shim works around this by
        // materialising the IR's constants in memory before compiling (see
        // CASCADIA_MATERIALIZE_CONSTANTS in shim.cpp), which is the graph form
        // the plugin builds its fused op from without any offload path.
        // Default: no offload — 3.4 ms per padded decode row and 30 ms per
        // 23-row prefill at Inkling's shape, against 5.5 / 55 ms through the
        // plugin's on-disk offload path. Set CASCADIA_INKLING_OV_MOE_OFFLOAD=N
        // (1..99) to stream that fraction of experts from disk instead.
        let offload = std::env::var("CASCADIA_INKLING_OV_MOE_OFFLOAD")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty() && v != "0");
        // The plugin's batched-GEMV decode kernel crashes on the Arc B390; the
        // grouped-GEMM path is selected by this plugin option, read from the
        // process environment at compile time. Honour an operator's own value.
        if std::env::var_os("OV_GPU_MOE_BATCHED_GEMV_THRESHOLD").is_none() {
            set_process_env("OV_GPU_MOE_BATCHED_GEMV_THRESHOLD", "0");
        }
        let ov = Self::new(
            dir,
            device,
            hidden,
            k_total,
            n_experts,
            cache_dir.as_deref(),
            offload,
        );
        tracing::info!(
            target: "cascadia::inkling",
            event = "ov_moe_config",
            device = %ov.device,
            k_total,
            offload = ov.offload.as_deref().unwrap_or("0"),
            cache_dir = cache_dir.as_deref().unwrap_or("<unset>"),
            dir = %ov.dir.display(),
        );
        Some(ov)
    }

    /// Explicit constructor (in-process hosts and tests); `dir` is `moe_ov/`.
    pub fn new(
        dir: PathBuf,
        device: String,
        hidden: usize,
        k_total: usize,
        n_experts: usize,
        cache_dir: Option<&str>,
        offload: Option<String>,
    ) -> Self {
        if std::env::var_os("OV_GPU_MOE_BATCHED_GEMV_THRESHOLD").is_none() {
            set_process_env("OV_GPU_MOE_BATCHED_GEMV_THRESHOLD", "0");
        }
        // f32, not the plugin's f16: on the full 66-layer model the fused layers of the
        // deeper ranks overflow half precision and every logit comes out NaN (the
        // pipeline then emits token 0, '!', at every step). Seen on an 11-box fleet;
        // layers 0-10 alone (the four-box bed) never showed it. f32 is correct on all
        // 33 fused layers of that fleet. Override with the env var to experiment.
        let precision =
            std::env::var("CASCADIA_INKLING_OV_MOE_PRECISION").unwrap_or_else(|_| "f32".into());
        let mut plugin = PluginConfig::new().with("INFERENCE_PRECISION_HINT", precision);
        match &offload {
            Some(r) => {
                plugin = plugin.with("OFFLOAD_RATIO", r.clone());
                if let Some(cd) = cache_dir {
                    plugin = plugin.with("CACHE_DIR", cd);
                }
            }
            None => {
                // A blob imported from the cache restores its weights from
                // the IR file, i.e. as file-backed constants again — measured
                // 28 ms per decode call against 3.7 ms compiled fresh — so
                // the materialised path never uses the blob cache.
                plugin = plugin.with("CASCADIA_MATERIALIZE_CONSTANTS", "1");
                if cache_dir.is_some() {
                    warn!("CASCADIA_INKLING_OV_MOE_CACHE_DIR ignored: the materialised fused-MoE path does not use the blob cache");
                }
            }
        }
        Self {
            dir,
            device,
            plugin,
            hidden,
            k_total,
            n_experts,
            offload,
            require_fused: false,
            profiles: Mutex::new(HashMap::new()),
            layers: Mutex::new(HashMap::new()),
            failed: Mutex::new(Default::default()),
            noted: Mutex::new(Default::default()),
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            call_ns: AtomicU64::new(0),
            compiles: AtomicU64::new(0),
            compile_ns: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
        }
    }

    /// Reject execution unless the runtime reports the compressed fused MoE
    /// node. Profiling is enabled before compile, and evidence retained.
    pub fn requiring_fusion(mut self) -> Self {
        self.require_fused = true;
        self.plugin = self.plugin.with("PERF_COUNT", "YES");
        self
    }

    pub fn fusion_profiles(&self) -> HashMap<u32, String> {
        self.profiles.lock().unwrap().clone()
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn k_total(&self) -> usize {
        self.k_total
    }

    pub fn stats(&self) -> OvMoeStats {
        OvMoeStats {
            calls: self.calls.load(Ordering::Relaxed),
            rows: self.rows.load(Ordering::Relaxed),
            call_ns: self.call_ns.load(Ordering::Relaxed),
            compiles: self.compiles.load(Ordering::Relaxed),
            compile_ns: self.compile_ns.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
        }
    }

    fn xml(&self, lid: u32) -> PathBuf {
        self.dir
            .join(format!("layer_{lid:02}"))
            .join("openvino_model.xml")
    }

    /// Whether an IR exists for layer `lid`.
    pub fn has_layer(&self, lid: u32) -> bool {
        self.xml(lid).is_file()
    }

    /// Compiled model for `lid`, compiling on first use; `None` once the IR
    /// proved unusable.
    fn compiled(&self, lid: u32) -> Option<Arc<Mutex<Runtime>>> {
        if self.failed.lock().unwrap().contains(&lid) {
            return None;
        }
        let mut layers = self.layers.lock().expect("OV MoE layer table lock");
        if let Some(rt) = layers.get(&lid) {
            return Some(Arc::clone(rt));
        }
        let xml = self.xml(lid);
        let Some(p) = xml.to_str() else {
            drop(layers);
            self.mark_failed(lid, "non-utf8 IR path");
            return None;
        };
        let mut plugin = self.plugin.clone();
        if self.offload.is_some() {
            // The offload path streams experts from the IR's weights file.
            let bin = xml.with_extension("bin");
            plugin = plugin.with("WEIGHTS_PATH", bin.to_string_lossy().to_string());
        }
        let t0 = Instant::now();
        match Runtime::compile(p, &self.device, &plugin) {
            Ok(rt) => {
                self.compiles.fetch_add(1, Ordering::Relaxed);
                self.compile_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let rt = Arc::new(Mutex::new(rt));
                layers.insert(lid, Arc::clone(&rt));
                Some(rt)
            }
            Err(e) => {
                drop(layers);
                self.mark_failed(lid, &format!("compile on {}: {e}", self.device));
                None
            }
        }
    }

    /// Compile layer `lid` ahead of time and touch every real expert once,
    /// so the plugin's offload slots hold them all before timing starts
    /// (its slot cache starts empty: the first touch of each expert streams
    /// it from the IR file at ~1 GB/s, ~8 s per Inkling layer).
    pub fn warm(&self, lid: u32) -> bool {
        if self.compiled(lid).is_none() {
            return false;
        }
        let k = self.k_total;
        let x = vec![0.0f32; self.hidden];
        let w = vec![0.0f32; k];
        let before = (
            self.calls.load(Ordering::Relaxed),
            self.rows.load(Ordering::Relaxed),
            self.call_ns.load(Ordering::Relaxed),
        );
        let mut ok = true;
        if self.offload.is_some() {
            // Fill the plugin's slot cache: touch every real expert once.
            let mut ids: Vec<i32> = (0..self.n_experts as i32).collect();
            while !ids.len().is_multiple_of(k) {
                ids.push(ids[0]);
            }
            for chunk in ids.chunks(k) {
                ok &= self.forward(lid, &x, 1, chunk, &w).is_some();
            }
        }
        // The first call at a row count pays the plugin's kernel setup for
        // that shape (~140 ms for the padded decode shape on the B390); take
        // the decode bucket and the smallest prefill bucket here.
        let ids: Vec<i32> = (0..k as i32).collect();
        ok &= self.forward(lid, &x, 1, &ids, &w).is_some();
        let x8 = vec![0.0f32; 8 * self.hidden];
        let ids8: Vec<i32> = ids.iter().copied().cycle().take(8 * k).collect();
        let w8 = vec![0.0f32; 8 * k];
        ok &= self.forward(lid, &x8, 8, &ids8, &w8).is_some();
        // Warm-up calls are not benchmark calls.
        self.calls.store(before.0, Ordering::Relaxed);
        self.rows.store(before.1, Ordering::Relaxed);
        self.call_ns.store(before.2, Ordering::Relaxed);
        ok
    }

    /// The MoE output for `rows` rows of `xs` (`[rows, hidden]`) with the
    /// selected expert ids (`[rows, k_total]`, shared experts as
    /// `n_routed + s`) and their weights, or `None` when this layer must take
    /// another path. One row is padded to two (see the module doc).
    pub fn forward(
        &self,
        lid: u32,
        xs: &[f32],
        rows: usize,
        ids: &[i32],
        weights: &[f32],
    ) -> Option<Vec<f32>> {
        debug_assert_eq!(xs.len(), rows * self.hidden);
        debug_assert_eq!(ids.len(), rows * self.k_total);
        debug_assert_eq!(weights.len(), rows * self.k_total);
        if rows == 0 {
            return Some(Vec::new());
        }
        let Some(rt) = self.compiled(lid) else {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let t0 = Instant::now();
        // Pad to the shape bucket: copies of the last row with zero weights
        // (the kernel still touches their experts, which are the same ones).
        let prow = bucket_rows(rows);
        let (xs_p, ids_p, w_p);
        let (xs, ids, weights) = if prow != rows {
            let h = self.hidden;
            let k = self.k_total;
            let mut x2 = xs.to_vec();
            let mut i2 = ids.to_vec();
            let mut w2 = weights.to_vec();
            for _ in rows..prow {
                x2.extend_from_slice(&xs[(rows - 1) * h..rows * h]);
                i2.extend_from_slice(&ids[(rows - 1) * k..rows * k]);
                w2.extend(std::iter::repeat_n(0.0f32, k));
            }
            xs_p = x2;
            ids_p = i2;
            w_p = w2;
            (&xs_p[..], &ids_p[..], &w_p[..])
        } else {
            (xs, ids, weights)
        };
        let out = {
            let mut rt = rt.lock().expect("OV MoE runtime lock");
            let step: Result<(), String> = rt
                .set_input("x", DType::F32, &[1, prow, self.hidden], f32_bytes(xs))
                .map_err(|e| format!("set_input x: {e}"))
                .and_then(|_| {
                    rt.set_input(
                        "topk_indices",
                        DType::I32,
                        &[prow, self.k_total],
                        i32_bytes(ids),
                    )
                    .map_err(|e| format!("set_input topk_indices: {e}"))
                })
                .and_then(|_| {
                    rt.set_input(
                        "routing_weights",
                        DType::F32,
                        &[prow, self.k_total],
                        f32_bytes(weights),
                    )
                    .map_err(|e| format!("set_input routing_weights: {e}"))
                })
                .and_then(|_| rt.infer().map_err(|e| format!("infer: {e}")));
            if let Err(why) = step {
                // Not latched (a device-side error can be transient), but said
                // once per layer so a benchmark cannot silently fall back.
                self.note_call_failure(lid, &why);
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            if self.require_fused && !self.profiles.lock().unwrap().contains_key(&lid) {
                let profile = rt.profiling().unwrap_or_default();
                let fused = profile.lines().any(|line| {
                    let fields: Vec<_> = line.split('\t').collect();
                    fields.len() >= 3
                        && ((fields[1] == "MOECompressed"
                            && fields[2].contains("ocl::moe::moe_3gemm_"))
                            || fields[1..3].iter().any(|field| {
                                field
                                    .chars()
                                    .filter(|c| c.is_ascii_alphanumeric())
                                    .collect::<String>()
                                    .to_ascii_lowercase()
                                    .contains("moe3gemmfusedcompressed")
                            }))
                });
                if !fused {
                    self.mark_failed(
                        lid,
                        &format!("required fused compressed MoE absent from profiling: {profile}"),
                    );
                    self.fallbacks.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                self.profiles.lock().unwrap().insert(lid, profile);
            }
            let (_, _, bytes) = match rt.output(0) {
                Ok(o) => o,
                Err(e) => {
                    self.note_call_failure(lid, &format!("output: {e}"));
                    self.fallbacks.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        };
        if out.len() != prow * self.hidden {
            self.mark_failed(
                lid,
                &format!("output len {} != {} x {}", out.len(), prow, self.hidden),
            );
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(if prow != rows {
            out[..rows * self.hidden].to_vec()
        } else {
            out
        })
    }

    fn note_call_failure(&self, lid: u32, why: &str) {
        if self.noted.lock().unwrap().insert(lid) {
            warn!(
                layer = lid,
                "inkling fused-MoE call failed ({why}); falling back for this call"
            );
            eprintln!("[inkling] fused-MoE layer {lid} call failed: {why}");
        }
    }

    fn mark_failed(&self, lid: u32, why: &str) {
        if self.failed.lock().unwrap().insert(lid) {
            warn!(
                layer = lid,
                "inkling fused-MoE IR unusable ({why}); other expert paths for this layer"
            );
            // Also on stderr: the bench examples run without a tracing
            // subscriber, and a silent fallback is the one thing a benchmark
            // must not do.
            eprintln!("[inkling] fused-MoE layer {lid} unusable: {why}");
        }
    }

    /// The compiled fused IR for `lid` sums `k_total` experts per row, but the
    /// layer routes `layer_k` (`top_k + n_shared`): the IR can never serve this
    /// layer, so latch it unusable — it then shows in [`Self::failed_layers`],
    /// so `--warm-ov` reports it FAILED — and report it once, like any bad IR.
    pub fn mark_k_mismatch(&self, lid: u32, layer_k: usize) {
        self.mark_failed(
            lid,
            &format!(
                "IR k_total {} != layer top_k+n_shared {layer_k}",
                self.k_total
            ),
        );
    }

    /// A token hit a layer whose fused IR's K disagrees with the layer's
    /// (see [`Self::mark_k_mismatch`]): count the bypass as a fallback so the
    /// benchmark read-out cannot understate it, and latch + report it once.
    pub fn note_k_mismatch(&self, lid: u32, layer_k: usize) {
        self.fallbacks.fetch_add(1, Ordering::Relaxed);
        self.mark_k_mismatch(lid, layer_k);
    }

    pub fn failed_layers(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.failed.lock().unwrap().iter().copied().collect();
        v.sort_unstable();
        v
    }
}

impl std::fmt::Debug for OvMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OvMoe")
            .field("dir", &self.dir)
            .field("device", &self.device)
            .field("hidden", &self.hidden)
            .field("k_total", &self.k_total)
            .finish()
    }
}
