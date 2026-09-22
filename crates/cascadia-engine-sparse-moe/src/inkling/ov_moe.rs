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
/// `CASCADIA_INKLING_OV_PERF=1`: compile the device graphs with per-primitive
/// profiling and account, per call, the time inside `infer()` and the time the
/// DEVICE spent executing. Wall minus device time is when the GPU sat idle
/// (host-side shape inference, argument setting, submission, copies): the
/// number that says whether a second frame could use the device meanwhile.
/// A diagnostic: profiling itself costs a little, so set it on one rank.
pub(crate) fn ov_perf() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("CASCADIA_INKLING_OV_PERF").is_ok_and(|v| v.trim() == "1"))
}

/// Device execution time of the last `infer()` (sum over primitives), ns.
pub(crate) fn device_ns(rt: &Runtime) -> u64 {
    rt.profiling()
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split('\t').nth(3)?.parse::<u64>().ok())
        .sum::<u64>()
        * 1000
}

/// `CASCADIA_INKLING_OV_MOE_DECODE_DIR`: folder (next to `moe_ov/`) of layers
/// that run through the plugin's decode kernels.
fn decode_dir_name() -> Option<&'static str> {
    use std::sync::OnceLock;
    static D: OnceLock<Option<String>> = OnceLock::new();
    D.get_or_init(|| {
        std::env::var("CASCADIA_INKLING_OV_MOE_DECODE_DIR")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty() && !v.contains('/') && !v.contains(".."))
    })
    .as_deref()
}

/// `CASCADIA_INKLING_OV_MOE_DECODE_LAYERS` (`all` or a comma list): layers
/// that take the plugin's decode kernels from their ordinary group-32 IR.
/// That needs a plugin whose MoE kernels run with sub-group 16 on this GPU
/// (Xe2 and newer default to 32, which refuses group 32: autolab 026, 029).
fn decode_layer_listed(lid: u32) -> bool {
    let (all, list) = decode_layer_list();
    *all || list.contains(&lid)
}

/// Whether any layer is listed (the per-layer plugin knob must then be set
/// for every layer this process compiles, listed or not).
fn decode_layers_configured() -> bool {
    let (all, list) = decode_layer_list();
    *all || !list.is_empty()
}

fn decode_layer_list() -> &'static (bool, Vec<u32>) {
    use std::sync::OnceLock;
    static L: OnceLock<(bool, Vec<u32>)> = OnceLock::new();
    L.get_or_init(|| {
        let v = std::env::var("CASCADIA_INKLING_OV_MOE_DECODE_LAYERS").unwrap_or_default();
        let v = v.trim();
        (
            v.eq_ignore_ascii_case("all"),
            v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        )
    })
}

/// `CASCADIA_INKLING_OV_MOE_DECODE_ROWS` (default 1): calls of at most this
/// many rows take the decode kernels on a decode layer, unpadded. They read
/// at the bus limit but share no expert between rows: measured equal to the
/// prefill path at two rows and slower beyond (026), so one row by default.
fn decode_rows() -> usize {
    use std::sync::OnceLock;
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("CASCADIA_INKLING_OV_MOE_DECODE_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 32)
    })
}

pub(crate) fn bucket_rows(rows: usize) -> usize {
    if rows > 32 {
        return rows.div_ceil(32) * 32;
    }
    small_buckets()
        .iter()
        .copied()
        .find(|&b| b >= rows)
        .unwrap_or(32)
}

/// The row counts (<= 32) device calls are padded to. Default `2,8,16,24,32`:
/// a one-row call used to crash the plugin's decode kernel (its 32-bit expert
/// offset, see autolab 022), so decode was padded to two rows. With a fixed
/// plugin `CASCADIA_INKLING_OV_BUCKETS=1,2,4,8,16,24,32` lets a frame of one
/// row read one row's experts, and a frame of three pad to four, not eight.
fn small_buckets() -> &'static [usize] {
    use std::sync::OnceLock;
    static B: OnceLock<Vec<usize>> = OnceLock::new();
    B.get_or_init(|| {
        let mut v: Vec<usize> = std::env::var("CASCADIA_INKLING_OV_BUCKETS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse().ok())
                    .filter(|&b| (1..=32).contains(&b))
                    .collect()
            })
            .unwrap_or_default();
        if v.is_empty() {
            v = vec![2, 8, 16, 24, 32];
        }
        v.sort_unstable();
        v.dedup();
        if v.last() != Some(&32) {
            v.push(32);
        }
        v
    })
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
    /// With `CASCADIA_INKLING_OV_PERF=1`: time inside `infer()` and time the
    /// device executed, of `call_ns`.
    pub infer_ns: u64,
    pub device_ns: u64,
    pub compiles: u64,
    pub compile_ns: u64,
    pub fallbacks: u64,
    /// Calls whose output held a NaN or infinity (half-precision overflow on
    /// the device) and were handed to the other expert path; part of `fallbacks`.
    pub nonfinite: u64,
}

/// `CASCADIA_INKLING_OV_MOE_WEIGHT_RESCALE` (default on): divide each row's
/// routing weights by a power of two before the device call and multiply the
/// row's output back on the host. Inkling's routing weights sum to
/// `8 * mlp.gate.global_scale`, which grows with depth (about 100 per weight at
/// layer 40): at the plugin's f16 the weighted sum of expert outputs passes
/// 65504 and the layer returns infinities, then NaN logits (the fleet printed
/// `!!!!`). A power of two changes no mantissa bit, so the result equals the
/// unscaled one wherever that one was finite.
fn weight_rescale() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("CASCADIA_INKLING_OV_MOE_WEIGHT_RESCALE")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Smallest power of two >= `v` (1 for anything at or below 1, or not finite).
fn pow2_ceil(v: f32) -> f32 {
    if !v.is_finite() || v <= 1.0 {
        return 1.0;
    }
    2.0f32.powi(v.log2().ceil() as i32)
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
    infer_ns: AtomicU64,
    device_ns: AtomicU64,
    compiles: AtomicU64,
    compile_ns: AtomicU64,
    fallbacks: AtomicU64,
    nonfinite: AtomicU64,
    /// Per layer: what the device's output must be multiplied by. A layer
    /// generated with `--up-scale-exponent N` (its `cascadia_moe.json` says so)
    /// returns `y * 2^-N`, which keeps an expert whose own output passes f16's
    /// range finite on the device (Inkling layer 8's shared expert reaches
    /// -94909); 1.0 for every other layer.
    out_scale: Mutex<HashMap<u32, f32>>,
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
        if ov_perf() {
            plugin = plugin.with("PERF_COUNT", "YES");
        }
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
            infer_ns: AtomicU64::new(0),
            device_ns: AtomicU64::new(0),
            compiles: AtomicU64::new(0),
            compile_ns: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
            nonfinite: AtomicU64::new(0),
            out_scale: Mutex::new(HashMap::new()),
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
            infer_ns: self.infer_ns.load(Ordering::Relaxed),
            device_ns: self.device_ns.load(Ordering::Relaxed),
            compiles: self.compiles.load(Ordering::Relaxed),
            compile_ns: self.compile_ns.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            nonfinite: self.nonfinite.load(Ordering::Relaxed),
        }
    }

    /// `2^N` for a layer generated with `--up-scale-exponent N`, else 1.
    fn layer_out_scale(&self, lid: u32) -> f32 {
        if let Some(&f) = self.out_scale.lock().unwrap().get(&lid) {
            return f;
        }
        let side = self.layer_dir(lid).join("cascadia_moe.json");
        let n = std::fs::read_to_string(&side)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get("up_scale_exponent").and_then(|e| e.as_u64()))
            .filter(|&n| n <= 16)
            .unwrap_or(0);
        let f = 2.0f32.powi(n as i32);
        if n > 0 {
            tracing::info!(
                target: "cascadia::inkling",
                event = "ov_moe_up_scale",
                layer = lid,
                exponent = n,
            );
        }
        self.out_scale.lock().unwrap().insert(lid, f);
        f
    }

    fn xml(&self, lid: u32) -> PathBuf {
        self.layer_dir(lid).join("openvino_model.xml")
    }

    /// The folder of `lid`'s IR. `CASCADIA_INKLING_OV_MOE_DECODE_DIR=moe_ov_g64`
    /// names a sibling of `moe_ov/` holding the same layers re-quantised to a
    /// wider int4 group (run.sh writes them when `CASCADIA_FUSE_GROUP` asks):
    /// the GPU plugin's MoE decode kernels (batched GEMV: three kernels and no
    /// host synchronisation per call) refuse group 32 on Xe2 and newer. A
    /// layer found there is compiled with the decode threshold ON, every other
    /// layer keeps the prefill path, so one rank can run both side by side.
    fn layer_dir(&self, lid: u32) -> PathBuf {
        if let Some(alt) = decode_dir_name() {
            if let Some(parent) = self.dir.parent() {
                let d = parent.join(alt).join(format!("layer_{lid:02}"));
                if d.join("openvino_model.xml").is_file() {
                    return d;
                }
            }
        }
        self.dir.join(format!("layer_{lid:02}"))
    }

    /// Whether `lid` runs from the re-quantised folder (see [`Self::layer_dir`]).
    pub fn is_regrouped(&self, lid: u32) -> bool {
        decode_dir_name().is_some()
            && self.layer_dir(lid) != self.dir.join(format!("layer_{lid:02}"))
    }

    /// Whether `lid`'s small calls take the plugin's decode kernels: a layer
    /// from the re-quantised folder, or one listed in
    /// `CASCADIA_INKLING_OV_MOE_DECODE_LAYERS`.
    pub fn uses_decode_kernels(&self, lid: u32) -> bool {
        self.is_regrouped(lid) || decode_layer_listed(lid)
    }

    /// Rows a call of `rows` rows is padded to on layer `lid`.
    fn padded_rows(&self, lid: u32, rows: usize) -> usize {
        if rows <= decode_rows() && self.uses_decode_kernels(lid) {
            rows
        } else {
            bucket_rows(rows)
        }
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
        // The plugin reads this knob from the process environment when it
        // builds a model: set per layer, under the layer-table lock.
        // (The plugin turns 0 into 1: a one-row call takes the decode kernels
        // on every layer, which is why other layers pad one row to two.)
        let decode = self.uses_decode_kernels(lid);
        let threshold = if decode {
            decode_rows().to_string()
        } else {
            "0".to_string()
        };
        if std::env::var("OV_GPU_MOE_BATCHED_GEMV_THRESHOLD").as_deref() != Ok(threshold.as_str())
            && (decode || decode_dir_name().is_some() || decode_layers_configured())
        {
            set_process_env("OV_GPU_MOE_BATCHED_GEMV_THRESHOLD", &threshold);
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
        // On the HIGHEST ids: the plugin's decode kernels overflow a 32-bit
        // expert offset from id 228 on (autolab 022); a plugin that does must
        // fail here, while the rank loads, not inside the first request.
        let ids: Vec<i32> = (self.n_experts.saturating_sub(k)..self.n_experts)
            .map(|e| e as i32)
            .collect();
        let mut shapes: Vec<usize> = small_buckets().iter().copied().filter(|&b| b <= 8).collect();
        if self.uses_decode_kernels(lid) {
            shapes.extend(1..=decode_rows());
            shapes.sort_unstable();
            shapes.dedup();
        }
        for b in shapes {
            let xb = vec![0.0f32; b * self.hidden];
            let idsb: Vec<i32> = ids.iter().copied().cycle().take(b * k).collect();
            let wb = vec![0.0f32; b * k];
            ok &= self.forward(lid, &xb, b, &idsb, &wb).is_some();
        }
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
        // Keep the device's weighted sum inside half precision: per row, the
        // routing weights go down by a power of two and the output comes back
        // up by it on the host (see `weight_rescale`).
        let row_scale: Vec<f32> = if weight_rescale() {
            weights
                .chunks_exact(self.k_total)
                .map(|w| pow2_ceil(w.iter().map(|v| v.abs()).sum()))
                .collect()
        } else {
            Vec::new()
        };
        let scaled_w: Vec<f32>;
        let weights = if row_scale.iter().any(|&f| f != 1.0) {
            scaled_w = weights
                .chunks_exact(self.k_total)
                .zip(&row_scale)
                .flat_map(|(w, &f)| w.iter().map(move |v| v / f))
                .collect();
            &scaled_w[..]
        } else {
            weights
        };
        // Pad to the shape bucket: copies of the last row with zero weights
        // (the kernel still touches their experts, which are the same ones).
        let prow = self.padded_rows(lid, rows);
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
                .and_then(|_| {
                    let t_infer = Instant::now();
                    let r = rt.infer().map_err(|e| format!("infer: {e}"));
                    if ov_perf() {
                        self.infer_ns
                            .fetch_add(t_infer.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        self.device_ns.fetch_add(device_ns(&rt), Ordering::Relaxed);
                    }
                    r
                });
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
        let mut out = out;
        out.truncate(rows * self.hidden);
        // An expert's own output can still pass 65504 on the device (seen on a
        // shared expert of layer 8: -94909): that call goes to the other
        // expert path instead of poisoning the residual stream with NaN.
        if out.iter().any(|v| !v.is_finite()) {
            self.nonfinite.fetch_add(1, Ordering::Relaxed);
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            self.note_call_failure(lid, "non-finite output (half-precision overflow)");
            return None;
        }
        let layer_scale = self.layer_out_scale(lid);
        if row_scale.is_empty() {
            if layer_scale != 1.0 {
                for v in out.iter_mut() {
                    *v *= layer_scale;
                }
            }
        } else {
            for (row, &f) in out.chunks_exact_mut(self.hidden).zip(&row_scale) {
                let f = f * layer_scale;
                if f != 1.0 {
                    for v in row {
                        *v *= f;
                    }
                }
            }
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(out)
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
    /// Take `lid` off the device for good, with the reason (a load-time check
    /// failed): its rows go to the host kernels from now on.
    pub fn fail_layer(&self, lid: u32, why: &str) {
        self.mark_failed(lid, why);
    }

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

#[cfg(test)]
mod rescale_tests {
    #[test]
    fn weight_rescale_factor_is_an_exact_power_of_two() {
        use super::pow2_ceil;
        assert_eq!(pow2_ceil(0.0), 1.0);
        assert_eq!(pow2_ceil(0.7), 1.0);
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.5), 2.0);
        assert_eq!(pow2_ceil(800.0), 1024.0);
        assert_eq!(pow2_ceil(1024.0), 1024.0);
        assert_eq!(pow2_ceil(f32::NAN), 1.0);
        assert_eq!(pow2_ceil(f32::INFINITY), 1.0);
        // Dividing a weight by the factor and multiplying the product back is
        // exact: only the exponent moves.
        for &(w, y) in &[(97.3f32, 1873.25f32), (105.0, -0.0371), (3.1e-3, 4.2e2)] {
            let f = pow2_ceil(800.0);
            assert_eq!((w / f) * y * f, w * y);
        }
    }
}
