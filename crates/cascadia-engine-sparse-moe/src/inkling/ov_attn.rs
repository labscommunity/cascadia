//! OpenVINO backend for Inkling's attention projections (iGPU / NPU / CPU).
//!
//! The five GEMVs of a layer's attention — `q`, `k`, `v`, `r` (the
//! relative-position query) from the input-normed hidden state, and `o` from
//! the attended context — are the bf16 bytes the CPU streams at ~80 GB/s
//! (~264 MB per layer). `tools/inkling_attn_ov.py` writes them as two
//! per-layer IRs, `<model>/attn_ov/layer_NN/{qkvr,o}/openvino_model.xml`,
//! by default re-quantised to int4 on the experts' grid (~66 MB per layer:
//! the byte saving is the speed-up, a f16 copy on the device would not be),
//! and this backend runs them; the head norms, position bias, softmax, KV
//! cache and convolutions stay in [`super::attn`].
//!
//! Outputs are rounded to bf16 like the Rust `linear_bf16_w`, so what differs
//! from the CPU path is the weights' quantisation (int8 by default, optionally
//! int4 — see `tools/inkling_attn_ov.py --weights`) and the device's f16
//! accumulation. Enabled by `CASCADIA_INKLING_OV_ATTN=1`;
//! `CASCADIA_INKLING_OV_ATTN_DEVICE` (default `GPU`). Layers without IRs and
//! calls the device refuses keep the Rust kernel (reported once per layer).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use tracing::warn;

use crate::dsv4::math::to_bf16;

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bf16_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OvAttnStats {
    pub calls: u64,
    pub rows: u64,
    pub call_ns: u64,
    /// With `CASCADIA_INKLING_OV_PERF=1` (see `ov_moe::ov_perf`).
    pub infer_ns: u64,
    pub device_ns: u64,
    pub compiles: u64,
    pub fallbacks: u64,
}

/// The two compiled models of one layer.
struct LayerRt {
    qkvr: Arc<Mutex<Runtime>>,
    o: Arc<Mutex<Runtime>>,
}

pub struct OvAttn {
    dir: PathBuf, // <model>/attn_ov
    device: String,
    plugin: PluginConfig,
    layers: Mutex<HashMap<u32, Arc<LayerRt>>>,
    failed: Mutex<HashSet<u32>>,
    noted: Mutex<HashSet<u32>>,
    calls: AtomicU64,
    rows: AtomicU64,
    call_ns: AtomicU64,
    infer_ns: AtomicU64,
    device_ns: AtomicU64,
    compiles: AtomicU64,
    fallbacks: AtomicU64,
}

impl OvAttn {
    pub fn from_env(model_dir: &Path) -> Option<Self> {
        if !super::env_flag("CASCADIA_INKLING_OV_ATTN") {
            return None;
        }
        // `CASCADIA_INKLING_OV_ATTN_DIR` selects a variant directory (e.g.
        // `attn_ov_int8`); default `attn_ov`.
        let dir = model_dir.join(
            std::env::var("CASCADIA_INKLING_OV_ATTN_DIR").unwrap_or_else(|_| "attn_ov".into()),
        );
        if !dir.is_dir() {
            warn!(
                dir = %dir.display(),
                "CASCADIA_INKLING_OV_ATTN set but the model has no attn_ov/ dir \
                 (tools/inkling_attn_ov.py); keeping the Rust attention kernels"
            );
            return None;
        }
        let device =
            std::env::var("CASCADIA_INKLING_OV_ATTN_DEVICE").unwrap_or_else(|_| "GPU".into());
        let ov = Self::new(dir, device);
        tracing::info!(
            target: "cascadia::inkling",
            event = "ov_attn_config",
            device = %ov.device,
            dir = %ov.dir.display(),
        );
        Some(ov)
    }

    pub fn new(dir: PathBuf, device: String) -> Self {
        Self {
            dir,
            device,
            plugin: if super::ov_moe::ov_perf() {
                PluginConfig::new()
                    .with("INFERENCE_PRECISION_HINT", "f16")
                    .with("PERF_COUNT", "YES")
            } else {
                PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f16")
            },
            layers: Mutex::new(HashMap::new()),
            failed: Mutex::new(HashSet::new()),
            noted: Mutex::new(HashSet::new()),
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            call_ns: AtomicU64::new(0),
            infer_ns: AtomicU64::new(0),
            device_ns: AtomicU64::new(0),
            compiles: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
        }
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn stats(&self) -> OvAttnStats {
        OvAttnStats {
            calls: self.calls.load(Ordering::Relaxed),
            rows: self.rows.load(Ordering::Relaxed),
            call_ns: self.call_ns.load(Ordering::Relaxed),
            infer_ns: self.infer_ns.load(Ordering::Relaxed),
            device_ns: self.device_ns.load(Ordering::Relaxed),
            compiles: self.compiles.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
        }
    }

    fn xml(&self, lid: u32, which: &str) -> PathBuf {
        self.dir
            .join(format!("layer_{lid:02}"))
            .join(which)
            .join("openvino_model.xml")
    }

    pub fn has_layer(&self, lid: u32) -> bool {
        self.xml(lid, "qkvr").is_file() && self.xml(lid, "o").is_file()
    }

    fn compiled(&self, lid: u32) -> Option<Arc<LayerRt>> {
        if self.failed.lock().unwrap().contains(&lid) {
            return None;
        }
        let mut layers = self.layers.lock().expect("OV attn layer table lock");
        if let Some(rt) = layers.get(&lid) {
            return Some(Arc::clone(rt));
        }
        let mut compile = |which: &str| -> Result<Runtime, String> {
            let xml = self.xml(lid, which);
            let p = xml.to_str().ok_or("non-utf8 IR path")?;
            Runtime::compile(p, &self.device, &self.plugin)
                .map_err(|e| format!("{which}: compile on {}: {e}", self.device))
        };
        match (compile("qkvr"), compile("o")) {
            (Ok(q), Ok(o)) => {
                self.compiles.fetch_add(2, Ordering::Relaxed);
                let rt = Arc::new(LayerRt {
                    qkvr: Arc::new(Mutex::new(q)),
                    o: Arc::new(Mutex::new(o)),
                });
                layers.insert(lid, Arc::clone(&rt));
                Some(rt)
            }
            (Err(e), _) | (_, Err(e)) => {
                drop(layers);
                self.mark_failed(lid, &e);
                None
            }
        }
    }

    /// Compile layer `lid` and take the first-call cost of the one-row shape.
    pub fn warm(&self, lid: u32, hidden: usize, ctx_dim: usize) -> bool {
        if self.compiled(lid).is_none() {
            return false;
        }
        let before = (
            self.calls.load(Ordering::Relaxed),
            self.rows.load(Ordering::Relaxed),
            self.call_ns.load(Ordering::Relaxed),
        );
        let ok = self.qkvr(lid, &vec![0.0f32; hidden], 1).is_some()
            && self.o(lid, &vec![0.0f32; ctx_dim], 1).is_some()
            && self.qkvr(lid, &vec![0.0f32; 8 * hidden], 8).is_some()
            && self.o(lid, &vec![0.0f32; 8 * ctx_dim], 8).is_some();
        self.calls.store(before.0, Ordering::Relaxed);
        self.rows.store(before.1, Ordering::Relaxed);
        self.call_ns.store(before.2, Ordering::Relaxed);
        ok
    }

    /// `q`, `k`, `v`, `r` for `rows` input-normed rows (`xs` = `[rows, hidden]`),
    /// each `[rows, dim]` and bf16-rounded; `None` to take the Rust kernel.
    pub fn qkvr(&self, lid: u32, xs: &[f32], rows: usize) -> Option<[Vec<f32>; 4]> {
        let Some(rt) = self.compiled(lid) else {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let hidden = xs.len() / rows.max(1);
        let prow = super::ov_moe::bucket_rows(rows);
        let xs_p;
        let xs = if prow != rows {
            let mut v = xs.to_vec();
            v.resize(prow * hidden, 0.0);
            xs_p = v;
            &xs_p[..]
        } else {
            xs
        };
        let t0 = Instant::now();
        let out = {
            let mut r = rt.qkvr.lock().expect("OV attn qkvr lock");
            let step = r
                .set_input("x", DType::F32, &[1, prow, hidden], f32_bytes(xs))
                .map_err(|e| format!("qkvr set_input: {e}"))
                .and_then(|_| {
                    let t_infer = Instant::now();
                    let res = r.infer().map_err(|e| format!("qkvr infer: {e}"));
                    if super::ov_moe::ov_perf() {
                        self.infer_ns
                            .fetch_add(t_infer.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        self.device_ns
                            .fetch_add(super::ov_moe::device_ns(&r), Ordering::Relaxed);
                    }
                    res
                });
            if let Err(why) = step {
                self.note(lid, &why);
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let mut outs: Vec<Vec<f32>> = Vec::with_capacity(4);
            for i in 0..4 {
                match r.output(i) {
                    Ok((_, _, bytes)) => {
                        let mut v = bf16_vec(&bytes);
                        // The IR emits [1, prow, dim]; a stale export or the
                        // wrong CASCADIA_INKLING_OV_ATTN_DIR variant infers
                        // cleanly but returns a length that is not a clean
                        // [prow, dim], which the truncation below would silently
                        // mis-shape into attend(). Latch the layer like the
                        // expert/MoE backends do on a length mismatch.
                        if v.is_empty() || v.len() % prow != 0 {
                            self.mark_failed(
                                lid,
                                &format!(
                                    "qkvr output {i} len {} not a multiple of prow {prow}",
                                    v.len()
                                ),
                            );
                            self.fallbacks.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                        if prow != rows {
                            v.truncate(v.len() / prow * rows);
                        }
                        outs.push(v)
                    }
                    Err(e) => {
                        self.note(lid, &format!("qkvr output {i}: {e}"));
                        self.fallbacks.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                }
            }
            outs
        };
        let [q, k, v, r]: [Vec<f32>; 4] = out.try_into().ok()?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some([q, k, v, r])
    }

    /// The output projection for `rows` context rows (`ctx` = `[rows, Hq·D]`),
    /// `[rows, hidden]` bf16-rounded; `None` to take the Rust kernel.
    pub fn o(&self, lid: u32, ctx: &[f32], rows: usize) -> Option<Vec<f32>> {
        let Some(rt) = self.compiled(lid) else {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let dim = ctx.len() / rows.max(1);
        let prow = super::ov_moe::bucket_rows(rows);
        let ctx_p;
        let ctx = if prow != rows {
            let mut v = ctx.to_vec();
            v.resize(prow * dim, 0.0);
            ctx_p = v;
            &ctx_p[..]
        } else {
            ctx
        };
        let t0 = Instant::now();
        let out = {
            let mut r = rt.o.lock().expect("OV attn o lock");
            let step = r
                .set_input("ctx", DType::F32, &[1, prow, dim], f32_bytes(ctx))
                .map_err(|e| format!("o set_input: {e}"))
                .and_then(|_| {
                    let t_infer = Instant::now();
                    let res = r.infer().map_err(|e| format!("o infer: {e}"));
                    if super::ov_moe::ov_perf() {
                        self.infer_ns
                            .fetch_add(t_infer.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        self.device_ns
                            .fetch_add(super::ov_moe::device_ns(&r), Ordering::Relaxed);
                    }
                    res
                });
            if let Err(why) = step {
                self.note(lid, &why);
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match r.output(0) {
                Ok((_, _, bytes)) => {
                    let mut v = bf16_vec(&bytes);
                    // The IR emits [1, prow, dim]; a stale export or the wrong
                    // CASCADIA_INKLING_OV_ATTN_DIR variant infers cleanly but
                    // returns a length that is not a clean [prow, dim], which
                    // the truncation below would silently mis-shape into
                    // attend(). Latch the layer like the expert/MoE backends do
                    // on a length mismatch.
                    if v.is_empty() || v.len() % prow != 0 {
                        self.mark_failed(
                            lid,
                            &format!("o output len {} not a multiple of prow {prow}", v.len()),
                        );
                        self.fallbacks.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                    if prow != rows {
                        v.truncate(v.len() / prow * rows);
                    }
                    v
                }
                Err(e) => {
                    self.note(lid, &format!("o output: {e}"));
                    self.fallbacks.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
        };
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(out)
    }

    fn note(&self, lid: u32, why: &str) {
        if self.noted.lock().unwrap().insert(lid) {
            warn!(
                layer = lid,
                "inkling OV attention call failed ({why}); Rust kernel for this call"
            );
            eprintln!("[inkling] OV attention layer {lid} call failed: {why}");
        }
    }

    fn mark_failed(&self, lid: u32, why: &str) {
        if self.failed.lock().unwrap().insert(lid) {
            warn!(
                layer = lid,
                "inkling OV attention IR unusable ({why}); Rust kernels for this layer"
            );
            eprintln!("[inkling] OV attention layer {lid} unusable: {why}");
        }
    }

    pub fn failed_layers(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.failed.lock().unwrap().iter().copied().collect();
        v.sort_unstable();
        v
    }
}

impl std::fmt::Debug for OvAttn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OvAttn")
            .field("dir", &self.dir)
            .field("device", &self.device)
            .finish()
    }
}
