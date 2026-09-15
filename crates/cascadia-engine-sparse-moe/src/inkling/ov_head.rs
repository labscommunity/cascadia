//! OpenVINO backend for Inkling's unembed head (iGPU / NPU / CPU).
//!
//! `unembed · rmsnorm(x)/mup` reads the whole `[vocab, hidden]` table every
//! token — 2.46 GB of bf16 for Inkling's 200k × 6144, ~31 ms on the CPU at
//! ~80 GB/s, and the last rank of a pipeline pays it once per token.
//! `tools/inkling_attn_ov.py --head` writes it as one compressed-FC IR
//! (int8 per row by default, ~1.2 GB) and this backend runs it; the RMSNorm,
//! the mup divide and the slice to `unpadded_vocab` stay in [`super::model`].
//!
//! Enabled by `CASCADIA_INKLING_OV_HEAD=1` with `<model>/head_ov`;
//! `CASCADIA_INKLING_OV_HEAD_DEVICE` (default `GPU`). A missing or
//! uncompilable IR, or a refused call, keeps the Rust `matvec_f32` (reported
//! once).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use tracing::warn;

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OvHeadStats {
    pub calls: u64,
    pub call_ns: u64,
    pub fallbacks: u64,
}

pub struct OvHead {
    xml: PathBuf,
    device: String,
    plugin: PluginConfig,
    /// Logits the caller keeps (`unpadded_vocab`); the IR may emit the padded
    /// vocabulary, and anything shorter is a wrong IR.
    unpadded_vocab: usize,
    rt: Mutex<Option<Runtime>>,
    failed: AtomicBool,
    noted: AtomicBool,
    calls: AtomicU64,
    call_ns: AtomicU64,
    fallbacks: AtomicU64,
}

impl OvHead {
    pub fn from_env(model_dir: &Path, unpadded_vocab: usize) -> Option<Self> {
        if !super::env_flag("CASCADIA_INKLING_OV_HEAD") {
            return None;
        }
        let xml = model_dir.join("head_ov").join("openvino_model.xml");
        if !xml.is_file() {
            warn!(
                path = %xml.display(),
                "CASCADIA_INKLING_OV_HEAD set but the model has no head_ov/ IR \
                 (tools/inkling_attn_ov.py --head); keeping the Rust head"
            );
            return None;
        }
        let device =
            std::env::var("CASCADIA_INKLING_OV_HEAD_DEVICE").unwrap_or_else(|_| "GPU".into());
        tracing::info!(
            target: "cascadia::inkling",
            event = "ov_head_config",
            device = %device,
            path = %xml.display(),
        );
        Some(Self::new(xml, device, unpadded_vocab))
    }

    pub fn new(xml: PathBuf, device: String, unpadded_vocab: usize) -> Self {
        Self {
            xml,
            device,
            plugin: PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f16"),
            unpadded_vocab,
            rt: Mutex::new(None),
            failed: AtomicBool::new(false),
            noted: AtomicBool::new(false),
            calls: AtomicU64::new(0),
            call_ns: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
        }
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn stats(&self) -> OvHeadStats {
        OvHeadStats {
            calls: self.calls.load(Ordering::Relaxed),
            call_ns: self.call_ns.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
        }
    }

    /// Compile the head ahead of time and take the one-row shape's first-call
    /// cost; `false` once the IR proved unusable.
    pub fn warm(&self) -> bool {
        if self.failed.load(Ordering::Relaxed) {
            return false;
        }
        let mut g = self.rt.lock().expect("OV head lock");
        if g.is_none() {
            let Some(p) = self.xml.to_str() else {
                drop(g);
                self.mark_failed("non-utf8 IR path");
                return false;
            };
            match Runtime::compile(p, &self.device, &self.plugin) {
                Ok(rt) => *g = Some(rt),
                Err(e) => {
                    drop(g);
                    self.mark_failed(&format!("compile on {}: {e}", self.device));
                    return false;
                }
            }
        }
        true
    }

    /// Logits for one normed, mup-divided hidden state (`x` = `[hidden]`),
    /// sliced to `unpadded_vocab`; `None` to take the Rust head.
    pub fn logits(&self, x: &[f32]) -> Option<Vec<f32>> {
        if !self.warm() {
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let t0 = Instant::now();
        let mut g = self.rt.lock().expect("OV head lock");
        let rt = g.as_mut()?;
        let step = rt
            .set_input("x", DType::F32, &[1, 1, x.len()], f32_bytes(x))
            .map_err(|e| format!("set_input: {e}"))
            .and_then(|_| rt.infer().map_err(|e| format!("infer: {e}")));
        if let Err(why) = step {
            drop(g);
            self.note(&why);
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let (_, _, bytes) = match rt.output(0) {
            Ok(o) => o,
            Err(e) => {
                drop(g);
                self.note(&format!("output: {e}"));
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        drop(g);
        let n = bytes.len() / 4;
        if n < self.unpadded_vocab {
            self.mark_failed(&format!(
                "output {n} logits < unpadded_vocab {}",
                self.unpadded_vocab
            ));
            self.fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let out: Vec<f32> = bytes
            .chunks_exact(4)
            .take(self.unpadded_vocab)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(out)
    }

    fn note(&self, why: &str) {
        if !self.noted.swap(true, Ordering::Relaxed) {
            warn!("inkling OV head call failed ({why}); Rust head for this call");
            eprintln!("[inkling] OV head call failed: {why}");
        }
    }

    fn mark_failed(&self, why: &str) {
        if !self.failed.swap(true, Ordering::Relaxed) {
            warn!("inkling OV head IR unusable ({why}); Rust head from here on");
            eprintln!("[inkling] OV head unusable: {why}");
        }
    }
}

impl std::fmt::Debug for OvHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OvHead")
            .field("xml", &self.xml)
            .field("device", &self.device)
            .finish()
    }
}
