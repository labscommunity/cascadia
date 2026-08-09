//! OpenVINO attention backend for glm5 prefill (iGPU offload, T3).
//!
//! Runs one MLA-attention window (`x[rows,hidden]` + padded past Lc/Rc +
//! runtime mask) through a per-layer compiled IR instead of the Rust absorbed
//! decode kernel, for prefill only. Counterpart of [`super::ov_expert`] for
//! the attention block: same shim compile/infer plumbing, same poison-latch
//! shape, but its OWN latch (not shared) and NOT the same silent-degrade
//! failure mode — see below.
//!
//! Graphs come from `tools/glm5_attn_ov.py` (`<model>/attn_ov/layer_NN.{xml,bin}`
//! and `export_stamp.json`), one graph per GLOBAL layer index, ONE compile
//! per rank-owned layer (spec r3 Sec3.2: no per-past-length bucket variants —
//! they duplicate the attention shell's weights into the same iGPU pool that
//! killed ranks at 24 GiB held). The graph's row capacity (`rows_cap`, baked
//! at export = `MAX_BATCH_COUNT`) and past capacity (`p_max`) are FIXED; the
//! true past length enters only through the additive mask input built per
//! call (spec Sec3.1) — never through the tensor shape.
//!
//! # Silent-degrade discipline (why this file exists as a separate module)
//!
//! [`super::ov_expert::OvExperts`] returns `None` on a transient device error
//! without logging, so a persistently failing device degrades silently (a
//! documented weakness, not copied here). This module:
//! - gives every startup failure mode (missing dir, stale stamp, W mismatch,
//!   incomplete stamp coverage, unavailable runtime, precision drift, bf16
//!   canary failure, partial compile) its own named `event=...` at WARN/ERROR,
//!   never a bare `None`;
//! - latches on N (default 3) CONSECUTIVE transient infer failures, not just
//!   on typed resource exhaustion — a permanent compute-then-discard-then-redo
//!   cycle is a ~2x prefill regression that per-request breadcrumbs alone
//!   would not surface.
//!
//! # Local vs global layer indexing
//!
//! [`OvAttn::compiled`] is keyed by LOCAL offset (this rank's position in the
//! `owned_layers` slice passed to [`OvAttn::from_opts`]), matching
//! `layer_local` in [`OvAttn::prefill_window`]. `GlmRunner` stores no global
//! layer index and `GlmLayer`/`forward_layers_batch` iterate without one, so
//! keying locally avoids a signature change rippling into
//! `GlmModel::prefill_h`/`forward_batch_h`. On-disk filenames and the export
//! stamp keep GLOBAL indices; [`OvAttn::from_opts`] translates once.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use cascadia_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};
use serde::Deserialize;
use tracing::{error, info, warn};

use super::loader::GlmManifest;
use super::ov_expert::is_fatal_resource_error;
use super::stage::StageOpts;
use crate::dsv4::math::to_bf16;

/// This crate's expected `cpp/shim.h` ABI version. Bump alongside
/// `CASCADIA_SHIM_ABI_VERSION` in the shim header whenever a symbol this file
/// depends on (`cascadia_runtime_get_property`, `cascadia_runtime_compile_bf16_canary`)
/// changes. A mismatch means an un-redeployed shim — see the module doc.
const EXPECTED_SHIM_ABI_VERSION: i32 = 2;

/// Consecutive transient infer failures that latch the path (spec r3 Sec5).
const TRANSIENT_LATCH_THRESHOLD: u32 = 3;

/// Dense-exact offload budget cap for a no-indexer model (spec Sec4): the
/// validated static-shape ceiling, never `usize::MAX`/0.
pub const NO_INDEXER_W: usize = 2048;

/// Additive mask value for a masked (non-attendable) column — matches
/// `tools/glm5_attn_ov.py::MASK_NEG` exactly (an arbitrary-but-shared huge
/// negative f32, not `f32::NEG_INFINITY`: the graphs are exported with this
/// literal baked into `--validate-graph`'s reference, so the host-built mask
/// must match bit-for-bit-equivalent semantics, not merely "very negative").
const MASK_NEG: f32 = -1e30;

/// One window's real-row outputs (`[rows, dim]`, row-major f32) — the
/// real-rows-only slice of the graph's `[rows_cap, dim]` outputs.
pub struct AttnWindowOut {
    pub attn_out: Vec<f32>, // [rows, hidden]
    pub lc: Vec<f32>,       // [rows, kv_lora]
    pub rc: Vec<f32>,       // [rows, qk_rope]
}

/// Why one prefill window did NOT run on the OV path — the CLOSED vocabulary
/// the `event=ov_attn_prefill` line and the cumulative [`stats`] summary both
/// report. Every fallback must map to exactly one of these: a window that
/// silently reverts to Rust while the rank still advertises `Active` is this
/// project's signature defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvSkipReason {
    /// The backend latched off (see [`OvAttnState::Poisoned`]).
    Poisoned,
    /// This local layer's IR never compiled — permanently ineligible.
    LayerNotCompiled,
    /// `base + rows` exceeds the offload budget `W`, or `rows` exceeds the
    /// graph's baked row capacity: the window does not fit the static shape.
    WindowTooLarge,
    /// Fewer real rows than the resolved `min_rows` floor — too small to pay
    /// for the past-KV upload.
    RowsBelowMin,
    /// [`OvAttn::prefill_window`] returned `None` (device error, shape
    /// mismatch, latch tripped mid-call).
    InferFailed,
    /// The graph produced a NaN/Inf in a real row.
    NonFinite,
    /// The window could not be committed:
    /// [`super::attn::AttentionLayer::commit_prefill_rows`] returned `Err`, or
    /// the pre-commit shape check rejected `attn_out`. The two are told apart by
    /// the WARN that accompanies them (`ov_attn_commit_failed` vs
    /// `ov_attn_bad_shape`); neither wrote anything.
    CommitFailed,
}

impl OvSkipReason {
    /// Number of variants — the width of [`stats`]'s per-reason counter array.
    const COUNT: usize = 7;

    fn idx(self) -> usize {
        match self {
            OvSkipReason::Poisoned => 0,
            OvSkipReason::LayerNotCompiled => 1,
            OvSkipReason::WindowTooLarge => 2,
            OvSkipReason::RowsBelowMin => 3,
            OvSkipReason::InferFailed => 4,
            OvSkipReason::NonFinite => 5,
            OvSkipReason::CommitFailed => 6,
        }
    }

    /// Stable log token. `"none"` is reserved for "nothing was skipped" and is
    /// deliberately NOT a variant — it is not a reason.
    pub fn as_str(self) -> &'static str {
        match self {
            OvSkipReason::Poisoned => "poisoned",
            OvSkipReason::LayerNotCompiled => "layer_not_compiled",
            OvSkipReason::WindowTooLarge => "window_too_large",
            OvSkipReason::RowsBelowMin => "rows_below_min",
            OvSkipReason::InferFailed => "infer_failed",
            OvSkipReason::NonFinite => "non_finite",
            OvSkipReason::CommitFailed => "commit_failed",
        }
    }

    fn from_idx(i: usize) -> Self {
        [
            OvSkipReason::Poisoned,
            OvSkipReason::LayerNotCompiled,
            OvSkipReason::WindowTooLarge,
            OvSkipReason::RowsBelowMin,
            OvSkipReason::InferFailed,
            OvSkipReason::NonFinite,
            OvSkipReason::CommitFailed,
        ][i]
    }
}

/// Per-`forward_layers_batch` accounting for the OV attention route, filled in
/// layer by layer and emitted once as `event=ov_attn_prefill`.
#[derive(Default)]
pub struct OvPrefillTally {
    pub layers_ov: u32,
    pub layers_rust: u32,
    /// The FIRST layer's skip reason in this call. First (not last, not "most
    /// severe") because the layers run in order and the earliest fallback is
    /// the one an operator should chase — a later `infer_failed` is usually a
    /// consequence of whatever made the first layer bail.
    first_skip: Option<OvSkipReason>,
}

impl OvPrefillTally {
    /// Record that one layer's window ran on OV.
    pub fn note_used(&mut self) {
        self.layers_ov += 1;
        stats::record_used();
    }

    /// Record that one layer's window fell back to Rust, and why.
    pub fn note_skip(&mut self, reason: OvSkipReason) {
        self.layers_rust += 1;
        self.first_skip.get_or_insert(reason);
        stats::record_skip(reason);
    }

    /// The `skipped_reason` log field: `"none"` when every layer ran on OV.
    pub fn skipped_reason(&self) -> &'static str {
        self.first_skip.map(OvSkipReason::as_str).unwrap_or("none")
    }

    /// At least one layer's window ran on OV.
    pub fn used(&self) -> bool {
        self.layers_ov > 0
    }
}

/// One layer's OV routing context, handed to
/// [`super::model::GlmLayer::forward_prefill`]. Bundled into a struct (rather
/// than three more parameters) so the OFF path stays a single `None`.
pub struct OvRoute<'a> {
    pub ov: &'a OvAttn,
    /// LOCAL layer offset — see the module doc's indexing contract.
    pub layer_local: usize,
    pub tally: &'a mut OvPrefillTally,
}

/// The closed enablement-state vocabulary this module and its callers share.
///
/// KNOWN GAP: spec Sec7 asks for this on a rank `event=ready` line. No such
/// line exists and there is no engine→control-plane path to carry it (see
/// `GlmRunner::ov_attn`), so today it is only reachable in-process. The
/// shipped operator-facing signals are the startup `event=ov_attn_*` logs and
/// doctor's stamp advisory.
///
/// A
/// LIVE [`OvAttn`] (i.e. [`OvAttn::from_opts`] returned `Some`) only ever
/// reports `Enabled`/`Active`/`Partial`/`Poisoned`: the other three name the
/// reason a `None` was returned, which callers recover from the specific
/// `event=...` this module logged at that `None` (there is no instance left
/// to query once construction itself failed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvAttnState {
    /// `ov_attn` not enabled (config/env). No instance exists in this case.
    Off,
    /// Compiled and passed every precondition, but has not served a window yet.
    Enabled,
    /// Has served at least one window successfully.
    Active,
    /// Some owned layers never compiled; those windows always fall back.
    Partial,
    /// Latched off after a fatal or repeated-transient device error.
    Poisoned,
    /// Startup probe failed (stub shim build, ABI mismatch, device probe,
    /// or zero layers compiled). No instance exists in this case.
    Unavailable,
    /// Export stamp missing, stale, or W-mismatched. No instance exists in
    /// this case.
    StaleIr,
}

/// The exporter generation this engine accepts. Bumped whenever the exported
/// graph's numerics change, so an IR set built by an older `glm5_attn_ov.py`
/// is rejected rather than silently serving different math. Must track
/// `tools/glm5_attn_ov.py::EXPORTER_VERSION`.
const EXPECTED_EXPORTER_VERSION: &str = "2";

#[derive(Deserialize)]
struct ExportStamp {
    manifest_sha256: String,
    /// sha256 of each owned layer's `shells/layer_NN.safetensors` at export
    /// time — the "built from the same weights" half.
    per_layer_digest: HashMap<String, String>,
    /// sha256 of `attn_ov/layer_NN.xml` concatenated with `layer_NN.bin` —
    /// the "these are the artifacts that export produced" half. Defaulted so
    /// a pre-v2 stamp reaches the version check with a legible reason instead
    /// of failing as an unparseable blob.
    #[serde(default)]
    ir_digest: HashMap<String, String>,
    exporter_version: String,
    w: usize,
    p_max: usize,
    rows: usize,
    hidden: usize,
    kv_lora: usize,
    qk_rope: usize,
}

/// Pure result of comparing a loaded [`ExportStamp`] against this rank's live
/// config — factored out so it is testable without touching the filesystem or
/// OpenVINO (Task 5 Step 2's "return-reason enum tested directly" fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StampCheck {
    Ok,
    /// `p_max + rows != w` — the stamp's own internal invariant (the
    /// exporter guarantees this by construction: `p_max = w - rows`) doesn't
    /// hold. A corrupt/hand-edited/truncated stamp, checked BEFORE comparing
    /// against the live manifest at all, since there's no point asking "is
    /// this stale" about an artifact that was never internally consistent.
    CorruptDims,
    /// `manifest_sha256` disagrees: the model was re-exported (weights
    /// changed) without re-running the attention exporter.
    ManifestMismatch,
    /// The stamp's baked `W` disagrees with what THIS manifest derives.
    WMismatch,
    /// The stamp was written by a different exporter generation, whose graph
    /// numerics this engine makes no claim about.
    ExporterVersionMismatch,
    /// The stamp's `per_layer_digest` / `ir_digest` has no entry for an owned
    /// global layer (e.g. exported with a `--layers` filter that missed this
    /// rank's slice).
    MissingCoverage,
}

fn check_stamp(
    stamp: &ExportStamp,
    manifest_sha256: &str,
    derived_w: usize,
    owned_layers: &[usize],
) -> StampCheck {
    if stamp.p_max + stamp.rows != stamp.w {
        return StampCheck::CorruptDims;
    }
    if stamp.exporter_version != EXPECTED_EXPORTER_VERSION {
        return StampCheck::ExporterVersionMismatch;
    }
    if stamp.manifest_sha256 != manifest_sha256 {
        return StampCheck::ManifestMismatch;
    }
    if stamp.w != derived_w {
        return StampCheck::WMismatch;
    }
    for &gl in owned_layers {
        let key = format!("{gl:02}");
        if !stamp.per_layer_digest.contains_key(&key) || !stamp.ir_digest.contains_key(&key) {
            return StampCheck::MissingCoverage;
        }
    }
    StampCheck::Ok
}

/// Streaming sha256 of a file — the shells and IR `.bin`s are large enough
/// that reading them whole just to hash them is not worth the resident bytes.
fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let mut s = String::with_capacity(64);
    for b in hasher.finalize() {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    Ok(s)
}

/// Re-derive the digests the stamp recorded and compare them, so spec Sec6's
/// "built from the same weights is verified, not assumed" covers all three
/// artifacts and not just `manifest.json`. Returns the failing layer and the
/// artifact class on the first mismatch.
///
/// `contains_key` alone (what this used to be) accepts a stale `layer_NN.bin`
/// from an older exporter sitting beside a fresh stamp — the fleet's named
/// stale-artifact class, and the one thing the handshake exists to stop.
fn verify_artifact_digests(
    model_dir: &Path,
    ir_dir: &Path,
    stamp: &ExportStamp,
    owned_layers: &[usize],
) -> Result<(), String> {
    for &gl in owned_layers {
        let key = format!("{gl:02}");

        let shell = model_dir.join(format!("shells/layer_{gl:02}.safetensors"));
        let got = sha256_file(&shell).map_err(|e| format!("layer {key}: {shell:?}: {e}"))?;
        if Some(&got) != stamp.per_layer_digest.get(&key) {
            return Err(format!(
                "layer {key}: shell weights changed since export (shells/layer_{key}.safetensors)"
            ));
        }

        // One digest over xml||bin: the pair is only ever produced and only
        // ever consumed together, so a per-file split buys no extra signal.
        let xml = ir_dir.join(format!("layer_{gl:02}.xml"));
        let bin = ir_dir.join(format!("layer_{gl:02}.bin"));
        let joined = {
            use sha2::{Digest, Sha256};
            use std::fmt::Write;
            let mut h = Sha256::new();
            for p in [&xml, &bin] {
                h.update(std::fs::read(p).map_err(|e| format!("layer {key}: {p:?}: {e}"))?);
            }
            let mut s = String::with_capacity(64);
            for b in h.finalize() {
                let _ = write!(s, "{b:02x}");
            }
            s
        };
        if Some(&joined) != stamp.ir_digest.get(&key) {
            return Err(format!(
                "layer {key}: IR does not match the stamp (layer_{key}.xml/.bin)"
            ));
        }
    }
    Ok(())
}

/// `W` per spec Sec4: `index_topk` when the manifest attaches an indexer with
/// a real (nonzero) `index_topk`; otherwise the unbounded-causal cap. Mirrors
/// `glm/loader.rs`'s indexer-attach clamp AND `tools/glm5_attn_ov.py::derive_w`
/// — kept in sync by hand (no shared crate between the Python exporter and
/// this engine); a divergence here would silently accept a graph exported for
/// a different W. `index_n_heads>0 && index_topk==0` mirrors the exporter's
/// hard failure at export time (no valid stamp can exist for that manifest);
/// returning `usize::MAX` here — never used as a real dimension, since this
/// value can only ever reach [`check_stamp`]'s equality check — guarantees a
/// stamp-W mismatch rather than an accidental match.
fn derive_w(m: &GlmManifest) -> usize {
    if m.index_n_heads > 0 {
        if m.index_topk == 0 {
            usize::MAX
        } else {
            m.index_topk
        }
    } else {
        NO_INDEXER_W
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// `x[n]` cast to a `&[u8]` byte view for `set_input` — mirrors
/// `ov_expert.rs::f32_bytes`, duplicated locally rather than shared across
/// modules for a 3-line helper.
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns; lifetime tied to `v`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Config-wins-over-env resolution for a boolean knob, plus the source label
/// every startup log records (spec Sec7: "config wins over env when both
/// set... logs the effective value AND its source").
/// The env value is PARSED, not merely probed for presence: the runbook's
/// emergency rollback tells an operator to turn this knob off, and
/// `CASCADIA_GLM5_OV_ATTN=0` must not be the thing that enables it.
fn resolve_bool(cfg: Option<bool>, env_key: &str) -> (bool, &'static str) {
    match cfg {
        Some(v) => (v, "engine_arg"),
        None => match std::env::var_os(env_key) {
            Some(raw) => (env_value_is_on(&raw), "env"),
            None => (false, "default"),
        },
    }
}

/// `0` / `false` / `no` / `off` / empty (any case, surrounding space ignored)
/// mean OFF; every other present value means ON. Deliberately permissive on
/// the ON side — an operator who mistypes the enable value gets the safe
/// answer only for the spellings that unambiguously say "off".
fn env_value_is_on(raw: &std::ffi::OsStr) -> bool {
    match raw.to_str() {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        // Non-UTF-8 is not any of the off spellings, so it reads as "set".
        None => true,
    }
}

fn resolve_string(cfg: Option<String>, env_key: &str, default: &str) -> (String, &'static str) {
    if let Some(v) = cfg {
        return (v, "engine_arg");
    }
    if let Ok(v) = std::env::var(env_key) {
        if !v.trim().is_empty() {
            return (v, "env");
        }
    }
    (default.to_string(), "default")
}

fn resolve_u32(cfg: Option<u32>, env_key: &str, default: u32) -> (u32, &'static str) {
    if let Some(v) = cfg {
        return (v, "engine_arg");
    }
    if let Ok(v) = std::env::var(env_key) {
        if let Ok(n) = v.trim().parse::<u32>() {
            return (n, "env");
        }
    }
    (default, "default")
}

/// Whether `ov_attn` is REQUESTED and targets a non-CPU device, resolved with
/// the same config→env precedence [`OvAttn::from_opts`] uses.
///
/// Read before construction, and keyed on the request rather than on a
/// successful compile, because both callers act on the *attempt*: the
/// experts-OV conflict must be rejected even if the attention compile would
/// later have failed, and expert pinning must be skipped because the compile
/// is what needs the GPU's RAM pool — a run where pinning already starved it
/// into failing is exactly the case a construction-keyed check would miss.
pub fn requested_on_accelerator(opts: &StageOpts) -> bool {
    let (enabled, _) = resolve_bool(opts.ov_attn, "CASCADIA_GLM5_OV_ATTN");
    if !enabled {
        return false;
    }
    let (device, _) = resolve_string(
        opts.ov_attn_device.clone(),
        "CASCADIA_GLM5_OV_ATTN_DEVICE",
        "GPU",
    );
    !device.eq_ignore_ascii_case("CPU")
}

/// Host-side additive mask, matching `tools/glm5_attn_ov.py::build_window_mask`
/// exactly: real past occupies columns `[0,past_len)`, padding
/// `[past_len,p_max)` is masked, and window columns `[p_max,p_max+rows_cap)`
/// are causal per row (row `i` sees `p_max..=p_max+i`). Applied uniformly to
/// EVERY row of the graph's fixed `rows_cap`, including padding rows beyond
/// the real window length — a padding row still sees its own diagonal (never
/// fully masked, so it cannot soften-NaN and trip the latch); its output row
/// is discarded by the caller's real-rows slice regardless.
fn build_window_mask(rows_cap: usize, p_max: usize, past_len: usize) -> Vec<f32> {
    let w = p_max + rows_cap;
    let mut m = vec![MASK_NEG; rows_cap * w];
    let past_real = past_len.min(p_max);
    for i in 0..rows_cap {
        let row = &mut m[i * w..(i + 1) * w];
        row[..past_real].fill(0.0);
        let hi = (p_max + i + 1).min(w);
        row[p_max..hi].fill(0.0);
    }
    m
}

// --------------------------------------------------------------------------
// bf16 device canary (Task 5 Step 0b / spec Sec9 gate 0) — ported from
// tools/glm5_attn_ov.py::canary_cases()/BF16_TIE_CASES. Pure host-side
// generation (no OV needed to BUILD the battery, only to RUN it on-device).
// --------------------------------------------------------------------------

/// Bases deliberately not at a power-of-two magnitude boundary (see the
/// Python `_CANARY_BASES` comment: at an exact power of two, the next bf16
/// grid point toward zero crosses into a finer-spaced exponent bucket and the
/// `_bf16_ulp_at`-then-add trick silently computes the wrong `g_hi`). Mix of
/// even/odd bf16 mantissas and both signs so RNE and truncation's
/// round-toward-zero direction are both exercised both ways.
const CANARY_BASES: [f32; 12] = [
    1.5, 1.0234375, -1.5, -1.0234375, 12.0, 12.1875, -12.0, -12.1875, 100.0, -100.0, 0.0234375,
    -0.0234375,
];

/// `{x_bits, want_bf16_bits}` pairs — IDENTICAL to Python's `BF16_TIE_CASES`
/// (ties, zero/signed-zero, infinities, NaN canonicalization, and two
/// non-tie near-boundary cases). `want` is the bf16 bit pattern in the LOW 16
/// bits; the graph's f32 output widens it to `want << 16`.
const BF16_TIE_CASES: [(u32, u16); 20] = [
    (0x3F808000, 0x3F80),
    (0x3F818000, 0x3F82),
    (0x3F808001, 0x3F81),
    (0x3F817FFF, 0x3F81),
    (0xBF808000, 0xBF80),
    (0xBF818000, 0xBF82),
    (0x40FF8000, 0x4100),
    (0x3F800000, 0x3F80),
    (0x00000000, 0x0000),
    (0x80000000, 0x8000),
    (0x7F800000, 0x7F80),
    (0xFF800000, 0xFF80),
    (0x7FC00000, 0x7FC0),
    (0x7F800001, 0x7FC0),
    (0x00000001, 0x0000),
    (0x3F7FFFFF, 0x3F80),
    (0x42F60000, 0x42F6),
    (0x42F68000, 0x42F6),
    (0x42F78000, 0x42F8),
    (0xC2F68000, 0xC2F6),
];

/// The bf16 ULP (spacing between adjacent bf16-representable values) at the
/// magnitude of `x` — `math.frexp`'s exponent convention (`x = m*2^e`,
/// `0.5<=|m|<1`) computed via the f32 exponent bit field, exact for the
/// non-subnormal `x` this battery only ever calls it with.
fn bf16_ulp_at(x: f32) -> f32 {
    debug_assert!(x.is_finite() && x != 0.0);
    let bits = x.to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    debug_assert!(exp_field != 0, "bf16_ulp_at: unexpected subnormal");
    let frexp_e = exp_field - 127 + 1;
    2f32.powi(frexp_e - 8)
}

/// One canary case plus the OTHER three rounding hypotheses it could have
/// matched instead — ported from Python's `canary_cases()`, which computes
/// these so a device mismatch is diagnosable without re-deriving the battery
/// by hand (`run_bf16_canary` reports which hypothesis, if any, the observed
/// bits actually matched).
struct CanaryCase {
    label: String,
    x: f32,
    rne: f32,
    trunc: f32,
    half_away: f32,
    identity: f32,
}

fn canary_cases() -> Vec<CanaryCase> {
    let mut cases = Vec::with_capacity(CANARY_BASES.len() * 3);
    for &base in CANARY_BASES.iter() {
        let g_lo = to_bf16(base);
        let ulp = bf16_ulp_at(g_lo);
        let g_hi = to_bf16(g_lo + ulp);
        for (frac, tag) in [(0.25f32, "f=.25"), (0.5, "f=.5(tie)"), (0.75, "f=.75")] {
            let x = g_lo + frac * ulp;
            let rne = to_bf16(x);
            // Matches Python's canary_cases(). Truncation rounds toward ZERO,
            // so it lands on the smaller-magnitude of g_lo/g_hi -- and since
            // g_hi = g_lo + ulp, that flips with the sign of the base, at
            // every frac including the tie. half-away-from-zero is plain
            // nearest except at the exact tie, where it breaks toward the
            // larger magnitude (g_hi for base>0; g_lo, the more-negative one,
            // for base<0).
            let trunc = if base > 0.0 { g_lo } else { g_hi };
            let half_away = if frac < 0.5 || (frac == 0.5 && base < 0.0) {
                g_lo
            } else {
                g_hi
            };
            cases.push(CanaryCase {
                label: format!("base={base} {tag}"),
                x,
                rne,
                trunc,
                half_away,
                identity: x,
            });
        }
    }
    cases
}

/// Runs the shipped bf16-roundtrip canary graph on `device` with `plugin`
/// (the SAME compile config the real per-layer graphs use), feeds
/// [`canary_cases`] plus [`BF16_TIE_CASES`], and asserts RAW f32 bit patterns
/// match RNE exactly. `true` = every case matched. Logs its own
/// `event=ov_attn_bf16_canary_failed` on any failure path (compile, infer, or
/// a bit mismatch) so the caller only needs to act on the boolean.
fn run_bf16_canary(device: &str, plugin: &PluginConfig) -> bool {
    let battery = canary_cases();
    let mut xin: Vec<f32> = battery.iter().map(|c| c.x).collect();
    xin.extend(BF16_TIE_CASES.iter().map(|&(bits, _)| f32::from_bits(bits)));
    let mut want_bits: Vec<u32> = battery.iter().map(|c| c.rne.to_bits()).collect();
    want_bits.extend(BF16_TIE_CASES.iter().map(|&(_, w16)| (w16 as u32) << 16));

    let mut rt = match Runtime::compile_bf16_canary(xin.len(), device, plugin) {
        Ok(rt) => rt,
        Err(e) => {
            warn!(
                target: "cascadia::glm5",
                event = "ov_attn_bf16_canary_failed",
                device = %device,
                reason = %format!("compile: {e}"),
            );
            return false;
        }
    };
    if let Err(e) = rt.set_input("x", DType::F32, &[xin.len()], f32_bytes(&xin)) {
        warn!(
            target: "cascadia::glm5",
            event = "ov_attn_bf16_canary_failed",
            device = %device,
            reason = %format!("set_input: {e}"),
        );
        return false;
    }
    if let Err(e) = rt.infer() {
        warn!(
            target: "cascadia::glm5",
            event = "ov_attn_bf16_canary_failed",
            device = %device,
            reason = %format!("infer: {e}"),
        );
        return false;
    }
    let bytes = match rt.output(0) {
        Ok((_, _, b)) => b,
        Err(e) => {
            warn!(
                target: "cascadia::glm5",
                event = "ov_attn_bf16_canary_failed",
                device = %device,
                reason = %format!("output: {e}"),
            );
            return false;
        }
    };
    if bytes.len() != xin.len() * 4 {
        warn!(
            target: "cascadia::glm5",
            event = "ov_attn_bf16_canary_failed",
            device = %device,
            reason = %format!("output byte len {} != {}", bytes.len(), xin.len() * 4),
        );
        return false;
    }
    let mut mismatches = 0usize;
    let n_battery = battery.len();
    for (i, &want) in want_bits.iter().enumerate() {
        let got = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        if got != want {
            mismatches += 1;
            if i < n_battery {
                let c = &battery[i];
                // Which OTHER hypothesis (if any) the observed bits match --
                // makes a real device failure diagnosable without re-deriving
                // the battery by hand (ported from Python's run_canary).
                let matched = if got == c.trunc.to_bits() {
                    "trunc"
                } else if got == c.half_away.to_bits() {
                    "half_away"
                } else if got == c.identity.to_bits() {
                    "identity"
                } else {
                    "unknown"
                };
                warn!(
                    target: "cascadia::glm5",
                    case = %c.label,
                    want = format!("0x{want:08x}"),
                    got = format!("0x{got:08x}"),
                    matches = matched,
                    "bf16 canary case mismatch",
                );
            } else {
                warn!(
                    target: "cascadia::glm5",
                    case = %format!("tie_case idx={}", i - n_battery),
                    want = format!("0x{want:08x}"),
                    got = format!("0x{got:08x}"),
                    "bf16 canary case mismatch",
                );
            }
        }
    }
    if mismatches > 0 {
        warn!(
            target: "cascadia::glm5",
            event = "ov_attn_bf16_canary_failed",
            device = %device,
            mismatches,
            total = xin.len(),
        );
        false
    } else {
        true
    }
}

/// Reads back `INFERENCE_PRECISION_HINT`/`DYNAMIC_QUANTIZATION_GROUP_SIZE`
/// from a COMPILED model and asserts they landed on the requested values (a
/// hint is not a guarantee). `Err` carries a human-readable reason.
fn check_precision(rt: &Runtime) -> Result<(), String> {
    let prec = rt
        .property("INFERENCE_PRECISION_HINT")
        .map_err(|e| format!("INFERENCE_PRECISION_HINT read failed: {e}"))?;
    if !prec.trim().eq_ignore_ascii_case("f32") {
        return Err(format!(
            "INFERENCE_PRECISION_HINT effective={prec:?}, requested f32"
        ));
    }
    let dq = rt
        .property("DYNAMIC_QUANTIZATION_GROUP_SIZE")
        .map_err(|e| format!("DYNAMIC_QUANTIZATION_GROUP_SIZE read failed: {e}"))?;
    if dq.trim() != "0" {
        return Err(format!(
            "DYNAMIC_QUANTIZATION_GROUP_SIZE effective={dq:?}, requested 0"
        ));
    }
    Ok(())
}

/// Eagerly-compiled per-layer OpenVINO attention backend. See the module doc
/// for the local-vs-global indexing contract and the silent-degrade
/// discipline every construction/runtime failure follows.
pub struct OvAttn {
    /// Compiled runtimes keyed by LOCAL offset (index into the `owned_layers`
    /// slice `from_opts` was built with). `None` at an index means that
    /// layer's IR did not compile — permanently ineligible, not retried.
    /// Single `Mutex` (not per-layer): `forward_prefill` drives layers
    /// sequentially within a rank, so this is uncontended in practice, same
    /// reasoning as `OvExperts::cache`.
    compiled: Mutex<Vec<Option<Runtime>>>,
    device: String,
    /// Offload budget (spec Sec4): the caller's eligibility check
    /// (`base + rows <= w`) reads this; not enforced inside this struct.
    pub w: usize,
    /// Fixed past-KV capacity every compiled graph expects (`w - rows_cap`).
    pub p_max: usize,
    /// Fixed window row capacity every compiled graph expects (baked at
    /// export, `MAX_BATCH_COUNT`). Real windows with fewer rows are
    /// host-side zero-padded up to this by `prefill_window`.
    pub rows_cap: usize,
    kv_lora: usize,
    qk_rope: usize,
    hidden: usize,
    layers_ok: usize,
    layers_expected: usize,
    /// Resolved minimum real rows a window needs to be eligible (spec
    /// requirement: config→env→64, see [`OvAttn::from_opts`]). Stored (not
    /// just logged) so the prefill-integration caller consumes the SAME
    /// resolved value `ov_attn_config` reported, instead of re-deriving the
    /// config/env/default precedence a second time.
    pub min_rows: u32,
    poisoned: AtomicBool,
    transient_fails: AtomicU32,
    used: AtomicBool,
    /// Latches the ONE-TIME warning for a caller-side contract violation in
    /// [`OvAttn::prefill_window`] (bad shapes, `rows>rows_cap`,
    /// `past_len>p_max`) — see `warn_contract_violation_once`'s doc.
    contract_violation_warned: AtomicBool,
    /// Test-only canned-output source; the field itself does not exist in a
    /// non-test build. See [`OvAttn::mock`].
    #[cfg(test)]
    mock: Option<MockBackend>,
}

/// Canned per-window outputs for the prefill-seam tests: `(layer_local, x,
/// rows, past_len) -> Option<AttnWindowOut>`.
#[cfg(test)]
pub(crate) type MockWindowFn =
    Box<dyn Fn(usize, &[f32], usize, usize) -> Option<AttnWindowOut> + Send + Sync>;

#[cfg(test)]
pub(crate) struct MockBackend {
    /// Per-LOCAL-offset compile success, standing in for the `compiled` vec a
    /// real build fills — so `layer_not_compiled` is reachable in tests.
    compiled: Vec<bool>,
    window: MockWindowFn,
}

/// Shape/budget knobs for [`OvAttn::mock`]. A struct, not eight arguments.
#[cfg(test)]
pub(crate) struct MockCfg {
    pub compiled: Vec<bool>,
    pub w: usize,
    pub p_max: usize,
    pub rows_cap: usize,
    pub hidden: usize,
    pub kv_lora: usize,
    pub qk_rope: usize,
    pub min_rows: u32,
}

impl OvAttn {
    /// `None` unless enabled + dir/stamp present and fresh + startup probe +
    /// bf16 canary all pass. Eager-compiles every owned layer (keyed by LOCAL
    /// offset, translated from `owned_layers`'s GLOBAL indices here); logs
    /// `event=ov_attn_config` with the spec Sec7 fingerprint on success. A
    /// partial compile (some but not all owned layers) still returns `Some`
    /// (those layers permanently fall back) but logs `event=ov_attn_partial`.
    ///
    /// Every OTHER failure mode returns `None` with its own named event —
    /// see the module doc's event table (also in the task report).
    pub fn from_opts(
        model_dir: &Path,
        owned_layers: &[usize],
        m: &GlmManifest,
        opts: &StageOpts,
    ) -> Option<Self> {
        let (enabled, enabled_src) = resolve_bool(opts.ov_attn, "CASCADIA_GLM5_OV_ATTN");
        if !enabled {
            return None; // off means off (spec Sec9 gate 3): no event, matches OvExperts precedent
        }
        let (device, device_src) = resolve_string(
            opts.ov_attn_device.clone(),
            "CASCADIA_GLM5_OV_ATTN_DEVICE",
            "GPU",
        );
        let (min_rows, min_rows_src) =
            resolve_u32(opts.ov_attn_min_rows, "CASCADIA_GLM5_OV_ATTN_MIN_ROWS", 64);

        let dir = model_dir.join("attn_ov");
        let stamp_path = dir.join("export_stamp.json");
        let stamp_raw = match std::fs::read(&stamp_path) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_ir_missing",
                    path = %stamp_path.display(),
                    error = %e,
                );
                return None;
            }
        };
        let stamp: ExportStamp = match serde_json::from_slice(&stamp_raw) {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_ir_missing",
                    path = %stamp_path.display(),
                    reason = %format!("stamp parse failed: {e}"),
                );
                return None;
            }
        };
        let stamp_hash = sha256_hex(&stamp_raw);

        let manifest_bytes = match std::fs::read(model_dir.join("manifest.json")) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_ir_missing",
                    reason = %format!("manifest.json unreadable: {e}"),
                );
                return None;
            }
        };
        let found_sha = sha256_hex(&manifest_bytes);
        let derived_w = derive_w(m);
        match check_stamp(&stamp, &found_sha, derived_w, owned_layers) {
            StampCheck::Ok => {}
            StampCheck::CorruptDims => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_ir_missing",
                    reason = "corrupt_stamp_dims",
                    p_max = stamp.p_max,
                    rows = stamp.rows,
                    w = stamp.w,
                );
                return None;
            }
            StampCheck::ManifestMismatch => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_stale_ir",
                    reason = "manifest_sha_mismatch",
                    expected = %stamp.manifest_sha256,
                    found = %found_sha,
                );
                return None;
            }
            StampCheck::WMismatch => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_w_mismatch",
                    expected = derived_w,
                    found = stamp.w,
                );
                return None;
            }
            StampCheck::ExporterVersionMismatch => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_stale_ir",
                    reason = "exporter_version_mismatch",
                    expected = EXPECTED_EXPORTER_VERSION,
                    found = %stamp.exporter_version,
                );
                return None;
            }
            StampCheck::MissingCoverage => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_stale_ir",
                    reason = "missing_layer_digest",
                );
                return None;
            }
        }

        // Values, not just presence. Runs before the device probe so a stale
        // artifact set is diagnosed as such rather than as a device problem.
        if let Err(reason) = verify_artifact_digests(model_dir, &dir, &stamp, owned_layers) {
            warn!(
                target: "cascadia::glm5",
                event = "ov_attn_stale_ir",
                reason = "artifact_digest_mismatch",
                detail = %reason,
            );
            return None;
        }

        // Startup probe (spec Sec7): a real (non-stub) shim at the ABI this
        // crate was built against, and the target device actually present —
        // BEFORE declaring the path usable. `shim_abi_version() == None`
        // covers the stub build; a version mismatch covers a shim compiled
        // from stale C++ source. Honest scope (see cpp/shim.h's
        // CASCADIA_SHIM_ABI_VERSION doc): `cc::Build` recompiles the shim
        // fresh alongside every `--features openvino` Rust build today, so
        // this check is defense-in-depth against a future build-model
        // change, NOT protection against this fleet's actual stale-artifact
        // hazard — a stale OpenVINO RUNTIME install behind
        // `INTEL_OPENVINO_DIR` — which is a different mechanism entirely and
        // this symbol cannot see.
        match cascadia_ov_genai_shim::shim_abi_version() {
            None => {
                error!(target: "cascadia::glm5", event = "ov_attn_unavailable", reason = "stub_shim_build");
                return None;
            }
            Some(v) if v != EXPECTED_SHIM_ABI_VERSION => {
                error!(
                    target: "cascadia::glm5",
                    event = "ov_attn_unavailable",
                    reason = "shim_abi_mismatch",
                    expected = EXPECTED_SHIM_ABI_VERSION,
                    found = v,
                );
                return None;
            }
            Some(_) => {}
        }
        let device_full_name = match cascadia_ov_genai_shim::device_full_name(&device) {
            Ok(n) => n,
            Err(e) => {
                error!(
                    target: "cascadia::glm5",
                    event = "ov_attn_unavailable",
                    reason = %format!("device probe failed: {e}"),
                    device = %device,
                );
                return None;
            }
        };

        let canary_plugin = PluginConfig::new()
            .with("INFERENCE_PRECISION_HINT", "f32")
            .with("DYNAMIC_QUANTIZATION_GROUP_SIZE", "0");
        if !run_bf16_canary(&device, &canary_plugin) {
            // run_bf16_canary already logged event=ov_attn_bf16_canary_failed.
            return None;
        }

        // Blob cache: CASCADIA_GLM5_OV_CACHE_DIR/<stamp_hash prefix>, so a
        // re-export can never hit a stale blob (spec Sec6) — the subdirectory
        // name changes whenever the stamp does. NOTE: this covers re-export
        // only; a driver bump is OV's own blob keying inside CACHE_DIR (the
        // plugin embeds its own compatibility markers in what it writes
        // there), not something this subdirectory scheme delivers.
        let cache_dir_env = std::env::var("CASCADIA_GLM5_OV_CACHE_DIR").ok();
        let mut layer_plugin = PluginConfig::new()
            .with("INFERENCE_PRECISION_HINT", "f32")
            .with("DYNAMIC_QUANTIZATION_GROUP_SIZE", "0");
        let mut blob_hit = false;
        let mut cache_subdir_str = String::new();
        if let Some(cd) = &cache_dir_env {
            let prefix = &stamp_hash[..16.min(stamp_hash.len())];
            let subdir = std::path::Path::new(cd).join(prefix);
            blob_hit = subdir
                .read_dir()
                .map(|mut it| it.next().is_some())
                .unwrap_or(false);
            let _ = std::fs::create_dir_all(&subdir);
            if let Some(s) = subdir.to_str() {
                cache_subdir_str = s.to_string();
                layer_plugin = layer_plugin.with("CACHE_DIR", cache_subdir_str.clone());
            }
        }

        let mut compiled: Vec<Option<Runtime>> = Vec::with_capacity(owned_layers.len());
        let mut layers_ok = 0usize;
        let mut precision_checked = false;
        // `(global_layer, reason)` for every layer that failed to compile —
        // named in `ov_attn_partial` below rather than left as a bare count.
        let mut failed_layers: Vec<(usize, String)> = Vec::new();
        for &gl in owned_layers {
            let xml = dir.join(format!("layer_{gl:02}.xml"));
            let path_str = match xml.to_str() {
                Some(s) => s,
                None => {
                    let reason = format!("non-UTF8 IR path: {}", xml.display());
                    warn!(
                        target: "cascadia::glm5",
                        event = "ov_attn_layer_compile_failed",
                        layer = gl,
                        reason = %reason,
                    );
                    failed_layers.push((gl, reason));
                    compiled.push(None);
                    continue;
                }
            };
            match Runtime::compile(path_str, &device, &layer_plugin) {
                Ok(rt) => {
                    if !precision_checked {
                        precision_checked = true;
                        if let Err(msg) = check_precision(&rt) {
                            warn!(
                                target: "cascadia::glm5",
                                event = "ov_attn_precision_mismatch",
                                layer = gl,
                                reason = %msg,
                            );
                            return None;
                        }
                    }
                    compiled.push(Some(rt));
                    layers_ok += 1;
                }
                Err(e) => {
                    let reason = e.to_string();
                    // WARN (not debug!): the most likely production failure
                    // (GPU compile fails for every layer) must be visible at
                    // default verbosity, not only to someone who already
                    // knew to raise it.
                    warn!(
                        target: "cascadia::glm5",
                        event = "ov_attn_layer_compile_failed",
                        layer = gl,
                        reason = %reason,
                    );
                    failed_layers.push((gl, reason));
                    compiled.push(None);
                }
            }
        }
        if layers_ok == 0 {
            error!(
                target: "cascadia::glm5",
                event = "ov_attn_unavailable",
                reason = "no owned layer compiled",
                layers_expected = owned_layers.len(),
            );
            return None;
        }
        if layers_ok < owned_layers.len() {
            let failed_summary: Vec<String> = failed_layers
                .iter()
                .map(|(gl, reason)| format!("{gl:02}: {reason}"))
                .collect();
            warn!(
                target: "cascadia::glm5",
                event = "ov_attn_partial",
                layers_ok,
                layers_expected = owned_layers.len(),
                failed_layers = %failed_summary.join("; "),
            );
        }

        let ir_bytes: u64 = owned_layers
            .iter()
            .filter_map(|&gl| std::fs::metadata(dir.join(format!("layer_{gl:02}.bin"))).ok())
            .map(|md| md.len())
            .sum();

        info!(
            target: "cascadia::glm5",
            event = "ov_attn_config",
            enabled,
            enabled_source = enabled_src,
            device = %device,
            device_source = device_src,
            device_full_name = %device_full_name,
            min_rows,
            min_rows_source = min_rows_src,
            layers_ok,
            layers_expected = owned_layers.len(),
            w = derived_w,
            p_max = stamp.p_max,
            rows_cap = stamp.rows,
            kv_lora = stamp.kv_lora,
            qk_rope = stamp.qk_rope,
            export_stamp_hash = %stamp_hash,
            ir_bytes,
            cache_dir = %cache_subdir_str,
            blob_hit,
        );

        Some(Self {
            compiled: Mutex::new(compiled),
            device,
            w: derived_w,
            p_max: stamp.p_max,
            rows_cap: stamp.rows,
            kv_lora: stamp.kv_lora,
            qk_rope: stamp.qk_rope,
            hidden: stamp.hidden,
            layers_ok,
            layers_expected: owned_layers.len(),
            min_rows,
            poisoned: AtomicBool::new(false),
            transient_fails: AtomicU32::new(0),
            used: AtomicBool::new(false),
            contract_violation_warned: AtomicBool::new(false),
            #[cfg(test)]
            mock: None,
        })
    }

    /// A device-free instance whose `prefill_window` returns canned outputs.
    /// Exists so the prefill-seam integration (Task 6) is testable on a machine
    /// with no OpenVINO SDK: the mock is consulted only AFTER the real poison
    /// latch and argument-contract checks, and routes its success/failure
    /// through the same `note_infer_success`/`note_logic_error` bookkeeping, so
    /// a mock-routed test exercises the production control flow rather than a
    /// parallel one.
    #[cfg(test)]
    pub(crate) fn mock(cfg: MockCfg, window: MockWindowFn) -> Self {
        let layers_expected = cfg.compiled.len();
        let layers_ok = cfg.compiled.iter().filter(|&&c| c).count();
        Self {
            compiled: Mutex::new(Vec::new()),
            device: "MOCK".to_string(),
            w: cfg.w,
            p_max: cfg.p_max,
            rows_cap: cfg.rows_cap,
            kv_lora: cfg.kv_lora,
            qk_rope: cfg.qk_rope,
            hidden: cfg.hidden,
            layers_ok,
            layers_expected,
            min_rows: cfg.min_rows,
            poisoned: AtomicBool::new(false),
            transient_fails: AtomicU32::new(0),
            used: AtomicBool::new(false),
            contract_violation_warned: AtomicBool::new(false),
            mock: Some(MockBackend {
                compiled: cfg.compiled,
                window,
            }),
        }
    }

    /// Whether local layer `layer_local`'s IR compiled. Part of the caller's
    /// eligibility check so an uncompiled layer is reported as
    /// [`OvSkipReason::LayerNotCompiled`] instead of being lumped into the
    /// generic `infer_failed` bucket a bare `None` would produce.
    pub fn layer_compiled(&self, layer_local: usize) -> bool {
        #[cfg(test)]
        if let Some(mock) = &self.mock {
            return mock.compiled.get(layer_local).copied().unwrap_or(false);
        }
        self.compiled
            .lock()
            .map(|g| g.get(layer_local).is_some_and(Option::is_some))
            .unwrap_or(false)
    }

    /// Current enablement state — see [`OvAttnState`]'s doc for why a live
    /// instance never reports `Off`/`Unavailable`/`StaleIr`. Poisoned takes
    /// priority over Partial (a latched path is fully off, coverage is now
    /// moot); Partial takes priority over Active (incomplete coverage is the
    /// more actionable fact even once the compiled subset has served traffic).
    pub fn state(&self) -> OvAttnState {
        if self.poisoned.load(Ordering::Relaxed) {
            return OvAttnState::Poisoned;
        }
        if self.layers_ok < self.layers_expected {
            return OvAttnState::Partial;
        }
        if self.used.load(Ordering::Relaxed) {
            OvAttnState::Active
        } else {
            OvAttnState::Enabled
        }
    }

    /// This rank's device string (as resolved at construction), for callers
    /// that log alongside per-window events.
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run one window of layer `layer_local` (LOCAL offset — see the module
    /// doc).
    ///
    /// `x` is `[rows, hidden]` (real rows only) and is the **RAW residual**,
    /// PRE-`input_layernorm`: the graph applies `rmsnorm(in_ln)` itself as its
    /// first op. Passing an already-normed row normalizes twice and produces
    /// wrong-but-finite, on-grid KV rows that no downstream guard can detect.
    ///
    /// `past_lc`/`past_rc` are
    /// `[past_len, kv_lora]`/`[past_len, qk_rope]` (real past only, NOT
    /// pre-padded to `p_max` — this function pads). Returns `None` on ANY
    /// failure (bad shapes, poisoned, layer never compiled, device error) —
    /// the caller falls back to the Rust path for THIS window only. Does NOT
    /// finite-check or commit the outputs; that is the caller's job (Task 6),
    /// so this stays a pure compute step callers can retry-free discard.
    pub fn prefill_window(
        &self,
        layer_local: usize,
        x: &[f32],
        rows: usize,
        past_lc: &[f32],
        past_rc: &[f32],
        past_len: usize,
    ) -> Option<AttnWindowOut> {
        if self.poisoned.load(Ordering::Relaxed) {
            return None;
        }
        if rows == 0 || rows > self.rows_cap || past_len > self.p_max {
            self.warn_contract_violation_once(&format!(
                "rows={rows} rows_cap={} past_len={past_len} p_max={}",
                self.rows_cap, self.p_max
            ));
            return None;
        }
        if x.len() != rows * self.hidden
            || past_lc.len() != past_len * self.kv_lora
            || past_rc.len() != past_len * self.qk_rope
        {
            self.warn_contract_violation_once(&format!(
                "x.len={} want={} past_lc.len={} want={} past_rc.len={} want={}",
                x.len(),
                rows * self.hidden,
                past_lc.len(),
                past_len * self.kv_lora,
                past_rc.len(),
                past_len * self.qk_rope,
            ));
            return None;
        }

        // Canned-output short-circuit — after the latch and contract checks
        // above, so those still gate a mock-routed call exactly as they gate a
        // real one.
        #[cfg(test)]
        if let Some(mock) = &self.mock {
            if !mock.compiled.get(layer_local).copied().unwrap_or(false) {
                return None;
            }
            return match (mock.window)(layer_local, x, rows, past_len) {
                Some(out) => {
                    self.note_infer_success();
                    Some(out)
                }
                None => {
                    self.note_logic_error(layer_local, "mock window failure");
                    None
                }
            };
        }

        let mut x_pad = vec![0.0f32; self.rows_cap * self.hidden];
        x_pad[..x.len()].copy_from_slice(x);
        let mut lc_pad = vec![0.0f32; self.p_max * self.kv_lora];
        lc_pad[..past_lc.len()].copy_from_slice(past_lc);
        let mut rc_pad = vec![0.0f32; self.p_max * self.qk_rope];
        rc_pad[..past_rc.len()].copy_from_slice(past_rc);
        let mask = build_window_mask(self.rows_cap, self.p_max, past_len);
        let w_cols = self.p_max + self.rows_cap;

        let mut guard = self.compiled.lock().expect("OvAttn compiled-layer lock");
        let Some(rt) = guard.get_mut(layer_local).and_then(|slot| slot.as_mut()) else {
            return None; // never compiled for this local offset -- always falls back
        };

        let infer_result = rt
            .set_input(
                "x",
                DType::F32,
                &[self.rows_cap, self.hidden],
                f32_bytes(&x_pad),
            )
            .and_then(|_| {
                rt.set_input(
                    "past_lc",
                    DType::F32,
                    &[self.p_max, self.kv_lora],
                    f32_bytes(&lc_pad),
                )
            })
            .and_then(|_| {
                rt.set_input(
                    "past_rc",
                    DType::F32,
                    &[self.p_max, self.qk_rope],
                    f32_bytes(&rc_pad),
                )
            })
            .and_then(|_| {
                rt.set_input(
                    "mask",
                    DType::F32,
                    &[self.rows_cap, w_cols],
                    f32_bytes(&mask),
                )
            })
            .and_then(|_| rt.infer());

        if let Err(e) = infer_result {
            // A real shim call just failed on THIS thread, so the typed
            // last-error code (if any) is trustworthy here.
            let rt_snapshot_device = self.device.clone();
            drop(guard);
            self.note_shim_failure(layer_local, &rt_snapshot_device, &e);
            return None;
        }

        // Output order is the exporter's `Model([attn_out, lc, rc], ...)`
        // (tools/glm5_attn_ov.py::build_layer_graph) — indices 0/1/2.
        let attn_out = Self::read_rows(rt, 0, self.rows_cap, rows, self.hidden);
        let lc = Self::read_rows(rt, 1, self.rows_cap, rows, self.kv_lora);
        let rc = Self::read_rows(rt, 2, self.rows_cap, rows, self.qk_rope);
        drop(guard);

        let (Some(attn_out), Some(lc), Some(rc)) = (attn_out, lc, rc) else {
            // Every shim call above SUCCEEDED (we only reach here past the
            // `infer_result` Err check) -- this is a Rust-side shape/dtype
            // mismatch, not a device error. Must NOT consult the typed
            // last-error code: it is thread-local and never cleared on
            // success, so it can still hold whatever an EARLIER, unrelated
            // failure on this thread left behind (e.g. an OvExperts GPU
            // compile failure classified as resource exhaustion) and would
            // misclassify this as fatal. `note_logic_error` never reads it.
            self.note_logic_error(layer_local, "unexpected output shape");
            return None;
        };

        self.note_infer_success();
        Some(AttnWindowOut { attn_out, lc, rc })
    }

    /// WARN once (not per-call) when `prefill_window` rejects its own
    /// arguments before ever touching the device — a caller-side stride bug
    /// would otherwise call this every window. Without this, a contract
    /// violation turns the path into a permanent silent no-op: `state()`
    /// still reports Enabled/Active and the startup `ov_attn_config` log
    /// still claims success, while every window quietly falls back — the
    /// exact silent-degrade mode this module's doc says it exists to
    /// prevent.
    fn warn_contract_violation_once(&self, detail: &str) {
        if !self.contract_violation_warned.swap(true, Ordering::Relaxed) {
            warn!(
                target: "cascadia::glm5",
                event = "ov_attn_contract_violation",
                detail = %detail,
                "prefill_window rejected its own arguments; every window on \
                 this backend will silently fall back until the caller is fixed",
            );
        }
    }

    /// Slice the real-rows prefix (`rows*dim` elements) out of output `idx`'s
    /// `[rows_cap, dim]` f32 buffer. `None` on a dtype/shape/length mismatch.
    fn read_rows(
        rt: &Runtime,
        idx: usize,
        rows_cap: usize,
        rows: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let (dtype, shape, bytes) = rt.output(idx).ok()?;
        if dtype != DType::F32 || shape != [rows_cap, dim] {
            return None;
        }
        let want = rows * dim * 4;
        if bytes.len() < want {
            return None;
        }
        Some(
            bytes[..want]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Reset the consecutive-transient-failure counter and mark the backend
    /// as having served at least one window — called from `prefill_window`'s
    /// ONLY success return path. An intervening success means the next
    /// failure starts a fresh consecutive run (the latch is "N IN A ROW",
    /// not "N ever"); factored into its own method (not inlined at the call
    /// site) so tests can drive the exact bookkeeping `prefill_window` uses
    /// without a real compiled device.
    fn note_infer_success(&self) {
        self.transient_fails.store(0, Ordering::Relaxed);
        self.used.store(true, Ordering::Relaxed);
    }

    /// Classify a failure where a SHIM CALL just returned `Err` on this
    /// thread — the typed last-error code is trustworthy here (set by the
    /// call that just failed, per `cpp/shim.cpp`'s `set_last_error`).
    /// Consults it; a fatal (typed OR string-classified) resource-exhaustion
    /// error latches immediately (same taxonomy as `OvExperts::poison`).
    fn note_shim_failure(&self, layer_local: usize, device: &str, err: &OvError) {
        let msg = err.to_string();
        let fatal = cascadia_ov_genai_shim::last_error_resource_exhausted()
            || is_fatal_resource_error(&msg);
        self.note_failure(
            layer_local,
            fatal.then(|| format!("resource exhaustion on device {device}: {msg}")),
            &msg,
        );
    }

    /// Classify a failure that is NOT a shim call failure — e.g. an output
    /// shape/dtype mismatch discovered AFTER every shim call in the window
    /// already succeeded. MUST NEVER consult the typed last-error code: it
    /// is thread-local and is never cleared on success, so it can still hold
    /// whatever an unrelated EARLIER failure on this thread left behind (a
    /// real observed source: `OvExperts` GPU compile failures on the same
    /// worker thread set exactly the `resource unavailable` class). Always
    /// non-fatal — only the consecutive-transient latch can apply here.
    fn note_logic_error(&self, layer_local: usize, msg: &str) {
        self.note_failure(layer_local, None, msg);
    }

    /// Shared bookkeeping: `fatal_reason = Some(..)` latches immediately;
    /// `None` counts toward the consecutive-transient latch (default 3, see
    /// the module doc). Every individual transient failure is logged (WARN)
    /// — the silent-degrade weakness this module exists to not repeat.
    fn note_failure(&self, layer_local: usize, fatal_reason: Option<String>, msg: &str) {
        if let Some(why) = fatal_reason {
            self.poison(&why);
            return;
        }
        let n = self.transient_fails.fetch_add(1, Ordering::Relaxed) + 1;
        warn!(
            target: "cascadia::glm5",
            event = "ov_attn_infer_error",
            layer_local,
            consecutive = n,
            error = %msg,
        );
        if n >= TRANSIENT_LATCH_THRESHOLD {
            self.poison(&format!(
                "{n} consecutive transient infer failures (layer_local={layer_local}): {msg}"
            ));
        }
    }

    /// Latch the path off for good and log once (idempotent — a second
    /// concurrent trigger is a no-op).
    fn poison(&self, why: &str) {
        if !self.poisoned.swap(true, Ordering::Relaxed) {
            error!(
                target: "cascadia::glm5",
                event = "ov_attn_poisoned",
                reason = %why,
                "OV attention offload disabled: {why}; every window now falls back to the Rust path",
            );
        }
    }
}

/// Cumulative OV-attention prefill accounting — the spec Sec7 periodic
/// summary, emitted on the same cadence and behind the same
/// `CASCADIA_GLM5_OV_STATS` gate as the expert-cache summary.
///
/// Counting is unconditional (a handful of relaxed adds per prefill layer) so
/// the shutdown emission is complete regardless of when the gate was read;
/// only the emission is gated and rate-limited.
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::OvSkipReason;

    static LAYERS_OV: AtomicU64 = AtomicU64::new(0);
    static SKIPPED: [AtomicU64; OvSkipReason::COUNT] =
        [const { AtomicU64::new(0) }; OvSkipReason::COUNT];

    pub(super) fn record_used() {
        LAYERS_OV.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_skip(reason: OvSkipReason) {
        SKIPPED[reason.idx()].fetch_add(1, Ordering::Relaxed);
    }

    fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CASCADIA_GLM5_OV_STATS").is_ok())
    }

    fn dump_interval_secs() -> u64 {
        std::env::var("CASCADIA_GLM5_OV_STATS_EVERY_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(10)
    }

    static LAST_DUMP_S: AtomicU64 = AtomicU64::new(0);

    fn since_start_secs() -> u64 {
        static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_secs()
    }

    /// Rate-limited emission (see [`dump_now`] for the unconditional one).
    pub fn dump() {
        if !enabled() {
            return;
        }
        let now = since_start_secs();
        if now.saturating_sub(LAST_DUMP_S.load(Ordering::Relaxed)) < dump_interval_secs() {
            return;
        }
        LAST_DUMP_S.store(now, Ordering::Relaxed);
        dump_now();
    }

    /// Emit unconditionally, ignoring the rate limit (shutdown).
    pub fn dump_now() {
        if !enabled() {
            return;
        }
        let ov = LAYERS_OV.load(Ordering::Relaxed);
        let skips: Vec<(OvSkipReason, u64)> = (0..OvSkipReason::COUNT)
            .map(|i| {
                (
                    OvSkipReason::from_idx(i),
                    SKIPPED[i].load(Ordering::Relaxed),
                )
            })
            .filter(|&(_, n)| n > 0)
            .collect();
        let total = ov + skips.iter().map(|&(_, n)| n).sum::<u64>();
        if total == 0 {
            return;
        }
        let by_reason: Vec<String> = skips
            .iter()
            .map(|(r, n)| format!("{}:{n}", r.as_str()))
            .collect();
        eprintln!(
            "GLM5_OVATTN layer_windows={total} ov={ov} rust={} ov_rate={:.1}% skipped=[{}]",
            total - ov,
            100.0 * ov as f64 / total as f64,
            by_reason.join(","),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_stamp() -> ExportStamp {
        ExportStamp {
            manifest_sha256: "abc123".to_string(),
            per_layer_digest: [
                ("00".to_string(), "x".to_string()),
                ("01".to_string(), "y".to_string()),
            ]
            .into_iter()
            .collect(),
            ir_digest: [
                ("00".to_string(), "ix".to_string()),
                ("01".to_string(), "iy".to_string()),
            ]
            .into_iter()
            .collect(),
            exporter_version: EXPECTED_EXPORTER_VERSION.to_string(),
            w: 8,
            p_max: 4,
            rows: 4,
            hidden: 16,
            kv_lora: 4,
            qk_rope: 4,
        }
    }

    fn test_manifest(index_n_heads: usize, index_topk: usize) -> GlmManifest {
        // Only the fields derive_w/check_stamp touch matter; the rest are
        // zeroed (GlmManifest has no #[derive(Default)] in loader.rs, so
        // build it field-by-field to keep this test independent of that).
        GlmManifest {
            arch: "glm5".to_string(),
            num_layers: 2,
            dense_layers: vec![],
            num_experts: 0,
            top_k: 0,
            hidden_size: 16,
            num_attention_heads: 1,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            vocab_size: 8,
            expert_intermediate: 8,
            dense_intermediate: 8,
            n_shared_experts: 1,
            routed_scaling_factor: 1.0,
            rope_theta: 8_000_000.0,
            rms_norm_eps: 1e-5,
            eos_token_ids: vec![],
            index_topk,
            index_n_heads,
            index_head_dim: 0,
            indexer_types: vec![],
            has_mtp: false,
        }
    }

    // ---- W derivation table (spec Sec4) -----------------------------------

    #[test]
    fn derive_w_no_indexer_is_the_dense_cap() {
        assert_eq!(derive_w(&test_manifest(0, 0)), NO_INDEXER_W);
        // index_topk set but index_n_heads==0 -- still "no indexer attached".
        assert_eq!(derive_w(&test_manifest(0, 512)), NO_INDEXER_W);
    }

    #[test]
    fn derive_w_with_indexer_is_index_topk() {
        assert_eq!(derive_w(&test_manifest(2, 512)), 512);
        assert_eq!(derive_w(&test_manifest(2, 8)), 8); // tiny parity model, W=8
    }

    #[test]
    fn derive_w_indexer_present_zero_topk_never_matches_a_real_stamp() {
        // Mirrors the exporter's hard failure at export time (no stamp can
        // exist for this manifest); the sentinel guarantees a W mismatch
        // rather than an accidental match against some other stamp's W.
        let w = derive_w(&test_manifest(2, 0));
        assert_eq!(w, usize::MAX);
        assert_ne!(w, 0);
    }

    // ---- Accelerator predicate (drives nopin + the dual-accel rejection) ---

    /// Both `load_staged` consumers — the expert-pinning exclusion and the
    /// "reject ov_attn + ov_experts on one device" load error — are one-line
    /// uses of this, so its polarity is the whole behaviour.
    #[test]
    fn requested_on_accelerator_tracks_enablement_and_device() {
        let off = StageOpts {
            ov_attn: Some(false),
            ov_attn_device: Some("GPU".to_string()),
            ..Default::default()
        };
        assert!(
            !requested_on_accelerator(&off),
            "disabled must never imply an accelerator, whatever the device says"
        );

        let cpu = StageOpts {
            ov_attn: Some(true),
            ov_attn_device: Some("cpu".to_string()), // case-insensitive
            ..Default::default()
        };
        assert!(
            !requested_on_accelerator(&cpu),
            "CPU is not an accelerator: it must not disable pinning"
        );

        let gpu = StageOpts {
            ov_attn: Some(true),
            ov_attn_device: Some("GPU.0".to_string()),
            ..Default::default()
        };
        assert!(requested_on_accelerator(&gpu));

        let npu = StageOpts {
            ov_attn: Some(true),
            ov_attn_device: Some("NPU".to_string()),
            ..Default::default()
        };
        assert!(npu.ov_attn_device.is_some() && requested_on_accelerator(&npu));
    }

    // ---- Stamp handshake (pure, no filesystem/OV) -------------------------

    #[test]
    fn check_stamp_ok_when_everything_agrees() {
        let stamp = test_stamp();
        assert_eq!(check_stamp(&stamp, "abc123", 8, &[0, 1]), StampCheck::Ok);
    }

    #[test]
    fn check_stamp_flags_manifest_mismatch() {
        let stamp = test_stamp();
        assert_eq!(
            check_stamp(&stamp, "different-hash", 8, &[0, 1]),
            StampCheck::ManifestMismatch
        );
    }

    #[test]
    fn check_stamp_flags_w_mismatch() {
        let stamp = test_stamp();
        assert_eq!(
            check_stamp(&stamp, "abc123", 999, &[0, 1]),
            StampCheck::WMismatch
        );
    }

    #[test]
    fn check_stamp_flags_missing_layer_coverage() {
        let stamp = test_stamp();
        // Layer 2 is owned but the stamp only covers 00/01.
        assert_eq!(
            check_stamp(&stamp, "abc123", 8, &[0, 1, 2]),
            StampCheck::MissingCoverage
        );
    }

    #[test]
    fn check_stamp_flags_corrupt_dims() {
        // The exporter guarantees p_max + rows == w by construction; a
        // stamp that violates its own invariant is corrupt, checked before
        // (and independent of) any comparison against the live manifest.
        let mut stamp = test_stamp();
        stamp.p_max = 3; // 3 + rows(4) = 7 != w(8)
        assert_eq!(
            check_stamp(&stamp, "abc123", 8, &[0, 1]),
            StampCheck::CorruptDims
        );
    }

    #[test]
    fn check_stamp_flags_an_older_exporter_generation() {
        let mut stamp = test_stamp();
        stamp.exporter_version = "1".to_string();
        assert_eq!(
            check_stamp(&stamp, "abc123", 8, &[0, 1]),
            StampCheck::ExporterVersionMismatch
        );
    }

    /// A pre-`ir_digest` stamp (the field defaults to empty) must not pass
    /// coverage — otherwise the IR files stay unverified exactly as before.
    #[test]
    fn check_stamp_flags_missing_ir_digest_coverage() {
        let mut stamp = test_stamp();
        stamp.ir_digest.remove("01");
        assert_eq!(
            check_stamp(&stamp, "abc123", 8, &[0, 1]),
            StampCheck::MissingCoverage
        );
    }

    /// The digest VALUES are compared, not merely present — a stale IR beside
    /// a fresh stamp used to pass the whole handshake.
    #[test]
    fn verify_artifact_digests_rejects_a_stale_ir_beside_a_fresh_stamp() {
        let td = std::env::temp_dir().join(format!("ov_attn_digest_{}", std::process::id()));
        let shells = td.join("shells");
        let ir = td.join("attn_ov");
        std::fs::create_dir_all(&shells).unwrap();
        std::fs::create_dir_all(&ir).unwrap();
        std::fs::write(shells.join("layer_00.safetensors"), b"weights").unwrap();
        std::fs::write(ir.join("layer_00.xml"), b"<net/>").unwrap();
        std::fs::write(ir.join("layer_00.bin"), b"ir-bytes").unwrap();

        let mut stamp = test_stamp();
        stamp
            .per_layer_digest
            .insert("00".into(), sha256_hex(b"weights"));
        stamp
            .ir_digest
            .insert("00".into(), sha256_hex(b"<net/>ir-bytes"));
        assert!(verify_artifact_digests(&td, &ir, &stamp, &[0]).is_ok());

        // Same filename, older exporter's bytes.
        std::fs::write(ir.join("layer_00.bin"), b"stale-ir-bytes").unwrap();
        let err = verify_artifact_digests(&td, &ir, &stamp, &[0]).unwrap_err();
        assert!(err.contains("IR does not match"), "got: {err}");

        // And the shell half: weights re-exported without re-running this tool.
        std::fs::write(ir.join("layer_00.bin"), b"ir-bytes").unwrap();
        std::fs::write(shells.join("layer_00.safetensors"), b"new-weights").unwrap();
        let err = verify_artifact_digests(&td, &ir, &stamp, &[0]).unwrap_err();
        assert!(err.contains("shell weights changed"), "got: {err}");

        std::fs::remove_dir_all(&td).ok();
    }

    // ---- env knob polarity (documented emergency rollback) ----------------

    #[test]
    fn env_off_spellings_disable_and_everything_else_enables() {
        use std::ffi::OsStr;
        for off in ["0", "false", "FALSE", "no", "off", "", "  off  "] {
            assert!(!env_value_is_on(OsStr::new(off)), "{off:?} must mean OFF");
        }
        for on in ["1", "true", "yes", "on", "GPU"] {
            assert!(env_value_is_on(OsStr::new(on)), "{on:?} must mean ON");
        }
    }

    #[test]
    fn resolve_bool_config_wins_and_env_value_is_honoured() {
        // Config wins over env in both directions, whatever env says.
        let key = "CASCADIA_GLM5_OV_ATTN_TEST_POLARITY";
        // SAFETY: single-threaded within this test; key is test-local.
        unsafe { std::env::set_var(key, "0") };
        assert_eq!(resolve_bool(Some(true), key), (true, "engine_arg"));
        assert_eq!(
            resolve_bool(None, key),
            (false, "env"),
            "=0 must NOT enable the feature — the runbook's rollback depends on it"
        );
        unsafe { std::env::set_var(key, "1") };
        assert_eq!(resolve_bool(None, key), (true, "env"));
        assert_eq!(resolve_bool(Some(false), key), (false, "engine_arg"));
        unsafe { std::env::remove_var(key) };
        assert_eq!(resolve_bool(None, key), (false, "default"));
    }

    // ---- Mask builder -------------------------------------------------

    #[test]
    fn mask_unmasks_real_past_and_causal_diagonal_only() {
        let (rows_cap, p_max, past_len) = (4usize, 4usize, 2usize);
        let w = p_max + rows_cap;
        let m = build_window_mask(rows_cap, p_max, past_len);
        assert_eq!(m.len(), rows_cap * w);
        for i in 0..rows_cap {
            let row = &m[i * w..(i + 1) * w];
            // Real past columns [0, past_len) unmasked.
            for c in 0..past_len {
                assert_eq!(
                    row[c], 0.0,
                    "row {i} col {c} (real past) should be unmasked"
                );
            }
            // Padding past columns [past_len, p_max) masked.
            for c in past_len..p_max {
                assert_eq!(
                    row[c], MASK_NEG,
                    "row {i} col {c} (pad past) should be masked"
                );
            }
            // Window columns: causal, p_max..=p_max+i unmasked, rest masked.
            for c in p_max..w {
                let expect_unmasked = c <= p_max + i;
                assert_eq!(
                    row[c] == 0.0,
                    expect_unmasked,
                    "row {i} col {c} causal expectation"
                );
            }
        }
    }

    #[test]
    fn mask_padding_row_still_sees_at_least_one_column() {
        // Every row (including padding rows beyond the "real" window length,
        // which this function has no notion of -- it always builds for the
        // full rows_cap) sees at least its own diagonal, so softmax over a
        // fully-masked row (-inf everywhere) can never happen.
        let m = build_window_mask(8, 0, 0); // p_max=0: no real/padded past at all
        let w = 8;
        for i in 0..8 {
            let row = &m[i * w..(i + 1) * w];
            assert!(row.iter().any(|&v| v == 0.0), "row {i} fully masked");
        }
    }

    // ---- bf16 canary battery generation (pure, no OV) ----------------------

    #[test]
    fn canary_battery_rne_matches_to_bf16() {
        // canary_cases()'s `rne` field is just `to_bf16(x)` -- this pins that
        // the battery's ground truth stays wired to the same function the
        // rest of the engine uses for bf16 rounding, so a refactor of one
        // can't silently desync from the other.
        for c in canary_cases() {
            assert_eq!(c.rne, to_bf16(c.x), "case {}", c.label);
        }
    }

    #[test]
    fn canary_battery_has_expected_case_count() {
        // 12 bases x 3 fractional offsets each.
        assert_eq!(canary_cases().len(), CANARY_BASES.len() * 3);
    }

    #[test]
    fn canary_battery_hypotheses_match_the_documented_rule() {
        // Ground truth for the competing-hypothesis diagnostic
        // (run_bf16_canary reports which of these an on-device mismatch
        // actually matched) -- pins the ported rule from
        // tools/glm5_attn_ov.py::canary_cases() directly.
        for c in canary_cases() {
            assert_eq!(
                c.identity, c.x,
                "case {}: identity must equal the raw input",
                c.label
            );
            // Truncation rounds toward zero, so whatever else is true, its
            // prediction is never larger in magnitude than the input's.
            assert!(
                c.trunc.abs() <= c.x.abs(),
                "case {}: trunc must round TOWARD zero (got {} for x={})",
                c.label,
                c.trunc,
                c.x
            );
            let negative = c.label.contains("base=-");
            if c.label.contains("f=.25") {
                // Nearest is g_lo; truncation agrees only for a positive base.
                if negative {
                    assert_ne!(
                        c.trunc.to_bits(),
                        c.half_away.to_bits(),
                        "case {}: negative base -> trunc(g_hi) != half_away(g_lo)",
                        c.label
                    );
                } else {
                    assert_eq!(
                        c.trunc, c.half_away,
                        "case {}: positive base, f<.5 -> both g_lo",
                        c.label
                    );
                }
            } else if c.label.contains("f=.75") {
                // Nearest is g_hi; truncation agrees only for a negative base.
                if negative {
                    assert_eq!(
                        c.trunc, c.half_away,
                        "case {}: negative base, f>.5 -> both g_hi",
                        c.label
                    );
                } else {
                    assert_ne!(
                        c.trunc.to_bits(),
                        c.half_away.to_bits(),
                        "case {}: positive base -> trunc(g_lo) != half_away(g_hi)",
                        c.label
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_tie_cases_want_matches_to_bf16() {
        // Ground-truth check on the ported table itself: to_bf16 (verified
        // bit-exact against half::bf16::from_f32 elsewhere in this crate)
        // must agree with every literal `want` in BF16_TIE_CASES, or the
        // ported table itself has a transcription bug independent of any
        // device.
        for &(bits, want16) in BF16_TIE_CASES.iter() {
            let x = f32::from_bits(bits);
            if x.is_nan() {
                // to_bf16(NaN) is a real NaN too; canonical-NaN comparison
                // only makes sense bit-for-bit on the graph's output, not
                // via Rust's to_bf16 (half::bf16 preserves NaN-ness, not
                // necessarily the exact 0x7FC0 payload) -- skip the value
                // comparison, just confirm NaN-in NaN-out.
                assert!(to_bf16(x).is_nan(), "NaN input must stay NaN");
                continue;
            }
            let got = to_bf16(x).to_bits();
            let want = (want16 as u32) << 16;
            assert_eq!(got, want, "tie case x_bits=0x{bits:08x}");
        }
    }

    // ---- Tracing capture: for asserting on `event=...` field values
    // without a global logger, since no in-crate capture pattern existed
    // (Task 5 Step 2 fallback). Thread-scoped (`with_default`), so parallel
    // `cargo test` runs don't cross-contaminate captures. ------------------

    fn capture_logs<F: FnOnce()>(f: F) -> String {
        use std::io;
        use std::sync::{Arc, Mutex as StdMutex};

        #[derive(Clone)]
        struct BufWriter(Arc<StdMutex<Vec<u8>>>);
        impl io::Write for BufWriter {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(data);
                Ok(data.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let buf = Arc::new(StdMutex::new(Vec::<u8>::new()));
        let writer = BufWriter(buf.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        let bytes = buf.lock().unwrap().clone();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // ---- from_opts handshake, end-to-end (tempfile, no OV) -----------------
    //
    // Every filesystem/stamp check runs BEFORE the shim probe, so a synthetic
    // export_stamp.json + manifest.json exercises the real `from_opts`
    // dispatch (not just the pure `check_stamp` function) with no device.

    /// Stand-in artifact bytes; the fixture writer and `stamp_json` agree on
    /// them so the digest comparison has something real to compare.
    fn artifact_bytes(layer: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            format!("shell-{layer}").into_bytes(),
            format!("xml-{layer}").into_bytes(),
            format!("bin-{layer}").into_bytes(),
        )
    }

    fn write_fixture(dir: &std::path::Path, stamp: &serde_json::Value, manifest_bytes: &[u8]) {
        let attn_dir = dir.join("attn_ov");
        let shells = dir.join("shells");
        std::fs::create_dir_all(&attn_dir).unwrap();
        std::fs::create_dir_all(&shells).unwrap();
        std::fs::write(
            attn_dir.join("export_stamp.json"),
            serde_json::to_vec(stamp).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("manifest.json"), manifest_bytes).unwrap();
        for l in stamp["per_layer_digest"].as_object().unwrap().keys() {
            let (shell, xml, bin) = artifact_bytes(l);
            std::fs::write(shells.join(format!("layer_{l}.safetensors")), shell).unwrap();
            std::fs::write(attn_dir.join(format!("layer_{l}.xml")), xml).unwrap();
            std::fs::write(attn_dir.join(format!("layer_{l}.bin")), bin).unwrap();
        }
    }

    fn stamp_json(
        manifest_sha256: &str,
        w: usize,
        p_max: usize,
        rows: usize,
        layers: &[&str],
    ) -> serde_json::Value {
        let mut digest: HashMap<String, String> = HashMap::new();
        let mut ir: HashMap<String, String> = HashMap::new();
        for l in layers {
            let (shell, xml, bin) = artifact_bytes(l);
            digest.insert(l.to_string(), sha256_hex(&shell));
            ir.insert(l.to_string(), sha256_hex(&[xml, bin].concat()));
        }
        serde_json::json!({
            "manifest_sha256": manifest_sha256,
            "per_layer_digest": digest,
            "ir_digest": ir,
            "exporter_version": EXPECTED_EXPORTER_VERSION,
            "w": w,
            "p_max": p_max,
            "rows": rows,
            "hidden": 16,
            "kv_lora": 4,
            "qk_rope": 4,
        })
    }

    #[test]
    fn from_opts_ok_stamp_falls_through_to_the_shim_probe() {
        // Proves the StampCheck::Ok arm does NOT erroneously short-circuit:
        // the only event reaching the log must be the shim probe's (stub
        // build), never a stale_ir/w_mismatch false positive.
        let dir = tempfile::tempdir().unwrap();
        let manifest_bytes = b"fake manifest bytes";
        let sha = sha256_hex(manifest_bytes);
        let stamp = stamp_json(&sha, 8, 4, 4, &["00", "01"]);
        write_fixture(dir.path(), &stamp, manifest_bytes);
        let opts = StageOpts {
            ov_attn: Some(true),
            ..Default::default()
        };
        let manifest = test_manifest(2, 8); // derive_w == 8, matches stamp.w
        let log = capture_logs(|| {
            let result = OvAttn::from_opts(dir.path(), &[0, 1], &manifest, &opts);
            assert!(result.is_none(), "stub build must fail the shim probe");
        });
        assert!(!log.contains("ov_attn_stale_ir"), "log:\n{log}");
        assert!(!log.contains("ov_attn_w_mismatch"), "log:\n{log}");
        assert!(log.contains("ov_attn_unavailable"), "log:\n{log}");
    }

    #[test]
    fn from_opts_rejects_manifest_sha_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_bytes = b"fake manifest bytes";
        let stamp = stamp_json("deadbeefdeadbeef", 8, 4, 4, &["00", "01"]); // wrong hash
        write_fixture(dir.path(), &stamp, manifest_bytes);
        let opts = StageOpts {
            ov_attn: Some(true),
            ..Default::default()
        };
        let manifest = test_manifest(2, 8);
        let log = capture_logs(|| {
            let result = OvAttn::from_opts(dir.path(), &[0, 1], &manifest, &opts);
            assert!(result.is_none());
        });
        assert!(log.contains("ov_attn_stale_ir"), "log:\n{log}");
        assert!(log.contains("manifest_sha_mismatch"), "log:\n{log}");
    }

    #[test]
    fn from_opts_rejects_w_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_bytes = b"fake manifest bytes";
        let sha = sha256_hex(manifest_bytes);
        // Internally consistent (995+4==999) but disagrees with the live
        // manifest's derived W (8).
        let stamp = stamp_json(&sha, 999, 995, 4, &["00", "01"]);
        write_fixture(dir.path(), &stamp, manifest_bytes);
        let opts = StageOpts {
            ov_attn: Some(true),
            ..Default::default()
        };
        let manifest = test_manifest(2, 8);
        let log = capture_logs(|| {
            let result = OvAttn::from_opts(dir.path(), &[0, 1], &manifest, &opts);
            assert!(result.is_none());
        });
        assert!(log.contains("ov_attn_w_mismatch"), "log:\n{log}");
    }

    #[test]
    fn from_opts_rejects_missing_layer_coverage() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_bytes = b"fake manifest bytes";
        let sha = sha256_hex(manifest_bytes);
        let stamp = stamp_json(&sha, 8, 4, 4, &["00", "01"]); // no digest for layer 2
        write_fixture(dir.path(), &stamp, manifest_bytes);
        let opts = StageOpts {
            ov_attn: Some(true),
            ..Default::default()
        };
        let manifest = test_manifest(2, 8);
        let log = capture_logs(|| {
            let result = OvAttn::from_opts(dir.path(), &[0, 1, 2], &manifest, &opts);
            assert!(result.is_none());
        });
        assert!(log.contains("ov_attn_stale_ir"), "log:\n{log}");
        assert!(log.contains("missing_layer_digest"), "log:\n{log}");
    }

    /// A stale `layer_NN.bin` from an older exporter beside a structurally
    /// valid, freshly written stamp — the exact case `contains_key` accepted.
    #[test]
    fn from_opts_rejects_ir_bytes_that_do_not_match_the_stamp() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_bytes = b"fake manifest bytes";
        let sha = sha256_hex(manifest_bytes);
        let stamp = stamp_json(&sha, 8, 4, 4, &["00", "01"]);
        write_fixture(dir.path(), &stamp, manifest_bytes);
        std::fs::write(dir.path().join("attn_ov/layer_01.bin"), b"older-generation").unwrap();

        let opts = StageOpts {
            ov_attn: Some(true),
            ..Default::default()
        };
        let manifest = test_manifest(2, 8);
        let log = capture_logs(|| {
            assert!(OvAttn::from_opts(dir.path(), &[0, 1], &manifest, &opts).is_none());
        });
        assert!(log.contains("artifact_digest_mismatch"), "log:\n{log}");
        assert!(!log.contains("ov_attn_unavailable"), "log:\n{log}");
    }

    #[test]
    fn from_opts_reports_ir_missing_when_stamp_absent() {
        let dir = tempfile::tempdir().unwrap(); // no attn_ov/ created at all
        let opts = StageOpts {
            ov_attn: Some(true),
            ..Default::default()
        };
        let manifest = test_manifest(2, 8);
        let log = capture_logs(|| {
            let result = OvAttn::from_opts(dir.path(), &[0, 1], &manifest, &opts);
            assert!(result.is_none());
        });
        assert!(log.contains("ov_attn_ir_missing"), "log:\n{log}");
    }

    // ---- State machine: poison latch (pure, no OV) -------------------------

    fn test_instance() -> OvAttn {
        OvAttn {
            compiled: Mutex::new(Vec::new()),
            device: "CPU".to_string(),
            w: 8,
            p_max: 4,
            rows_cap: 4,
            kv_lora: 4,
            qk_rope: 4,
            hidden: 16,
            layers_ok: 2,
            layers_expected: 2,
            min_rows: 64,
            poisoned: AtomicBool::new(false),
            transient_fails: AtomicU32::new(0),
            used: AtomicBool::new(false),
            contract_violation_warned: AtomicBool::new(false),
            mock: None,
        }
    }

    #[test]
    fn state_starts_enabled() {
        let ov = test_instance();
        assert_eq!(ov.state(), OvAttnState::Enabled);
    }

    #[test]
    fn state_reports_partial_when_coverage_is_incomplete() {
        let mut ov = test_instance();
        ov.layers_ok = 1;
        assert_eq!(ov.state(), OvAttnState::Partial);
    }

    #[test]
    fn state_reports_active_after_a_marked_success() {
        let ov = test_instance();
        ov.used.store(true, Ordering::Relaxed);
        assert_eq!(ov.state(), OvAttnState::Active);
    }

    #[test]
    fn three_consecutive_transient_failures_latch() {
        let ov = test_instance();
        let err = OvError::Native("device busy".to_string());
        ov.note_shim_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Enabled,
            "1st transient: not yet latched"
        );
        ov.note_shim_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Enabled,
            "2nd transient: not yet latched"
        );
        ov.note_shim_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Poisoned,
            "3rd consecutive transient: latched"
        );
    }

    /// Drives the REAL success path `prefill_window` calls
    /// (`note_infer_success`), not the raw atomic — a version of this test
    /// that pokes `transient_fails.store(0, ..)` directly proves nothing
    /// about whether `prefill_window`'s success branch actually resets it
    /// (verified: deleting the `store` inside `note_infer_success` turns
    /// this test red, since both `prefill_window` and this test now share
    /// that one method).
    #[test]
    fn a_success_between_failures_resets_the_consecutive_counter() {
        let ov = test_instance();
        let err = OvError::Native("device busy".to_string());
        ov.note_shim_failure(0, "CPU", &err);
        ov.note_shim_failure(0, "CPU", &err);
        ov.note_infer_success(); // the exact method prefill_window's success return calls
        ov.note_shim_failure(0, "CPU", &err);
        ov.note_shim_failure(0, "CPU", &err);
        assert_eq!(
            ov.transient_fails.load(Ordering::Relaxed),
            2,
            "the post-reset pair must not add onto the pre-reset pair"
        );
        assert_ne!(
            ov.state(),
            OvAttnState::Poisoned,
            "non-consecutive failures (reset by a success in between) must not latch"
        );
        assert_eq!(
            ov.state(),
            OvAttnState::Active,
            "the success marks it Active"
        );
    }

    #[test]
    fn typed_resource_exhaustion_latches_on_the_first_failure() {
        let ov = test_instance();
        let err = OvError::Native(
            "compile on GPU: openvino-genai error: resource unavailable try again".to_string(),
        );
        ov.note_shim_failure(0, "GPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Poisoned,
            "fatal resource error must latch immediately"
        );
    }

    /// The bug this guards: `note_logic_error` (output-shape-mismatch path,
    /// reached only after every shim call in the window already succeeded)
    /// must NEVER consult the typed/string resource-exhaustion classifier —
    /// the thread-local shim error code is never cleared on success and can
    /// still hold whatever an unrelated earlier failure left behind. Proven
    /// here by feeding it a message that WOULD trip the string classifier if
    /// it were consulted; it must still only count as one transient.
    #[test]
    fn logic_error_never_latches_on_first_occurrence_even_with_fatal_looking_text() {
        let ov = test_instance();
        ov.note_logic_error(0, "resource unavailable try again (unrelated stale text)");
        assert_ne!(
            ov.state(),
            OvAttnState::Poisoned,
            "a logic error must never fatal-latch on the first occurrence"
        );
        assert_eq!(ov.transient_fails.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn poison_is_idempotent_and_does_not_repanic_on_second_trigger() {
        let ov = test_instance();
        ov.poison("first");
        ov.poison("second"); // must not double-log-crash or flip anything further
        assert_eq!(ov.state(), OvAttnState::Poisoned);
    }

    // ---- prefill_window contract-violation trace (item 7) ------------------

    #[test]
    fn prefill_window_contract_violation_warns_once_not_per_call() {
        let ov = test_instance(); // rows_cap=4, p_max=4, hidden=16, kv_lora=4, qk_rope=4
        let log = capture_logs(|| {
            // rows == 0 is invalid.
            assert!(ov.prefill_window(0, &[], 0, &[], &[], 0).is_none());
            // rows > rows_cap is invalid.
            assert!(ov
                .prefill_window(0, &vec![0.0; 5 * 16], 5, &[], &[], 0)
                .is_none());
        });
        let count = log.matches("ov_attn_contract_violation").count();
        assert_eq!(count, 1, "warn should fire once, not per-call; log:\n{log}");
    }
}
