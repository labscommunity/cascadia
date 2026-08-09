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

/// Enablement state surfaced at the rank's `event=ready` line / doctor output
/// (spec Sec7) — the closed vocabulary this module and its callers share. A
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

#[derive(Deserialize)]
struct ExportStamp {
    manifest_sha256: String,
    per_layer_digest: HashMap<String, String>,
    #[allow(dead_code)] // not compared today; kept for forward compat / logging
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
    /// `manifest_sha256` disagrees: the model was re-exported (weights
    /// changed) without re-running the attention exporter.
    ManifestMismatch,
    /// The stamp's baked `W` disagrees with what THIS manifest derives.
    WMismatch,
    /// The stamp's `per_layer_digest` has no entry for an owned global layer
    /// (e.g. exported with a `--layers` filter that missed this rank's slice).
    MissingCoverage,
}

fn check_stamp(
    stamp: &ExportStamp,
    manifest_sha256: &str,
    derived_w: usize,
    owned_layers: &[usize],
) -> StampCheck {
    if stamp.manifest_sha256 != manifest_sha256 {
        return StampCheck::ManifestMismatch;
    }
    if stamp.w != derived_w {
        return StampCheck::WMismatch;
    }
    for &gl in owned_layers {
        if !stamp.per_layer_digest.contains_key(&format!("{gl:02}")) {
            return StampCheck::MissingCoverage;
        }
    }
    StampCheck::Ok
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
fn resolve_bool(cfg: Option<bool>, env_key: &str) -> (bool, &'static str) {
    match cfg {
        Some(v) => (v, "engine_arg"),
        None => {
            if std::env::var_os(env_key).is_some() {
                (true, "env")
            } else {
                (false, "default")
            }
        }
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

struct CanaryCase {
    label: String,
    x: f32,
    rne: f32,
}

fn canary_cases() -> Vec<CanaryCase> {
    let mut cases = Vec::with_capacity(CANARY_BASES.len() * 3);
    for &base in CANARY_BASES.iter() {
        let g_lo = to_bf16(base);
        let ulp = bf16_ulp_at(g_lo);
        for (frac, tag) in [(0.25f32, "f=.25"), (0.5, "f=.5(tie)"), (0.75, "f=.75")] {
            let x = g_lo + frac * ulp;
            let rne = to_bf16(x);
            cases.push(CanaryCase {
                label: format!("base={base} {tag}"),
                x,
                rne,
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
            let label = if i < n_battery {
                battery[i].label.clone()
            } else {
                format!("tie_case idx={}", i - n_battery)
            };
            warn!(
                target: "cascadia::glm5",
                case = %label,
                want = format!("0x{want:08x}"),
                got = format!("0x{got:08x}"),
                "bf16 canary case mismatch",
            );
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
    poisoned: AtomicBool,
    transient_fails: AtomicU32,
    used: AtomicBool,
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
            StampCheck::MissingCoverage => {
                warn!(
                    target: "cascadia::glm5",
                    event = "ov_attn_stale_ir",
                    reason = "missing_layer_digest",
                );
                return None;
            }
        }

        // Startup probe (spec Sec7): a real (non-stub) shim at the ABI this
        // crate was built against, and the target device actually present —
        // BEFORE declaring the path usable. `shim_abi_version() == None`
        // covers both the stub build AND (defensively) any future build mode
        // where the shim could be linked without being recompiled alongside
        // this crate; today's `cc::Build` always recompiles it together, but
        // the check costs nothing and is what the brief explicitly asked for
        // given this fleet's stale-DLL history.
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
        // re-export or driver bump can never hit a stale blob (spec Sec6) —
        // the subdirectory name changes whenever the stamp does.
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
        for &gl in owned_layers {
            let xml = dir.join(format!("layer_{gl:02}.xml"));
            let path_str = match xml.to_str() {
                Some(s) => s,
                None => {
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
                    tracing::debug!(
                        target: "cascadia::glm5",
                        layer = gl,
                        error = %e,
                        "attn_ov layer compile failed; this layer stays on the Rust path",
                    );
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
            warn!(
                target: "cascadia::glm5",
                event = "ov_attn_partial",
                layers_ok,
                layers_expected = owned_layers.len(),
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
            device = %device,
            device_source = enabled_src, // enabled + device share one config-precedence story; both sources logged below explicitly
            device_full_name = %device_full_name,
            min_rows,
            min_rows_source = min_rows_src,
            device_source_actual = device_src,
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
            poisoned: AtomicBool::new(false),
            transient_fails: AtomicU32::new(0),
            used: AtomicBool::new(false),
        })
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
    /// doc). `x` is `[rows, hidden]` (real rows only); `past_lc`/`past_rc` are
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
            return None;
        }
        if x.len() != rows * self.hidden
            || past_lc.len() != past_len * self.kv_lora
            || past_rc.len() != past_len * self.qk_rope
        {
            return None;
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
            let rt_snapshot_device = self.device.clone();
            drop(guard);
            self.note_infer_failure(layer_local, &rt_snapshot_device, &e);
            return None;
        }

        // Output order is the exporter's `Model([attn_out, lc, rc], ...)`
        // (tools/glm5_attn_ov.py::build_layer_graph) — indices 0/1/2.
        let attn_out = Self::read_rows(rt, 0, self.rows_cap, rows, self.hidden);
        let lc = Self::read_rows(rt, 1, self.rows_cap, rows, self.kv_lora);
        let rc = Self::read_rows(rt, 2, self.rows_cap, rows, self.qk_rope);
        drop(guard);

        let (Some(attn_out), Some(lc), Some(rc)) = (attn_out, lc, rc) else {
            self.note_infer_failure(
                layer_local,
                &self.device,
                &OvError::Native("unexpected output shape".to_string()),
            );
            return None;
        };

        self.transient_fails.store(0, Ordering::Relaxed);
        self.used.store(true, Ordering::Relaxed);
        Some(AttnWindowOut { attn_out, lc, rc })
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

    /// Classify one infer failure: typed resource exhaustion latches
    /// immediately (same taxonomy as `OvExperts::poison`); otherwise counts
    /// toward the consecutive-transient latch (default 3, see the module
    /// doc). Every individual transient failure is logged (WARN) — the
    /// silent-degrade weakness this module exists to not repeat.
    fn note_infer_failure(&self, layer_local: usize, device: &str, err: &OvError) {
        let msg = err.to_string();
        let fatal = cascadia_ov_genai_shim::last_error_resource_exhausted()
            || is_fatal_resource_error(&msg);
        if fatal {
            self.poison(&format!("resource exhaustion on device {device}: {msg}"));
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
            exporter_version: "1".to_string(),
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
            poisoned: AtomicBool::new(false),
            transient_fails: AtomicU32::new(0),
            used: AtomicBool::new(false),
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
        ov.note_infer_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Enabled,
            "1st transient: not yet latched"
        );
        ov.note_infer_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Enabled,
            "2nd transient: not yet latched"
        );
        ov.note_infer_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Poisoned,
            "3rd consecutive transient: latched"
        );
    }

    #[test]
    fn a_reset_transient_counter_does_not_accumulate_across_successes() {
        let ov = test_instance();
        let err = OvError::Native("device busy".to_string());
        ov.note_infer_failure(0, "CPU", &err);
        ov.note_infer_failure(0, "CPU", &err);
        // Simulate an intervening success (prefill_window resets the counter
        // on every successful window).
        ov.transient_fails.store(0, Ordering::Relaxed);
        ov.note_infer_failure(0, "CPU", &err);
        ov.note_infer_failure(0, "CPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Enabled,
            "non-consecutive failures (reset by a success in between) must not latch"
        );
    }

    #[test]
    fn typed_resource_exhaustion_latches_on_the_first_failure() {
        let ov = test_instance();
        let err = OvError::Native(
            "compile on GPU: openvino-genai error: resource unavailable try again".to_string(),
        );
        ov.note_infer_failure(0, "GPU", &err);
        assert_eq!(
            ov.state(),
            OvAttnState::Poisoned,
            "fatal resource error must latch immediately"
        );
    }

    #[test]
    fn poison_is_idempotent_and_does_not_repanic_on_second_trigger() {
        let ov = test_instance();
        ov.poison("first");
        ov.poison("second"); // must not double-log-crash or flip anything further
        assert_eq!(ov.state(), OvAttnState::Poisoned);
    }
}
