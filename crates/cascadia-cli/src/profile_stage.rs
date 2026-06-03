//! `profile-devices --per-stage` — build the per-(stage, device) cost table
//! the #41 placement ILP consumes (step 1.5, between `profile-devices` and
//! `cascadia place`).
//!
//! For each stage IR in a multi-stage shard it compiles the stage on each
//! available device and times a forward pass with **zeroed inputs** (compute
//! latency is shape-, not value-, dependent), recording: the stage's resident
//! weight bytes, the per-device latency, and op-support — a device that fails
//! to *compile* a stage is simply omitted from that stage's latency map, which
//! the solver reads as "can't place here" ([`crate::placement`]). Per-device
//! memory budgets come from OV (`GPU_DEVICE_TOTAL_MEM_SIZE` /
//! `NPU_DEVICE_TOTAL_MEM_SIZE`); the shared UMA pool + CPU budget are
//! operator-provided via `--pool-gb` (cross-platform RAM detection would need
//! a dependency we deliberately avoid).
//!
//! Operates on a **static** export (`cascadia shard --target npu`): static
//! input shapes let us size the zeroed inputs from the compiled model, and the
//! static shards run on GPU/CPU/NPU alike via the #63 static-KV runtime — so
//! one profile covers all three tiers with one set of IRs.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Args;

// The free helpers below are consumed by the `openvino`-gated submodule and
// the unit tests; in a stub (no-feature) non-test build they read as dead, so
// each carries `#[cfg_attr(not(feature = "openvino"), allow(dead_code))]` to
// silence exactly that case without blanket-allowing real dead code.

#[derive(Args, Debug, Clone)]
pub struct PerStageArgs {
    /// Multi-stage shard directory (contains `stage_0/`, `stage_1/`, … each
    /// with `openvino_model.xml`). Must be a STATIC export
    /// (`cascadia shard --target npu …`).
    #[arg(long)]
    pub shard: PathBuf,

    /// Output placement-profile JSON — the input to `cascadia place`.
    #[arg(long, default_value = "placement_profile.json")]
    pub output: PathBuf,

    /// Devices to profile: `auto` (every plugin OV sees) or a comma list
    /// like `GPU,NPU,CPU`.
    #[arg(long, default_value = "auto")]
    pub devices: String,

    /// Timed forward passes per (stage, device); the best (min) is recorded.
    #[arg(long, default_value_t = 5)]
    pub runs: u32,

    /// Warmup forward passes per (stage, device), not counted (clears
    /// shader-cache / first-touch effects).
    #[arg(long, default_value_t = 2)]
    pub warmup: u32,

    /// Fraction of each device's reported memory to treat as usable — leaves
    /// room for the KV cache, activations, and runtime overhead that share
    /// the same budget as the weights.
    #[arg(long, default_value_t = 0.9)]
    pub mem_headroom: f64,

    /// Usable shared UMA pool in GiB (system RAM minus OS headroom). Sets the
    /// solver's global memory gate AND the CPU tier's budget (the CPU can
    /// address the whole pool). Omit to skip the global gate (per-device caps
    /// only) and leave the CPU budget unbounded.
    #[arg(long)]
    pub pool_gb: Option<f64>,
}

#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Parse `stage_<n>` → `n`. `None` for anything else.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
fn stage_index(name: &str) -> Option<u32> {
    name.strip_prefix("stage_")
        .and_then(|s| s.parse::<u32>().ok())
}

/// The shard's `stage_<n>` directories, sorted by `n`.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
fn stage_dirs(shard: &Path) -> Result<Vec<PathBuf>> {
    let mut indexed: Vec<(u32, PathBuf)> = Vec::new();
    let rd = std::fs::read_dir(shard)
        .with_context(|| format!("reading shard dir {}", shard.display()))?;
    for entry in rd {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        if let Some(idx) = name.to_str().and_then(stage_index) {
            indexed.push((idx, entry.path()));
        }
    }
    indexed.sort_by_key(|(i, _)| *i);
    Ok(indexed.into_iter().map(|(_, p)| p).collect())
}

/// Sum the resident-weight bytes of a stage IR (the `.bin` blobs + the
/// `.xml` topology). Approximates the memory the compiled model occupies.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
fn dir_weight_bytes(stage_dir: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(stage_dir)
        .with_context(|| format!("reading stage dir {}", stage_dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase);
            if matches!(ext.as_deref(), Some("bin") | Some("xml")) {
                total += entry.metadata()?.len();
            }
        }
    }
    Ok(total)
}

/// Order a device set by placement preference (fastest tier first, so the
/// solver's latency-tie break favours it): GPU, then NPU, then CPU, then any
/// others alphabetically. Pure — unit-tested.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
fn order_devices(mut devices: Vec<String>) -> Vec<String> {
    fn rank(d: &str) -> (u8, String) {
        let up = d.to_ascii_uppercase();
        let tier = if up.starts_with("GPU") {
            0
        } else if up.starts_with("NPU") {
            1
        } else if up.starts_with("CPU") {
            2
        } else {
            3
        };
        (tier, up)
    }
    devices.sort_by_key(|d| rank(d));
    devices.dedup();
    devices
}

/// The CPU tier can address the whole shared pool; GPU/NPU report their own
/// (UMA-shared) addressing limit. Apply the headroom fraction in all cases.
/// `reported` is the device's raw byte budget (from OV) or the pool for CPU.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
fn usable_cap(reported: u64, headroom: f64) -> u64 {
    (reported as f64 * headroom) as u64
}

pub fn cmd_profile_per_stage(args: PerStageArgs) -> Result<()> {
    #[cfg(not(feature = "openvino"))]
    {
        let _ = args;
        bail!(
            "`profile-devices --per-stage` needs a real OpenVINO runtime; this \
             binary was built without the `openvino` feature (stub). Rebuild \
             with `--features openvino` on an Intel host."
        );
    }
    #[cfg(feature = "openvino")]
    {
        openvino_impl::run(args)
    }
}

#[cfg(feature = "openvino")]
mod openvino_impl {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Instant;

    use cascadia_ov_genai_shim::{self as shim, PluginConfig};
    use tracing::{info, warn};

    use crate::placement::{DeviceCap, PlacementProfile, StageCost};

    pub fn run(args: PerStageArgs) -> Result<()> {
        let stages = stage_dirs(&args.shard)?;
        if stages.is_empty() {
            bail!(
                "no `stage_*` directories under {} — point --shard at a \
                 multi-stage export",
                args.shard.display()
            );
        }

        let device_names = resolve_devices(&args.devices)?;
        if device_names.is_empty() {
            bail!("no devices to profile (OV enumerated none, or --devices was empty)");
        }
        info!(devices = ?device_names, stages = stages.len(), "per-stage profiling");

        // Device memory budgets.
        let pool_bytes = args.pool_gb.map(|g| (g * BYTES_PER_GIB) as u64);
        let mut dev_caps: Vec<DeviceCap> = Vec::with_capacity(device_names.len());
        for d in &device_names {
            let reported = device_reported_mem(d, pool_bytes)?;
            dev_caps.push(DeviceCap {
                device: d.clone(),
                mem_bytes: usable_cap(reported, args.mem_headroom),
            });
        }

        // Per-stage cost.
        let mut stage_costs: Vec<StageCost> = Vec::with_capacity(stages.len());
        for (i, sd) in stages.iter().enumerate() {
            let xml = sd.join("openvino_model.xml");
            let xml_s = xml
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 path {}", xml.display()))?;
            let mem = dir_weight_bytes(sd)?;
            let mut lat: BTreeMap<String, f64> = BTreeMap::new();
            for d in &device_names {
                match time_stage(xml_s, d, args.warmup, args.runs) {
                    Ok(ms) => {
                        info!(stage = i, device = %d, ms, "timed");
                        lat.insert(d.clone(), ms);
                    }
                    Err(e) => {
                        warn!(stage = i, device = %d, error = %e,
                            "stage unsupported on device (compile/infer failed) — gated out");
                    }
                }
            }
            if lat.is_empty() {
                bail!("stage {i} compiled on no available device — cannot place it");
            }
            stage_costs.push(StageCost {
                stage: i as u32,
                mem_bytes: mem,
                lat_ms: lat,
            });
        }

        let model = args
            .shard
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let profile = PlacementProfile {
            model,
            devices: dev_caps,
            stages: stage_costs,
            pool_bytes,
        };

        let json = serde_json::to_string_pretty(&profile)?;
        std::fs::write(&args.output, &json)
            .with_context(|| format!("writing {}", args.output.display()))?;
        print_summary(&profile);
        println!("wrote {}", args.output.display());
        Ok(())
    }

    /// Resolve `--devices` to a preference-ordered list. `auto` enumerates
    /// every OV plugin on the host.
    fn resolve_devices(arg: &str) -> Result<Vec<String>> {
        let raw: Vec<String> = if arg.eq_ignore_ascii_case("auto") {
            shim::list_devices().context("OV list_devices")?
        } else {
            arg.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
        Ok(order_devices(raw))
    }

    /// Reported byte budget for a device: OV's total-mem property for
    /// GPU/NPU, the operator's pool for CPU (or a large sentinel if no pool
    /// was given, since the CPU can use whatever RAM exists).
    fn device_reported_mem(device: &str, pool_bytes: Option<u64>) -> Result<u64> {
        let up = device.to_ascii_uppercase();
        let prop = if up.starts_with("GPU") {
            Some("GPU_DEVICE_TOTAL_MEM_SIZE")
        } else if up.starts_with("NPU") {
            Some("NPU_DEVICE_TOTAL_MEM_SIZE")
        } else {
            None
        };
        if let Some(key) = prop {
            let raw = shim::device_property(device, key)
                .with_context(|| format!("querying {key} on {device}"))?;
            return raw
                .trim()
                .parse::<u64>()
                .with_context(|| format!("parsing {key}='{raw}' as bytes"));
        }
        // CPU / other: the shared pool, or a large sentinel (256 GiB) when
        // the operator didn't bound it.
        Ok(pool_bytes.unwrap_or(256 * 1024 * 1024 * 1024))
    }

    /// Compile a stage on a device and time a forward pass with zeroed
    /// inputs. Returns the best (min) of `runs` timings in milliseconds, or
    /// an error if the device can't compile/run the stage (op-support gate).
    fn time_stage(xml: &str, device: &str, warmup: u32, runs: u32) -> Result<f64> {
        let plugin = PluginConfig::default();
        let mut rt = shim::Runtime::compile(xml, device, &plugin)
            .with_context(|| format!("compile on {device}"))?;

        let n_in = rt.input_count();
        for i in 0..n_in {
            let name = rt.input_name(i)?;
            let shape = rt.input_shape(i)?;
            let dtype = rt.input_dtype(i)?;
            if shape.iter().any(|&d| d == 0) {
                bail!(
                    "input '{name}' has a dynamic/zero dim {shape:?} — the \
                     per-stage profiler needs a STATIC export (`--target npu`)"
                );
            }
            let elems: usize = shape.iter().product();
            let bytes = vec![0u8; elems * dtype.bytes_per_element()];
            rt.set_input(&name, dtype, &shape, &bytes)
                .with_context(|| format!("set zeroed input '{name}' {shape:?}"))?;
        }

        for _ in 0..warmup {
            rt.infer().context("warmup infer")?;
        }
        let mut best = f64::INFINITY;
        for _ in 0..runs.max(1) {
            let t = Instant::now();
            rt.infer().context("timed infer")?;
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(best)
    }

    fn print_summary(p: &PlacementProfile) {
        let gib = |b: u64| b as f64 / BYTES_PER_GIB;
        println!("per-stage profile: {} ({} stages)", p.model, p.stages.len());
        print!("  devices:");
        for d in &p.devices {
            print!(" {}={:.1}GiB", d.device, gib(d.mem_bytes));
        }
        println!();
        if let Some(pool) = p.pool_bytes {
            println!("  shared UMA pool gate: {:.1} GiB", gib(pool));
        }
        let total: u64 = p.stages.iter().map(|s| s.mem_bytes).sum();
        println!(
            "  total resident: {:.1} GiB across {} stages",
            gib(total),
            p.stages.len()
        );
        for s in &p.stages {
            let mut lats: Vec<String> = s
                .lat_ms
                .iter()
                .map(|(d, ms)| format!("{d}={ms:.2}ms"))
                .collect();
            lats.sort();
            println!(
                "  stage {:>2} ({:.2} GiB): {}",
                s.stage,
                gib(s.mem_bytes),
                lats.join(" ")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_index_parses_only_stage_dirs() {
        assert_eq!(stage_index("stage_0"), Some(0));
        assert_eq!(stage_index("stage_12"), Some(12));
        assert_eq!(stage_index("stage_x"), None);
        assert_eq!(stage_index("tokenizer"), None);
        assert_eq!(stage_index("stage_"), None);
    }

    #[test]
    fn order_devices_prefers_gpu_then_npu_then_cpu() {
        assert_eq!(
            order_devices(vec!["CPU".into(), "NPU".into(), "GPU".into()]),
            vec!["GPU", "NPU", "CPU"]
        );
        // unknown plugins sort last, alphabetically
        assert_eq!(
            order_devices(vec!["CPU".into(), "GNA".into(), "GPU".into()]),
            vec!["GPU", "CPU", "GNA"]
        );
    }

    #[test]
    fn usable_cap_applies_headroom() {
        assert_eq!(usable_cap(10_000, 0.9), 9_000);
        assert_eq!(usable_cap(0, 0.9), 0);
    }

    #[test]
    fn stage_dirs_sorts_numerically_and_skips_non_stages() {
        let tmp = std::env::temp_dir().join(format!("cascadia-pstest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        for d in ["stage_0", "stage_10", "stage_2", "tokenizer", "stage_x"] {
            std::fs::create_dir_all(tmp.join(d)).unwrap();
        }
        let dirs = stage_dirs(&tmp).unwrap();
        let names: Vec<String> = dirs
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["stage_0", "stage_2", "stage_10"]);
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
