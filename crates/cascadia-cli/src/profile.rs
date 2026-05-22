//! `cascadia profile-devices` — measure compile-time + decode tok/s
//! across OV plugins on the worker host.
//!
//! Step 1 of [#41](https://github.com/labscommunity/cascadia/issues/41)
//! (three-tier {iGPU, NPU, CPU} ILP placement). The placement ILP needs
//! per-(layer, device) latency to make decisions; this tool produces the
//! per-(model, device) baseline that justifies — or invalidates — the
//! ILP itself. Without this measurement an operator can't tell whether
//! `--device GPU` already wins, whether `HETERO:GPU,CPU` regresses
//! (it did on Lunar Lake in our beta sweep), or whether NPU is even
//! reachable for a given model class.
//!
//! Output is a `device_profile.json` with the schema in
//! [`DeviceProfile`]. The same file is consumed by the (future) ILP
//! step and by `docs/perf/DEVICE_PROFILE.md`'s recommendation matrix.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use cascadia_ov_genai_shim::{self as shim, GenConfig, LlmPipeline, PluginConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Profile each available OV device against a single model. Writes a
/// device-profile JSON; emits a short summary to stderr.
#[derive(Parser, Debug, Clone)]
pub struct ProfileDevicesArgs {
    /// Path to an exported OV-GenAI model directory (containing
    /// `openvino_model.xml`, `tokenizer*.{xml,bin}`, etc.). Same path
    /// you'd pass to `cascadia worker --engine ov-genai --model <PATH>`.
    #[arg(long)]
    pub model: String,

    /// Where to write the device profile JSON. Defaults to
    /// `device_profile.json` in the current directory.
    #[arg(long, default_value = "device_profile.json")]
    pub output: PathBuf,

    /// Prompt to use for measurement. Should be a short, deterministic
    /// query; the bench measures decode latency, so a long prompt would
    /// inflate the wall-time without measuring more.
    #[arg(long, default_value = "Explain Intel Lunar Lake in three sentences.")]
    pub prompt: String,

    /// Tokens to generate per run. Default is small enough that several
    /// runs across several devices fits comfortably under five minutes,
    /// large enough that compile overhead doesn't dominate the timing.
    #[arg(long, default_value_t = 32)]
    pub max_tokens: u32,

    /// Number of measured runs per device (best is reported as tok/s).
    #[arg(long, default_value_t = 3)]
    pub runs: u32,

    /// Warmup runs per device (not counted in tok/s). One warmup is
    /// usually enough on Intel iGPU to clear shader-cache + KV first-touch
    /// effects; bump to two if you see large first-vs-second-run variance.
    #[arg(long, default_value_t = 1)]
    pub warmup: u32,

    /// Devices to profile. `auto` enumerates every plugin OV sees on
    /// this host (CPU, GPU, NPU, etc.); a comma-separated list lets you
    /// pin the set (e.g. `CPU,GPU` to skip NPU on hosts where it
    /// reliably fails). HETERO strings are accepted verbatim — pass
    /// e.g. `auto,HETERO:GPU,CPU` to combine the auto-enum with an
    /// explicit HETERO probe.
    #[arg(long, default_value = "auto")]
    pub devices: String,

    /// Also profile every `HETERO:` priority permutation of the auto-
    /// enumerated devices. Off by default because the permutation count
    /// grows factorially; on a 3-device host (CPU/GPU/NPU) we add 6
    /// extra runs. Use when triaging "which HETERO order wins?".
    #[arg(long, default_value_t = false)]
    pub include_hetero_permutations: bool,

    /// Optional OV plugin CACHE_DIR; passed through to every pipeline
    /// construction. Off by default so the bench measures cold-compile
    /// times honestly; set to amortise across re-runs of the same model
    /// on the same host.
    #[arg(long)]
    pub ov_cache_dir: Option<PathBuf>,

    /// Suppress the result table that's printed to stdout by default.
    /// JSON output is still written either way.
    #[arg(long, default_value_t = false)]
    pub no_summary: bool,
}

/// One device's measurements. `error == Some(_)` ⇒ compile or run
/// failed; all `_s` fields are `None` in that case. A `None` here is
/// not a JSON `null` — serde drops the field entirely.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceResult {
    /// Device string as passed to OV (`CPU`, `GPU`, `HETERO:GPU,CPU`).
    pub device: String,
    /// Pipeline construction wall time (includes compile + plugin init).
    /// First-load only; cached compiles don't reflect this number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compile_s: Option<f64>,
    /// Warmup wall time. Reported for diagnosis; not part of best_run_s.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_s: Option<f64>,
    /// Best (min) measured run wall time across `runs` repetitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_run_s: Option<f64>,
    /// All measured runs in order. Useful for spotting warm-up bias the
    /// `warmup` flag didn't filter out. Empty on failure.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runs_s: Vec<f64>,
    /// tok/s = max_tokens / best_run_s. None on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tok_per_sec: Option<f64>,
    /// First 200 characters of the best-run output. Used as a
    /// quality-equivalence smoke check: if devices disagree wildly here
    /// the placement isn't safe even if the timings look good.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_preview: Option<String>,
    /// OV-side compile or runtime error. Plain prose; the operator
    /// triages from here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One enumerated host plugin. `full_name` is `FULL_DEVICE_NAME` from
/// the OV core; missing if the plugin loads but refuses the property
/// (rare; some custom plugins).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostDevice {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostInfo {
    pub host_devices: Vec<HostDevice>,
}

/// Full profile output. JSON-stable; consumed by both the (future) ILP
/// step and human readers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// `device_profile.json` v1 — bump if the schema changes
    /// incompatibly. Readers should refuse unknown major versions.
    pub schema_version: u32,
    pub hardware: HostInfo,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub runs: u32,
    pub warmup: u32,
    pub results: Vec<DeviceResult>,
    /// Device with the highest measured tok/s; `None` if every probe
    /// failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_tok_per_sec: Option<f64>,
}

pub const SCHEMA_VERSION: u32 = 1;

/// Enumerate OV plugins on this host, fetching FULL_DEVICE_NAME for
/// each. Returns `Ok(empty)` rather than `Err` when no plugins enumerate;
/// the caller still emits a usable (if empty) profile in that case.
pub fn enumerate_host_devices() -> Result<Vec<HostDevice>> {
    let names = shim::list_devices().context("OV list_devices")?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let full = match shim::device_full_name(&name) {
            Ok(s) => Some(s),
            Err(e) => {
                warn!(device = %name, error = %e, "FULL_DEVICE_NAME query failed");
                None
            }
        };
        out.push(HostDevice {
            name,
            full_name: full,
        });
    }
    Ok(out)
}

/// Resolve the user's `--devices` argument into a concrete probe list.
/// `auto` expands to the host's enumerated devices; an explicit
/// comma-list passes through. The two forms can be combined
/// (`auto,HETERO:GPU,CPU`) to add HETERO probes on top of the auto enum.
///
/// We deliberately do NOT split on `:` so that `HETERO:GPU,CPU,NPU`
/// stays a single token — the comma inside `HETERO:...` is part of the
/// device string, not a list separator. We achieve that by splitting on
/// commas at the top level only when no preceding `HETERO:` is open.
pub fn resolve_device_list(arg: &str, host: &[HostDevice]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for tok in split_top_level(arg) {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if tok.eq_ignore_ascii_case("auto") {
            for hd in host {
                if seen.insert(hd.name.clone()) {
                    out.push(hd.name.clone());
                }
            }
        } else if seen.insert(tok.to_string()) {
            out.push(tok.to_string());
        }
    }
    out
}

/// Split a comma-separated `--devices` argument into individual probe
/// tokens, treating each `HETERO:`/`MULTI:`/`BATCH:`/`AUTO:` token as a
/// greedy run that consumes subsequent bare chunks (the comma in
/// `HETERO:GPU,CPU` is OV's own list separator, not ours).
///
/// Rule: a plugin token starts at any chunk matching `<PREFIX>:...` and
/// ends at the next plugin-prefixed chunk OR end-of-input. Bare chunks
/// (`CPU`, `GPU`, `auto`) outside a plugin are their own tokens.
///
/// Examples:
/// - `CPU,GPU,NPU`                       → `[CPU, GPU, NPU]`
/// - `auto,HETERO:GPU,CPU`               → `[auto, HETERO:GPU,CPU]`
/// - `auto,HETERO:GPU,CPU,HETERO:NPU,GPU`→ `[auto, HETERO:GPU,CPU, HETERO:NPU,GPU]`
/// - `HETERO:GPU,CPU,NPU`                → `[HETERO:GPU,CPU,NPU]` (all 3 in one HETERO)
///
/// Ambiguous case: `HETERO:GPU,CPU` followed by bare `NPU` cannot be
/// expressed with this grammar — the bare NPU is consumed into HETERO.
/// Operators who want both must put bares first: `NPU,HETERO:GPU,CPU`.
fn split_top_level(s: &str) -> Vec<String> {
    const PLUGIN_PREFIXES: [&str; 4] = ["HETERO:", "MULTI:", "BATCH:", "AUTO:"];
    fn starts_with_plugin_prefix(s: &str) -> bool {
        let up = s.trim_start().to_uppercase();
        PLUGIN_PREFIXES.iter().any(|p| up.starts_with(p))
    }

    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_plugin = false;
    for part in s.split(',') {
        let next_is_plugin = starts_with_plugin_prefix(part);
        if in_plugin && !next_is_plugin {
            // Append this bare chunk to the open HETERO/MULTI token —
            // the comma was OV's, not ours.
            buf.push(',');
            buf.push_str(part);
        } else {
            // Either we're between tokens (in_plugin=false) or we're
            // closing the previous HETERO because a new plugin prefix
            // arrived. Flush, then start the new token.
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            buf.push_str(part);
            in_plugin = next_is_plugin;
            if !in_plugin {
                // Bare device — single-chunk token, flush immediately.
                out.push(std::mem::take(&mut buf));
            }
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Build the cartesian permutations of `devices` as HETERO priority
/// strings: `HETERO:DEV1,DEV2,...`. For 3 devices this produces 6
/// strings; for 2 devices, 2. Excludes single-device permutations
/// (those duplicate the non-HETERO probes).
pub fn hetero_permutations(devices: &[String]) -> Vec<String> {
    if devices.len() < 2 {
        return Vec::new();
    }
    let mut perms: Vec<Vec<&String>> = vec![Vec::new()];
    for d in devices {
        let mut next = Vec::new();
        for p in &perms {
            for i in 0..=p.len() {
                let mut q = p.clone();
                q.insert(i, d);
                next.push(q);
            }
        }
        perms = next;
    }
    perms
        .into_iter()
        .map(|p| {
            let parts: Vec<&str> = p.iter().map(|s| s.as_str()).collect();
            format!("HETERO:{}", parts.join(","))
        })
        .collect()
}

/// Probe one device. On failure, returns `DeviceResult` with `error =
/// Some(_)` so the caller can write a complete profile (the operator
/// wants to see WHICH devices failed and why, not just the survivors).
pub fn probe_device(device: &str, args: &ProfileDevicesArgs) -> DeviceResult {
    let mut plugin = PluginConfig::new();
    if let Some(cache) = args.ov_cache_dir.as_ref() {
        // OV plugin CACHE_DIR; same property name on all plugins.
        plugin = plugin.with("CACHE_DIR", cache.to_string_lossy().to_string());
    }

    info!(device = %device, "compiling pipeline");
    let t_compile = Instant::now();
    let pipe = match LlmPipeline::new(&args.model, device, &plugin) {
        Ok(p) => p,
        Err(e) => {
            return DeviceResult {
                device: device.to_string(),
                compile_s: None,
                warmup_s: None,
                best_run_s: None,
                runs_s: Vec::new(),
                tok_per_sec: None,
                output_preview: None,
                error: Some(format!("compile-fail: {e}")),
            };
        }
    };
    let compile_s = t_compile.elapsed().as_secs_f64();

    let cfg = GenConfig {
        max_new_tokens: args.max_tokens,
        do_sample: false,
        temperature: 1.0,
        num_assistant_tokens: 0,
        max_ngram_size: 0,
    };

    // Warmup. Failure here is fatal for this device (we can't trust
    // measured numbers if even warmup didn't compile codegen).
    let mut warmup_s: Option<f64> = None;
    for i in 0..args.warmup {
        let t = Instant::now();
        match pipe.generate(&args.prompt, &cfg) {
            Ok(_) => {
                if i + 1 == args.warmup {
                    warmup_s = Some(t.elapsed().as_secs_f64());
                }
            }
            Err(e) => {
                return DeviceResult {
                    device: device.to_string(),
                    compile_s: Some(compile_s),
                    warmup_s: None,
                    best_run_s: None,
                    runs_s: Vec::new(),
                    tok_per_sec: None,
                    output_preview: None,
                    error: Some(format!("warmup-fail: {e}")),
                };
            }
        }
    }

    // Measured runs.
    let mut runs_s: Vec<f64> = Vec::with_capacity(args.runs as usize);
    let mut last_output: Option<String> = None;
    for _ in 0..args.runs {
        let t = Instant::now();
        match pipe.generate(&args.prompt, &cfg) {
            Ok(r) => {
                runs_s.push(t.elapsed().as_secs_f64());
                last_output = Some(r.text);
            }
            Err(e) => {
                return DeviceResult {
                    device: device.to_string(),
                    compile_s: Some(compile_s),
                    warmup_s,
                    best_run_s: None,
                    runs_s,
                    tok_per_sec: None,
                    output_preview: None,
                    error: Some(format!("run-fail: {e}")),
                };
            }
        }
    }

    let best_run_s = runs_s.iter().copied().fold(f64::INFINITY, f64::min);
    let tok_per_sec = if best_run_s > 0.0 {
        Some(args.max_tokens as f64 / best_run_s)
    } else {
        None
    };
    let output_preview = last_output.map(|s| truncate_preview(&s, 200));

    DeviceResult {
        device: device.to_string(),
        compile_s: Some(compile_s),
        warmup_s,
        best_run_s: Some(best_run_s),
        runs_s,
        tok_per_sec,
        output_preview,
        error: None,
    }
}

fn truncate_preview(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{truncated}…")
}

/// Pick the best device by tok/s, ignoring failures. Returns
/// (device_name, tok_per_sec) or `None` if every probe failed.
pub fn pick_best(results: &[DeviceResult]) -> Option<(String, f64)> {
    results
        .iter()
        .filter_map(|r| r.tok_per_sec.map(|tps| (r.device.clone(), tps)))
        .fold(None, |acc, (dev, tps)| match acc {
            None => Some((dev, tps)),
            Some((_, best_tps)) if tps > best_tps => Some((dev, tps)),
            other => other,
        })
}

/// Run the full profile: enumerate, resolve, probe, write JSON.
pub fn run_profile(args: &ProfileDevicesArgs) -> Result<DeviceProfile> {
    let host_devices = enumerate_host_devices().unwrap_or_else(|e| {
        warn!(error = %e, "failed to enumerate OV devices; auto resolves to empty");
        Vec::new()
    });

    let mut probes = resolve_device_list(&args.devices, &host_devices);
    if args.include_hetero_permutations {
        let auto_names: Vec<String> = host_devices.iter().map(|d| d.name.clone()).collect();
        for h in hetero_permutations(&auto_names) {
            if !probes.contains(&h) {
                probes.push(h);
            }
        }
    }

    if probes.is_empty() {
        return Err(anyhow!(
            "no devices to probe — pass --devices CPU,GPU,NPU explicitly or install an OV plugin"
        ));
    }

    info!(devices = ?probes, model = %args.model, "starting profile");
    let mut results = Vec::with_capacity(probes.len());
    for dev in &probes {
        let t = Instant::now();
        let r = probe_device(dev, args);
        let elapsed = t.elapsed();
        match &r.error {
            None => info!(device = %dev, tok_s = ?r.tok_per_sec, elapsed = ?elapsed, "device done"),
            Some(e) => warn!(device = %dev, error = %e, elapsed = ?elapsed, "device failed"),
        }
        results.push(r);
    }

    let (best_device, best_tps) = match pick_best(&results) {
        Some((d, t)) => (Some(d), Some(t)),
        None => (None, None),
    };

    let profile = DeviceProfile {
        schema_version: SCHEMA_VERSION,
        hardware: HostInfo { host_devices },
        model: args.model.clone(),
        prompt: args.prompt.clone(),
        max_tokens: args.max_tokens,
        runs: args.runs,
        warmup: args.warmup,
        results,
        best_device,
        best_tok_per_sec: best_tps,
    };

    let json = serde_json::to_string_pretty(&profile).context("serialise profile")?;
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create parent dir {parent:?}"))?;
        }
    }
    fs::write(&args.output, json).with_context(|| format!("write {}", args.output.display()))?;
    info!(path = %args.output.display(), "wrote device profile");

    if !args.no_summary {
        print_summary(&profile);
    }

    Ok(profile)
}

fn print_summary(p: &DeviceProfile) {
    println!();
    println!("== device profile ==");
    println!("model:  {}", p.model);
    println!("prompt: {:?}", p.prompt);
    println!(
        "config: max_tokens={} runs={} warmup={}",
        p.max_tokens, p.runs, p.warmup
    );
    println!();
    println!(
        "{:36}  {:>10}  {:>10}  {:>10}  {}",
        "device", "compile_s", "best_s", "tok/s", "status"
    );
    println!("{}", "-".repeat(86));
    for r in &p.results {
        let compile = r
            .compile_s
            .map(|v| format!("{:>10.2}", v))
            .unwrap_or_else(|| "        --".into());
        let best = r
            .best_run_s
            .map(|v| format!("{:>10.3}", v))
            .unwrap_or_else(|| "        --".into());
        let tps = r
            .tok_per_sec
            .map(|v| format!("{:>10.2}", v))
            .unwrap_or_else(|| "        --".into());
        let status = match &r.error {
            Some(e) => {
                let short: String = e.chars().take(40).collect();
                format!("FAIL: {short}")
            }
            None => "ok".to_string(),
        };
        println!(
            "{:36}  {}  {}  {}  {}",
            r.device, compile, best, tps, status
        );
    }
    println!();
    match (&p.best_device, p.best_tok_per_sec) {
        (Some(d), Some(t)) => println!("best: {d} @ {t:.2} tok/s"),
        _ => println!("best: (no device succeeded)"),
    }
    println!();
}

/// CLI dispatch entry point. Currently a thin wrapper over
/// [`run_profile`]; promotes to a Tokio-aware multi-host loop when the
/// (future) ILP step needs concurrent per-device probing.
pub fn cmd_profile_devices(args: ProfileDevicesArgs) -> Result<()> {
    if args.runs == 0 {
        // Not fatal — an operator may want device enumeration only.
        warn!("runs=0 — measurements will be empty; only enumeration is recorded");
    }
    run_profile(&args).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(devs: &[&str]) -> Vec<HostDevice> {
        devs.iter()
            .map(|d| HostDevice {
                name: d.to_string(),
                full_name: None,
            })
            .collect()
    }

    #[test]
    fn resolve_auto_expands_to_host() {
        let r = resolve_device_list("auto", &host(&["CPU", "GPU", "NPU"]));
        assert_eq!(r, vec!["CPU", "GPU", "NPU"]);
    }

    #[test]
    fn resolve_dedups() {
        let r = resolve_device_list("CPU,auto,GPU", &host(&["CPU", "GPU"]));
        assert_eq!(r, vec!["CPU", "GPU"]);
    }

    #[test]
    fn resolve_keeps_hetero_intact() {
        let r = resolve_device_list("GPU,HETERO:GPU,CPU", &host(&[]));
        // The HETERO:GPU,CPU should remain a single token, not get
        // split on its internal comma.
        assert_eq!(r, vec!["GPU", "HETERO:GPU,CPU"]);
    }

    #[test]
    fn resolve_auto_combined_with_hetero() {
        let r = resolve_device_list("auto,HETERO:GPU,CPU,NPU", &host(&["CPU", "GPU", "NPU"]));
        assert_eq!(r, vec!["CPU", "GPU", "NPU", "HETERO:GPU,CPU,NPU"]);
    }

    #[test]
    fn resolve_multiple_hetero_tokens() {
        // Regression test for the bug where consecutive HETERO tokens
        // collapsed into a single malformed device string. Each
        // `HETERO:` should start a new token; bare chunks before the
        // next plugin prefix get appended to the open one.
        let r = resolve_device_list(
            "auto,HETERO:GPU,CPU,HETERO:GPU,CPU,NPU,HETERO:NPU,GPU,CPU",
            &host(&["CPU", "GPU", "NPU"]),
        );
        assert_eq!(
            r,
            vec![
                "CPU",
                "GPU",
                "NPU",
                "HETERO:GPU,CPU",
                "HETERO:GPU,CPU,NPU",
                "HETERO:NPU,GPU,CPU",
            ]
        );
    }

    #[test]
    fn resolve_bare_first_then_hetero() {
        // Bare devices that precede a HETERO are correctly NOT
        // swallowed into it.
        let r = resolve_device_list("CPU,HETERO:GPU,NPU", &host(&[]));
        assert_eq!(r, vec!["CPU", "HETERO:GPU,NPU"]);
    }

    #[test]
    fn split_top_level_multi_hetero() {
        let r = split_top_level("HETERO:GPU,CPU,HETERO:NPU,GPU");
        assert_eq!(r, vec!["HETERO:GPU,CPU", "HETERO:NPU,GPU"]);
    }

    #[test]
    fn split_top_level_lowercase_hetero_is_bare_token() {
        // OV's device plugin names are case-insensitive in our parser
        // for the prefix detection (uppercased before match). Verify a
        // lowercase form still routes correctly.
        let r = split_top_level("hetero:gpu,cpu");
        assert_eq!(r, vec!["hetero:gpu,cpu"]);
    }

    #[test]
    fn hetero_permutations_3() {
        let p = hetero_permutations(&["CPU".into(), "GPU".into(), "NPU".into()]);
        assert_eq!(p.len(), 6, "3! = 6 permutations expected, got {p:?}");
        assert!(p.contains(&"HETERO:CPU,GPU,NPU".to_string()));
        assert!(p.contains(&"HETERO:NPU,GPU,CPU".to_string()));
    }

    #[test]
    fn hetero_permutations_empty_below_2() {
        assert!(hetero_permutations(&[]).is_empty());
        assert!(hetero_permutations(&["CPU".into()]).is_empty());
    }

    #[test]
    fn pick_best_skips_failures() {
        let results = vec![
            DeviceResult {
                device: "CPU".into(),
                compile_s: Some(1.0),
                warmup_s: Some(1.0),
                best_run_s: Some(1.0),
                runs_s: vec![1.0],
                tok_per_sec: Some(32.0),
                output_preview: Some("ok".into()),
                error: None,
            },
            DeviceResult {
                device: "NPU".into(),
                compile_s: None,
                warmup_s: None,
                best_run_s: None,
                runs_s: vec![],
                tok_per_sec: None,
                output_preview: None,
                error: Some("compile-fail".into()),
            },
            DeviceResult {
                device: "GPU".into(),
                compile_s: Some(8.0),
                warmup_s: Some(0.8),
                best_run_s: Some(0.5),
                runs_s: vec![0.5],
                tok_per_sec: Some(64.0),
                output_preview: Some("ok".into()),
                error: None,
            },
        ];
        let best = pick_best(&results).unwrap();
        assert_eq!(best.0, "GPU");
        assert!((best.1 - 64.0).abs() < 1e-9);
    }

    #[test]
    fn pick_best_all_failed_returns_none() {
        let results = vec![DeviceResult {
            device: "NPU".into(),
            compile_s: None,
            warmup_s: None,
            best_run_s: None,
            runs_s: vec![],
            tok_per_sec: None,
            output_preview: None,
            error: Some("fail".into()),
        }];
        assert!(pick_best(&results).is_none());
    }

    #[test]
    fn truncate_preview_keeps_short() {
        assert_eq!(truncate_preview("hi", 10), "hi");
    }

    #[test]
    fn truncate_preview_trims_long() {
        let s: String = "a".repeat(300);
        let t = truncate_preview(&s, 10);
        assert!(t.starts_with("aaaaaaaaaa"));
        assert!(t.ends_with('…'));
    }

    #[test]
    fn schema_version_is_1() {
        // Bumping is fine; this test exists so a careless bump is
        // an obvious code review hit.
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn json_round_trips() {
        let p = DeviceProfile {
            schema_version: SCHEMA_VERSION,
            hardware: HostInfo {
                host_devices: vec![HostDevice {
                    name: "CPU".into(),
                    full_name: Some("Intel Xeon".into()),
                }],
            },
            model: "/x".into(),
            prompt: "hi".into(),
            max_tokens: 32,
            runs: 3,
            warmup: 1,
            results: vec![DeviceResult {
                device: "CPU".into(),
                compile_s: Some(1.0),
                warmup_s: Some(0.5),
                best_run_s: Some(1.1),
                runs_s: vec![1.1, 1.2, 1.3],
                tok_per_sec: Some(29.09),
                output_preview: Some("ok".into()),
                error: None,
            }],
            best_device: Some("CPU".into()),
            best_tok_per_sec: Some(29.09),
        };
        let s = serde_json::to_string(&p).unwrap();
        let p2: DeviceProfile = serde_json::from_str(&s).unwrap();
        assert_eq!(p2.results[0].runs_s.len(), 3);
        assert_eq!(p2.best_device.as_deref(), Some("CPU"));
    }
}
