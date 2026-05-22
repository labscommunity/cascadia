//! Compute per-channel FFN sparsity thresholds (CHESS / issue #38) from
//! a directory of `layer_<lid>.bin` capture files produced by a
//! `cascadia worker --ffn-sparsity-capture-dir <PATH>` run.
//!
//! ## Inputs
//!
//! - `--capture-dir <PATH>`: directory containing `layer_<lid>.bin`
//!   files in the format documented in `ffn_capture.rs`.
//! - `--target-active-frac <0.0..1.0>`: target fraction of intermediate
//!   lanes that should remain active after thresholding. The tool
//!   picks the `1 - target_active_frac` percentile of each channel's
//!   `|silu(gate[c])| / max_j |silu(gate[j])|` distribution as that
//!   channel's threshold.
//! - `--model-id <STRING>`: free-form identifier baked into the file
//!   header. Convention: HF repo id or local cache name. Matched
//!   loosely (printed to logs; not enforced) by the runner at load.
//! - `--output <PATH>`: where to write the resulting threshold JSON.
//!   Atomic-rename write.
//! - `--n-intermediate <N>` (default 2048): expected intermediate dim.
//!   The tool rejects any capture file with a different `n_intermediate`
//!   header field.
//! - `--notes "<TEXT>"`: optional free-form provenance string.
//!
//! ## Output
//!
//! A `PerChannelThresholds` v1 JSON file (see `ffn_thresholds.rs`):
//! one entry per covered layer with `[f32; n_intermediate]` per-
//! channel thresholds. Layers without a capture file are simply
//! absent — the runtime falls back to the global-τ (or dense) path
//! for them.
//!
//! ## Method
//!
//! For each (layer, channel):
//!   1. Read the histogram bin counts `[u32; N_BINS]`.
//!   2. Compute the cumulative distribution.
//!   3. Find the smallest bin index `b` such that the CDF at `b`
//!      reaches `1 - target_active_frac`.
//!   4. Set `τ[layer, channel] = (b + 0.5) / N_BINS`
//!      (midpoint of bin `b`).
//!
//! Step 4's midpoint estimator is the standard non-interpolating
//! quantile read-out from a uniform-bin histogram. At `N_BINS=128`
//! the bin width is 0.0078, so the quantile estimate is accurate to
//! ±0.4%. If a channel saw zero samples, its threshold defaults to
//! 0.0 (always-active fallback — won't ever drop the channel).
//!
//! ## Example
//!
//! ```text
//! # Step 1: capture from a representative corpus
//! CASCADIA_FFN_SPARSITY_CAPTURE_DIR=/tmp/caps \
//!   cascadia worker --model <K26-DIR> --device CPU
//! # ... serve a representative prompt set, then stop the worker ...
//!
//! # Step 2: calibrate
//! cargo run --release --bin calibrate_ffn_thresholds -- \
//!     --capture-dir /tmp/caps \
//!     --target-active-frac 0.5 \
//!     --model-id kimi-k2.6-instruct \
//!     --output /tmp/k26_thresholds_50.json
//!
//! # Step 3: serve with per-channel thresholds
//! cascadia worker --model <K26-DIR> --device CPU \
//!     --ffn-sparsity-thresholds-file /tmp/k26_thresholds_50.json
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cascadia_int4_gemm::ffn_capture::{CAPTURE_FILE_MAGIC, N_BINS};
use cascadia_int4_gemm::PerChannelThresholds;

struct Args {
    capture_dir: PathBuf,
    target_active_frac: f32,
    model_id: String,
    output: PathBuf,
    n_intermediate: usize,
    notes: String,
}

fn print_usage() {
    eprintln!(
        "calibrate_ffn_thresholds — convert capture histograms → per-channel τ\n\n\
        Required:\n\
        \t--capture-dir <PATH>           directory of layer_<lid>.bin files\n\
        \t--target-active-frac <F>       target active fraction in [0, 1]\n\
        \t--model-id <STR>               free-form model identifier\n\
        \t--output <PATH>                output threshold JSON path\n\n\
        Optional:\n\
        \t--n-intermediate <N>           expected intermediate dim (default 2048)\n\
        \t--notes <STR>                  free-form provenance string\n"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut capture_dir: Option<PathBuf> = None;
    let mut target_active_frac: Option<f32> = None;
    let mut model_id: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_intermediate: usize = 2048;
    let mut notes = String::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--capture-dir" => capture_dir = it.next().map(PathBuf::from),
            "--target-active-frac" => {
                target_active_frac = it.next().and_then(|s| s.parse().ok());
            }
            "--model-id" => model_id = it.next(),
            "--output" => output = it.next().map(PathBuf::from),
            "--n-intermediate" => {
                n_intermediate = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--n-intermediate requires a positive integer".to_string())?;
            }
            "--notes" => notes = it.next().unwrap_or_default(),
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    let capture_dir = capture_dir.ok_or_else(|| "--capture-dir is required".to_string())?;
    let target_active_frac = target_active_frac
        .ok_or_else(|| "--target-active-frac is required (e.g. 0.5)".to_string())?;
    let model_id = model_id.ok_or_else(|| "--model-id is required".to_string())?;
    let output = output.ok_or_else(|| "--output is required".to_string())?;
    if !(0.0..=1.0).contains(&target_active_frac) {
        return Err(format!(
            "--target-active-frac must be in [0, 1]; got {target_active_frac}"
        ));
    }
    if n_intermediate == 0 {
        return Err("--n-intermediate must be > 0".into());
    }
    Ok(Args {
        capture_dir,
        target_active_frac,
        model_id,
        output,
        n_intermediate,
        notes,
    })
}

/// One layer's parsed histogram data, ready for quantile read-out.
struct LayerHist {
    layer_id: u32,
    n_intermediate: usize,
    total_samples: u32,
    /// `counts.len() == n_intermediate * N_BINS`, channel-major.
    counts: Vec<u32>,
}

/// Parse one `layer_<lid>.bin` file. Rejects mismatched magic /
/// n_bins / n_intermediate.
fn read_layer_capture(path: &Path, expected_intermediate: usize) -> Result<LayerHist, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 32 {
        return Err(format!(
            "{} too short ({} bytes)",
            path.display(),
            bytes.len()
        ));
    }
    if &bytes[..16] != CAPTURE_FILE_MAGIC {
        return Err(format!(
            "{} magic mismatch (got {:?}, expected {:?})",
            path.display(),
            &bytes[..16],
            CAPTURE_FILE_MAGIC,
        ));
    }
    let layer_id = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let n_intermediate = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let n_bins = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
    let total_samples = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
    if n_intermediate != expected_intermediate {
        return Err(format!(
            "{}: n_intermediate {} != expected {}",
            path.display(),
            n_intermediate,
            expected_intermediate,
        ));
    }
    if n_bins != N_BINS {
        return Err(format!(
            "{}: n_bins {} != runtime N_BINS {} (capture from a different cascadia build?)",
            path.display(),
            n_bins,
            N_BINS,
        ));
    }
    let expected_body = n_intermediate * n_bins * 4;
    if bytes.len() < 32 + expected_body {
        return Err(format!(
            "{}: body too short (got {} bytes, expected {})",
            path.display(),
            bytes.len() - 32,
            expected_body,
        ));
    }
    let mut counts = Vec::with_capacity(n_intermediate * n_bins);
    for chunk in bytes[32..32 + expected_body].chunks_exact(4) {
        counts.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(LayerHist {
        layer_id,
        n_intermediate,
        total_samples,
        counts,
    })
}

/// Compute the per-channel threshold vector for one layer's
/// histograms at the given active fraction. Channels with zero
/// samples → threshold 0.0 (always-active fallback).
///
/// `target_active_frac` is the *fraction of lanes we want active*,
/// which corresponds to the `1 - target_active_frac` quantile of the
/// magnitude-ratio distribution: a higher τ drops more lanes.
fn compute_thresholds_for_layer(layer: &LayerHist, target_active_frac: f32) -> Vec<f32> {
    let n_chan = layer.n_intermediate;
    let inv_n_bins = 1.0f32 / N_BINS as f32;
    let target_cdf = 1.0 - target_active_frac;
    let mut out = Vec::with_capacity(n_chan);
    for c in 0..n_chan {
        let base = c * N_BINS;
        let row = &layer.counts[base..base + N_BINS];
        let total: u32 = row.iter().sum();
        if total == 0 {
            // No samples for this channel — leave threshold at 0
            // (always-active). The runner's per-channel mask treats
            // that as "drop nothing for this channel."
            out.push(0.0);
            continue;
        }
        let target_count = ((total as f32) * target_cdf).ceil() as u32;
        let mut cum: u32 = 0;
        let mut chosen_bin: usize = N_BINS - 1;
        for (b, &c_b) in row.iter().enumerate() {
            cum = cum.saturating_add(c_b);
            if cum >= target_count {
                chosen_bin = b;
                break;
            }
        }
        // Midpoint of the chosen bin in [0, 1].
        let τ = (chosen_bin as f32 + 0.5) * inv_n_bins;
        out.push(τ);
    }
    out
}

fn run(args: Args) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&args.capture_dir)
        .map_err(|e| format!("read_dir {}: {e}", args.capture_dir.display()))?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("layer_") && s.ends_with(".bin"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    if entries.is_empty() {
        return Err(format!(
            "no layer_*.bin files in {}",
            args.capture_dir.display()
        ));
    }
    eprintln!(
        "calibrate_ffn_thresholds: scanning {} layer files in {}",
        entries.len(),
        args.capture_dir.display(),
    );

    let mut out =
        PerChannelThresholds::new(args.model_id, args.n_intermediate, args.target_active_frac);
    out.notes = args.notes;
    let mut total_samples: u64 = 0;
    for path in &entries {
        let layer = read_layer_capture(path, args.n_intermediate)?;
        let thr = compute_thresholds_for_layer(&layer, args.target_active_frac);
        total_samples += layer.total_samples as u64;
        eprintln!(
            "  layer {:4}: samples={:8}  min_τ={:.4}  max_τ={:.4}  mean_τ={:.4}",
            layer.layer_id,
            layer.total_samples,
            thr.iter().cloned().fold(f32::INFINITY, f32::min),
            thr.iter().cloned().fold(0.0, f32::max),
            thr.iter().sum::<f32>() / thr.len() as f32,
        );
        out.upsert_layer(layer.layer_id, thr);
    }
    out.calibration_n_tokens = total_samples;
    out.save(&args.output)
        .map_err(|e| format!("save {}: {e}", args.output.display()))?;
    eprintln!(
        "calibrate_ffn_thresholds: wrote {} layers, {} total samples → {}",
        out.n_layers(),
        total_samples,
        args.output.display(),
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a one-channel layer with the given bin counts.
    fn one_channel_layer(counts_b: [u32; N_BINS]) -> LayerHist {
        LayerHist {
            layer_id: 0,
            n_intermediate: 1,
            total_samples: counts_b.iter().sum(),
            counts: counts_b.to_vec(),
        }
    }

    /// Empty histograms get τ = 0.0 (always-active fallback).
    #[test]
    fn empty_histogram_yields_zero_threshold() {
        let layer = one_channel_layer([0u32; N_BINS]);
        let τ = compute_thresholds_for_layer(&layer, 0.5);
        assert_eq!(τ[0], 0.0);
    }

    /// All samples in bin 0 → the median quantile lands in bin 0 →
    /// τ at the midpoint of bin 0.
    #[test]
    fn all_samples_in_bin_zero_yields_low_threshold() {
        let mut counts = [0u32; N_BINS];
        counts[0] = 1000;
        let layer = one_channel_layer(counts);
        let τ = compute_thresholds_for_layer(&layer, 0.5);
        assert!((τ[0] - 0.5 / N_BINS as f32).abs() < 1e-6, "got τ={}", τ[0]);
    }

    /// All samples in the last bin → quantile is the last bin → τ
    /// at the midpoint of bin N_BINS-1.
    #[test]
    fn all_samples_in_last_bin_yields_high_threshold() {
        let mut counts = [0u32; N_BINS];
        counts[N_BINS - 1] = 1000;
        let layer = one_channel_layer(counts);
        let τ = compute_thresholds_for_layer(&layer, 0.5);
        let expected = (N_BINS as f32 - 0.5) / N_BINS as f32;
        assert!((τ[0] - expected).abs() < 1e-6, "got τ={}", τ[0]);
    }

    /// Uniform distribution across all bins: the median is at bin
    /// N_BINS/2 (or 1 below due to the ceil() rounding). Confirms
    /// the quantile is the smallest bin index whose CDF >= target.
    #[test]
    fn uniform_distribution_median() {
        let counts = [10u32; N_BINS];
        let layer = one_channel_layer(counts);
        let τ = compute_thresholds_for_layer(&layer, 0.5);
        // Cumulative reaches `ceil(0.5 * 10 * 128) = 640` at bin 63
        // (cumulative = 64 * 10 = 640 inclusive at the end of bin
        // 63). So chosen_bin = 63, τ = 63.5 / 128 ≈ 0.496.
        let expected = 63.5_f32 / N_BINS as f32;
        assert!((τ[0] - expected).abs() < 1e-6, "got τ={}", τ[0]);
    }

    /// Larger target_active_frac → lower threshold (we want MORE
    /// active lanes, so we accept a lower-magnitude cutoff).
    #[test]
    fn higher_active_frac_yields_lower_threshold() {
        let counts = [10u32; N_BINS];
        let layer = one_channel_layer(counts);
        let τ_low = compute_thresholds_for_layer(&layer, 0.2)[0];
        let τ_mid = compute_thresholds_for_layer(&layer, 0.5)[0];
        let τ_hi = compute_thresholds_for_layer(&layer, 0.8)[0];
        assert!(
            τ_low > τ_mid && τ_mid > τ_hi,
            "expected τ to decrease as active_frac increases: {τ_low} > {τ_mid} > {τ_hi}",
        );
    }

    /// active_frac = 1.0 ⇒ keep every lane ⇒ threshold near 0.
    /// active_frac = 0.0 ⇒ keep no lanes ⇒ threshold near max.
    #[test]
    fn extreme_active_fracs_pin_thresholds() {
        let counts = [10u32; N_BINS];
        let layer = one_channel_layer(counts);
        let τ_all = compute_thresholds_for_layer(&layer, 1.0)[0];
        let τ_none = compute_thresholds_for_layer(&layer, 0.0)[0];
        // active_frac=1 → target_cdf=0 → target_count=0 → first
        // bin (b=0) wins (cum 10 >= 0). τ = 0.5 / 128.
        assert!(
            (τ_all - 0.5 / N_BINS as f32).abs() < 1e-6,
            "got τ_all={}",
            τ_all
        );
        // active_frac=0 → target_cdf=1 → target_count=ceil(1*1280)=1280
        // → cumulative reaches 1280 only at the last bin. τ at
        // midpoint of bin N_BINS-1.
        assert!(
            (τ_none - (N_BINS as f32 - 0.5) / N_BINS as f32).abs() < 1e-6,
            "got τ_none={}",
            τ_none,
        );
    }

    /// Smoke test: build a capture in a tempdir, run the full
    /// pipeline, parse the output JSON, verify the layer count.
    #[test]
    fn end_to_end_with_capture_in_tempdir() {
        use cascadia_int4_gemm::ffn_capture::GateCaptureState;
        let dir = tempfile::tempdir().expect("tempdir");
        let st = GateCaptureState::new(dir.path().to_owned(), 4);
        // Record a few snapshots into two layers.
        for _ in 0..50 {
            st.record(0, &[0.1, 0.3, 0.5, 0.7], 0.7);
            st.record(3, &[0.5, 0.5, 0.5, 0.5], 0.5);
        }
        let (n_layers, total) = st.dump().expect("dump");
        assert_eq!(n_layers, 2);
        assert_eq!(total, 100);

        let output = dir.path().join("th.json");
        let args = Args {
            capture_dir: dir.path().to_owned(),
            target_active_frac: 0.5,
            model_id: "test".into(),
            output: output.clone(),
            n_intermediate: 4,
            notes: "smoke".into(),
        };
        run(args).expect("run");

        let loaded = PerChannelThresholds::load(&output).expect("load");
        assert_eq!(loaded.n_layers(), 2);
        assert_eq!(loaded.n_intermediate, 4);
        assert_eq!(loaded.target_active_frac, 0.5);
        // Layer 3 has every channel at ratio 1.0 (all silu values
        // equal max) → every channel's median quantile is the last
        // bin → τ ≈ (N_BINS-0.5)/N_BINS for every channel.
        let l3 = loaded.get(3).expect("layer 3 present");
        for &t in l3 {
            assert!(
                (t - (N_BINS as f32 - 0.5) / N_BINS as f32).abs() < 1e-6,
                "got τ={t}"
            );
        }
    }
}
