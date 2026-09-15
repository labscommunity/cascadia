//! Complete autoregressive Inkling decode, separate from the synthetic layer probe.
//!
//! --export DIR --cases cases.json [--tokens 64] [--samples 3] [--out result.json] [--warm-ov]
//! [--tolerate-divergence] records the first token that parts from greedy_ids
//! instead of aborting (device numerics); such a run is not correctness-verified.
//! [--route-trace routes.json] captures routed expert IDs without changing logits.
//! [--layer-profile profile.json] records attention/MLP branch timings per layer.
//! cases.json: [{"name":"case", "prompt_ids":[...], "greedy_ids":[...]}].
//! Omit greedy_ids only when recording an initial baseline (not correctness-verified).
//! The large 975B architecture is required unless --allow-fixture is explicit.
//! Fixture runs emit fixture_decode_tokens_per_s and can never satisfy the target.
//! Generation stops at EOS. The first generated token belongs to prefill; decode
//! throughput counts only subsequent tokens, with complete layers/head/argmax.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::inkling::loader::{load_model_with, read_manifest};
use cascadia_engine_sparse_moe::inkling::model::{argmax, Model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    name: String,
    prompt_ids: Vec<u32>,
    #[serde(default)]
    greedy_ids: Option<Vec<u32>>,
}

#[derive(Serialize)]
struct Sample {
    case: String,
    repetition: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    prefill_started_unix: f64,
    prefill_ended_unix: f64,
    decode_started_unix: f64,
    decode_ended_unix: f64,
    decode_steps: usize,
    generated_ids: Vec<u32>,
    /// Index of the first token that differs from the case's `greedy_ids`
    /// (`--tolerate-divergence` only; `None` = exact match or no reference).
    #[serde(skip_serializing_if = "Option::is_none")]
    first_divergence: Option<usize>,
}

#[derive(Serialize)]
struct LayerRoutes {
    layer: usize,
    routed_experts_per_position: Vec<Vec<usize>>,
}

#[derive(Serialize)]
struct RoutingSample {
    case: String,
    repetition: usize,
    prefill_positions: usize,
    decode_positions: usize,
    layers: Vec<LayerRoutes>,
}

#[derive(Serialize)]
struct TimingEvent {
    rows: usize,
    prefill: bool,
    attention_seconds: f64,
    mlp_seconds: f64,
    total_seconds: f64,
}

#[derive(Serialize)]
struct TimedLayer {
    layer: usize,
    events: Vec<TimingEvent>,
}

#[derive(Serialize)]
struct TimingSample {
    case: String,
    repetition: usize,
    layers: Vec<TimedLayer>,
}

fn hash_logits(hash: &mut u64, logits: &[f32]) {
    for value in logits {
        assert!(value.is_finite(), "non-finite logits");
        for byte in value.to_bits().to_le_bytes() {
            *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
}

fn generate(
    model: &mut Model,
    case: &Case,
    tokens: usize,
    eos: &[u32],
    hash: &mut u64,
    tolerate_divergence: bool,
) -> Sample {
    model.reset();
    let prefill_started_unix = unix_seconds();
    let start = Instant::now();
    let logits = model.prefill(&case.prompt_ids);
    let mut next = argmax(&logits) as u32;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let prefill_ended_unix = unix_seconds();
    hash_logits(hash, &logits);
    let mut generated_ids = vec![next];
    let decode_started_unix = unix_seconds();
    let start = Instant::now();
    while generated_ids.len() < tokens && !eos.contains(&next) {
        let logits = model.forward_token(next);
        hash_logits(hash, &logits);
        next = argmax(&logits) as u32;
        generated_ids.push(next);
    }
    let decode_seconds = start.elapsed().as_secs_f64();
    let decode_ended_unix = unix_seconds();
    let mut first_divergence = None;
    if let Some(expected) = &case.greedy_ids {
        if tolerate_divergence {
            // Device numerics (int8/f16 projections) are not expected to hold
            // the bf16 reference's greedy path for 64 tokens; record where it
            // parts instead of aborting the campaign, and say so in the output.
            first_divergence = generated_ids
                .iter()
                .zip(expected.iter())
                .position(|(a, b)| a != b)
                .or_else(|| {
                    (generated_ids.len() != expected.len())
                        .then_some(generated_ids.len().min(expected.len()))
                });
            if let Some(i) = first_divergence {
                println!(
                    "greedy_divergence case={} first_divergent_token={i} matched_prefix={i}/{}",
                    case.name,
                    expected.len()
                );
            }
        } else {
            assert_eq!(&generated_ids, expected, "greedy mismatch: {}", case.name);
        }
    }
    Sample {
        first_divergence,
        case: case.name.clone(),
        repetition: 0,
        prefill_seconds,
        decode_seconds,
        prefill_started_unix,
        prefill_ended_unix,
        decode_started_unix,
        decode_ended_unix,
        decode_steps: generated_ids.len() - 1,
        generated_ids,
    }
}

// Wall timestamps align diagnostic resource samples with phases. Throughput
// still uses monotonic Instant durations, independently of wall-clock changes.
fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_secs_f64()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut export = None;
    let mut cases_path = None;
    let mut out = None;
    let mut route_trace = None;
    let mut layer_profile = None;
    let mut prediction_trace = None;
    let mut prediction_lead_layers = 0usize;
    let mut tokens = 64usize;
    let mut repetitions = 3usize;
    let mut allow_fixture = false;
    let mut warm_ov = false;
    let mut tolerate_divergence = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--warm-ov" {
            warm_ov = true;
            continue;
        }
        if flag == "--tolerate-divergence" {
            tolerate_divergence = true;
            continue;
        }
        if flag == "--allow-fixture" {
            allow_fixture = true;
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--export" => export = Some(PathBuf::from(value)),
            "--cases" => cases_path = Some(PathBuf::from(value)),
            "--tokens" => tokens = value.parse()?,
            "--samples" => repetitions = value.parse()?,
            "--out" => out = Some(PathBuf::from(value)),
            "--route-trace" => route_trace = Some(PathBuf::from(value)),
            "--layer-profile" => layer_profile = Some(PathBuf::from(value)),
            "--prediction-trace" => prediction_trace = Some(PathBuf::from(value)),
            "--prediction-lead-layers" => prediction_lead_layers = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    assert!(
        tokens >= 2 && repetitions >= 1,
        "need >=2 tokens and >=1 samples"
    );
    assert!(
        prediction_lead_layers <= 1,
        "prediction lead must be 0 or 1"
    );
    assert!(
        prediction_lead_layers == 0 || prediction_trace.is_some(),
        "one-layer-early prediction requires --prediction-trace"
    );
    if let Some(path) = &route_trace {
        assert!(!path.exists(), "refusing to overwrite a routing trace");
        assert!(
            out.as_ref() != Some(path),
            "trace and result need distinct paths"
        );
    }
    if let Some(path) = &layer_profile {
        assert!(!path.exists(), "refusing to overwrite a layer profile");
        assert!(
            out.as_ref() != Some(path) && route_trace.as_ref() != Some(path),
            "profile, trace and result need distinct paths"
        );
    }
    if let Some(path) = &prediction_trace {
        assert!(!path.exists(), "refusing to overwrite a prediction trace");
        assert!(
            route_trace.is_some(),
            "prediction diagnostics require --route-trace for actual selections"
        );
        assert!(
            out.as_ref() != Some(path)
                && route_trace.as_ref() != Some(path)
                && layer_profile.as_ref() != Some(path),
            "prediction, profile, trace and result need distinct paths"
        );
    }
    let export = export.ok_or("--export is required")?;
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(cases_path.ok_or("--cases is required")?)?)?;
    assert!(!cases.is_empty(), "need at least one prompt");
    let manifest = read_manifest(&export)?;
    let large = manifest.num_layers == 66
        && manifest.hidden_size == 6144
        && manifest.vocab_size == 201024
        && manifest.num_attention_heads == 64
        && manifest.num_kv_heads == 8
        && manifest.head_dim == 128
        && manifest.dense_layers == [0, 1]
        && manifest.dense_intermediate == 24576
        && manifest.moe_intermediate == 3072
        && manifest.num_experts == 256
        && manifest.top_k == 6
        && manifest.n_shared_experts == 2;
    assert!(
        large || allow_fixture,
        "expected the complete large Inkling architecture"
    );
    let full_model = large && !allow_fixture;
    // Exact greedy parity is asserted per sample unless `--tolerate-divergence`,
    // in which case the run is NOT correctness-verified (divergences are
    // recorded per sample instead).
    let correctness_verified = cases.iter().all(|c| c.greedy_ids.is_some()) && !tolerate_divergence;
    for c in &cases {
        assert!(!c.prompt_ids.is_empty(), "empty prompt: {}", c.name);
        assert!(c
            .prompt_ids
            .iter()
            .all(|&t| (t as usize) < manifest.vocab_size));
    }
    let max_seq = cases
        .iter()
        .map(|c| c.prompt_ids.len())
        .max()
        .unwrap()
        .checked_add(tokens)
        .ok_or("sequence length overflow")?;
    let load = Instant::now();
    let mut model = load_model_with(&export, max_seq, ExpertsMode::Mmap)?;
    if warm_ov {
        // Compile the attached OpenVINO backends before any timed region.
        let t0 = Instant::now();
        let (ok, bad) = model.warm_ov_backends();
        println!(
            "warm_ov_backends={ok} warm_ov_failed={bad} warm_ov_seconds={:.1}",
            t0.elapsed().as_secs_f64()
        );
    }
    assert_eq!(model.layers().len(), manifest.num_layers);
    println!("load_seconds={}", load.elapsed().as_secs_f64());
    let embedding_mapped = model.embedding_is_mapped();
    println!("embedding_mapped={}", u8::from(embedding_mapped));
    let owned_shared_bytes = model.owned_shared_bytes();
    println!("owned_shared_bytes={owned_shared_bytes}");
    println!("full_model={}", u8::from(full_model));
    let mut captures = Vec::new();
    if route_trace.is_some() {
        for (li, layer) in model.layers_mut().iter_mut().enumerate() {
            if let Some(moe) = layer.moe_mut() {
                let routes = Arc::new(Mutex::new(Vec::new()));
                let target = Arc::clone(&routes);
                moe.set_route_observer(Some(Arc::new(move |gate| {
                    target.lock().unwrap().push(gate.idx.clone());
                })));
                captures.push((li, routes));
            }
        }
    }
    let mut prediction_captures = Vec::new();
    if prediction_trace.is_some() {
        for (li, layer) in model.layers_mut().iter_mut().enumerate() {
            if layer.moe().is_some() && (prediction_lead_layers == 0 || li > 0) {
                let routes = Arc::new(Mutex::new(Vec::new()));
                let target = Arc::clone(&routes);
                if prediction_lead_layers == 0 {
                    layer.set_pre_attention_route_observer(Some(Arc::new(move |gate| {
                        target.lock().unwrap().push(gate.idx.clone());
                    })));
                }
                prediction_captures.push((li, routes));
            }
        }
        if prediction_lead_layers == 1 {
            let targets = prediction_captures.clone();
            model.set_previous_layer_route_observer(Some(Arc::new(move |li, gate| {
                let (_, target) = targets.iter().find(|(index, _)| *index == li).unwrap();
                target.lock().unwrap().push(gate.idx.clone());
            })));
        }
    }
    let mut prediction_samples = Vec::new();
    let mut routing_samples = Vec::new();
    let mut timers = Vec::new();
    if layer_profile.is_some() {
        for layer in model.layers_mut() {
            let events = Arc::new(Mutex::new(Vec::new()));
            let target = Arc::clone(&events);
            layer.set_timing_observer(Some(Arc::new(move |timing| {
                target.lock().unwrap().push(TimingEvent {
                    rows: timing.rows,
                    prefill: timing.prefill,
                    attention_seconds: timing.attention.as_secs_f64(),
                    mlp_seconds: timing.mlp.as_secs_f64(),
                    total_seconds: timing.total.as_secs_f64(),
                });
            })));
            timers.push(events);
        }
    }
    let mut timing_samples = Vec::new();
    let mut samples = Vec::new();
    let mut reference_hash = None;
    for rep in 0..repetitions {
        let mut hash = 0xcbf29ce484222325;
        for case in &cases {
            let mut sample = generate(
                &mut model,
                case,
                tokens,
                &manifest.eos_token_ids,
                &mut hash,
                tolerate_divergence,
            );
            sample.repetition = rep;
            assert!(
                sample.decode_steps > 0,
                "{} ended during prefill; choose a longer prompt",
                case.name
            );
            println!("sample_json={}", serde_json::to_string(&sample)?);
            if route_trace.is_some() {
                let layers = captures
                    .iter()
                    .map(|(li, routes)| {
                        let rows = std::mem::take(&mut *routes.lock().unwrap());
                        assert_eq!(rows.len(), case.prompt_ids.len() + sample.decode_steps);
                        LayerRoutes {
                            layer: *li,
                            routed_experts_per_position: rows,
                        }
                    })
                    .collect();
                routing_samples.push(RoutingSample {
                    case: case.name.clone(),
                    repetition: rep,
                    prefill_positions: case.prompt_ids.len(),
                    decode_positions: sample.decode_steps,
                    layers,
                });
            }
            if prediction_trace.is_some() {
                let layers = prediction_captures
                    .iter()
                    .map(|(li, routes)| {
                        let rows = std::mem::take(&mut *routes.lock().unwrap());
                        assert_eq!(rows.len(), sample.decode_steps);
                        LayerRoutes {
                            layer: *li,
                            routed_experts_per_position: rows,
                        }
                    })
                    .collect();
                prediction_samples.push(RoutingSample {
                    case: case.name.clone(),
                    repetition: rep,
                    prefill_positions: 0,
                    decode_positions: sample.decode_steps,
                    layers,
                });
            }
            if layer_profile.is_some() {
                let layers = timers
                    .iter()
                    .enumerate()
                    .map(|(li, events)| {
                        let events = std::mem::take(&mut *events.lock().unwrap());
                        assert_eq!(events.len(), 1 + sample.decode_steps);
                        assert!(events[0].prefill && events[0].rows == case.prompt_ids.len());
                        assert!(events[1..]
                            .iter()
                            .all(|event| !event.prefill && event.rows == 1));
                        TimedLayer { layer: li, events }
                    })
                    .collect();
                timing_samples.push(TimingSample {
                    case: case.name.clone(),
                    repetition: rep,
                    layers,
                });
            }
            samples.push(sample);
        }
        if let Some(expected) = reference_hash {
            assert_eq!(hash, expected, "non-repeatable logits");
        } else {
            reference_hash = Some(hash);
        }
    }
    // Conservative primary metric: slowest complete case/repetition, including
    // the first decode and all validation/hash overhead. No warm-cache exclusions.
    let rate = samples
        .iter()
        .map(|s| s.decode_steps as f64 / s.decode_seconds)
        .fold(f64::INFINITY, f64::min);
    let steps = samples.iter().map(|s| s.decode_steps).min().unwrap();
    let hash = format!("{:016x}", reference_hash.unwrap());
    let scope = if full_model {
        "full_large_model_decode"
    } else {
        "fixture_model_decode"
    };
    println!("scope={scope}");
    println!("output_hash={hash}");
    println!("correctness_verified={}", u8::from(correctness_verified));
    println!("decode_steps_min={steps}");
    println!("repetitions={repetitions}");
    let metric = if full_model {
        "decode_tokens_per_s"
    } else {
        "fixture_decode_tokens_per_s"
    };
    println!("{metric}={rate}");
    let (uncached_read_bytes, uncached_read_fallbacks) =
        cascadia_engine_sparse_moe::inkling::uncached_read_statistics();
    let uncached_read_effective = uncached_read_bytes > 0 && uncached_read_fallbacks == 0;
    println!("uncached_read_bytes={uncached_read_bytes}");
    println!("uncached_read_fallbacks={uncached_read_fallbacks}");
    println!(
        "uncached_read_effective={}",
        u8::from(uncached_read_effective)
    );
    let pipelined_read_layers =
        cascadia_engine_sparse_moe::inkling::moe::pipeline_read_layer_count();
    println!("pipelined_read_layers={pipelined_read_layers}");
    println!(
        "pipeline_reads_effective={}",
        u8::from(pipelined_read_layers > 0)
    );
    let expert_cache = model.expert_cache_stats();
    println!(
        "expert_cache_capacity_bytes={}",
        expert_cache.capacity_bytes
    );
    println!(
        "expert_cache_retained_bytes={}",
        expert_cache.retained_bytes
    );
    println!("expert_cache_hits={}", expert_cache.hits);
    println!("expert_cache_hit_bytes={}", expert_cache.hit_bytes);
    println!("expert_cache_misses={}", expert_cache.misses);
    println!("expert_cache_admissions={}", expert_cache.admissions);
    println!("expert_cache_evictions={}", expert_cache.evictions);
    println!(
        "expert_cache_history_resets={}",
        expert_cache.history_resets
    );
    println!(
        "expert_cache_frequency_decays={}",
        expert_cache.frequency_decays
    );
    println!("expert_cache_effective={}", u8::from(expert_cache.hits > 0));
    println!(
        "expert_cache_recent_tie_admissions={}",
        expert_cache.recent_tie_admissions
    );
    let (prefill_read_experts, prefill_uncached_read_bytes, prefill_uncached_read_fallbacks) =
        cascadia_engine_sparse_moe::inkling::prefill_read_statistics();
    println!("prefill_read_experts={prefill_read_experts}");
    println!("prefill_uncached_read_bytes={prefill_uncached_read_bytes}");
    println!("prefill_uncached_read_fallbacks={prefill_uncached_read_fallbacks}");
    println!(
        "prefill_reads_effective={}",
        u8::from(prefill_read_experts > 0)
    );
    let prediction_reads = cascadia_engine_sparse_moe::inkling::prediction_read_statistics();
    let second_prediction_reads =
        cascadia_engine_sparse_moe::inkling::second_prediction_read_statistics();
    let prediction_read_workers =
        cascadia_engine_sparse_moe::inkling::prediction_read_worker_count();
    let second_prediction_rank_ceiling =
        cascadia_engine_sparse_moe::inkling::second_prediction_rank_ceiling();
    println!("second_prediction_rank_ceiling={second_prediction_rank_ceiling}");
    let second_prediction_reads_effective = second_prediction_reads.scheduled > 0;
    println!("prediction_read_workers={prediction_read_workers}");
    println!(
        "second_prediction_reads_effective={}",
        u8::from(second_prediction_reads_effective)
    );
    for (name, value) in serde_json::to_value(&second_prediction_reads)?
        .as_object()
        .unwrap()
    {
        println!("second_prediction_read_{name}={value}");
    }
    let third_prediction_reads =
        cascadia_engine_sparse_moe::inkling::third_prediction_read_statistics();
    let third_prediction_reads_effective = third_prediction_reads.scheduled > 0;
    let read_buffer_idle_limit_bytes =
        cascadia_engine_sparse_moe::inkling::read_buffer_idle_limit_bytes();
    println!("read_buffer_idle_limit_bytes={read_buffer_idle_limit_bytes}");
    let bf16_gemv_min_rows_configured =
        cascadia_engine_sparse_moe::dsv4::math::bf16_gemv_min_rows();
    println!("bf16_gemv_min_rows_configured={bf16_gemv_min_rows_configured}");
    println!(
        "third_prediction_reads_effective={}",
        u8::from(third_prediction_reads_effective)
    );
    for (name, value) in serde_json::to_value(&third_prediction_reads)?
        .as_object()
        .unwrap()
    {
        println!("third_prediction_read_{name}={value}");
    }
    let early_prediction_reads_effective =
        model.early_prediction_reads_enabled() && prediction_reads.scheduled > 0;
    println!(
        "early_prediction_reads_effective={}",
        u8::from(early_prediction_reads_effective)
    );
    for (name, value) in serde_json::to_value(&prediction_reads)?
        .as_object()
        .unwrap()
    {
        println!("prediction_read_{name}={value}");
    }
    println!(
        "prediction_read_effective={}",
        u8::from(
            prediction_reads.scheduled > 0
                && prediction_reads.read_failures == 0
                && prediction_reads.worker_failures == 0
                && prediction_reads.dispatch_failures == 0
        )
    );
    if let Some(out) = out {
        std::fs::write(
            out,
            serde_json::to_vec_pretty(&serde_json::json!({
                "scope": scope, "export": export, "manifest": {"layers":manifest.num_layers,
                    "hidden":manifest.hidden_size, "experts":manifest.num_experts},
                "output_hash":hash, "correctness_verified":correctness_verified,
                "embedding_mapped":embedding_mapped,
                "owned_shared_bytes":owned_shared_bytes,
                "uncached_read_bytes":uncached_read_bytes,
                "uncached_read_fallbacks":uncached_read_fallbacks,
                "uncached_read_effective":uncached_read_effective,
                "pipelined_read_layers":pipelined_read_layers,
                "expert_cache":expert_cache,
                "prediction_reads":prediction_reads,
                "second_prediction_reads":second_prediction_reads,
                "second_prediction_rank_ceiling":second_prediction_rank_ceiling,
                "prediction_read_workers":prediction_read_workers,
                "third_prediction_reads":third_prediction_reads,
                "third_prediction_reads_effective":third_prediction_reads_effective,
                "read_buffer_idle_limit_bytes":read_buffer_idle_limit_bytes,
                "bf16_gemv_min_rows_configured":bf16_gemv_min_rows_configured,
                "second_prediction_reads_effective":second_prediction_reads_effective,
                "early_prediction_reads_effective":early_prediction_reads_effective,
                "prefill_read_experts":prefill_read_experts,
                "prefill_uncached_read_bytes":prefill_uncached_read_bytes,
                "prefill_uncached_read_fallbacks":prefill_uncached_read_fallbacks,
                "slowest_case_decode_tokens_per_s":rate, "samples":samples
            }))?,
        )?;
    }
    if let Some(path) = route_trace {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({
                "scope": "routing_diagnostics", "generation_scope": scope,
                "full_model": full_model, "correctness_verified": correctness_verified,
                "output_hash": hash, "export": export,
                "manifest": {"layers": manifest.num_layers, "routed_experts": manifest.num_experts,
                    "shared_experts": manifest.n_shared_experts, "top_k": manifest.top_k,
                    "hidden": manifest.hidden_size, "intermediate": manifest.moe_intermediate},
                "samples": routing_samples,
            }),
        )?;
    }
    if let Some(path) = prediction_trace {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({
                "scope": if prediction_lead_layers == 0 { "pre_attention_route_prediction_diagnostics" }
                    else { "previous_layer_route_prediction_diagnostics" }, "generation_scope": scope,
                "full_model": full_model, "correctness_verified": correctness_verified,
                "output_hash": hash, "export": export,
                "prediction_input": if prediction_lead_layers == 0 {
                    "current_layer_residual_before_attention_with_existing_mlp_norm_and_router"
                } else {
                    "previous_layer_residual_before_attention_with_target_layer_mlp_norm_and_router"
                },
                "prediction_lead_layers": prediction_lead_layers,
                "actual_routing_changed": false, "prefetch_performed": prediction_reads.scheduled > 0,
                "observer_overhead_included_in_benchmark_time": true,
                "samples": prediction_samples,
            }),
        )?;
    }
    if let Some(path) = layer_profile {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({
                "scope": "layer_timing_diagnostics", "generation_scope": scope,
                "full_model": full_model, "correctness_verified": correctness_verified,
                "output_hash": hash, "export": export, "layers": manifest.num_layers,
                "branch_times_include_norms_convs_residuals": true,
                "head_and_embedding_excluded_from_layer_times": true,
                "observer_overhead_included_in_benchmark_time": true,
                "samples": timing_samples,
            }),
        )?;
    }
    Ok(())
}
