//! End-to-end correctness gate for the MiniMax-M2 OV-IR backend.
//!
//! Gated on `M2_MODEL_DIR` pointing at an export produced by
//! `tools/export_minimax_m2.py --tiny --no-quant` (which also writes a
//! `reference.json` containing the canonical HF greedy output). The test
//! runs the exact OV graphs through [`OvMoeRunner`] and asserts the greedy
//! token stream matches the reference — i.e. the Rust runtime reproduces
//! the PyTorch model. Skips (passes) when the env var is unset, so the
//! suite stays green on machines without OpenVINO or the fixture.
//!
//! Run on a Xeon host:
//!   M2_MODEL_DIR=/path/to/m2/tiny_fp32 \
//!     cargo test -p cascadia-engine-sparse-moe --test minimax_m2_eval -- --nocapture

use std::path::PathBuf;

use cascadia_engine_sparse_moe::OvMoeRunner;
use cascadia_ov_genai_shim::PluginConfig;

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn minimax_m2_tiny_matches_hf_reference() {
    let Some(model_dir) = env_dir("M2_MODEL_DIR") else {
        eprintln!("M2_MODEL_DIR not set / missing; skipping MiniMax-M2 e2e test");
        return;
    };

    let ref_path = model_dir.join("reference.json");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).expect("read reference.json"))
            .expect("parse reference.json");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let greedy: Vec<u32> = reference["greedy_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let first_next = reference["first_next_token"].as_u64().unwrap() as u32;

    let device = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let mut runner =
        OvMoeRunner::load(model_dir, &device, PluginConfig::new(), None).expect("load OvMoeRunner");

    let n_new = greedy.len() - prompt.len();
    let generated = runner
        .generate_argmax(&prompt, n_new)
        .expect("generate_argmax");

    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    eprintln!("reference greedy: {greedy:?}");
    eprintln!("ours     greedy : {full:?}");

    assert_eq!(
        generated.first().copied(),
        Some(first_next),
        "first generated token must match the HF reference"
    );
    assert_eq!(
        full, greedy,
        "full greedy token stream must match HF reference"
    );
}

/// iGPU router-split correctness gate (runs on CPU). When the shells have
/// been split by `tools/split_m2_shells.py`, running shell_core + the carved
/// CPU router must be byte-identical to the monolithic shell — the property
/// the iGPU path relies on (only the device differs in production). Forces
/// the split on CPU (`force_split = true`) so it needs no GPU and runs in CI.
/// Gated on `M2_MODEL_DIR`; skips if the split files aren't present.
#[test]
fn minimax_m2_split_path_matches_monolithic() {
    let Some(model_dir) = env_dir("M2_MODEL_DIR") else {
        eprintln!("M2_MODEL_DIR not set / missing; skipping split-path test");
        return;
    };
    let core0 = model_dir
        .join("shells")
        .join("layer_00")
        .join("shell_core.xml");
    if !core0.exists() {
        eprintln!("shell_core.xml absent (run tools/split_m2_shells.py); skipping split-path test");
        return;
    }
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_dir.join("reference.json")).expect("read reference.json"),
    )
    .expect("parse reference.json");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let greedy: Vec<u32> = reference["greedy_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let n_new = greedy.len() - prompt.len();

    // Monolithic shell (CPU).
    let mut mono = OvMoeRunner::load(model_dir.clone(), "CPU", PluginConfig::new(), None)
        .expect("load monolithic");
    let mono_gen = mono
        .generate_argmax(&prompt, n_new)
        .expect("monolithic generate");

    // shell_core + carved CPU router, forced on CPU.
    let mut split = OvMoeRunner::load_staged(
        model_dir.clone(),
        "CPU",
        PluginConfig::new(),
        None,
        0,
        1,
        0,
        0,
        true,
    )
    .expect("load split");
    let split_gen = split
        .generate_argmax(&prompt, n_new)
        .expect("split generate");

    eprintln!("monolithic: {mono_gen:?}");
    eprintln!("split     : {split_gen:?}");
    assert_eq!(
        split_gen, mono_gen,
        "shell_core + CPU router must match the monolithic shell byte-for-byte"
    );
    let mut full = prompt.clone();
    full.extend_from_slice(&split_gen);
    assert_eq!(full, greedy, "split path must match the HF reference");
}

/// Pipeline-parallel correctness gate: the SAME tiny model split across two
/// ranks (rank 0 = embed + first half of the layers, rank 1 = second half +
/// head) must reproduce the canonical HF greedy stream — i.e. slicing the
/// model and threading the hidden state rank→rank is numerically identical
/// to running it whole. Drives the runner slices directly (no sockets) so it
/// isolates the layer-slice math; the wire format is covered by
/// `dist_wire.rs` and the CLI loopback run. Gated on `M2_MODEL_DIR`.
#[test]
fn minimax_m2_two_rank_pipeline_matches_hf_reference() {
    let Some(model_dir) = env_dir("M2_MODEL_DIR") else {
        eprintln!("M2_MODEL_DIR not set / missing; skipping MiniMax-M2 pipeline test");
        return;
    };

    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_dir.join("reference.json")).expect("read reference.json"),
    )
    .expect("parse reference.json");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let greedy: Vec<u32> = reference["greedy_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let device = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let mut r0 = OvMoeRunner::load_staged(
        model_dir.clone(),
        &device,
        PluginConfig::new(),
        None,
        0,
        2,
        0,
        0,
        false,
    )
    .expect("load rank 0 slice");
    let mut r1 = OvMoeRunner::load_staged(
        model_dir.clone(),
        &device,
        PluginConfig::new(),
        None,
        1,
        2,
        0,
        0,
        false,
    )
    .expect("load rank 1 slice");
    assert!(
        r0.is_first() && !r0.is_last(),
        "rank 0 should own the embedding but not the head"
    );
    assert!(
        r1.is_last() && !r1.is_first(),
        "rank 1 should own the head but not the embedding"
    );

    let n_new = greedy.len() - prompt.len();
    let generated = drive_two_rank_greedy(&mut r0, &mut r1, &prompt, n_new);

    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    eprintln!("reference greedy : {greedy:?}");
    eprintln!("2-rank   greedy  : {full:?}");
    assert_eq!(
        full, greedy,
        "2-rank pipeline greedy stream must match the HF reference"
    );
}

/// Greedy-drive two pipeline slices in-process: rank 0 embeds + runs its
/// layers, hands the hidden state to rank 1, which runs its layers + head;
/// the argmax of rank 1's logits is the next token. Mirrors
/// [`OvMoeRunner::generate_argmax`] (default/greedy sampling), just with the
/// layers split across two runners and the hidden state passed by value the
/// way `cascadia-transport` carries it on the wire.
fn drive_two_rank_greedy(
    r0: &mut OvMoeRunner,
    r1: &mut OvMoeRunner,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    // First-max-wins, matching crate::sampling::argmax used by the runner.
    fn argmax(xs: &[f32]) -> u32 {
        let mut best = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in xs.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i as u32;
            }
        }
        best
    }
    fn fwd(r0: &mut OvMoeRunner, r1: &mut OvMoeRunner, tok: u32, pos: usize) -> Vec<f32> {
        let h = r0.embed_token(tok).expect("embed");
        let h = r0.forward_layers(h, pos).expect("rank 0 layers");
        let h = r1.forward_layers(h, pos).expect("rank 1 layers");
        r1.head_logits(&h).expect("head")
    }

    r0.reset();
    r1.reset();
    let eos = r0.eos_token_ids().to_vec();
    let mut pos = 0usize;
    let mut logits = Vec::new();
    for &t in prompt {
        logits = fwd(r0, r1, t, pos);
        pos += 1;
    }
    let mut out = Vec::with_capacity(max_new);
    loop {
        let next = argmax(&logits);
        out.push(next);
        if out.len() >= max_new || eos.contains(&next) {
            break;
        }
        logits = fwd(r0, r1, next, pos);
        pos += 1;
    }
    out
}

/// Real-model generation smoke test. Gated on `M2_GEN_DIR` pointing at a
/// full MiniMax-M2 export (with tokenizer.json). There's no exact HF
/// reference at 230B (won't fit in RAM), so this just confirms the Rust
/// runtime loads the real INT4 graphs, generates tokens through the
/// single-stage OV-IR pipeline, and decodes to non-empty text — the
/// single-stage-run-on-one-host deliverable. Prompt overridable via
/// `M2_PROMPT`; token budget via `M2_MAX_NEW` (default 30).
#[test]
fn minimax_m2_full_generate_smoke() {
    use tokenizers::Tokenizer;

    let Some(model_dir) = env_dir("M2_GEN_DIR") else {
        eprintln!("M2_GEN_DIR not set / missing; skipping MiniMax-M2 full-model smoke test");
        return;
    };
    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    let prompt = std::env::var("M2_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let max_new: usize = std::env::var("M2_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    eprintln!("prompt={prompt:?} prompt_ids={ids:?}");

    let device = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let cap = std::env::var("CASCADIA_MAX_EXPERTS_CACHED")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .and_then(std::num::NonZeroUsize::new);
    let mut plugin = PluginConfig::new();
    if let Ok(dir) = std::env::var("CASCADIA_OV_CACHE") {
        plugin = plugin.with("CACHE_DIR", dir);
    }
    let mut runner = OvMoeRunner::load(model_dir, &device, plugin, cap).expect("load OvMoeRunner");

    // Sampling knobs for the greedy-vs-repetition-penalty comparison.
    let env_f32 = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let cfg = cascadia_engine_sparse_moe::SamplingConfig {
        temperature: env_f32("M2_TEMP", 0.0),
        top_p: env_f32("M2_TOP_P", 1.0),
        repetition_penalty: env_f32("M2_REP_PENALTY", 1.0),
        repetition_window: std::env::var("M2_REP_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        seed: Some(42),
        ..cascadia_engine_sparse_moe::SamplingConfig::default()
    };
    eprintln!(
        "sampling: temp={} top_p={} rep_penalty={} rep_window={}",
        cfg.temperature, cfg.top_p, cfg.repetition_penalty, cfg.repetition_window
    );

    let started = std::time::Instant::now();
    let (generated, stats) = runner
        .generate_timed(&ids, max_new, &cfg)
        .expect("generate");
    let secs = started.elapsed().as_secs_f64();
    let text = tokenizer.decode(&generated, true).unwrap_or_default();
    let first_decode = stats.decode_secs.first().copied().unwrap_or(0.0);
    eprintln!(
        "generated {} tokens in {:.1}s | prefill {:.1}s | first decode-step {:.1}s | warm {:.3} tok/s | overall {:.3} tok/s",
        generated.len(),
        secs,
        stats.prefill_secs,
        first_decode,
        stats.warm_tok_s(),
        generated.len() as f64 / secs.max(1e-9)
    );
    eprintln!("completion: {text:?}");

    assert!(!generated.is_empty(), "model produced no tokens");
    assert!(!text.trim().is_empty(), "decoded completion is empty");
}

/// Snapshot/restore round-trip must be byte-identical to a continuous run.
/// Isolates the OvMoeEngine warm-resume: a warm resume (prefill a prefix,
/// snapshot, reset, restore, continue) must produce the SAME greedy tokens
/// as a single continuous generation. Runs on ONE runner (all layers) so it
/// bypasses the multi-stage transport — if this fails, the bug is in
/// `snapshot_kv`/`restore_kv`/`forward_layers` (e.g. hidden OV graph state
/// beyond the host-held KV); if it passes, the bug is multi-stage-only.
/// Gated on `M2_MODEL_DIR`.
#[test]
fn minimax_m2_snapshot_restore_matches_continuous() {
    let Some(model_dir) = env_dir("M2_MODEL_DIR") else {
        eprintln!("M2_MODEL_DIR not set / missing; skipping snapshot/restore test");
        return;
    };
    let device = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let mut r = OvMoeRunner::load(model_dir, &device, PluginConfig::new(), None).expect("load");

    // Pure argmax step (no rep-penalty), mirroring the engine's temp=0 path.
    fn step(r: &mut OvMoeRunner, tok: u32, pos: usize) -> u32 {
        let h = r.embed_token(tok).expect("embed");
        let h = r.forward_layers(h, pos).expect("forward");
        let logits = r.head_logits(&h).expect("head");
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .expect("argmax")
    }

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_gen = 8usize;

    // COLD: prefill prompt, decode n_gen.
    r.reset();
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &prompt {
        next = step(&mut r, t, pos);
        pos += 1;
    }
    let mut cold = Vec::new();
    for _ in 0..n_gen {
        cold.push(next);
        next = step(&mut r, next, pos);
        pos += 1;
    }

    // WARM: prefill all but the last prompt token, snapshot, reset, restore,
    // feed the last prompt token from the restored depth, decode n_gen.
    let split = prompt.len() - 1;
    r.reset();
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &prompt[..split] {
        next = step(&mut r, t, pos);
        pos += 1;
    }
    let snap = r.snapshot_kv().expect("snapshot");
    assert_eq!(snap.past_seq_len, split, "snapshot depth == fed tokens");
    r.reset();
    r.restore_kv(&snap).expect("restore");
    let mut pos = split;
    let mut next = 0u32;
    for &t in &prompt[split..] {
        next = step(&mut r, t, pos);
        pos += 1;
    }
    let mut warm = Vec::new();
    for _ in 0..n_gen {
        warm.push(next);
        next = step(&mut r, next, pos);
        pos += 1;
    }

    eprintln!("cold: {cold:?}");
    eprintln!("warm: {warm:?}");
    assert_eq!(warm, cold, "snapshot/restore warm run must match continuous cold run");
}

/// Two-stage (rank0 + rank1, in-process, no wire) snapshot/restore round-trip
/// must match a continuous run. Isolates the per-rank capture/restore under a
/// pipeline split from the frame/epoch transport layer: if this fails the bug
/// is in per-rank restore; if it passes the bug is in the CAPTURE/RESTORE
/// frame plumbing. Gated on `M2_MODEL_DIR`.
#[test]
fn minimax_m2_two_stage_snapshot_restore_matches_continuous() {
    let Some(model_dir) = env_dir("M2_MODEL_DIR") else {
        eprintln!("M2_MODEL_DIR not set / missing; skipping 2-stage snapshot/restore test");
        return;
    };
    let dev = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let mut r0 =
        OvMoeRunner::load_staged(model_dir.clone(), &dev, PluginConfig::new(), None, 0, 2, 0, 0, false)
            .expect("load rank0");
    let mut r1 =
        OvMoeRunner::load_staged(model_dir, &dev, PluginConfig::new(), None, 1, 2, 0, 0, false)
            .expect("load rank1");

    // One pipeline step: rank0 (embed+its layers) -> rank1 (its layers+head) -> argmax.
    fn pstep(r0: &mut OvMoeRunner, r1: &mut OvMoeRunner, tok: u32, pos: usize) -> u32 {
        let h = r0.embed_token(tok).expect("embed");
        let h = r0.forward_layers(h, pos).expect("r0 forward");
        let h = r1.forward_layers(h, pos).expect("r1 forward");
        let logits = r1.head_logits(&h).expect("head");
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .expect("argmax")
    }

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_gen = 8usize;

    // COLD
    r0.reset();
    r1.reset();
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &prompt {
        next = pstep(&mut r0, &mut r1, t, pos);
        pos += 1;
    }
    let mut cold = Vec::new();
    for _ in 0..n_gen {
        cold.push(next);
        next = pstep(&mut r0, &mut r1, next, pos);
        pos += 1;
    }

    // WARM: prefill all but last prompt token, snapshot BOTH ranks, reset BOTH,
    // restore BOTH, feed the last prompt token, decode n_gen.
    let split = prompt.len() - 1;
    r0.reset();
    r1.reset();
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &prompt[..split] {
        next = pstep(&mut r0, &mut r1, t, pos);
        pos += 1;
    }
    let s0 = r0.snapshot_kv().expect("snap r0");
    let s1 = r1.snapshot_kv().expect("snap r1");
    assert_eq!(s0.past_seq_len, split);
    assert_eq!(s1.past_seq_len, split);
    r0.reset();
    r1.reset();
    r0.restore_kv(&s0).expect("restore r0");
    r1.restore_kv(&s1).expect("restore r1");
    let mut pos = split;
    let mut next = 0u32;
    for &t in &prompt[split..] {
        next = pstep(&mut r0, &mut r1, t, pos);
        pos += 1;
    }
    let mut warm = Vec::new();
    for _ in 0..n_gen {
        warm.push(next);
        next = pstep(&mut r0, &mut r1, next, pos);
        pos += 1;
    }

    eprintln!("2stage cold: {cold:?}");
    eprintln!("2stage warm: {warm:?}");
    assert_eq!(warm, cold, "2-stage snapshot/restore must match continuous");
}
