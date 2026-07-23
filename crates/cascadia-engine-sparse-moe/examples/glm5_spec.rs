//! GLM-5.2 MTP speculative-decode bench — single process, mmap experts.
//!
//! Loads the whole int4 export as one `GlmModel` (main-layer experts mmap'd from
//! disk, MTP draft head bf16 in RAM), then runs plain greedy AND MTP speculative
//! decode on the SAME warm process, reporting tok/s for each plus the MTP
//! acceptance rate. Speculative output is asserted token-identical to greedy.
//!
//!   cargo run --release --example glm5_spec -- <model_dir> "<prompt>" [n_gen]
//!   CASCADIA_GLM5_SPEC_G=3 CASCADIA_GLM5_MAX_SEQ=2048 \
//!     cargo run --release --example glm5_spec -- <dir> --ids "1 2 3 4" 32
//!
//! Needs the export to carry an MTP head (manifest has_mtp; export_glm5.py emits
//! it). tokenizer.json in <model_dir> for text prompts / decoding.

use std::path::PathBuf;
use std::time::Instant;

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::glm::loader::load_model_with;
use cascadia_engine_sparse_moe::glm::stage::GLM5_DEFAULT_MAX_SEQ;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: glm5_spec <model_dir> \"<prompt>\" [n_gen]");
        eprintln!("   or: glm5_spec <model_dir> --ids \"1 2 3 4\" [n_gen]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let raw_ids = args[2] == "--ids";
    let prompt_arg = if raw_ids { &args[3] } else { &args[2] };
    let n_gen: usize = args
        .get(if raw_ids { 4 } else { 3 })
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let max_seq: usize = std::env::var("CASCADIA_GLM5_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GLM5_DEFAULT_MAX_SEQ);
    let g: usize = std::env::var("CASCADIA_GLM5_SPEC_G")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).ok();
    let prompt: Vec<u32> = if raw_ids {
        prompt_arg
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    } else {
        let t = tok
            .as_ref()
            .ok_or("text prompt needs tokenizer.json in <model_dir>")?;
        t.encode(prompt_arg.as_str(), true)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec()
    };
    println!(
        "[glm5_spec] dir={} max_seq={} prompt_ids={} n_gen={} g={g}",
        dir.display(),
        max_seq,
        prompt.len(),
        n_gen
    );

    let t_load = Instant::now();
    let mut model = load_model_with(&dir, max_seq, ExpertsMode::Mmap)?;
    println!(
        "[glm5_spec] loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );
    if !model.has_mtp() {
        return Err(
            "export has no MTP head (re-export with export_glm5.py; manifest has_mtp=false)".into(),
        );
    }

    // Baseline greedy (also warms the mmap page cache for the spec run).
    let t0 = Instant::now();
    let base = model.generate(&prompt, n_gen);
    let dt_base = t0.elapsed().as_secs_f64();
    println!(
        "[glm5_spec] greedy : {} tok in {:.1}s = {:.3} tok/s",
        base.len(),
        dt_base,
        base.len() as f64 / dt_base.max(1e-9)
    );

    // MTP speculative decode.
    let t1 = Instant::now();
    let spec = model.generate_spec(&prompt, n_gen, g);
    let dt_spec = t1.elapsed().as_secs_f64();
    let acc = if spec.drafted > 0 {
        100.0 * spec.accepted as f64 / spec.drafted as f64
    } else {
        0.0
    };
    println!(
        "[glm5_spec] spec(g={g}): {} tok in {:.1}s = {:.3} tok/s | forwards={} accepted={}/{} ({acc:.0}%)",
        spec.tokens.len(), dt_spec, spec.tokens.len() as f64 / dt_spec.max(1e-9), spec.forwards, spec.accepted, spec.drafted,
    );
    println!(
        "[glm5_spec] speedup: {:.2}x  ({:.3} -> {:.3} tok/s)",
        dt_base / dt_spec.max(1e-9),
        base.len() as f64 / dt_base.max(1e-9),
        spec.tokens.len() as f64 / dt_spec.max(1e-9)
    );

    assert_eq!(
        spec.tokens, base,
        "spec-decode output diverged from greedy!"
    );
    println!("[glm5_spec] ✓ spec output == greedy (exact)");
    if let Some(t) = tok.as_ref() {
        if let Ok(text) = t.decode(&base, true) {
            println!("[glm5_spec] out text: {text:?}");
        }
    }
    Ok(())
}
