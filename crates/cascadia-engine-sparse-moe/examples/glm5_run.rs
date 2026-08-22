//! Single-process GLM-5.2 greedy run — the first-real-tokens harness.
//!
//! Loads the whole int4 export on one rank (mmap experts stream from disk),
//! encodes a prompt with the export's tokenizer (or takes raw token ids), runs
//! greedy generation, and prints the tokens + tok/s.
//!
//!   cargo run --release --example glm5_run -- <model_dir> "<prompt>" [n_gen]
//!   CASCADIA_GLM5_MAX_SEQ=2048 cargo run --release --example glm5_run -- <dir> --ids "1 2 3 4" 8
//!
//! Needs tokenizer.json in <model_dir> for text prompts (stage it from the FP8
//! checkpoint). With `--ids` no tokenizer is required.
//!
//! `--force "<ids>"` is the teacher-forced DECODE bench: every id is driven
//! through the per-token decode path (`forward_layers`, not batch-union
//! prefill), so each step routes from a genuinely different hidden state —
//! the expert-streaming regime greedy generation can't reach on a synthetic
//! model whose argmax collapses to a fixed point. No sampling; prints tok/s
//! plus an FNV-1a hash of the final head logits as a cross-config parity
//! anchor (read-path changes must reproduce it bit-for-bit).

use std::path::PathBuf;
use std::time::Instant;

use cascadia_engine_sparse_moe::glm::stage::{GlmRunner, GLM5_DEFAULT_MAX_SEQ};
use cascadia_engine_sparse_moe::staged::StagedRunner;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: glm5_run <model_dir> \"<prompt>\" [n_gen]");
        eprintln!("   or: glm5_run <model_dir> --ids \"1 2 3 4\" [n_gen]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let forced = args[2] == "--force";
    let raw_ids = args[2] == "--ids" || forced;
    let prompt_arg = if raw_ids { &args[3] } else { &args[2] };
    let n_gen: usize = args
        .get(if raw_ids { 4 } else { 3 })
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let max_seq: usize = std::env::var("CASCADIA_GLM5_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GLM5_DEFAULT_MAX_SEQ);

    // tokenizer (needed only for text prompts / decoding).
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
        "[glm5_run] dir={} max_seq={} prompt_ids={} n_gen={}",
        dir.display(),
        max_seq,
        prompt.len(),
        n_gen
    );

    let t_load = Instant::now();
    let mut runner = GlmRunner::load_staged(&dir, max_seq, 0, 1, 0, 0, Default::default())?;
    println!(
        "[glm5_run] loaded in {:.1}s (arch={})",
        t_load.elapsed().as_secs_f64(),
        runner.arch_name()
    );

    if forced {
        assert!(
            prompt.len() <= runner.max_seq(),
            "forced ids ({}) exceed max_seq ({})",
            prompt.len(),
            runner.max_seq()
        );
        runner.reset();
        let t0 = Instant::now();
        let mut h = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            let e = runner.embed_token(t);
            h = runner.forward_layers(e, pos, Some(t));
        }
        let dt = t0.elapsed().as_secs_f64();
        let logits = runner.head_logits(&h);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
        for b in logits.iter().flat_map(|v| v.to_le_bytes()) {
            fnv ^= b as u64;
            fnv = fnv.wrapping_mul(0x0100_0000_01b3);
        }
        println!(
            "[glm5_run] forced {} tokens in {:.1}s = {:.3} tok/s",
            prompt.len(),
            dt,
            prompt.len() as f64 / dt.max(1e-9)
        );
        println!("[glm5_run] forced argmax={argmax} logits_fnv={fnv:#018x}");
        return Ok(());
    }

    let t_gen = Instant::now();
    let out = runner.generate_argmax(&prompt, n_gen);
    let dt = t_gen.elapsed().as_secs_f64();

    println!(
        "[glm5_run] {} tokens in {:.1}s = {:.3} tok/s",
        out.len(),
        dt,
        out.len() as f64 / dt.max(1e-9)
    );
    println!("[glm5_run] out ids: {out:?}");
    if let Some(t) = tok.as_ref() {
        match t.decode(&out, true) {
            Ok(text) => println!("[glm5_run] out text: {text:?}"),
            Err(e) => eprintln!("[glm5_run] decode failed: {e}"),
        }
    }
    Ok(())
}
