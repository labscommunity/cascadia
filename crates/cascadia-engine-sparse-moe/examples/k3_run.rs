//! Single-process Kimi-K3 greedy run — the first-real-tokens harness.
//!
//! Loads the whole export on one rank (experts mmap and stream from disk),
//! encodes a prompt with the export's tokenizer (or takes raw ids), generates
//! greedily, and prints tokens + tok/s.
//!
//!   cargo run --release --example k3_run -- <model_dir> "<prompt>" [n_gen]
//!   cargo run --release --example k3_run -- <dir> --ids "1 2 3" 8
//!
//! Works the same whether the expert bins sit in `<dir>/experts/` or are
//! symlinked onto several filesystems by `--expert-roots` — the loader just
//! opens the path. The banner resolves each bin so you can see which devices are
//! actually backing the run. Note that spreading them buys CAPACITY, not read
//! parallelism: a layer's 16 expert reads all live in one bin on one device and
//! layers are visited serially, so the queue is one device deep however many
//! roots there are.
//!
//! Set `CASCADIA_K3_PROFILE=1` for the per-token section split and residency,
//! and `CASCADIA_K3_AUTOPIN=1` to mlock the hottest experts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cascadia_engine_sparse_moe::k3::stage::{K3Runner, K3_DEFAULT_MAX_SEQ};
use cascadia_engine_sparse_moe::staged::StagedRunner;
use tokenizers::Tokenizer;

/// Report where the expert bins physically live: `symlink target dir -> count`.
/// A single-source export shows one entry; a split one shows several.
fn expert_sources(dir: &Path) -> BTreeMap<PathBuf, (usize, u64)> {
    let mut by: BTreeMap<PathBuf, (usize, u64)> = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir.join("experts")) else {
        return by;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("bin") {
            continue;
        }
        // canonicalize follows the symlink to the real backing file
        let real = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
        let sz = std::fs::metadata(&real).map(|m| m.len()).unwrap_or(0);
        let root = real.parent().unwrap_or(Path::new("/")).to_path_buf();
        let ent = by.entry(root).or_insert((0, 0));
        ent.0 += 1;
        ent.1 += sz;
    }
    by
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: k3_run <model_dir> \"<prompt>\" [n_gen]");
        eprintln!("   or: k3_run <model_dir> --ids \"1 2 3\" [n_gen]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let raw_ids = args[2] == "--ids";
    if raw_ids && args.len() < 4 {
        eprintln!("usage: k3_run <model_dir> --ids \"1 2 3\" [n_gen]");
        std::process::exit(2);
    }
    let prompt_arg = if raw_ids { &args[3] } else { &args[2] };
    let n_gen: usize = args
        .get(if raw_ids { 4 } else { 3 })
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let max_seq: usize = std::env::var("CASCADIA_K3_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(K3_DEFAULT_MAX_SEQ);

    let srcs = expert_sources(&dir);
    let total: u64 = srcs.values().map(|(_, b)| b).sum();
    println!(
        "[k3_run] expert bins across {} source{} ({:.1} GB total):",
        srcs.len(),
        if srcs.len() == 1 { "" } else { "s" },
        total as f64 / 1e9
    );
    for (root, (n, bytes)) in &srcs {
        println!(
            "           {n:>3} bins  {:7.1} GB  {}",
            *bytes as f64 / 1e9,
            root.display()
        );
    }

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
        "[k3_run] dir={} max_seq={} prompt_ids={} n_gen={}",
        dir.display(),
        max_seq,
        prompt.len(),
        n_gen
    );

    // `CASCADIA_K3_TOP_K=<k>` lowers the routed experts per token. Routed bytes
    // scale with it exactly, so this is the knob the throughput/quality curve is
    // swept over — which is the only way to choose a value.
    let top_k_override = std::env::var("CASCADIA_K3_TOP_K")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&k| k > 0);
    if let Some(k) = top_k_override {
        println!("[k3_run] top_k override = {k}");
    }

    let t_load = Instant::now();
    let mut runner = K3Runner::load(&dir, 0, 1, max_seq, top_k_override)?;
    println!(
        "[k3_run] loaded in {:.1}s (arch={}, pinned={} experts)",
        t_load.elapsed().as_secs_f64(),
        runner.arch_name(),
        runner.pinned_experts()
    );

    let t_gen = Instant::now();
    let out = runner.generate_argmax(&prompt, n_gen);
    let dt = t_gen.elapsed().as_secs_f64();

    println!(
        "[k3_run] {} tokens in {:.1}s = {:.4} tok/s ({:.1} s/token)",
        out.len(),
        dt,
        out.len() as f64 / dt.max(1e-9),
        dt / (out.len().max(1) as f64)
    );
    println!("[k3_run] out ids: {out:?}");
    if let Some(t) = tok.as_ref() {
        match t.decode(&out, true) {
            Ok(text) => println!("[k3_run] out text: {text:?}"),
            Err(e) => eprintln!("[k3_run] decode failed: {e}"),
        }
    }
    Ok(())
}
