//! How much does lowering `top_k` change what K3 predicts?
//!
//! K3 streams `top_k * moe_layers * 17.5 MB` = 25.8 GB per token at the shipped
//! `top_k = 16`, and decode is bound by that, so routed bytes fall EXACTLY in
//! proportion to k. That half needs no measurement — it is arithmetic, and this
//! probe prints it rather than timing it.
//!
//! What does need measuring is quality, and the obvious way to measure it is
//! unaffordable. K2.6's sweep ran a 10-prompt x 64-token eval per k; at K3's
//! ~150 s/token that is ~26 hours PER K POINT. So this probe measures the one
//! thing a single forward pass can see: how far the next-token distribution
//! moves when k is lowered, against the full-k run on the same prompt.
//!
//! ```text
//!   cargo run --release --example k3_topk_probe -- <export-dir> [k-list] [prompt-file]
//!   cargo run --release --example k3_topk_probe -- /models/k3 16,12,10,8,6,4
//! ```
//!
//! ## What the numbers mean, and what they do not
//!
//! `argmax` is the decisive one. If the top token already changes at some k,
//! generation diverges from there and no amount of eval will call it equivalent
//! — that k is ruled out cheaply. If argmax holds and KL is small, that is
//! **weak evidence for**, not proof: this is one step, and divergence compounds
//! over a generation. K2.6 recorded exactly that trap — a substring pass "doesn't
//! imply semantic equivalence".
//!
//! So: use this to ELIMINATE k values quickly, then run a real generation eval
//! on the two or three survivors. It is a screen, not a verdict.
//!
//! Prefix caching is forced off. It keys on the prompt, not on k, so a cached
//! prefill from an earlier k would be reused under the next one and every point
//! after the first would silently measure k=16.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::k3::stage::K3Runner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

/// Next-token logits for `prompt`, from a fresh state.
///
/// Mirrors the batched-prefill branch of `StagedRunner::generate`: embed every
/// row, run the layers once, take the head at the final position only. `reset`
/// matters — the KDA recurrent state and the MLA cache carry across calls, and
/// this is invoked once per k against the same prompt.
fn prefill_logits(runner: &mut K3Runner, prompt: &[u32]) -> Vec<f32> {
    runner.reset();
    let hs = runner.hidden_size();
    let n = prompt.len();
    let mut batch = vec![0.0f32; n * hs];
    for (r, &t) in prompt.iter().enumerate() {
        batch[r * hs..(r + 1) * hs].copy_from_slice(&runner.embed_token(t));
    }
    let h = runner.forward_layers_batch(batch, 0, n);
    runner.head_logits(&h[(n - 1) * hs..n * hs])
}

/// Bytes one token streams at a given width, at K3's real shape.
fn routed_bytes(top_k: usize) -> u64 {
    const MOE_LAYERS: u64 = 92;
    const EXPERT_BYTES: u64 = 17_547_264;
    top_k as u64 * MOE_LAYERS * EXPERT_BYTES
}

/// Index of the largest logit.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            },
        )
        .0
}

/// Softmax in f64, max-shifted. The vocab is 163,840 wide and the tail matters
/// for KL, so this is not a place to accumulate in f32.
fn softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::MIN, f32::max) as f64;
    let mut e: Vec<f64> = v.iter().map(|&x| ((x as f64) - m).exp()).collect();
    let s: f64 = e.iter().sum();
    for x in e.iter_mut() {
        *x /= s;
    }
    e
}

/// `KL(reference || candidate)` in nats.
///
/// Reference-first on purpose: it weights by what the FULL-k model believes,
/// so mass the lowered-k model drops from a token the reference was confident
/// about is penalised, which is the failure mode that matters.
fn kl(reference: &[f64], candidate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(candidate)
        .filter(|(&p, _)| p > 0.0)
        .map(|(&p, &q)| p * (p / q.max(1e-30)).ln())
        .sum()
}

/// The `n` highest-scoring indices.
fn top_n(v: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(n);
    idx
}

fn overlap(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|i| b.contains(i)).count()
}

/// The full-`top_k` run every lowered k is scored against.
struct Reference {
    probs: Vec<f64>,
    argmax: usize,
    top5: Vec<usize>,
}

/// One prompt: a label for the report and its token ids.
struct Prompt {
    name: String,
    ids: Vec<u32>,
}

/// Read prompts from a file.
///
/// One prompt per line, `name: id id id …`. The name is free text up to the
/// first colon; a line with no colon is taken as ids with a positional name.
/// Blank lines and `#` comments are skipped.
///
/// Several prompts matter more than several k values. The first sweep used a
/// single factual-recall prompt ("the capital of Italy is") and found argmax
/// unchanged down to k=4 — but that is the easiest possible case, because the
/// top token is so dominant that dropping 12 of 16 experts cannot dislodge it.
/// Prompts where the next token is genuinely contested are what the result
/// needs, and they cannot be had from one line of input.
fn load_prompts(path: &str) -> Result<Vec<Prompt>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, rest) = match line.split_once(':') {
            Some((n, r)) => (n.trim().to_string(), r),
            None => (format!("prompt{i}"), line),
        };
        let ids: Vec<u32> = rest
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if ids.is_empty() {
            return Err(format!("{path}:{}: no token ids", i + 1).into());
        }
        out.push(Prompt { name, ids });
    }
    if out.is_empty() {
        return Err(format!("{path}: no prompts").into());
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        args.get(1)
            .ok_or("usage: k3_topk_probe <export-dir> [k-list] [prompt-file]")?,
    );
    let ks: Vec<usize> = args
        .get(2)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 12, 10, 8, 6, 4]);
    if ks.is_empty() {
        return Err("k-list parsed to nothing".into());
    }

    // A cached prefill is keyed on the prompt, not on k. Leaving it on would
    // make every point after the first re-use the first point's k.
    std::env::set_var("CASCADIA_K3_PREFIX_CACHE", "0");

    let prompts = match args.get(3) {
        Some(p) => load_prompts(p)?,
        // Deterministic filler when none is supplied: the probe compares k
        // against k on the SAME input, so the tokens need only be fixed, not
        // meaningful. Real prompts give a more trustworthy answer.
        None => vec![Prompt {
            name: "synthetic".into(),
            ids: (0..64u32).map(|i| 1000 + i * 7).collect(),
        }],
    };
    let widest = prompts.iter().map(|p| p.ids.len()).max().unwrap_or(0);
    println!(
        "[topk_probe] dir={} prompts={} (max {widest} tokens) ks={:?}",
        dir.display(),
        prompts.len(),
        ks
    );

    // Load ONCE for every (prompt, k) pair — the load is ~1000 s and the sweep
    // has a lot of pairs.
    let mut runner = K3Runner::load(&dir, 0, 1, widest + 8, None)?;
    let full = runner.top_k();
    println!("[topk_probe] manifest top_k = {full}");

    let mut changed_at: Vec<String> = Vec::new();

    for p in &prompts {
        println!("\n=== {} ({} tokens) ===", p.name, p.ids.len());
        println!(
            "{:>4} {:>10} {:>8} {:>9} {:>9} {:>10}",
            "k", "GB/token", "vs full", "argmax", "top5", "KL(nats)"
        );
        // Each prompt is scored against ITS OWN full-k run — comparing across
        // prompts would be meaningless.
        let mut reference: Option<Reference> = None;

        for &k in &ks {
            let eff = runner.set_top_k(k);
            if eff != k {
                println!("[topk_probe] k={k} clamped to {eff}");
            }
            let logits = prefill_logits(&mut runner, &p.ids);
            let probs = softmax(&logits);
            let am = argmax(&logits);
            let t5 = top_n(&logits, 5);
            let gb = routed_bytes(eff) as f64 / 1e9;

            match &reference {
                None => {
                    // Print the reference token id, not just "ref": without it
                    // there is no way to tell afterwards whether the full-k model
                    // predicted anything sensible, and the whole table is then
                    // only self-consistency.
                    println!(
                        "{eff:>4} {gb:>10.2} {:>8} {:>9} {:>9} {:>10}",
                        "1.00x",
                        format!("id {am}"),
                        "ref",
                        "ref"
                    );
                    reference = Some(Reference {
                        probs,
                        argmax: am,
                        top5: t5,
                    });
                }
                Some(r) => {
                    let ratio = routed_bytes(full) as f64 / routed_bytes(eff) as f64;
                    let verdict = if am == r.argmax {
                        "same".to_string()
                    } else {
                        changed_at.push(format!("{} at k={eff}", p.name));
                        format!("-> {am}")
                    };
                    println!(
                        "{eff:>4} {gb:>10.2} {:>7.2}x {:>9} {:>9} {:>10.5}",
                        ratio,
                        verdict,
                        format!("{}/5", overlap(&t5, &r.top5)),
                        kl(&r.probs, &probs)
                    );
                }
            }
        }
    }

    println!("\n===== verdict =====");
    if changed_at.is_empty() {
        println!("argmax held on every prompt at every k tested.");
        println!("That is weak evidence FOR, not proof: one step per prompt, and");
        println!("error compounds over a generation. Run a generation eval before");
        println!("lowering top_k in production.");
    } else {
        println!("argmax CHANGED — these are ruled out, no further eval needed:");
        for c in &changed_at {
            println!("  {c}");
        }
        println!("The lowest k that held on EVERY prompt is the only candidate.");
    }
    Ok(())
}
