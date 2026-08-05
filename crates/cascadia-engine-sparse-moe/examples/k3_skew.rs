//! Is K3's expert routing skewed enough for pinning to be worth anything?
//!
//! `CASCADIA_K3_AUTOPIN=1` records a `(layer, expert)` histogram and pins the
//! hottest experts. Nothing read that histogram back, so an autopin run produced
//! data that answered no question. This reads it.
//!
//! ```text
//!   cargo run --release --example k3_skew -- <usage-file> [n_experts] [budget]
//! ```
//!
//! ## The number that decides it
//!
//! `coverage(b)` is the share of routed selections the hottest `b` experts of
//! each layer account for — i.e. exactly the hit rate pinning `b` would buy, and
//! therefore the fraction by which the I/O term falls.
//!
//! Compare it against `b / n_experts`, the uniform baseline. If routing is
//! uniform, pinning buys its budget fraction and nothing more.
//!
//! The budget is PER LAYER, and getting that wrong is easy: a 4-node AI-PC rank
//! has ~0.5 GB of expert cache, which is ~28 experts — but spread over the 23
//! layers that rank holds, so the per-layer budget is ~1, not 28. (This file
//! said "28 fits in RAM at 4 nodes" and it was the 6-8 node figure.)
//!
//!   4 nodes:  0.5 GB / 23 layers -> ~1 expert per layer
//!   6 nodes:  9.4 GB / 15 layers -> ~35
//!   8 nodes: 13.8 GB / 12 layers -> ~68
//!
//! So:
//!
//! ```text
//!   coverage ~= 3%    -> uniform. Pinning is noise; prefetch is the only move.
//!   coverage 10-20%   -> real but modest.
//!   coverage > 20%    -> worth shipping; ~7x the uniform baseline.
//! ```
//!
//! That bar is written down HERE, before the number is known, because two
//! earlier ideas on this model (cross-layer gate prediction, lane-lazy reads)
//! were killed by pre-registered thresholds and would have been easy to argue
//! for afterwards.
//!
//! ## What makes the answer meaningless
//!
//! A short run. K3 records `92 x top_k` selections per token, and the histogram
//! has 92 x 896 = 82,432 slots. A 3-token run fills ~4,400 of them, so almost
//! every expert has a count of 0 or 1 and "hottest" is arbitrary. The report
//! prints observations-per-slot and refuses to call it skew below 1.0.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::k3::residency::UsageStats;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(
        args.get(1)
            .ok_or("usage: k3_skew <usage-file> [n_experts] [budget]")?,
    );
    let n_experts: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(896);
    // PER-LAYER budget. Default is the 4-node AI-PC rank: ~0.5 GB of expert
    // cache over 23 layers is ~1 expert per layer. Pass the value for whatever
    // node count is being considered — this is the knob node count moves.
    let real_budget: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut u = UsageStats::new();
    u.load(&path)?;
    if u.is_empty() {
        return Err(format!("{}: histogram is empty", path.display()).into());
    }

    let total = u.total();
    let layers = u.layers();
    let widest = u.widest_layer();
    let slots = layers * n_experts;
    let per_slot = total as f64 / slots as f64;

    println!("file            {}", path.display());
    println!("layers          {layers}");
    println!("selections      {total}");
    println!("experts seen    {widest} of {n_experts} (widest layer)");
    println!("obs per slot    {per_slot:.2}");
    println!();

    println!(
        "{:>8} {:>10} {:>10} {:>8}",
        "budget", "coverage", "uniform", "ratio"
    );
    for &b in &[1usize, 8, real_budget, 35, 68, 90, 224] {
        if b > n_experts {
            continue;
        }
        let cov = u.coverage(b);
        let uni = b as f64 / n_experts as f64;
        let mark = match b {
            _ if b == real_budget => "  <- the budget asked for",
            1 => "  <- 4-node AI-PC rank",
            35 => "  <- 6-node",
            68 => "  <- 8-node",
            _ => "",
        };
        println!(
            "{b:>8} {:>9.1}% {:>9.1}% {:>7.1}x{mark}",
            cov * 100.0,
            uni * 100.0,
            if uni > 0.0 { cov / uni } else { 0.0 }
        );
    }
    println!();

    // Verdict. TWO questions, not one — the first version asked only whether
    // absolute coverage cleared a bar, and reported a 38.8x-skewed distribution
    // as "UNIFORM" because the budget it was given could only capture 4.3%.
    // Those call for opposite actions: a flat distribution means abandon
    // pinning, a small budget means buy more budget.
    let cov = u.coverage(real_budget) * 100.0;
    let uni = real_budget as f64 / n_experts as f64 * 100.0;
    let ratio = if uni > 0.0 { cov / uni } else { 0.0 };

    if per_slot < 1.0 {
        println!("NOT ENOUGH DATA: {per_slot:.2} observations per slot.");
        println!("Most experts were seen 0 or 1 times, so 'hottest' is arbitrary and");
        println!("any coverage figure above is an artefact of sparsity, not skew.");
        println!("K3 records 92 x top_k selections per token — run longer, or across");
        println!("more runs (the histogram merges on load).");
        return Ok(());
    }

    // Is the distribution skewed at all?
    if ratio < 2.0 {
        println!("ROUTING IS FLAT ({ratio:.1}x the uniform baseline).");
        println!("Pinning buys about its budget fraction whatever the budget, so more");
        println!("RAM does not help disproportionately. Residency has to come from");
        println!("prefetch that predicts routing, not from caching what was hot.");
        return Ok(());
    }
    println!("ROUTING IS SKEWED: {ratio:.1}x the uniform baseline.");

    // Given that it is skewed, does THIS budget capture enough of it?
    if cov > 20.0 {
        println!("At {real_budget} experts/layer that captures {cov:.1}% of routed reads —");
        println!("worth shipping. I/O falls by roughly that fraction.");
    } else {
        println!("But at {real_budget} experts/layer it captures only {cov:.1}%. The BUDGET is");
        println!("the limit here, not the distribution — the hot set exists, there is");
        println!("just nowhere to put it. Extra RAM converts at {ratio:.0}x, so node count");
        println!("is the lever: see the 6- and 8-node rows above.");
    }
    Ok(())
}
