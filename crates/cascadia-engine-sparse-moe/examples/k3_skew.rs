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
//! K3 streams 25.8 GB/token and an AI-PC node fits roughly 3% of a layer in RAM
//! at 4 nodes, so:
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
    // ~0.5 GB of expert cache per node at 4 nodes / 17.5 MB per expert.
    let real_budget: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(28);

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
    for &b in &[1usize, 8, real_budget, 45, 90, 224] {
        if b > n_experts {
            continue;
        }
        let cov = u.coverage(b);
        let uni = b as f64 / n_experts as f64;
        let mark = if b == real_budget {
            "  <- fits in RAM at 4 nodes"
        } else {
            ""
        };
        println!(
            "{b:>8} {:>9.1}% {:>9.1}% {:>7.1}x{mark}",
            cov * 100.0,
            uni * 100.0,
            if uni > 0.0 { cov / uni } else { 0.0 }
        );
    }
    println!();

    // Verdict, against the bar stated in this file's header.
    let cov = u.coverage(real_budget) * 100.0;
    if per_slot < 1.0 {
        println!("NOT ENOUGH DATA: {per_slot:.2} observations per slot.");
        println!("Most experts were seen 0 or 1 times, so 'hottest' is arbitrary and");
        println!("any coverage figure above is an artefact of sparsity, not skew.");
        println!("K3 records 92 x top_k selections per token — run longer, or across");
        println!("more runs (the histogram merges on load).");
    } else if cov > 20.0 {
        println!("SKEWED ({cov:.1}% at the real budget): pinning is worth shipping.");
    } else if cov > 10.0 {
        println!("MILD ({cov:.1}%): real but modest. Weigh against the page cache the");
        println!("pins displace — anonymous pinned memory that evicts cache has lost");
        println!("before on a sibling engine.");
    } else {
        println!("UNIFORM ({cov:.1}%): pinning buys about its budget fraction and no");
        println!("more. Residency has to come from somewhere else — more nodes, or");
        println!("prefetch that predicts the NEXT layer's routing.");
    }
    Ok(())
}
