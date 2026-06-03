//! Three-tier {iGPU, NPU, CPU} placement solver — step 2 of issue #41.
//!
//! Takes a per-(stage, device) cost profile (latency + memory + op-support,
//! produced by `profile-devices --per-stage`) and assigns each pipeline stage
//! to exactly one device so as to **minimise total forward latency subject to
//! each device's memory budget**. This is the offline ILP of PowerInfer §6.3
//! adapted to Intel UMA — see `docs/perf/THREE_TIER_PLACEMENT.md`.
//!
//! The problem is tiny (stages ≤ ~16, devices = 3), so we solve it *exactly*
//! with branch-and-bound in pure Rust — no `good_lp`/CBC system dependency,
//! keeping the single-static-binary, Rust-only invariant.
//!
//! For a model that fits the iGPU the optimum is trivially "all stages on the
//! fastest device" (and the placed run equals `--device GPU`); the solver
//! earns its keep when a device's memory cap forces overflow onto the next-
//! cheapest tier, or when an op-support gap excludes a (stage, device) pair.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};

/// One device's placement budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCap {
    /// OV device string: `"GPU"` / `"NPU"` / `"CPU"`. Order in
    /// [`PlacementProfile::devices`] encodes the tie-break preference
    /// (list the device you'd rather use first).
    pub device: String,
    /// Usable memory budget in bytes (e.g. the iGPU's
    /// `GPU_DEVICE_TOTAL_MEM_SIZE`, the NPU's `NPU_DEVICE_TOTAL_MEM_SIZE`,
    /// or free system RAM for CPU).
    pub mem_bytes: u64,
}

/// Per-stage cost: resident memory and per-device forward latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCost {
    /// Stage index (0-based, matches the multi-stage shard layout).
    pub stage: u32,
    /// Resident memory of this stage in bytes (≈ IR weight bytes + KV).
    pub mem_bytes: u64,
    /// device → single-forward latency (ms). A device **absent** from this
    /// map can't run the stage (op-support gate) and is never assigned.
    pub lat_ms: BTreeMap<String, f64>,
}

/// The solver input — written by `profile-devices --per-stage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementProfile {
    pub model: String,
    pub devices: Vec<DeviceCap>,
    pub stages: Vec<StageCost>,
    /// Usable shared-memory pool in bytes — the **UMA** total (system RAM
    /// minus headroom). On Intel AI PCs the iGPU/NPU/CPU "device budgets"
    /// above are addressing limits over ONE physical pool, so a placement
    /// that satisfies every per-device cap can still exceed physical memory.
    /// When set, the total resident memory of all stages must fit here.
    /// `None` (the default, and for non-UMA topologies) skips the global gate.
    #[serde(default)]
    pub pool_bytes: Option<u64>,
    /// Fingerprint of the (shard, device-set, pool) this profile was measured
    /// for — `profile-stages` reuses an existing profile whose fingerprint
    /// matches rather than re-running the (expensive) measurement. `None` on
    /// hand-written profiles; the cache is simply skipped then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
}

/// The solver output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Placement {
    /// `assignment[i]` = the device string for stage `i`.
    pub assignment: Vec<String>,
    /// Sum of the assigned per-stage latencies (ms). The single-stream
    /// pipeline objective (transport between tiers is added by the caller
    /// when stages span hosts; intra-host UMA transfer is negligible).
    pub total_lat_ms: f64,
    /// device → bytes of stage memory placed on it.
    pub per_device_mem: BTreeMap<String, u64>,
}

/// Why a profile has no valid placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    /// No stages to place.
    Empty,
    /// A stage that no device can run (op-support) or that exceeds every
    /// device's total budget on its own.
    UnplaceableStage(u32),
    /// Every stage is individually placeable, but no assignment fits all of
    /// them within the per-device memory budgets simultaneously.
    Infeasible,
    /// The model's total resident memory exceeds the shared UMA pool — no
    /// placement across any mix of tiers can hold it on this host.
    ExceedsPool { needed_bytes: u64, pool_bytes: u64 },
}

impl std::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementError::Empty => write!(f, "placement profile has no stages"),
            PlacementError::UnplaceableStage(s) => write!(
                f,
                "stage {s} cannot be placed on any device (no supported device, \
                 or it exceeds every device's memory budget)"
            ),
            PlacementError::Infeasible => write!(
                f,
                "no assignment fits all stages within the device memory budgets \
                 (total model memory exceeds total device capacity)"
            ),
            PlacementError::ExceedsPool {
                needed_bytes,
                pool_bytes,
            } => write!(
                f,
                "model needs {:.1} GiB resident but the shared UMA pool is only \
                 {:.1} GiB — too big for this host on any tier mix",
                *needed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                *pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
        }
    }
}

impl std::error::Error for PlacementError {}

/// Solve the placement ILP exactly. Returns the minimum-total-latency
/// assignment that fits every device's memory budget, or a
/// [`PlacementError`] explaining why none exists.
pub fn solve(profile: &PlacementProfile) -> std::result::Result<Placement, PlacementError> {
    let n = profile.stages.len();
    if n == 0 {
        return Err(PlacementError::Empty);
    }
    let devices = &profile.devices;

    // Global UMA gate. Total resident memory is assignment-independent — every
    // stage is placed exactly once and consumes its bytes from the one shared
    // pool no matter which tier — so a pool overflow can't be fixed by any
    // tier mix. Reject up front (and before the per-device search).
    if let Some(pool) = profile.pool_bytes {
        let needed: u64 = profile.stages.iter().map(|s| s.mem_bytes).sum();
        if needed > pool {
            return Err(PlacementError::ExceedsPool {
                needed_bytes: needed,
                pool_bytes: pool,
            });
        }
    }

    // Per stage: the feasible (device_index, latency) options, sorted by
    // (latency, device_index) so the DFS explores the cheapest — and, on a
    // latency tie, the earlier-listed (preferred) — device first.
    let mut options: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for st in &profile.stages {
        let mut opts: Vec<(usize, f64)> = Vec::new();
        for (di, dev) in devices.iter().enumerate() {
            if let Some(&lat) = st.lat_ms.get(&dev.device) {
                // Skip non-finite latencies (NaN/±inf): a NaN would poison the
                // sort/prune and could be chosen with a NaN total; treat it as
                // "not a usable option" (a hand-built profile, or a future
                // producer using +inf as a soft op-support marker).
                if lat.is_finite() && dev.mem_bytes >= st.mem_bytes {
                    opts.push((di, lat));
                }
            }
        }
        if opts.is_empty() {
            return Err(PlacementError::UnplaceableStage(st.stage));
        }
        opts.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        options.push(opts);
    }

    // Admissible lower bound for the suffix [i..n): the sum of each remaining
    // stage's cheapest latency, ignoring capacity. Never overestimates the
    // true remaining cost, so pruning against it is safe.
    let mut suffix_lb = vec![0.0f64; n + 1];
    for i in (0..n).rev() {
        let min_lat = options[i]
            .iter()
            .map(|&(_, l)| l)
            .fold(f64::INFINITY, f64::min);
        suffix_lb[i] = suffix_lb[i + 1] + min_lat;
    }

    let stage_mem: Vec<u64> = profile.stages.iter().map(|s| s.mem_bytes).collect();
    let caps: Vec<u64> = devices.iter().map(|d| d.mem_bytes).collect();

    let mut solver = Bnb {
        options,
        stage_mem,
        caps,
        suffix_lb,
        best_cost: f64::INFINITY,
        best_assign: None,
    };
    let mut used = vec![0u64; devices.len()];
    let mut cur = vec![0usize; n];
    solver.dfs(0, 0.0, &mut used, &mut cur);

    match solver.best_assign {
        Some(assign) => {
            let mut per_device_mem: BTreeMap<String, u64> = BTreeMap::new();
            let assignment: Vec<String> = assign
                .iter()
                .enumerate()
                .map(|(i, &di)| {
                    *per_device_mem
                        .entry(devices[di].device.clone())
                        .or_insert(0) += profile.stages[i].mem_bytes;
                    devices[di].device.clone()
                })
                .collect();
            Ok(Placement {
                assignment,
                total_lat_ms: solver.best_cost,
                per_device_mem,
            })
        }
        None => Err(PlacementError::Infeasible),
    }
}

/// Branch-and-bound state for [`solve`].
struct Bnb {
    options: Vec<Vec<(usize, f64)>>,
    stage_mem: Vec<u64>,
    caps: Vec<u64>,
    suffix_lb: Vec<f64>,
    best_cost: f64,
    best_assign: Option<Vec<usize>>,
}

impl Bnb {
    fn dfs(&mut self, i: usize, cost: f64, used: &mut [u64], cur: &mut [usize]) {
        // Prune: this branch can't beat the incumbent. `>=` (not `>`) keeps
        // the first — i.e. most-preferred, since options are sorted — optimal
        // assignment when several tie on total latency.
        if cost + self.suffix_lb[i] >= self.best_cost {
            return;
        }
        if i == self.options.len() {
            self.best_cost = cost;
            self.best_assign = Some(cur.to_vec());
            return;
        }
        // Small (≤ #devices) option list; copy it to release the borrow on
        // `self` for the recursive call.
        let opts = self.options[i].clone();
        let mem = self.stage_mem[i];
        for (di, lat) in opts {
            if used[di] + mem <= self.caps[di] {
                used[di] += mem;
                cur[i] = di;
                self.dfs(i + 1, cost + lat, used, cur);
                used[di] -= mem;
            }
        }
    }
}

impl Placement {
    /// A compact human-readable summary, one line per device tier used.
    pub fn summary(&self) -> String {
        let mut by_device: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, d) in self.assignment.iter().enumerate() {
            by_device.entry(d.as_str()).or_default().push(i);
        }
        let mut lines = vec![format!(
            "placement: {} stage(s), total forward latency {:.3} ms",
            self.assignment.len(),
            self.total_lat_ms
        )];
        for (dev, stages) in &by_device {
            let mb = *self.per_device_mem.get(*dev).unwrap_or(&0) as f64 / (1024.0 * 1024.0);
            lines.push(format!("  {dev:<4} stages {stages:?}  ({mb:.0} MiB)"));
        }
        lines.join("\n")
    }
}

/// `cascadia place` — read a placement profile, solve, write `placement.json`.
#[derive(Args, Debug)]
pub struct PlaceArgs {
    /// Path to the `placement_profile.json` produced by
    /// `profile-devices --per-stage`.
    #[arg(long)]
    pub profile: PathBuf,

    /// Where to write the solved `placement.json`.
    #[arg(long, default_value = "placement.json")]
    pub output: PathBuf,
}

pub fn cmd_place(args: PlaceArgs) -> Result<()> {
    let raw = std::fs::read_to_string(&args.profile)
        .with_context(|| format!("reading placement profile {}", args.profile.display()))?;
    let profile: PlacementProfile = serde_json::from_str(&raw)
        .with_context(|| format!("parsing placement profile {}", args.profile.display()))?;

    let placement = solve(&profile)
        .map_err(|e| anyhow::anyhow!("no valid placement for {}: {e}", profile.model))?;

    let json = serde_json::to_string_pretty(&placement)?;
    std::fs::write(&args.output, &json)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!("{}", placement.summary());
    println!("wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn dev(name: &str, gb: u64) -> DeviceCap {
        DeviceCap {
            device: name.to_string(),
            mem_bytes: gb * GB,
        }
    }

    /// Build a stage whose latency on each (device, ms) pair is given; devices
    /// not listed are unsupported (op-support gate).
    fn stage(idx: u32, mem_gb: u64, lats: &[(&str, f64)]) -> StageCost {
        StageCost {
            stage: idx,
            mem_bytes: mem_gb * GB,
            lat_ms: lats.iter().map(|(d, l)| (d.to_string(), *l)).collect(),
        }
    }

    /// Devices ordered by preference: GPU fastest, then NPU, then CPU.
    fn three_tier(gpu_gb: u64, npu_gb: u64, cpu_gb: u64) -> Vec<DeviceCap> {
        vec![dev("GPU", gpu_gb), dev("NPU", npu_gb), dev("CPU", cpu_gb)]
    }

    /// A profile with no global pool gate (per-device caps only).
    fn profile(model: &str, devices: Vec<DeviceCap>, stages: Vec<StageCost>) -> PlacementProfile {
        PlacementProfile {
            model: model.into(),
            devices,
            stages,
            pool_bytes: None,
            fingerprint: None,
        }
    }

    #[test]
    fn fitting_model_goes_all_gpu() {
        // 4 equal stages, GPU fastest for all, everything fits → all GPU.
        let profile = profile(
            "fits",
            three_tier(16, 8, 16),
            (0..4)
                .map(|i| stage(i, 1, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)]))
                .collect(),
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.assignment, vec!["GPU", "GPU", "GPU", "GPU"]);
        assert!((p.total_lat_ms - 4.0).abs() < 1e-9);
        assert_eq!(p.per_device_mem.get("GPU"), Some(&(4 * GB)));
    }

    #[test]
    fn gpu_cap_forces_overflow_to_cheaper_feasible_tier() {
        // GPU holds only 2 of the 4 stages (2 GiB cap, 1 GiB each); the
        // remaining 2 overflow to the *cheaper* feasible tier. Here the
        // synthetic latencies make NPU (1.4) cheaper than CPU (3.0), so the
        // solver picks NPU — the data-driven mechanic. (On real Lunar Lake
        // hardware CPU < NPU for decode, so the same mechanic overflows to CPU
        // — see docs/perf/THREE_TIER_PLACEMENT.md.)
        let profile = profile(
            "spill",
            three_tier(2, 8, 16),
            (0..4)
                .map(|i| stage(i, 1, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)]))
                .collect(),
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.per_device_mem.get("GPU"), Some(&(2 * GB)));
        assert_eq!(p.per_device_mem.get("NPU"), Some(&(2 * GB)));
        assert_eq!(p.per_device_mem.get("CPU"), None);
        // 2×GPU(1.0) + 2×NPU(1.4) = 4.8
        assert!((p.total_lat_ms - 4.8).abs() < 1e-9);
    }

    #[test]
    fn prefers_earlier_listed_device_on_a_latency_tie() {
        // All devices equally fast + everything fits → the solver must pick the
        // earliest-listed (most-preferred) device for every stage. Locks in the
        // tie-break (the `.then(a.0.cmp(&b.0))` sort + `>=` prune).
        let profile = profile(
            "tie",
            three_tier(16, 16, 16),
            (0..3)
                .map(|i| stage(i, 1, &[("GPU", 2.0), ("NPU", 2.0), ("CPU", 2.0)]))
                .collect(),
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.assignment, vec!["GPU", "GPU", "GPU"]);
    }

    #[test]
    fn single_device_places_everything_there() {
        let profile = profile(
            "solo",
            vec![dev("CPU", 16)],
            (0..3).map(|i| stage(i, 1, &[("CPU", 5.0)])).collect(),
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.assignment, vec!["CPU", "CPU", "CPU"]);
        assert!((p.total_lat_ms - 15.0).abs() < 1e-9);
    }

    #[test]
    fn non_finite_latency_is_not_chosen() {
        // A NaN/inf latency must be ignored (not selected, not NaN-poisoning the
        // total). Here GPU is NaN, so the stage lands on CPU.
        let profile = profile(
            "nan",
            three_tier(16, 16, 16),
            vec![stage(0, 1, &[("GPU", f64::NAN), ("CPU", 4.0)])],
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.assignment, vec!["CPU"]);
        assert!((p.total_lat_ms - 4.0).abs() < 1e-9);
    }

    #[test]
    fn op_support_gate_excludes_device() {
        // Stage 1 can't run on GPU (e.g. NPU-only attention shape). It must go
        // to NPU/CPU even though GPU has room and would be faster.
        let profile = profile(
            "gate",
            three_tier(16, 8, 16),
            vec![
                stage(0, 1, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)]),
                stage(1, 1, &[("NPU", 1.4), ("CPU", 3.0)]), // no GPU
                stage(2, 1, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)]),
            ],
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.assignment[0], "GPU");
        assert_eq!(p.assignment[1], "NPU"); // cheapest feasible for the gated stage
        assert_eq!(p.assignment[2], "GPU");
    }

    #[test]
    fn infeasible_when_total_memory_exceeds_total_capacity() {
        // 10 GiB of stages, 6 GiB total capacity across all devices.
        let profile = profile(
            "too-big",
            three_tier(2, 2, 2),
            (0..10)
                .map(|i| stage(i, 1, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)]))
                .collect(),
        );
        assert_eq!(solve(&profile), Err(PlacementError::Infeasible));
    }

    #[test]
    fn unplaceable_stage_when_no_device_supports_it() {
        let profile = profile(
            "no-device",
            three_tier(16, 8, 16),
            vec![stage(0, 1, &[])], // supported nowhere
        );
        assert_eq!(solve(&profile), Err(PlacementError::UnplaceableStage(0)));
    }

    #[test]
    fn unplaceable_stage_when_it_exceeds_every_budget() {
        // A 4 GiB stage, but no device has 4 GiB.
        let profile = profile(
            "stage-too-big",
            three_tier(2, 2, 3),
            vec![stage(0, 4, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)])],
        );
        assert_eq!(solve(&profile), Err(PlacementError::UnplaceableStage(0)));
    }

    #[test]
    fn empty_profile_is_an_error() {
        let profile = profile("empty", three_tier(16, 8, 16), vec![]);
        assert_eq!(solve(&profile), Err(PlacementError::Empty));
    }

    #[test]
    fn chooses_globally_optimal_not_greedy() {
        // Greedy-by-stage would put stage 0 on GPU (it's cheapest there), but
        // GPU then has room for only one more stage. The cap is 2 GiB = 2
        // stages. Stage 2 benefits most from GPU (huge CPU penalty), so the
        // optimum keeps GPU for stages 1 and 2 and pushes stage 0 (which is
        // cheap everywhere) to NPU.
        let profile = profile(
            "global",
            three_tier(2, 8, 16),
            vec![
                stage(0, 1, &[("GPU", 1.0), ("NPU", 1.1), ("CPU", 1.2)]), // cheap anywhere
                stage(1, 1, &[("GPU", 1.0), ("NPU", 5.0), ("CPU", 9.0)]),
                stage(2, 1, &[("GPU", 1.0), ("NPU", 5.0), ("CPU", 9.0)]),
            ],
        );
        let p = solve(&profile).unwrap();
        assert_eq!(p.assignment, vec!["NPU", "GPU", "GPU"]);
        // 1.1 + 1.0 + 1.0 = 3.1  (vs greedy 1.0+1.0+5.0 = 7.0)
        assert!((p.total_lat_ms - 3.1).abs() < 1e-9);
    }

    #[test]
    fn exceeds_uma_pool_is_rejected_even_when_per_device_caps_allow() {
        // Per-device caps (16+16 = 32 GiB) would accept this 20 GiB model, but
        // the shared UMA pool is only 12 GiB → no tier mix can hold it.
        let mut prof = profile(
            "pool",
            three_tier(16, 16, 16),
            (0..20)
                .map(|i| stage(i, 1, &[("GPU", 1.0), ("NPU", 1.4), ("CPU", 3.0)]))
                .collect(),
        );
        prof.pool_bytes = Some(12 * GB);
        assert_eq!(
            solve(&prof),
            Err(PlacementError::ExceedsPool {
                needed_bytes: 20 * GB,
                pool_bytes: 12 * GB,
            })
        );
        // With a 24 GiB pool the same model places fine (still memory-forced
        // off the 16 GiB GPU onto the NPU).
        prof.pool_bytes = Some(24 * GB);
        let p = solve(&prof).unwrap();
        assert_eq!(p.assignment.len(), 20);
        assert_eq!(p.per_device_mem.get("GPU"), Some(&(16 * GB)));
        assert_eq!(p.per_device_mem.get("NPU"), Some(&(4 * GB)));
    }
}
