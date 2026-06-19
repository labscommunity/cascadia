//! `cascadia run-placement` — launch a heterogeneous pipeline from a
//! `placement.json` (step 3 of #41: *apply* the ILP's per-stage device map).
//!
//! Spawns one `cascadia worker` per stage, each pinned to its assigned device,
//! wired into the existing pipeline ring (rank 0 is the API head; non-first
//! stages bind a relay `--listen` socket; non-last stages dial `--next`). For
//! the single-host three-tier case every worker is local and the relay is TCP
//! over loopback — exactly the topology #63 validated for multi-stage NPU,
//! now with a *heterogeneous* device per stage.
//!
//! The argv planning ([`plan_workers`]) is pure and unit-tested; the spawner
//! is a thin supervisor that tears the whole ring down if any stage exits or
//! on Ctrl-C.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Args;

use crate::placement::Placement;

#[derive(Args, Debug, Clone)]
pub struct RunPlacementArgs {
    /// Multi-stage shard directory (the same one profiled/placed).
    #[arg(long)]
    pub shard: PathBuf,

    /// `placement.json` from `cascadia place`.
    #[arg(long)]
    pub placement: PathBuf,

    /// API bind address for the pipeline head (rank 0).
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub api: String,

    /// Host the relay `--listen`/`--next` sockets bind/dial. This launcher
    /// spawns all stages as **local** child processes, so this is normally
    /// loopback; to span machines, launch `cascadia worker` per box by hand.
    #[arg(long, default_value = "127.0.0.1")]
    pub relay_host: String,

    /// Base TCP port for inter-stage relay. Stage `r` (r≥1) listens on
    /// `relay_base + r`; consecutive runs on one host must not overlap.
    #[arg(long, default_value_t = 9100)]
    pub relay_base: u16,

    /// OpenVINO compiled-blob cache dir, forwarded to every worker.
    #[arg(long)]
    pub ov_cache_dir: Option<String>,

    /// Print the worker command lines and exit without spawning.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
}

/// Build the argv (after the `worker` subcommand token) for one stage.
#[allow(clippy::too_many_arguments)]
fn worker_argv(
    rank: usize,
    total: usize,
    device: &str,
    model: &str,
    listen: Option<String>,
    next: Option<String>,
    api: Option<&str>,
    ov_cache: Option<&str>,
) -> Vec<String> {
    let mut a = vec![
        "worker".to_string(),
        "--rank".into(),
        rank.to_string(),
        "--total".into(),
        total.to_string(),
        "--engine".into(),
        // Static shards run on GPU/CPU/NPU via the #63 static-KV path.
        "ov-runtime".into(),
        "--device".into(),
        device.to_string(),
        "--model".into(),
        model.to_string(),
    ];
    if let Some(l) = listen {
        a.push("--listen".into());
        a.push(l);
    }
    if let Some(n) = next {
        a.push("--next".into());
        a.push(n);
    }
    if let Some(api) = api {
        a.push("--api".into());
        a.push(api.to_string());
    }
    if let Some(c) = ov_cache {
        a.push("--ov-cache-dir".into());
        a.push(c.to_string());
    }
    a
}

/// Plan every stage's worker argv from a placement. Pure — unit-tested.
///
/// Topology (matches `cmd_worker`): rank 0 holds `--api` and dials rank 1 via
/// `--next` (no `--listen`); rank `r` (0<r<N-1) listens on `relay_base+r` and
/// dials `relay_base+r+1`; the last rank listens but has no `--next`.
pub fn plan_workers(
    placement: &Placement,
    shard: &str,
    api: &str,
    relay_host: &str,
    relay_base: u16,
    ov_cache: Option<&str>,
) -> Result<Vec<Vec<String>>> {
    let n = placement.assignment.len();
    if n == 0 {
        bail!("placement has no stages");
    }
    let port = |rank: usize| -> Result<u16> {
        u16::try_from(relay_base as usize + rank)
            .map_err(|_| anyhow::anyhow!("relay port {relay_base}+{rank} overflows u16"))
    };
    let mut plans = Vec::with_capacity(n);
    for rank in 0..n {
        let device = &placement.assignment[rank];
        let listen = if rank == 0 {
            None
        } else {
            Some(format!("{relay_host}:{}", port(rank)?))
        };
        let next = if rank == n - 1 {
            None
        } else {
            Some(format!("{relay_host}:{}", port(rank + 1)?))
        };
        let api = if rank == 0 { Some(api) } else { None };
        plans.push(worker_argv(
            rank, n, device, shard, listen, next, api, ov_cache,
        ));
    }
    Ok(plans)
}

pub async fn cmd_run_placement(args: RunPlacementArgs) -> Result<()> {
    let raw = std::fs::read_to_string(&args.placement)
        .with_context(|| format!("reading placement {}", args.placement.display()))?;
    let placement: Placement = serde_json::from_str(&raw)
        .with_context(|| format!("parsing placement {}", args.placement.display()))?;
    let shard = args
        .shard
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 shard path {}", args.shard.display()))?;

    let plans = plan_workers(
        &placement,
        shard,
        &args.api,
        &args.relay_host,
        args.relay_base,
        args.ov_cache_dir.as_deref(),
    )?;

    println!(
        "run-placement: {} stage(s) across {:?}; API on {}",
        plans.len(),
        placement.per_device_mem.keys().cloned().collect::<Vec<_>>(),
        args.api
    );
    for (rank, argv) in plans.iter().enumerate() {
        println!(
            "  stage {rank} [{}]: cascadia {}",
            placement.assignment[rank],
            argv.join(" ")
        );
    }
    if args.dry_run {
        return Ok(());
    }

    spawn::supervise(plans).await
}

mod spawn {
    use super::*;
    use futures::future::{select_all, FutureExt};

    /// Spawn all stages as child `cascadia worker` processes and supervise:
    /// tear the whole ring down when any stage exits or on Ctrl-C. Listeners
    /// (higher ranks) are spawned first; the 60 s peer-connect retry in the
    /// transport covers any remaining startup races.
    pub async fn supervise(plans: Vec<Vec<String>>) -> Result<()> {
        let exe = std::env::current_exe().context("locating the cascadia executable")?;
        let mut children = Vec::with_capacity(plans.len());
        // Reverse order: bring up the listeners (ranks N-1..1) before rank 0
        // dials them.
        for argv in plans.iter().enumerate().rev() {
            let (rank, argv) = argv;
            // kill_on_drop: if a later spawn fails (the `?` below) or we return
            // early, dropping `children` tears down the already-spawned workers
            // instead of leaking them as orphans holding relay ports + devices.
            let child = tokio::process::Command::new(&exe)
                .args(argv)
                .kill_on_drop(true)
                .spawn()
                .with_context(|| format!("spawning stage {rank}"))?;
            children.push((rank, child));
        }

        // Wait for the first stage to exit, or Ctrl-C.
        let waits: Vec<_> = children
            .iter_mut()
            .map(|(rank, c)| {
                let rank = *rank;
                async move { (rank, c.wait().await) }.boxed()
            })
            .collect();

        tokio::select! {
            (res, _idx, _rest) = select_all(waits) => {
                let (rank, status) = res;
                tracing::warn!(rank, ?status, "a pipeline stage exited — tearing down the ring");
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Ctrl-C — tearing down the pipeline");
            }
        }

        for (rank, mut child) in children {
            let _ = child.start_kill();
            let _ = child.wait().await;
            tracing::debug!(rank, "stage stopped");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn placement(devices: &[&str]) -> Placement {
        Placement {
            assignment: devices.iter().map(|s| s.to_string()).collect(),
            total_lat_ms: 0.0,
            per_device_mem: BTreeMap::new(),
        }
    }

    fn argv_of(plans: &[Vec<String>], rank: usize) -> String {
        plans[rank].join(" ")
    }

    #[test]
    fn single_stage_is_api_head_with_no_relay() {
        let p = placement(&["GPU"]);
        let plans = plan_workers(&p, "/m", "127.0.0.1:8000", "127.0.0.1", 9100, None).unwrap();
        assert_eq!(plans.len(), 1);
        let a = argv_of(&plans, 0);
        assert!(a.contains("--rank 0"));
        assert!(a.contains("--total 1"));
        assert!(a.contains("--device GPU"));
        assert!(a.contains("--api 127.0.0.1:8000"));
        assert!(!a.contains("--listen"));
        assert!(!a.contains("--next"));
    }

    #[test]
    fn three_stage_ring_is_wired_head_to_tail() {
        let p = placement(&["GPU", "NPU", "CPU"]);
        let plans = plan_workers(&p, "/m", "127.0.0.1:8000", "127.0.0.1", 9100, None).unwrap();
        assert_eq!(plans.len(), 3);

        // rank 0: API head, dials rank 1, no listen.
        let r0 = argv_of(&plans, 0);
        assert!(r0.contains("--rank 0") && r0.contains("--device GPU"));
        assert!(r0.contains("--api 127.0.0.1:8000"));
        assert!(r0.contains("--next 127.0.0.1:9101"));
        assert!(!r0.contains("--listen"));

        // rank 1: listens on 9101, dials 9102, no API.
        let r1 = argv_of(&plans, 1);
        assert!(r1.contains("--rank 1") && r1.contains("--device NPU"));
        assert!(r1.contains("--listen 127.0.0.1:9101"));
        assert!(r1.contains("--next 127.0.0.1:9102"));
        assert!(!r1.contains("--api"));

        // rank 2 (last): listens on 9102, no --next, no API.
        let r2 = argv_of(&plans, 2);
        assert!(r2.contains("--rank 2") && r2.contains("--device CPU"));
        assert!(r2.contains("--listen 127.0.0.1:9102"));
        assert!(!r2.contains("--next"));
        assert!(!r2.contains("--api"));
    }

    #[test]
    fn every_stage_uses_ov_runtime_and_the_shard() {
        let p = placement(&["GPU", "CPU"]);
        let plans =
            plan_workers(&p, "/srv/shard", "127.0.0.1:8000", "127.0.0.1", 9100, None).unwrap();
        for a in &plans {
            let s = a.join(" ");
            assert!(s.contains("--engine ov-runtime"));
            assert!(s.contains("--model /srv/shard"));
        }
    }

    #[test]
    fn ov_cache_dir_is_forwarded_to_all_stages() {
        let p = placement(&["GPU", "NPU"]);
        let plans =
            plan_workers(&p, "/m", "127.0.0.1:8000", "127.0.0.1", 9100, Some("/c")).unwrap();
        for a in &plans {
            assert!(a.join(" ").contains("--ov-cache-dir /c"));
        }
    }

    #[test]
    fn empty_placement_is_an_error() {
        let p = placement(&[]);
        assert!(plan_workers(&p, "/m", "127.0.0.1:8000", "127.0.0.1", 9100, None).is_err());
    }
}
