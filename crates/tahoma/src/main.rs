//! tahoma binary entry point. All logic lives in the tahoma-cli crate.
//!
//! `main` is intentionally not `#[tokio::main]` — we need to plan
//! and install the CPU-affinity layout (which builds the global
//! rayon pool + sizes the tokio runtime's worker pool) BEFORE the
//! tokio runtime spawns any threads. Once tokio has its workers,
//! the `on_thread_start` closure is no longer applied to them.

use anyhow::Result;
use clap::Parser;
use tahoma_cli::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let rt = tahoma_cli::install_cpu_affinity_and_build_runtime(&cli)?;
    rt.block_on(tahoma_cli::run(cli))
}
