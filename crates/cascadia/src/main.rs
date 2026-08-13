//! cascadia binary entry point. All logic lives in the cascadia-cli crate.

use anyhow::Result;
use cascadia_cli::Cli;
use clap::Parser;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // The elastic memory posture (`--elastic`) may re-execute this process with
    // an allocator interposer preloaded. That must happen BEFORE the async
    // runtime spins up worker threads: `execv` and the `LD_PRELOAD`/env setup
    // are only safe while the process is single-threaded. On Linux success the
    // call replaces the image and never returns; otherwise it falls through and
    // the run continues (with OV memory knobs still seeded).
    cascadia_cli::activate_elastic_if_requested(&cli);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(cascadia_cli::run(cli))
}
