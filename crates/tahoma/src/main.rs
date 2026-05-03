//! tahoma binary entry point. All logic lives in the tahoma-cli crate.

use anyhow::Result;
use clap::Parser;
use tahoma_cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tahoma_cli::run(cli).await
}
