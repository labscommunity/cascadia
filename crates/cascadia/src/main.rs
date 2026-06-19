//! cascadia binary entry point. All logic lives in the cascadia-cli crate.

use anyhow::Result;
use cascadia_cli::Cli;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    cascadia_cli::run(cli).await
}
