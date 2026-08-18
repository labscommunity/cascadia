//! Fail `--features embed-spa` builds with instructions when the SPA
//! hasn't been built, instead of the three cryptic E0599 errors the
//! `#[derive(Embed)]` on a missing `web/dist/` folder produces.

use std::env;
use std::path::Path;

fn main() {
    // Set by cargo iff the `embed-spa` feature is enabled for this build.
    if env::var_os("CARGO_FEATURE_EMBED_SPA").is_some() {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let index = Path::new(&manifest_dir)
            .join("web")
            .join("dist")
            .join("index.html");
        if !index.exists() {
            panic!(
                "\n\nthe `embed-spa` / `dashboard-embed` feature is enabled, but the dashboard \
                 SPA has not been built (crates/cascadia-dashboard/web/dist/ is missing — it is \
                 generated, not checked in).\n\nBuild it first:\n\n    cd crates/cascadia-dashboard/web\n    \
                 npm ci && npm run build\n\nthen re-run the cargo build.\n"
            );
        }
        // Re-run this check (and rust-embed's re-embed) when the built SPA
        // appears or changes; without the feature the crate doesn't touch
        // web/dist, so no dependency is registered.
        println!("cargo:rerun-if-changed=web/dist");
    }
}
