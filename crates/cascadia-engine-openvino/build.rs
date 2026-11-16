//! `CASCADIA_GIT_SHA`: the commit this crate is built from, for
//! `policy::engine_build`; `unknown` outside a git checkout.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

fn main() {
    let sha = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=CASCADIA_GIT_SHA={sha}");
    // Rebuild on a new commit or checkout, not only when this crate changes.
    let mut watched = vec!["HEAD".to_string(), "packed-refs".to_string()];
    if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(head_ref);
    }
    for p in watched {
        // A missing watched path would rebuild this crate every time.
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &p])
            .filter(|path| std::path::Path::new(path).exists())
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
