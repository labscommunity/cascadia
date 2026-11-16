//! `CASCADIA_GIT_SHA`: the commit this crate is built from, for
//! `policy::engine_build`. The build environment's `CASCADIA_GIT_SHA` wins
//! (a container build without `.git`), then the git checkout's `HEAD`, else
//! `unknown`. A `CASCADIA_GIT_SHA` left set in a shell overrides git too.

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
    // A container or tarball build has no .git: pass the commit in.
    println!("cargo:rerun-if-env-changed=CASCADIA_GIT_SHA");
    let sha = std::env::var("CASCADIA_GIT_SHA")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=CASCADIA_GIT_SHA={sha}");
    // Rebuild on a new commit or checkout, not only when this crate changes.
    // Cargo reruns a build script on every build while a watched path is
    // missing, so a missing ref is watched through its nearest existing
    // directory under `refs/` (its loose file appears there on the next
    // commit), and a missing `packed-refs` needs no watch: packing removes
    // the loose ref, which is watched.
    let mut watched = vec!["HEAD".to_string(), "packed-refs".to_string()];
    if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(head_ref);
    }
    for p in watched {
        let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &p]) else {
            continue;
        };
        let mut path = std::path::PathBuf::from(path);
        if p.starts_with("refs/") {
            while !path.exists()
                && path
                    .parent()
                    .is_some_and(|d| d.components().any(|c| c.as_os_str() == "refs"))
            {
                path.pop();
            }
        }
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
