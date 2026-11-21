//! The engine sha `build.rs` stamps into `CASCADIA_GIT_SHA`, split out so
//! `tests/build_sha.rs` can exercise it.

use std::path::Path;
use std::process::Command;

pub fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

/// Whether tracked files differ from `HEAD`, staged or not. Untracked files
/// do not count: a cargo git checkout always holds an untracked `.cargo-ok`.
/// A failed `git status` counts as dirty: a build that cannot prove it is
/// clean must not claim to be.
pub fn dirty(dir: &Path) -> bool {
    let Ok(out) = Command::new("git")
        // No index refresh: a write here would look like a change next build.
        .args([
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ])
        .current_dir(dir)
        .output()
    else {
        return true;
    };
    !out.status.success() || !out.stdout.is_empty()
}

/// The sha for `dir` and whether it names a clean checkout. A non-empty
/// `from_env` wins, then `HEAD` (`-dirty` when tracked files changed), else
/// `unknown`.
pub fn engine_sha(from_env: Option<&str>, dir: &Path) -> (String, bool) {
    if let Some(sha) = from_env.map(str::trim).filter(|s| !s.is_empty()) {
        return (sha.to_string(), false);
    }
    match git(dir, &["rev-parse", "HEAD"]) {
        Some(head) if dirty(dir) => (format!("{head}-dirty"), false),
        Some(head) => (head, true),
        None => ("unknown".into(), false),
    }
}
