//! The engine sha from `build_sha.rs`: only tracked changes make a build
//! `-dirty`, so a cargo git checkout (untracked `.cargo-ok`) stays clean.

#[path = "../build_sha.rs"]
mod build_sha;

use std::path::Path;
use std::process::Command;

fn run(dir: &Path, args: &[&str]) {
    let st = Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(dir)
        .status()
        .expect("git runs");
    assert!(st.success(), "git {args:?}");
}

/// A one-commit repo and its clone, the clone shaped like a cargo git checkout.
fn origin_and_clone() -> (tempfile::TempDir, tempfile::TempDir, String) {
    let origin = tempfile::tempdir().unwrap();
    run(origin.path(), &["init", "-q"]);
    std::fs::write(origin.path().join("lib.rs"), "fn a() {}\n").unwrap();
    run(origin.path(), &["add", "lib.rs"]);
    run(origin.path(), &["commit", "-q", "-m", "init"]);
    let clone = tempfile::tempdir().unwrap();
    run(
        clone.path(),
        &["clone", "-q", origin.path().to_str().unwrap(), "."],
    );
    let head = build_sha::git(clone.path(), &["rev-parse", "HEAD"]).unwrap();
    (origin, clone, head)
}

#[test]
fn untracked_cargo_ok_is_clean() {
    let (_o, clone, head) = origin_and_clone();
    assert_eq!(
        build_sha::engine_sha(None, clone.path()),
        (head.clone(), true)
    );
    std::fs::write(clone.path().join(".cargo-ok"), "{\"v\":1}").unwrap();
    assert!(!build_sha::dirty(clone.path()));
    assert_eq!(build_sha::engine_sha(None, clone.path()), (head, true));
}

#[test]
fn modified_tracked_file_is_dirty() {
    let (_o, clone, head) = origin_and_clone();
    std::fs::write(clone.path().join("lib.rs"), "fn b() {}\n").unwrap();
    assert!(build_sha::dirty(clone.path()));
    assert_eq!(
        build_sha::engine_sha(None, clone.path()),
        (format!("{head}-dirty"), false)
    );
}

#[test]
fn staged_new_file_is_dirty() {
    let (_o, clone, head) = origin_and_clone();
    std::fs::write(clone.path().join("new.rs"), "\n").unwrap();
    run(clone.path(), &["add", "new.rs"]);
    assert_eq!(
        build_sha::engine_sha(None, clone.path()),
        (format!("{head}-dirty"), false)
    );
}

#[test]
fn env_override_wins_over_git() {
    let (_o, clone, _head) = origin_and_clone();
    std::fs::write(clone.path().join("lib.rs"), "fn b() {}\n").unwrap();
    assert_eq!(
        build_sha::engine_sha(Some(" abc123 \n"), clone.path()),
        ("abc123".to_string(), false)
    );
    // Blank counts as unset.
    assert!(build_sha::engine_sha(Some("  "), clone.path())
        .0
        .ends_with("-dirty"));
}
