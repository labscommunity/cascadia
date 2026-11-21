//! `CASCADIA_GIT_SHA`: the commit this crate is built from, for
//! `policy::engine_build`. The build environment's `CASCADIA_GIT_SHA` wins
//! (a container build without `.git`), then the git checkout's `HEAD`, with
//! `-dirty` appended when tracked files have changes, else `unknown`. A
//! `CASCADIA_GIT_SHA` left set in a shell overrides git too.

#[path = "build_sha.rs"]
mod build_sha;

use std::path::Path;

fn git(args: &[&str]) -> Option<String> {
    build_sha::git(Path::new(env!("CARGO_MANIFEST_DIR")), args)
}

fn main() {
    // A container or tarball build has no .git: pass the commit in.
    println!("cargo:rerun-if-env-changed=CASCADIA_GIT_SHA");
    let from_env = std::env::var("CASCADIA_GIT_SHA").ok();
    let (sha, clean) =
        build_sha::engine_sha(from_env.as_deref(), Path::new(env!("CARGO_MANIFEST_DIR")));
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
    // A clean build must notice the first edit, which touches no git file.
    // Only while clean: a dirty build already says so, and watching the
    // sources then would rerun this script (and rebuild the crate) on every
    // edit anywhere in the workspace. Leaving dirty takes a commit, which
    // the refs above catch; reverting an edit by hand stays `-dirty` until
    // the next commit or checkout, the safe direction.
    if clean {
        if let Some(top) = git(&["rev-parse", "--show-toplevel"]) {
            let top = std::path::Path::new(&top);
            for p in ["crates", "Cargo.toml", "Cargo.lock"] {
                let path = top.join(p);
                if path.exists() {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}
