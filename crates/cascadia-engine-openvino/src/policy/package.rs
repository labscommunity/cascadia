//! A policy package's identity: a digest of every file in it, and the build
//! of the engine that computes its actions. Two servers whose package digest,
//! precision and engine build agree return the same actions for one request.

use std::path::{Path, PathBuf};

use cascadia_engine::EngineError;
use sha2::{Digest, Sha256};

/// The commit the engine was built from: `CASCADIA_GIT_SHA` at build time,
/// else the git checkout's `HEAD`, else `unknown`.
const GIT_SHA: &str = env!("CASCADIA_GIT_SHA");

/// SHA-256 (lowercase hex) over every file under `dir`, so two packages that
/// differ in any file, listed by the manifest or not, differ here.
///
/// It is `sha256sum` of the `sha256sum` listing (`<hex>  <path>`) of every
/// file, by `/`-separated path relative to `dir`, sorted bytewise. A symlink
/// to a file counts as that file's content under the link's own path, wherever
/// the target is (a Hub snapshot links out to its blob store). A symlink to a
/// directory, a dangling one and anything but a file or directory are refused
/// by name, as is a name `sha256sum` would escape (not UTF-8, `\\` or a newline).
/// Nothing is ignored, OS metadata files included: an extra file can only
/// refuse a match, never make two different packages match.
pub fn package_sha256(dir: &Path) -> std::io::Result<String> {
    package_sha256_with(dir, &[])
}

/// [`package_sha256`] without the top-level files named in `ignore`: files a
/// deployer writes next to the package that never reach the engine (a staging
/// checksum, say). Each is a file name, no glob and no path, matched without
/// case (as APFS and NTFS open it); the same name in a subdirectory, or a
/// directory of that name, still counts. A name with a separator, a glob
/// character, `.` or `..` is refused (`InvalidInput`).
///
/// This does not know which files the engine reads: ignoring one makes two
/// packages that compute different actions match. Check the list against the
/// package's manifest with [`check_digest_ignore`] first, as
/// `OvPolicyEngine::load` does.
pub fn package_sha256_with(dir: &Path, ignore: &[String]) -> std::io::Result<String> {
    digest(dir, ignore, None)
}

/// [`package_sha256_with`], also without the directory `skip` (the OpenVINO
/// compile cache, which never changes outputs) when it lies inside `dir`.
pub(crate) fn digest(
    dir: &Path,
    ignore: &[String],
    skip: Option<&Path>,
) -> std::io::Result<String> {
    check_plain_names(ignore)?;
    let ignore: Vec<String> = ignore.iter().map(|n| n.to_lowercase()).collect();
    let skip = skip.and_then(|p| p.canonicalize().ok());
    let mut files = Vec::new();
    walk(dir, "", &ignore, skip.as_deref(), &mut files)?;
    files.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut listing = Sha256::new();
    for (rel, path) in files {
        listing.update(format!("{}  {rel}\n", hash_file(&rel, &path)?));
    }
    Ok(hex(&listing.finalize()))
}

/// A digest failure as a load error: a deliberate refusal (an entry the
/// identity cannot cover, a bad ignore name) is the package's or the
/// caller's fault and stays one; any other I/O error (a file shared or
/// replaced mid-walk, an interrupted read) may pass, so it is retryable.
#[cfg_attr(not(feature = "openvino"), allow(dead_code))]
pub(crate) fn digest_error(dir: &Path, e: std::io::Error) -> EngineError {
    let msg = format!("{}: {e}", dir.display());
    match e.kind() {
        std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => {
            EngineError::InvalidConfig(msg)
        }
        _ => EngineError::Backend(msg),
    }
}

fn check_plain_names(ignore: &[String]) -> std::io::Result<()> {
    match ignore.iter().find(|n| {
        n.is_empty() || *n == "." || *n == ".." || n.contains(['/', '\\', '*', '?', '[', '\0'])
    }) {
        Some(bad) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("digest ignore entry {bad:?} is not a plain file name"),
        )),
        None => Ok(()),
    }
}

/// Read beside the tokenizer IR for the added special tokens.
pub(crate) const TOKENIZER_JSON: &str = "tokenizer.json";

/// Refuse a digest-ignore list [`package_sha256_with`] would accept but the
/// engine could notice: anything but plain names, or a top-level name (without
/// case) the engine may open for this manifest. Those are `manifest.json`,
/// `NOTICE`, every manifest-named artifact and any file sharing its stem
/// (its `.bin` weights, a `.json` beside it), and the `tokenizer.json` beside
/// the tokenizer IR.
pub fn check_digest_ignore(
    manifest: &super::manifest::PolicyManifest,
    ignore: &[String],
) -> Result<(), EngineError> {
    check_plain_names(ignore).map_err(|e| EngineError::InvalidConfig(e.to_string()))?;
    let top = |p: &str| !p.contains(['/', '\\']);
    let mut names = vec!["manifest.json".to_string(), "NOTICE".to_string()];
    let mut stems = Vec::new();
    for artifact in [
        manifest.model_artifact.as_str(),
        manifest.tokenizer_artifact(),
    ] {
        if top(artifact) {
            names.push(artifact.to_string());
            if let Some(stem) = Path::new(artifact).file_stem() {
                stems.push(stem.to_string_lossy().to_lowercase());
            }
        }
    }
    if top(manifest.tokenizer_artifact()) {
        names.push(TOKENIZER_JSON.to_string());
    }
    for n in ignore {
        let lower = n.to_lowercase();
        let stem = Path::new(&lower)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned());
        if names.iter().any(|r| r.to_lowercase() == lower)
            || stem.is_some_and(|s| stems.contains(&s))
        {
            return Err(EngineError::InvalidConfig(format!(
                "digest_ignore names {n:?}, which the engine may read; it must stay in the \
                 package identity"
            )));
        }
    }
    Ok(())
}

/// SHA-256 of one file, read in 1 MiB blocks. Opened without blocking where
/// the platform allows and checked on the open handle, so an entry swapped
/// for a FIFO after the walk is refused instead of waited on.
fn hash_file(rel: &str, path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NONBLOCK);
    }
    let mut f = opts.open(path).map_err(|e| at(rel, e))?;
    if !f.metadata().map_err(|e| at(rel, e))?.is_file() {
        return Err(refuse(rel, "is not a regular file"));
    }
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => h.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(at(rel, e)),
        }
    }
    Ok(hex(&h.finalize()))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn at(rel: &str, e: std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), format!("package file {rel}: {e}"))
}

fn refuse(rel: &str, why: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("package entry {rel} {why}; the package identity cannot cover it"),
    )
}

fn walk(
    dir: &Path,
    prefix: &str,
    ignore: &[String],
    skip: Option<&Path>,
    out: &mut Vec<(String, PathBuf)>,
) -> std::io::Result<()> {
    let shown = if prefix.is_empty() { "." } else { prefix };
    for entry in std::fs::read_dir(dir).map_err(|e| at(shown, e))? {
        let entry = entry.map_err(|e| at(shown, e))?;
        let name = entry.file_name();
        let rel = match name.to_str() {
            Some(n) => format!("{prefix}{n}"),
            None => {
                let lossy = format!("{prefix}{}", name.to_string_lossy());
                return Err(refuse(&lossy, "has a name that is not UTF-8"));
            }
        };
        if rel.contains(['\\', '\n']) {
            return Err(refuse(
                &rel.escape_debug().to_string(),
                "has '\\' or a newline in its name",
            ));
        }
        let path = entry.path();
        let kind = entry.file_type().map_err(|e| at(&rel, e))?;
        // `digest` lowercased `ignore`.
        if prefix.is_empty() && !kind.is_dir() && ignore.contains(&rel.to_lowercase()) {
            continue;
        }
        if kind.is_dir() {
            if skip.is_some() && path.canonicalize().ok().as_deref() == skip {
                continue;
            }
            walk(&path, &format!("{rel}/"), ignore, skip, out)?;
        } else if kind.is_file() {
            out.push((rel, path));
        } else if kind.is_symlink() {
            match std::fs::metadata(&path) {
                Err(_) => return Err(refuse(&rel, "is a dangling symlink")),
                Ok(m) if m.is_file() => out.push((rel, path)),
                Ok(m) if m.is_dir() => return Err(refuse(&rel, "is a symlink to a directory")),
                Ok(_) => return Err(refuse(&rel, "is a symlink to something not a file")),
            }
        } else {
            return Err(refuse(&rel, "is not a file or directory"));
        }
    }
    Ok(())
}

/// The engine that computes the actions: this engine's version and commit and
/// the OpenVINO runtime build actually loaded, as
/// `cascadia-ov-policy/<version>+<git sha> openvino/<runtime build>`. Two
/// builds of one commit on one runtime give the same string.
pub fn engine_build(openvino_runtime: &str) -> String {
    format!(
        "cascadia-ov-policy/{}+{GIT_SHA} openvino/{openvino_runtime}",
        env!("CARGO_PKG_VERSION")
    )
}

/// Whether an [`engine_build`] string names a known commit and runtime. One
/// that does not (`+unknown`, a build outside git without `CASCADIA_GIT_SHA`)
/// must never match another, itself included: it cannot tell two engines apart.
pub fn engine_build_is_known(build: &str) -> bool {
    let Some(rest) = build.strip_prefix("cascadia-ov-policy/") else {
        return false;
    };
    let Some((engine, runtime)) = rest.split_once(" openvino/") else {
        return false;
    };
    // The version may carry its own `+` build metadata; the sha comes last.
    let sha = engine.rsplit_once('+').map_or("", |(_, sha)| sha);
    !sha.is_empty() && sha != "unknown" && !runtime.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn package(root: &Path, name: &str, weights: &[u8]) -> std::path::PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(dir.join("ir")).unwrap();
        fs::write(dir.join("manifest.json"), br#"{"policy":{"name":"p"}}"#).unwrap();
        fs::write(dir.join("ir/policy.xml"), b"<net/>").unwrap();
        fs::write(dir.join("ir/policy.bin"), weights).unwrap();
        dir
    }

    #[test]
    fn a_file_outside_the_manifest_changes_the_digest() {
        let t = tempfile::tempdir().unwrap();
        let fp16 = package(t.path(), "a", b"fp16 weights");
        let int8 = package(t.path(), "b", b"int8 weights");
        assert_eq!(
            fs::read(fp16.join("manifest.json")).unwrap(),
            fs::read(int8.join("manifest.json")).unwrap()
        );
        assert_ne!(
            package_sha256(&fp16).unwrap(),
            package_sha256(&int8).unwrap()
        );
        // A file the manifest never names still counts.
        let extra = package(t.path(), "c", b"fp16 weights");
        let before = package_sha256(&extra).unwrap();
        assert_eq!(before, package_sha256(&fp16).unwrap());
        fs::write(extra.join("int8.bin"), b"x").unwrap();
        assert_ne!(before, package_sha256(&extra).unwrap());
    }

    #[test]
    fn a_copied_package_has_the_same_digest() {
        let t = tempfile::tempdir().unwrap();
        let a = package(t.path(), "a", b"weights");
        let b = t.path().join("elsewhere/b");
        fs::create_dir_all(b.join("ir")).unwrap();
        for f in ["manifest.json", "ir/policy.xml", "ir/policy.bin"] {
            fs::copy(a.join(f), b.join(f)).unwrap();
        }
        let d = package_sha256(&a).unwrap();
        assert_eq!(d.len(), 64);
        assert!(d.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')));
        assert_eq!(d, package_sha256(&b).unwrap());
    }

    /// The digest is `sha256sum` of `sha256sum`'s listing of the sorted
    /// relative paths, so it can be checked without Cascadia.
    #[test]
    fn the_digest_is_sha256sum_of_the_sorted_listing() {
        use sha2::Digest;
        let hex = |b: &[u8]| -> String {
            sha2::Sha256::digest(b)
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect()
        };
        let t = tempfile::tempdir().unwrap();
        let dir = package(t.path(), "a", b"w");
        fs::write(dir.join("Z.txt"), b"z").unwrap();
        let mut listing = String::new();
        for (path, body) in [
            ("Z.txt", &b"z"[..]),
            ("ir/policy.bin", b"w"),
            ("ir/policy.xml", b"<net/>"),
            ("manifest.json", br#"{"policy":{"name":"p"}}"#),
        ] {
            listing.push_str(&format!("{}  {path}\n", hex(body)));
        }
        assert_eq!(package_sha256(&dir).unwrap(), hex(listing.as_bytes()));
    }

    #[test]
    fn a_renamed_file_changes_the_digest() {
        let t = tempfile::tempdir().unwrap();
        let a = package(t.path(), "a", b"w");
        let before = package_sha256(&a).unwrap();
        fs::rename(a.join("ir/policy.bin"), a.join("ir/policy2.bin")).unwrap();
        assert_ne!(before, package_sha256(&a).unwrap());
    }

    #[test]
    fn the_skipped_cache_directory_does_not_count() {
        let t = tempfile::tempdir().unwrap();
        let a = package(t.path(), "a", b"w");
        let before = package_sha256(&a).unwrap();
        fs::create_dir_all(a.join("ov-cache")).unwrap();
        fs::write(a.join("ov-cache/123.blob"), b"compiled").unwrap();
        assert_eq!(before, digest(&a, &[], Some(&a.join("ov-cache"))).unwrap());
        assert_ne!(before, package_sha256(&a).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_count_by_content_and_a_directory_link_is_refused() {
        use std::os::unix::fs::symlink;
        let t = tempfile::tempdir().unwrap();
        // A Hub snapshot: the package's files link out to a blob store.
        let blobs = t.path().join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::write(blobs.join("w1"), b"weights").unwrap();
        let real = package(t.path(), "real", b"weights");
        let snap = package(t.path(), "snap", b"unused");
        fs::remove_file(snap.join("ir/policy.bin")).unwrap();
        symlink(blobs.join("w1"), snap.join("ir/policy.bin")).unwrap();
        assert_eq!(
            package_sha256(&real).unwrap(),
            package_sha256(&snap).unwrap()
        );
        // The content behind an escaping link is part of the identity.
        fs::write(blobs.join("w1"), b"other weights").unwrap();
        assert_ne!(
            package_sha256(&real).unwrap(),
            package_sha256(&snap).unwrap()
        );

        // A link to a directory outside the package is refused, by name.
        let outside = t.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, real.join("extra")).unwrap();
        let e = package_sha256(&real).unwrap_err().to_string();
        assert!(e.contains("extra"), "{e}");
        assert!(e.contains("directory"), "{e}");
        fs::remove_file(real.join("extra")).unwrap();

        // So is a dangling link.
        symlink(t.path().join("gone"), real.join("dangling")).unwrap();
        let e = package_sha256(&real).unwrap_err().to_string();
        assert!(e.contains("dangling"), "{e}");
    }

    #[test]
    fn an_ignored_top_level_file_does_not_count() {
        let t = tempfile::tempdir().unwrap();
        let a = package(t.path(), "a", b"w");
        let b = package(t.path(), "b", b"w");
        fs::write(b.join(".staging-sha256"), b"bookkeeping").unwrap();
        let ignore = [".staging-sha256".to_string()];
        assert_eq!(
            package_sha256_with(&a, &ignore).unwrap(),
            package_sha256_with(&b, &ignore).unwrap()
        );
        assert_eq!(
            package_sha256(&a).unwrap(),
            package_sha256_with(&b, &ignore).unwrap()
        );
        assert_ne!(package_sha256(&a).unwrap(), package_sha256(&b).unwrap());
        // Only at the top: the same name in a subdirectory still counts.
        fs::write(b.join("ir/.staging-sha256"), b"bookkeeping").unwrap();
        assert_ne!(
            package_sha256(&a).unwrap(),
            package_sha256_with(&b, &ignore).unwrap()
        );
    }

    #[test]
    fn ignore_entries_are_plain_names() {
        let t = tempfile::tempdir().unwrap();
        let a = package(t.path(), "a", b"w");
        for bad in ["", ".", "..", "ir/policy.bin", "a\\b", "*.json"] {
            let e = package_sha256_with(&a, &[bad.to_string()]).unwrap_err();
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{bad:?}: {e}");
        }
    }

    #[test]
    fn refusals_are_config_errors_and_io_failures_are_retryable() {
        use cascadia_engine::EngineError;
        let dir = Path::new("pkg");
        let refusal = refuse("x", "is a dangling symlink");
        assert!(matches!(
            digest_error(dir, refusal),
            EngineError::InvalidConfig(_)
        ));
        let bad_ignore = std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad name");
        assert!(matches!(
            digest_error(dir, bad_ignore),
            EngineError::InvalidConfig(_)
        ));
        for kind in [
            std::io::ErrorKind::NotFound,
            std::io::ErrorKind::Interrupted,
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::Other,
        ] {
            let e = at(
                "ir/policy.bin",
                std::io::Error::new(kind, "sharing violation"),
            );
            assert!(
                matches!(digest_error(dir, e), EngineError::Backend(_)),
                "{kind:?}"
            );
        }
        // A real refusal from a walk.
        #[cfg(unix)]
        {
            let t = tempfile::tempdir().unwrap();
            let a = package(t.path(), "a", b"w");
            std::os::unix::fs::symlink(t.path().join("gone"), a.join("x")).unwrap();
            let e = package_sha256(&a).unwrap_err();
            assert!(matches!(digest_error(&a, e), EngineError::InvalidConfig(_)));
        }
    }

    /// A file swapped for a FIFO after the walk is refused, not waited on.
    #[cfg(unix)]
    #[test]
    fn a_fifo_is_refused_without_blocking() {
        let t = tempfile::tempdir().unwrap();
        let fifo = t.path().join("fifo");
        let ok = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(ok.success());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(hash_file("fifo", &fifo).map_err(|e| e.kind()));
        });
        let res = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("hashing a FIFO blocked");
        assert_eq!(res.unwrap_err(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn an_unknown_commit_never_matches() {
        assert!(engine_build_is_known(
            "cascadia-ov-policy/0.2.7+02f48c2969d9a14bbdb5f09b2997dd1c9612dcd2 openvino/2026.4.1-1"
        ));
        assert!(!engine_build_is_known(
            "cascadia-ov-policy/0.2.7+unknown openvino/2026.4.1-1"
        ));
        assert!(!engine_build_is_known(
            "cascadia-ov-policy/0.2.7+abc openvino/"
        ));
        assert!(!engine_build_is_known(
            "cascadia-ov-policy/0.2.7 openvino/2026.4.1-1"
        ));
        assert!(!engine_build_is_known(""));
        assert_eq!(
            engine_build_is_known(&engine_build("2026.4.1-1")),
            GIT_SHA != "unknown"
        );
    }

    fn smolvla() -> super::super::manifest::PolicyManifest {
        super::super::manifest::PolicyManifest::parse(
            include_str!("../../tests/fixtures/policy/intel-smolvla-libero-manifest.json"),
            super::super::manifest::LoadOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn an_ignore_name_the_engine_can_open_is_refused() {
        let m = smolvla();
        // The fixture's artifacts sit at the top: smolvla.xml, tokenizer.xml.
        assert_eq!(m.model_artifact, "smolvla.xml");
        assert_eq!(m.tokenizer_artifact(), "tokenizer.xml");
        for read in [
            "manifest.json",
            "NOTICE",
            "smolvla.xml",
            "smolvla.bin",
            "smolvla.json",
            "tokenizer.xml",
            "tokenizer.bin",
            // Read beside the tokenizer IR for its added special tokens.
            "tokenizer.json",
            // Case-insensitive file systems open these as the same files.
            "Tokenizer.xml",
            "TOKENIZER.JSON",
            "Manifest.json",
            "notice",
        ] {
            let e = check_digest_ignore(&m, &[read.to_string()]).unwrap_err();
            assert!(matches!(e, EngineError::InvalidConfig(_)), "{read}");
            assert!(e.to_string().contains("engine"), "{read}: {e}");
        }
        check_digest_ignore(&m, &[".staging-sha256".into()]).unwrap();
        // Not a plain name: refused the same way.
        let e = check_digest_ignore(&m, &["ir/x".into()]).unwrap_err();
        assert!(matches!(e, EngineError::InvalidConfig(_)), "{e}");
    }

    #[test]
    fn ignore_names_match_without_case() {
        let t = tempfile::tempdir().unwrap();
        let a = package(t.path(), "a", b"w");
        let b = package(t.path(), "b", b"w");
        fs::write(b.join(".staging-sha256"), b"bookkeeping").unwrap();
        assert_eq!(
            package_sha256(&a).unwrap(),
            package_sha256_with(&b, &[".Staging-SHA256".to_string()]).unwrap()
        );
    }

    #[test]
    fn engine_build_is_known_reads_the_last_plus() {
        // A version with semver build metadata keeps its own '+'.
        assert!(engine_build_is_known(
            "cascadia-ov-policy/0.2.7+local+abc123 openvino/2026.4.1-1"
        ));
        assert!(!engine_build_is_known(
            "cascadia-ov-policy/0.2.7+local+unknown openvino/2026.4.1-1"
        ));
    }

    #[test]
    fn engine_build_is_stable_and_follows_the_runtime() {
        let a = engine_build("2026.4.1-20466-abc");
        assert_eq!(a, engine_build("2026.4.1-20466-abc"));
        assert_ne!(a, engine_build("2026.4.0-20000-def"));
        assert!(a.contains(env!("CARGO_PKG_VERSION")), "{a}");
        assert!(a.ends_with("openvino/2026.4.1-20466-abc"), "{a}");
        assert!(a.contains(super::GIT_SHA), "{a}");
        let sha = super::GIT_SHA;
        assert!(
            sha == "unknown" || (sha.len() == 40 && sha.bytes().all(|c| c.is_ascii_hexdigit())),
            "{sha}"
        );
    }
}
