//! A policy package's identity: a digest of every file in it, and the build
//! of the engine that computes its actions. Two servers whose package digest,
//! precision and engine build agree return the same actions for one request.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The commit the engine was built from (`unknown` outside a git checkout).
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
    package_sha256_skipping(dir, None)
}

/// [`package_sha256`] without the directory `skip` (the OpenVINO compile
/// cache, which never changes outputs) when it lies inside `dir`.
pub(crate) fn package_sha256_skipping(dir: &Path, skip: Option<&Path>) -> std::io::Result<String> {
    let skip = skip.and_then(|p| p.canonicalize().ok());
    let mut files = Vec::new();
    walk(dir, "", skip.as_deref(), &mut files)?;
    files.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut listing = Sha256::new();
    for (rel, path) in files {
        let mut h = Sha256::new();
        let mut f = std::fs::File::open(&path).map_err(|e| at(&rel, e))?;
        std::io::copy(&mut f, &mut h).map_err(|e| at(&rel, e))?;
        listing.update(format!("{}  {rel}\n", hex(&h.finalize())));
    }
    Ok(hex(&listing.finalize()))
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
        if kind.is_dir() {
            if skip.is_some() && path.canonicalize().ok().as_deref() == skip {
                continue;
            }
            walk(&path, &format!("{rel}/"), skip, out)?;
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
        assert_eq!(
            before,
            package_sha256_skipping(&a, Some(&a.join("ov-cache"))).unwrap()
        );
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
