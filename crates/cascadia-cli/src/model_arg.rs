//! How the exporters (`cascadia shard`, `cascadia export-policy`) read `--model`: a Hugging Face
//! model id or a local directory, decided by how it is written rather than by what happens to be
//! in the current directory.

use std::path::Path;

use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelArg {
    /// A path: written as one, or not a valid model id.
    Local,
    /// A Hugging Face model id.
    Hub,
}

/// Whether `s` is written as a path: absolute, `.`/`..`-relative, or a Windows drive path.
pub(crate) fn written_as_path(s: &str) -> bool {
    let b = s.as_bytes();
    let drive =
        b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && matches!(b[2], b'\\' | b'/');
    Path::new(s).is_absolute()
        || drive
        || s.starts_with('/')
        || s.starts_with('\\')
        || matches!(s, "." | "..")
        || ["./", "../", ".\\", "..\\"]
            .iter()
            .any(|p| s.starts_with(p))
}

/// `--model` as a local path or a model id. A valid id that also names a directory here is
/// refused rather than guessed: `owner/name` must not turn into a local export because a folder
/// of that name happens to be in the current directory.
pub(crate) fn classify(model: &str, is_valid_id: impl Fn(&str) -> bool) -> Result<ModelArg> {
    if written_as_path(model) || !is_valid_id(model) {
        return Ok(ModelArg::Local);
    }
    if Path::new(model).exists() {
        return Err(anyhow!(
            "--model {model} is both a Hugging Face model id and a directory here. Write \
             ./{model} for the directory; for the Hugging Face model, run from another directory"
        ));
    }
    Ok(ModelArg::Hub)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> bool {
        crate::export_policy::validate_model_id(s).is_ok()
    }

    #[test]
    fn path_spellings() {
        for p in [
            "./a", "../a/b", ".\\a", "..\\a", "/abs/x", ".", "C:\\m", "c:/m",
        ] {
            assert!(written_as_path(p), "{p}");
        }
        for p in ["owner/name", "name", "a/b/c"] {
            assert!(!written_as_path(p), "{p}");
        }
    }

    #[test]
    fn paths_and_ids_are_told_apart_by_spelling() {
        let root = tempfile::tempdir().unwrap();
        let owner = root.path().join("owner");
        std::fs::create_dir_all(owner.join("name")).unwrap();
        let abs = owner.join("name");
        assert_eq!(
            classify(abs.to_str().unwrap(), id).unwrap(),
            ModelArg::Local
        );
        assert_eq!(
            classify("lerobot/no_such_dir_here", id).unwrap(),
            ModelArg::Hub
        );
        // Not a valid id: a path, whatever exists.
        assert_eq!(classify("models/a/b", id).unwrap(), ModelArg::Local);
        assert_eq!(classify("plainname", id).unwrap(), ModelArg::Local);
    }

    #[test]
    fn an_existing_id_shaped_directory_is_ambiguous() {
        // A fresh `<name>/ckpt` under the test's cwd, so the bare id resolves to a directory.
        let cwd = std::env::current_dir().unwrap();
        let dir = tempfile::tempdir_in(&cwd).unwrap();
        let name = dir
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .replace('.', "x");
        let parent = cwd.join(&name);
        std::fs::create_dir_all(parent.join("ckpt")).unwrap();
        let model = format!("{name}/ckpt");
        let e = classify(&model, id).unwrap_err().to_string();
        assert!(e.contains(&format!("./{model}")), "{e}");
        assert_eq!(
            classify(&format!("./{model}"), id).unwrap(),
            ModelArg::Local
        );
        std::fs::remove_dir_all(&parent).unwrap();
    }
}
