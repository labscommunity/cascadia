//! Local model registry + HuggingFace snapshot puller.
//!
//! Mirrors the public surface of `tahoma/download/__init__.py`:
//! `register`, `unregister`, `get`, `list`, `pull`, plus a tiny
//! filesystem-backed registry at `~/.cache/tahoma/registry.json`.
//!
//! The registry is process-local; concurrent writes from sibling tahoma
//! processes on the same host are not coordinated (matches Python).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("hub error: {0}")]
    Hub(String),
    #[error("model not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub local_path: Option<String>,
    #[serde(default)]
    pub pulled_at: i64,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_source() -> String {
    "huggingface".into()
}

impl ModelEntry {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source: default_source(),
            local_path: None,
            pulled_at: chrono::Utc::now().timestamp(),
            size_bytes: 0,
            revision: None,
            tags: Vec::new(),
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct RegistryFile {
    #[serde(default)]
    models: HashMap<String, ModelEntry>,
}

/// Filesystem-backed registry. Cheap-clone (Arc inside).
#[derive(Clone)]
pub struct Registry {
    inner: std::sync::Arc<RwLock<RegistryFile>>,
    path: PathBuf,
}

impl Registry {
    pub fn open_default() -> Result<Self> {
        Self::open(default_registry_path()?)
    }

    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let inner = if path.exists() {
            let bytes = fs::read(&path)?;
            serde_json::from_slice::<RegistryFile>(&bytes).unwrap_or_default()
        } else {
            RegistryFile::default()
        };
        Ok(Self {
            inner: std::sync::Arc::new(RwLock::new(inner)),
            path,
        })
    }

    fn flush(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(&*self.inner.read())?;
        fs::write(&self.path, bytes)?;
        Ok(())
    }

    pub fn register(&self, entry: ModelEntry) -> Result<()> {
        self.inner.write().models.insert(entry.id.clone(), entry);
        self.flush()
    }

    pub fn unregister(&self, id: &str) -> Result<()> {
        self.inner.write().models.remove(id);
        self.flush()
    }

    pub fn get(&self, id: &str) -> Option<ModelEntry> {
        self.inner.read().models.get(id).cloned()
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        self.inner.read().models.values().cloned().collect()
    }
}

pub fn default_registry_path() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("TAHOMA_REGISTRY_DIR") {
        return Ok(PathBuf::from(dir).join("registry.json"));
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    Ok(PathBuf::from(home).join(".cache/tahoma/registry.json"))
}

/// Pull a HuggingFace snapshot synchronously and register it.
///
/// Returns the local path of the snapshot. On Mac/Linux this lands under
/// `~/.cache/huggingface/hub/`; the registry only stores a pointer.
pub async fn pull(registry: &Registry, repo_id: &str) -> Result<PathBuf> {
    use hf_hub::api::tokio::ApiBuilder;
    let api = ApiBuilder::new().build().map_err(|e| Error::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());
    // hf-hub doesn't expose snapshot_download(); pulling at least one file
    // populates the local cache. For real use the API caller pulls each
    // shard (e.g. config.json, tokenizer.json, model.safetensors).
    let probe = repo
        .get("config.json")
        .await
        .map_err(|e| Error::Hub(e.to_string()))?;
    info!(repo_id, ?probe, "config.json fetched");
    let local_dir = probe.parent().map(Path::to_path_buf);

    let mut entry = ModelEntry::new(repo_id);
    if let Some(dir) = &local_dir {
        entry.local_path = Some(dir.to_string_lossy().to_string());
    }
    registry.register(entry)?;
    Ok(local_dir.unwrap_or_else(|| probe.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn registry_roundtrips_register_and_list() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("registry.json");
        let reg = Registry::open(&path).unwrap();
        reg.register(ModelEntry::new("my/model")).unwrap();
        assert_eq!(reg.list().len(), 1);

        // Reopen reads the persisted state.
        let reg2 = Registry::open(&path).unwrap();
        assert_eq!(reg2.get("my/model").unwrap().id, "my/model");
    }

    #[test]
    fn registry_unregister_removes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("registry.json");
        let reg = Registry::open(&path).unwrap();
        reg.register(ModelEntry::new("a")).unwrap();
        reg.register(ModelEntry::new("b")).unwrap();
        reg.unregister("a").unwrap();
        let names: Vec<_> = reg.list().into_iter().map(|m| m.id).collect();
        assert_eq!(names, vec!["b".to_string()]);
    }

    #[test]
    fn default_registry_path_uses_env_when_set() {
        std::env::set_var("TAHOMA_REGISTRY_DIR", "/tmp/test-tahoma-reg");
        let path = default_registry_path().unwrap();
        assert!(path.starts_with("/tmp/test-tahoma-reg"));
        std::env::remove_var("TAHOMA_REGISTRY_DIR");
    }
}
