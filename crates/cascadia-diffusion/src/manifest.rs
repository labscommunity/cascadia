//! Immutable, fixed-shape artifact contracts. A manifest is trusted only when
//! its digest matches a value supplied by the operator, never by a remote peer.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

pub const MAX_TENSOR_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_PORTS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    F16,
    I64,
    I32,
    U8,
}
impl Dtype {
    pub fn width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
    pub fn finite(self, bytes: &[u8]) -> bool {
        match self {
            Self::F32 => bytes
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
            Self::F16 => bytes
                .chunks_exact(2)
                .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7c00 != 0x7c00),
            _ => true,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}
impl Port {
    pub fn bytes(&self) -> Result<usize> {
        identifier(&self.name)?;
        if self.shape.len() > 8 {
            return invalid("tensor rank exceeds eight");
        }
        let mut bytes = self.dtype.width();
        for &dim in &self.shape {
            if dim == 0 {
                return invalid("zero tensor dimension");
            }
            bytes = bytes
                .checked_mul(dim)
                .ok_or_else(|| Error::Invalid("tensor size overflow".into()))?;
            if bytes > MAX_TENSOR_BYTES {
                return invalid("tensor exceeds byte budget");
            }
        }
        Ok(bytes)
    }
}
pub fn ports_bytes(ports: &[Port]) -> Result<usize> {
    if ports.is_empty() || ports.len() > MAX_PORTS {
        return invalid("invalid tensor count");
    }
    let mut names = BTreeSet::new();
    let mut bytes = 0usize;
    for port in ports {
        if !names.insert(&port.name) {
            return invalid("duplicate tensor name");
        }
        bytes = bytes
            .checked_add(port.bytes()?)
            .ok_or_else(|| Error::Invalid("tensor size overflow".into()))?;
        if bytes > MAX_TENSOR_BYTES {
            return invalid("combined tensors exceed byte budget");
        }
    }
    Ok(bytes)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    /// Relative IR XML and BIN paths. Each is covered by `files`.
    pub xml: String,
    pub weights: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub architecture: String,
    pub source_revision: String,
    pub width: u32,
    pub height: u32,
    pub components: BTreeMap<String, Component>,
    /// Includes scheduler/tokenizer/config files as well as all model weights.
    pub files: BTreeMap<String, Artifact>,
}
impl Manifest {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 || self.architecture != "sdxl" {
            return invalid("unsupported manifest version or architecture");
        }
        if self.source_revision.len() != 40
            || !self
                .source_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return invalid("source revision must be an immutable 40-character commit");
        }
        if [self.width, self.height]
            .iter()
            .any(|d| *d < 64 || *d > 1024 || d % 8 != 0)
        {
            return invalid("invalid image dimensions");
        }
        if self.components.is_empty()
            || self.components.len() > 16
            || self.files.is_empty()
            || self.files.len() > 256
        {
            return invalid("invalid component or artifact count");
        }
        for (path, artifact) in &self.files {
            relative_path(path)?;
            digest(&artifact.sha256)?;
            if artifact.bytes == 0 {
                return invalid("empty artifact");
            }
        }
        for (name, component) in &self.components {
            identifier(name)?;
            ports_bytes(&component.inputs)?;
            ports_bytes(&component.outputs)?;
            for path in [&component.xml, &component.weights] {
                relative_path(path)?;
                if !self.files.contains_key(path) {
                    return invalid("component references an unhashed artifact");
                }
            }
            if !component.xml.ends_with(".xml")
                || !component.weights.ends_with(".bin")
                || component.xml.trim_end_matches(".xml")
                    != component.weights.trim_end_matches(".bin")
            {
                return invalid("IR XML and BIN must share a basename");
            }
        }
        Ok(())
    }
    pub fn id(&self) -> Result<String> {
        self.validate()?;
        let value = serde_json::to_value(self)?;
        let mut bytes = Vec::new();
        canonical(&value, &mut bytes)?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
    /// Verify just the selected worker bundle, plus all common metadata. Bundle
    /// directories must be operator-owned and remain immutable while serving.
    pub fn verify_bundle(&self, root: &Path, expected: &str, selected: &[String]) -> Result<()> {
        digest(expected)?;
        if self.id()? != expected {
            return invalid("manifest digest mismatch");
        }
        if selected.len() > self.components.len()
            || selected.iter().collect::<BTreeSet<_>>().len() != selected.len()
        {
            return invalid("invalid selected components");
        }
        let mut needed = BTreeSet::new();
        for name in selected {
            let component = self
                .components
                .get(name)
                .ok_or_else(|| Error::Invalid("unknown component".into()))?;
            needed.insert(component.xml.as_str());
            needed.insert(component.weights.as_str());
        }
        for path in self.files.keys() {
            if !path.ends_with(".xml") && !path.ends_with(".bin") {
                needed.insert(path);
            }
        }
        for path in needed {
            let mut full = root.to_path_buf();
            // Reject symlinks at every level, including the bundle root.
            if root.symlink_metadata()?.file_type().is_symlink() {
                return invalid("bundle root is a symlink");
            }
            for part in path.split('/') {
                full.push(part);
                if full.symlink_metadata()?.file_type().is_symlink() {
                    return invalid("artifact symlink");
                }
            }
            let artifact = &self.files[path];
            let mut file = File::open(full)?;
            let meta = file.metadata()?;
            if !meta.is_file() || meta.len() != artifact.bytes {
                return invalid("artifact length mismatch");
            }
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            if format!("{:x}", hash.finalize()) != artifact.sha256 {
                return invalid("artifact hash mismatch");
            }
        }
        let mut pending = vec![root.to_path_buf()];
        let mut count = 0;
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                count += 1;
                if count > 1024 {
                    return invalid("bundle has too many entries");
                }
                let kind = entry.file_type()?;
                if kind.is_symlink() {
                    return invalid("bundle symlink");
                }
                if kind.is_dir() {
                    pending.push(entry.path());
                } else if kind.is_file() {
                    let full = entry.path();
                    let name = full
                        .strip_prefix(root)
                        .map_err(|_| Error::Invalid("artifact outside bundle".into()))?
                        .to_string_lossy()
                        .replace('\\', "/");
                    if name != "manifest.json" && !self.files.contains_key(&name) {
                        return invalid("unlisted artifact");
                    }
                } else {
                    return invalid("nonregular artifact");
                }
            }
        }
        Ok(())
    }
}
pub fn digest(s: &str) -> Result<()> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return invalid("invalid SHA-256 digest");
    }
    Ok(())
}
pub fn identifier(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 128
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./:-".contains(&b))
    {
        return invalid("invalid identifier");
    }
    Ok(())
}
pub fn relative_path(s: &str) -> Result<()> {
    if s.len() > 240
        || s.split('/').any(|p| {
            p.is_empty()
                || p == "."
                || p == ".."
                || p.ends_with('.')
                || p.bytes()
                    .any(|b| !b.is_ascii_alphanumeric() && !b"_.-".contains(&b))
        })
    {
        return invalid("unsafe artifact path");
    }
    // Windows device names and alternate streams must never be accepted.
    for part in s.split('/') {
        let stem = part.split('.').next().unwrap().to_ascii_uppercase();
        if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || ((stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit())
        {
            return invalid("reserved artifact path");
        }
    }
    Ok(())
}
fn canonical(v: &serde_json::Value, out: &mut Vec<u8>) -> Result<()> {
    match v {
        serde_json::Value::Object(map) => {
            out.push(b'{');
            let sorted: BTreeMap<_, _> = map.iter().collect();
            for (i, (k, v)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(k)?);
                out.push(b':');
                canonical(v, out)?;
            }
            out.push(b'}');
        }
        serde_json::Value::Array(values) => {
            out.push(b'[');
            for (i, v) in values.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonical(v, out)?;
            }
            out.push(b']');
        }
        _ => out.extend(serde_json::to_vec(v)?),
    }
    Ok(())
}
pub(crate) fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::Invalid(message.into()))
}
