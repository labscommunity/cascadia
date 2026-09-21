//! Bounded, opt-in residual capture. One completed file per stream instance.
//!
//! INKCAP01 header (32 bytes, little endian): magic[8], hidden u32,
//! pipeline rank u32, slot u32, prompt rows u32, sequence u64.
//! Each position: sampled next-token i64 (-1 when unsampled), hidden f16[H].
//! Rewinds truncate records; slot reuse creates a new file. Only .bin files
//! are complete; an I/O or budget failure leaves .part evidence unfinalized.
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const HEADER: u64 = 32;
struct StreamFile {
    file: File,
    path: PathBuf,
    prompt_rows: usize,
}

pub struct StateCapture {
    dir: PathBuf,
    rank: u32,
    hidden: usize,
    record_bytes: u64,
    budget: u64,
    written: u64,
    sequence: u64,
    boot: String,
    failed: bool,
    streams: HashMap<usize, StreamFile>,
}

impl StateCapture {
    pub fn from_env(rank: u32, total: u32, hidden: usize) -> Option<Self> {
        let path = if rank + 1 == total {
            std::env::var_os("CASCADIA_STREAMS_CAPTURE_FINAL")
                .or_else(|| std::env::var_os("CASCADIA_STREAMS_CAPTURE_BOUNDARY"))
        } else {
            std::env::var_os("CASCADIA_STREAMS_CAPTURE_BOUNDARY")
        }?;
        if path.is_empty() {
            return None;
        }
        let budget = std::env::var("CASCADIA_STREAMS_CAPTURE_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024);
        match Self::new(Path::new(&path), rank, hidden, budget) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!("state capture unavailable: {e}");
                None
            }
        }
    }

    fn new(dir: &Path, rank: u32, hidden: usize, budget: u64) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let written = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        let micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        Ok(Self {
            dir: dir.to_owned(),
            rank,
            hidden,
            record_bytes: 8 + 2 * hidden as u64,
            budget,
            written,
            sequence: 0,
            boot: format!("{micros}-{}", std::process::id()),
            failed: false,
            streams: HashMap::new(),
        })
    }

    fn reserve(&mut self, bytes: u64) -> io::Result<()> {
        if self.written.saturating_add(bytes) > self.budget {
            return Err(io::Error::other("state capture byte budget exhausted"));
        }
        self.written += bytes;
        Ok(())
    }

    fn check(&mut self, result: io::Result<()>) {
        if let Err(e) = result {
            self.failed = true;
            self.streams.clear();
            tracing::warn!("state capture stopped, incomplete .part files retained: {e}");
        }
    }

    pub fn open(&mut self, slot: usize) {
        if self.failed {
            return;
        }
        let result = (|| {
            self.reserve(HEADER)?;
            self.sequence += 1;
            let path = self.dir.join(format!(
                "r{}-s{}-{}-{}.part",
                self.rank, slot, self.boot, self.sequence
            ));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            file.write_all(b"INKCAP01")?;
            for n in [self.hidden as u32, self.rank, slot as u32, 0] {
                file.write_all(&n.to_le_bytes())?;
            }
            file.write_all(&self.sequence.to_le_bytes())?;
            self.streams.insert(
                slot,
                StreamFile {
                    file,
                    path,
                    prompt_rows: 0,
                },
            );
            Ok(())
        })();
        self.check(result);
    }

    pub fn rows(&mut self, slot: usize, pos: usize, hidden: &[f32]) {
        if self.failed {
            return;
        }
        let result = (|| {
            let rows = hidden.len() / self.hidden;
            self.reserve(rows as u64 * self.record_bytes)?;
            let s = self
                .streams
                .get_mut(&slot)
                .ok_or_else(|| io::Error::other("capture slot not open"))?;
            s.file
                .seek(SeekFrom::Start(HEADER + pos as u64 * self.record_bytes))?;
            let mut bytes = Vec::with_capacity(rows * self.record_bytes as usize);
            for row in hidden.chunks_exact(self.hidden) {
                bytes.extend_from_slice(&(-1i64).to_le_bytes());
                for &x in row {
                    bytes.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
                }
            }
            s.file.write_all(&bytes)
        })();
        self.check(result);
    }

    pub fn token(&mut self, slot: usize, pos: usize, token: i64) {
        if self.failed {
            return;
        }
        let result = (|| {
            self.reserve(12)?;
            let s = self
                .streams
                .get_mut(&slot)
                .ok_or_else(|| io::Error::other("capture token slot not open"))?;
            if s.prompt_rows == 0 {
                s.prompt_rows = pos + 1;
                s.file.seek(SeekFrom::Start(20))?;
                s.file.write_all(&(s.prompt_rows as u32).to_le_bytes())?;
            }
            s.file
                .seek(SeekFrom::Start(HEADER + pos as u64 * self.record_bytes))?;
            s.file.write_all(&token.to_le_bytes())
        })();
        self.check(result);
    }

    pub fn truncate(&mut self, slot: usize, len: usize) {
        if self.failed {
            return;
        }
        let result = self.streams.get_mut(&slot).map_or(Ok(()), |s| {
            s.file.set_len(HEADER + len as u64 * self.record_bytes)
        });
        self.check(result);
    }

    pub fn close(&mut self, slot: usize) {
        if self.failed {
            return;
        }
        let result = (|| {
            if let Some(s) = self.streams.remove(&slot) {
                let path = s.path;
                let size = s.file.metadata()?.len();
                let finished = path.with_extension("bin");
                let mut line = serde_json::to_vec(&serde_json::json!({
                    "file": finished.file_name().unwrap().to_string_lossy(),
                    "size": size, "rank": self.rank, "slot": slot,
                    "prompt_rows": s.prompt_rows, "hidden": self.hidden,
                }))?;
                line.push(b'\n');
                self.reserve(line.len() as u64)?;
                drop(s.file);
                std::fs::rename(&path, finished)?;
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(self.dir.join("index.jsonl"))?
                    .write_all(&line)?;
            }
            Ok(())
        })();
        self.check(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rewind_preserves_accepted_positions_and_slot_reuse_is_separate() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = StateCapture::new(dir.path(), 10, 2, 4096).unwrap();
        c.open(1);
        c.rows(1, 0, &[1., 2., 3., 4.]);
        c.token(1, 1, 11);
        c.rows(1, 2, &[5., 6.]);
        c.token(1, 2, 99);
        c.truncate(1, 2);
        c.rows(1, 2, &[7., 8.]);
        c.token(1, 2, 12);
        c.close(1);
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "bin"))
            .unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), 32 + 3 * 12);
        assert_eq!(&bytes[..8], b"INKCAP01");
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 2);
        assert_eq!(i64::from_le_bytes(bytes[56..64].try_into().unwrap()), 12);
        assert_eq!(
            half::f16::from_le_bytes(bytes[64..66].try_into().unwrap()).to_f32(),
            7.
        );
        c.open(1);
        c.rows(1, 0, &[0., 0.]);
        c.close(1);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);
    }

    #[test]
    fn budget_failure_never_finalizes_partial_capture_and_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = StateCapture::new(dir.path(), 1, 2, 44).unwrap();
        c.open(0);
        c.rows(0, 0, &[1., 2.]);
        c.rows(0, 1, &[3., 4.]);
        c.close(0);
        assert!(c.failed);
        let files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].extension().unwrap(), "part");
        let mut again = StateCapture::new(dir.path(), 1, 2, 44).unwrap();
        again.open(1);
        assert!(again.failed);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
