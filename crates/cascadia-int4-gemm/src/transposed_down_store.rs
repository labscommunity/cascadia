//! On-disk cache for AXPY-form transposed-and-requantized down
//! weights — the fix for PR #43's mmap page-cache eviction
//! regression.
//!
//! ## Why this exists
//!
//! PR #43 (the AXPY-form kernel) cached each expert's transposed
//! down weight in **anonymous** heap memory (`Vec<u8>` inside a
//! `LruCache`). On K2.6 (~518 GiB mmap'd model, a 133 GiB-RAM
//! host) that 4 GiB of pinned anon memory displaced ~910 experts'
//! worth of model mmap pages from the page cache, causing
//! subsequent expert dispatches to hit cold NVMe seeks (~576 ms
//! per fully-cold expert). The net was a 2.5× end-to-end
//! regression at the kernel-level speedup we'd have otherwise gotten.
//!
//! ## How this fixes it
//!
//! Persist each expert's transposed weights to a file on disk; mmap
//! the file at runtime. Three properties matter:
//!
//! 1. **The mmap'd transposed pages are EVICTABLE.** The kernel can
//!    reclaim them when the model mmap needs space, then re-fault
//!    from disk on the next AXPY call. Anon memory is pinned (no
//!    swap on this box); mmap'd file memory rotates through the
//!    same LRU as the model.
//! 2. **The build cost is paid once per expert per disk.** First
//!    AXPY dispatch builds + writes the file. Subsequent
//!    dispatches (including across process restarts) just mmap.
//! 3. **The 56 MiB transient scratch is per-build, not per-call.**
//!    With on-disk caching the scratch is allocated → used → freed
//!    *once* per expert across the whole lifetime of the model on
//!    disk; never reaccumulates.
//!
//! This is structurally what PowerInfer does — they persist
//! `--transpose-down all` at GGUF conversion time
//! (`convert_hf_to_gguf.py:6275-6283`). We do it lazily at first
//! AXPY dispatch instead of at export time, but the runtime cost
//! model is identical: pure mmap reads, zero anon pinning.
//!
//! ## File format
//!
//! Per-expert file at `<cache_dir>/L{lid:04}_E{eid:04}.cxd`. Layout:
//!
//! ```text
//! offset 0   16  magic           "CSCD_TXP_DOWN_v1" (16 ASCII bytes, no NUL)
//! offset 16  4   n_intermediate  u32 LE
//! offset 20  4   n_hidden        u32 LE
//! offset 24  4   group_size      u32 LE
//! offset 28  4   reserved        u32 LE (padding to 32-byte header)
//! offset 32  ..  packed_t        n_intermediate * n_hidden / 2 bytes
//! ..             scale_t_bits    n_intermediate * n_hidden / group_size * 2 bytes
//! ```
//!
//! Magic is checked on load; mismatched dimensions or magic trigger
//! a rebuild. The on-disk format is versioned so future kernel
//! refactors can invalidate caches by bumping the version suffix.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;

use crate::ffn_axpy::transpose_requantize_down;
use crate::GROUP_SIZE;

/// On-disk file format magic. Bump the suffix when the layout
/// changes; existing `.cxd` files become invalid and rebuild.
const MAGIC: &[u8; 16] = b"CSCD_TXP_DOWN_v1";
const HEADER_BYTES: usize = 32;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("cache file too small: expected {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },
    #[error("cache file magic mismatch (got {got:?}, want {want:?})")]
    BadMagic { got: [u8; 16], want: [u8; 16] },
    #[error(
        "cache file dimension mismatch (file has n_intermediate={file_int} \
         n_hidden={file_hid} group_size={file_gs}, runtime wants \
         n_intermediate={want_int} n_hidden={want_hid} group_size={want_gs})"
    )]
    DimMismatch {
        file_int: u32,
        file_hid: u32,
        file_gs: u32,
        want_int: u32,
        want_hid: u32,
        want_gs: u32,
    },
    #[error("cache dir is not writable: {path} ({err})")]
    UnwritableCacheDir { path: PathBuf, err: std::io::Error },
}

/// One mmap'd transposed-down expert. The two slices below are
/// borrowed views into [`Self::mmap`]; the Mmap stays alive as long
/// as this struct is held in the store.
pub struct TransposedDownMmap {
    #[allow(dead_code)]
    mmap: Mmap,
    packed_t_off: usize,
    packed_t_len: usize,
    scale_t_off: usize,
    scale_t_len: usize,
}

impl TransposedDownMmap {
    pub fn packed_t(&self) -> &[u8] {
        &self.mmap[self.packed_t_off..self.packed_t_off + self.packed_t_len]
    }

    pub fn scale_t_bits(&self) -> &[u8] {
        &self.mmap[self.scale_t_off..self.scale_t_off + self.scale_t_len]
    }
}

/// Persistent cache: lazily builds + persists transposed-and-
/// requantized down weights per `(layer, expert)` to disk; serves
/// them via mmap on every subsequent dispatch.
///
/// `cache_dir` is created if it doesn't exist. If it can't be made
/// writable, `get_or_build` will error and the caller should fall
/// back to the in-memory or dense path.
///
/// Not Sync (the inner HashMap of mmaps is mutated on miss). The
/// caller (the runner) holds `&mut self` for the dispatch path so
/// single-threaded mutation is fine.
pub struct TransposedDownStore {
    cache_dir: PathBuf,
    mmaps: HashMap<(u32, u32), TransposedDownMmap>,
    /// How many cache files we've built+written from scratch
    /// since open — for instrumentation.
    builds: u64,
    /// How many cache files we mmap'd from a pre-existing file
    /// — for instrumentation.
    mmap_hits: u64,
}

impl TransposedDownStore {
    /// Open or create the cache directory. Doesn't pre-load any
    /// files — opening is cheap (just a `mkdir -p`).
    pub fn open(cache_dir: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let cache_dir = cache_dir.into();
        std::fs::create_dir_all(&cache_dir).map_err(|err| StoreError::UnwritableCacheDir {
            path: cache_dir.clone(),
            err,
        })?;
        // Verify writability by creating + removing a probe file.
        let probe = cache_dir.join(".cascadia_writable_probe");
        match File::create(&probe) {
            Ok(_) => {
                let _ = std::fs::remove_file(&probe);
            }
            Err(err) => {
                return Err(StoreError::UnwritableCacheDir {
                    path: cache_dir,
                    err,
                });
            }
        }
        Ok(Self {
            cache_dir,
            mmaps: HashMap::new(),
            builds: 0,
            mmap_hits: 0,
        })
    }

    /// Return `(packed_t, scale_t_bits)` slices for the given
    /// `(lid, eid)`. On miss, build them from the supplied source
    /// down weight + scales, persist to disk, then mmap.
    ///
    /// The returned references are valid for the lifetime of
    /// `&self` (= until the next `&mut self` mutation on this
    /// store).
    pub fn get_or_build(
        &mut self,
        lid: u32,
        eid: u32,
        src_packed: &[u8],
        src_scale_bits: &[u8],
        n_hidden: usize,
        n_intermediate: usize,
    ) -> Result<&TransposedDownMmap, StoreError> {
        let key = (lid, eid);
        if !self.mmaps.contains_key(&key) {
            let path = self.file_path(lid, eid);
            // Try to mmap an existing file; on any error
            // (truncation, magic mismatch, dim mismatch), fall back
            // to rebuild.
            let mmap = match Self::try_load(&path, n_hidden, n_intermediate) {
                Ok(m) => {
                    self.mmap_hits += 1;
                    m
                }
                Err(_load_err) => {
                    // Build + persist + mmap.
                    Self::build_and_persist(
                        &path,
                        src_packed,
                        src_scale_bits,
                        n_hidden,
                        n_intermediate,
                    )?;
                    self.builds += 1;
                    Self::try_load(&path, n_hidden, n_intermediate)?
                }
            };
            self.mmaps.insert(key, mmap);
        }
        Ok(self.mmaps.get(&key).expect("just inserted/checked"))
    }

    /// `(builds, mmap_hits)` since `open` — for the runner's
    /// instrumentation log.
    pub fn stats(&self) -> (u64, u64) {
        (self.builds, self.mmap_hits)
    }

    /// Number of mmap'd entries currently held in-process. Doesn't
    /// include files on disk that haven't been touched yet.
    pub fn live_mmaps(&self) -> usize {
        self.mmaps.len()
    }

    fn file_path(&self, lid: u32, eid: u32) -> PathBuf {
        self.cache_dir.join(format!("L{lid:04}_E{eid:04}.cxd"))
    }

    fn try_load(
        path: &Path,
        n_hidden: usize,
        n_intermediate: usize,
    ) -> Result<TransposedDownMmap, StoreError> {
        let f = File::open(path)?;
        let meta = f.metadata()?;
        let n_groups_h = n_hidden / GROUP_SIZE;
        let packed_bytes = n_intermediate * n_hidden / 2;
        let scale_bytes = n_intermediate * n_groups_h * 2;
        let expected_total = HEADER_BYTES + packed_bytes + scale_bytes;
        if (meta.len() as usize) < expected_total {
            return Err(StoreError::Truncated {
                expected: expected_total,
                actual: meta.len() as usize,
            });
        }
        let mmap = unsafe { Mmap::map(&f)? };
        // Validate header.
        let mut magic = [0u8; 16];
        magic.copy_from_slice(&mmap[0..16]);
        if &magic != MAGIC {
            return Err(StoreError::BadMagic {
                got: magic,
                want: *MAGIC,
            });
        }
        let file_int = u32::from_le_bytes([mmap[16], mmap[17], mmap[18], mmap[19]]);
        let file_hid = u32::from_le_bytes([mmap[20], mmap[21], mmap[22], mmap[23]]);
        let file_gs = u32::from_le_bytes([mmap[24], mmap[25], mmap[26], mmap[27]]);
        if file_int as usize != n_intermediate
            || file_hid as usize != n_hidden
            || file_gs as usize != GROUP_SIZE
        {
            return Err(StoreError::DimMismatch {
                file_int,
                file_hid,
                file_gs,
                want_int: n_intermediate as u32,
                want_hid: n_hidden as u32,
                want_gs: GROUP_SIZE as u32,
            });
        }
        Ok(TransposedDownMmap {
            mmap,
            packed_t_off: HEADER_BYTES,
            packed_t_len: packed_bytes,
            scale_t_off: HEADER_BYTES + packed_bytes,
            scale_t_len: scale_bytes,
        })
    }

    fn build_and_persist(
        path: &Path,
        src_packed: &[u8],
        src_scale_bits: &[u8],
        n_hidden: usize,
        n_intermediate: usize,
    ) -> Result<(), StoreError> {
        let (packed_t, scale_t_bits) =
            transpose_requantize_down(src_packed, src_scale_bits, n_hidden, n_intermediate);

        // Write atomically: write to a tmp path then rename, so a
        // crash mid-write doesn't leave a half-written `.cxd` file
        // that a future load would mistake as valid (it'd fail
        // truncation but the rebuild path would then race against
        // any concurrent loaders — better to never expose a
        // partial file).
        let tmp_path = path.with_extension("cxd.partial");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            let n_groups_h = n_hidden / GROUP_SIZE;
            // Header.
            f.write_all(MAGIC)?;
            f.write_all(&(n_intermediate as u32).to_le_bytes())?;
            f.write_all(&(n_hidden as u32).to_le_bytes())?;
            f.write_all(&(GROUP_SIZE as u32).to_le_bytes())?;
            f.write_all(&0u32.to_le_bytes())?; // reserved padding
                                               // Body.
            f.write_all(&packed_t)?;
            f.write_all(&scale_t_bits)?;
            f.sync_data()?; // durable write before rename
            debug_assert_eq!(
                packed_t.len(),
                n_intermediate * n_hidden / 2,
                "packed_t size invariant"
            );
            debug_assert_eq!(
                scale_t_bits.len(),
                n_intermediate * n_groups_h * 2,
                "scale_t_bits size invariant"
            );
        } // f drops + close

        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_synthetic_down(n_hidden: usize, n_intermediate: usize) -> (Vec<u8>, Vec<u8>) {
        let n_groups = n_intermediate / GROUP_SIZE;
        let packed: Vec<u8> = (0..n_hidden * n_intermediate / 2)
            .map(|i| {
                let lo = (i * 31 + 7) & 0x0F;
                let hi = (i * 53 + 11) & 0x0F;
                ((hi << 4) | lo) as u8
            })
            .collect();
        let scales: Vec<u8> = vec![0x80, 0x3F].repeat(n_hidden * n_groups);
        (packed, scales)
    }

    /// First `get_or_build` for a given `(lid, eid)` builds + writes
    /// the file; second call mmap-hits without rebuilding.
    #[test]
    fn second_get_or_build_is_mmap_hit() {
        let tmp = TempDir::new().unwrap();
        let mut store = TransposedDownStore::open(tmp.path()).unwrap();
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale) = make_synthetic_down(n_hidden, n_intermediate);

        // Initially empty.
        assert_eq!(store.stats(), (0, 0));

        // First call: build + write + mmap.
        let _first = store
            .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        assert_eq!(store.stats(), (1, 0));
        assert_eq!(store.live_mmaps(), 1);

        // Second call same (lid, eid): mmap is already in the
        // process cache — no build, no extra mmap_hit (we don't
        // re-mmap a file we already have).
        let _second = store
            .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        assert_eq!(store.stats(), (1, 0));

        // Different (lid, eid): build a second file.
        let _third = store
            .get_or_build(7, 43, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        assert_eq!(store.stats(), (2, 0));
        assert_eq!(store.live_mmaps(), 2);
    }

    /// A second `TransposedDownStore::open` on the same dir picks
    /// up the existing files via mmap, without rebuilding.
    #[test]
    fn second_open_reuses_existing_files() {
        let tmp = TempDir::new().unwrap();
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale) = make_synthetic_down(n_hidden, n_intermediate);

        // First store: build two files.
        {
            let mut store = TransposedDownStore::open(tmp.path()).unwrap();
            store
                .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
                .unwrap();
            store
                .get_or_build(7, 43, &src_packed, &src_scale, n_hidden, n_intermediate)
                .unwrap();
            assert_eq!(store.stats(), (2, 0));
        }

        // Second store: should mmap-hit on both.
        {
            let mut store = TransposedDownStore::open(tmp.path()).unwrap();
            store
                .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
                .unwrap();
            store
                .get_or_build(7, 43, &src_packed, &src_scale, n_hidden, n_intermediate)
                .unwrap();
            assert_eq!(store.stats(), (0, 2), "expected mmap-hit on both");
        }
    }

    /// Bit-identical: the bytes mmap'd from disk match what
    /// `transpose_requantize_down` would produce in-process.
    #[test]
    fn cached_bytes_match_in_process_transpose() {
        let tmp = TempDir::new().unwrap();
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale) = make_synthetic_down(n_hidden, n_intermediate);

        let mut store = TransposedDownStore::open(tmp.path()).unwrap();
        let mmap = store
            .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        let on_disk_packed = mmap.packed_t().to_vec();
        let on_disk_scale = mmap.scale_t_bits().to_vec();

        let (in_proc_packed, in_proc_scale) =
            transpose_requantize_down(&src_packed, &src_scale, n_hidden, n_intermediate);
        assert_eq!(on_disk_packed, in_proc_packed);
        assert_eq!(on_disk_scale, in_proc_scale);
    }

    /// A corrupted (truncated) cache file triggers rebuild rather
    /// than panic.
    #[test]
    fn truncated_file_triggers_rebuild() {
        let tmp = TempDir::new().unwrap();
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale) = make_synthetic_down(n_hidden, n_intermediate);

        // Build first.
        let mut store = TransposedDownStore::open(tmp.path()).unwrap();
        store
            .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        drop(store);

        // Truncate the file.
        let path = tmp.path().join("L0007_E0042.cxd");
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(100).unwrap(); // way too small
        drop(f);

        // Second open + get_or_build: should detect truncation and rebuild.
        let mut store = TransposedDownStore::open(tmp.path()).unwrap();
        store
            .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        // (1 build for the rebuild, 0 hits — the truncated file
        // failed the load attempt).
        assert_eq!(store.stats(), (1, 0));
    }

    /// Bad magic (e.g. an unrelated file in the cache dir)
    /// triggers rebuild.
    #[test]
    fn bad_magic_triggers_rebuild() {
        let tmp = TempDir::new().unwrap();
        let n_hidden = 64;
        let n_intermediate = 64;
        let (src_packed, src_scale) = make_synthetic_down(n_hidden, n_intermediate);

        // Plant a garbage file at the expected path with the right
        // size but wrong magic.
        let path = tmp.path().join("L0007_E0042.cxd");
        std::fs::create_dir_all(tmp.path()).unwrap();
        let n_groups = n_hidden / GROUP_SIZE;
        let total = HEADER_BYTES + n_intermediate * n_hidden / 2 + n_intermediate * n_groups * 2;
        std::fs::write(&path, vec![0xFFu8; total]).unwrap();

        let mut store = TransposedDownStore::open(tmp.path()).unwrap();
        store
            .get_or_build(7, 42, &src_packed, &src_scale, n_hidden, n_intermediate)
            .unwrap();
        // Magic mismatch → rebuild.
        assert_eq!(store.stats(), (1, 0));
    }
}
