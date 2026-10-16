//! Rank 0's cache of encoded images, keyed on [`MediaInput::sha256`].
//!
//! A chat resends its earlier images every turn (the OpenAI API is
//! stateless), so without a cache each turn runs the vision encoder again on
//! every image in the conversation: about a second per image on CPU for
//! Pixtral, three for Gemma 4 E2B. The key is the hash of the decoded pixels
//! and dimensions, never the placeholder ids, which are the same for every
//! image of one size. Each front end owns one cache, so its preprocessing
//! settings are part of the key implicitly.
//!
//! Bounded by bytes, least recently used out first. `CASCADIA_IMAGE_CACHE_MB`
//! sets the bound per front end (default 256; 0 disables the cache).
//!
//! [`MediaInput::sha256`]: cascadia_types::MediaInput::sha256

use std::collections::{HashMap, VecDeque};

const DEFAULT_MB: usize = 256;

/// Encoded rows of one image plus whatever its front end needs alongside
/// them (a patch grid, a soft-token count).
pub struct EncodeCache<M> {
    cap_bytes: usize,
    bytes: usize,
    map: HashMap<[u8; 32], (Vec<f32>, M)>,
    order: VecDeque<[u8; 32]>,
    pub hits: u64,
    pub misses: u64,
}

impl<M: Clone> EncodeCache<M> {
    pub fn with_capacity_bytes(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            bytes: 0,
            map: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// The bound from `CASCADIA_IMAGE_CACHE_MB` (default 256 MiB).
    pub fn from_env() -> Self {
        let mb = std::env::var("CASCADIA_IMAGE_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MB);
        Self::with_capacity_bytes(mb << 20)
    }

    /// A copy of the cached rows and metadata for `key`, refreshing its
    /// recency.
    pub fn get(&mut self, key: &[u8; 32]) -> Option<(Vec<f32>, M)> {
        match self.map.get(key) {
            Some((rows, meta)) => {
                self.hits += 1;
                let out = (rows.clone(), meta.clone());
                if let Some(at) = self.order.iter().position(|k| k == key) {
                    self.order.remove(at);
                }
                self.order.push_back(*key);
                Some(out)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Remember `rows` for `key`, evicting the least recently used entries
    /// to stay within the bound. An image larger than the whole bound is not
    /// kept.
    pub fn put(&mut self, key: [u8; 32], rows: &[f32], meta: M) {
        let size = std::mem::size_of_val(rows);
        if size > self.cap_bytes || self.map.contains_key(&key) {
            return;
        }
        while self.bytes + size > self.cap_bytes {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some((r, _)) = self.map.remove(&old) {
                self.bytes -= std::mem::size_of_val(&r[..]);
            }
        }
        self.bytes += size;
        self.map.insert(key, (rows.to_vec(), meta));
        self.order.push_back(key);
    }

    /// Log the counters after a task's images were prepared, when any of
    /// them came from the cache.
    pub fn report(&self, served_from_cache: usize, images: usize) {
        if served_from_cache > 0 {
            tracing::info!(
                served_from_cache,
                images,
                cached = self.len(),
                cached_mb = self.bytes >> 20,
                hits = self.hits,
                misses = self.misses,
                "image encode cache hit"
            );
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(b: u8) -> [u8; 32] {
        [b; 32]
    }

    #[test]
    fn hit_returns_the_rows_and_metadata() {
        let mut c = EncodeCache::with_capacity_bytes(1 << 20);
        assert!(c.get(&key(1)).is_none());
        c.put(key(1), &[1.0, 2.0], (3u32, 4u32));
        assert_eq!(c.get(&key(1)), Some((vec![1.0, 2.0], (3, 4))));
        assert_eq!((c.hits, c.misses), (1, 1));
    }

    #[test]
    fn evicts_least_recently_used_within_the_byte_bound() {
        // Room for two 4-float images (16 bytes each).
        let mut c = EncodeCache::with_capacity_bytes(32);
        c.put(key(1), &[0.0; 4], ());
        c.put(key(2), &[0.0; 4], ());
        assert!(c.get(&key(1)).is_some()); // 1 is now the most recent
        c.put(key(3), &[0.0; 4], ());
        assert!(c.get(&key(2)).is_none(), "2 was least recently used");
        assert!(c.get(&key(1)).is_some() && c.get(&key(3)).is_some());
        assert_eq!(c.bytes(), 32);
    }

    #[test]
    fn zero_capacity_keeps_nothing() {
        let mut c = EncodeCache::with_capacity_bytes(0);
        c.put(key(1), &[1.0], ());
        assert!(c.is_empty());
    }
}
