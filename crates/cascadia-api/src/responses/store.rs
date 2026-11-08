//! The item cache behind `store` and `item_reference` (spec D10).
//!
//! The output items of every response whose `store` is not `false` are kept
//! here, in memory only, so a later request's `item_reference` resolves: the
//! default `@ai-sdk/openai` multi-step loop sends the previous turn's items
//! back by id. The cache is bounded by item count and by bytes (each item
//! charged its serialized size), evicting the oldest first; an item larger
//! than the whole byte cap is not kept.
//!
//! Isolation between callers rests on the ids being unguessable: every item
//! id is `Uuid::new_v4` based (see `output::new_id`), and there is no
//! per-caller scoping.

use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// A stored item larger than the byte budget the lookup was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TooLarge {
    /// The item's stored size.
    pub size: usize,
}

#[derive(Default)]
struct Inner {
    /// id -> (item, its serialized size). Shared so a lookup clones a
    /// pointer under the lock, not the item.
    map: HashMap<String, (Arc<Value>, usize)>,
    /// Insertion order, oldest first, for eviction.
    order: VecDeque<String>,
    /// Sum of the stored sizes.
    bytes: usize,
}

/// Output items by id, bounded by [`Self::MAX_ITEMS`] and [`Self::MAX_BYTES`].
pub(crate) struct ItemCache {
    inner: Mutex<Inner>,
    max_items: usize,
    max_bytes: usize,
}

impl Default for ItemCache {
    fn default() -> Self {
        Self::with_limits(Self::MAX_ITEMS, Self::MAX_BYTES)
    }
}

impl ItemCache {
    pub const MAX_ITEMS: usize = 16384;
    pub const MAX_BYTES: usize = 32 << 20;

    pub fn with_limits(items: usize, bytes: usize) -> Self {
        Self {
            inner: Mutex::default(),
            max_items: items,
            max_bytes: bytes,
        }
    }

    /// Keep `items` by their `id`, replacing an item stored under the same
    /// id; an item with no string `id`, or larger than the byte cap, is
    /// skipped. The oldest items are evicted until both caps hold.
    pub fn insert_all(&self, items: &[Value]) {
        // Size and copy the items outside the lock.
        let entries: Vec<(String, Arc<Value>, usize)> = items
            .iter()
            .filter_map(|item| {
                let id = item.get("id")?.as_str()?;
                let size = serde_json::to_vec(item).map(|b| b.len()).ok()?;
                (size <= self.max_bytes).then(|| (id.to_owned(), Arc::new(item.clone()), size))
            })
            .collect();
        if entries.is_empty() {
            return;
        }
        // Replaced and evicted items are dropped after the lock is released.
        let mut dropped: Vec<Arc<Value>> = Vec::new();
        {
            let mut inner = self.inner.lock();
            for (id, item, size) in entries {
                if let Some((old, old_size)) = inner.map.remove(&id) {
                    inner.bytes -= old_size;
                    inner.order.retain(|k| *k != id);
                    dropped.push(old);
                }
                inner.order.push_back(id.clone());
                inner.map.insert(id, (item, size));
                inner.bytes += size;
            }
            while inner.map.len() > self.max_items || inner.bytes > self.max_bytes {
                let Some(oldest) = inner.order.pop_front() else {
                    break;
                };
                if let Some((old, size)) = inner.map.remove(&oldest) {
                    inner.bytes -= size;
                    dropped.push(old);
                }
            }
        }
        drop(dropped);
    }

    /// The item stored under `id` and its stored size, if that size fits
    /// `budget`; over it, [`TooLarge`] without cloning the item. The item is
    /// copied after the lock is released.
    pub fn get_within(&self, id: &str, budget: usize) -> Result<Option<(Value, usize)>, TooLarge> {
        let found = {
            let inner = self.inner.lock();
            match inner.map.get(id) {
                None => return Ok(None),
                Some(&(_, size)) if size > budget => return Err(TooLarge { size }),
                Some((item, size)) => (Arc::clone(item), *size),
            }
        };
        let (item, size) = found;
        Ok(Some((Value::clone(&item), size)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(id: &str) -> Value {
        json!({"id": id, "type": "message", "content": []})
    }

    fn size(v: &Value) -> usize {
        serde_json::to_vec(v).unwrap().len()
    }

    fn has(cache: &ItemCache, id: &str) -> bool {
        cache.get_within(id, usize::MAX).unwrap().is_some()
    }

    #[test]
    fn inserted_items_are_found_by_id() {
        let cache = ItemCache::default();
        let a = item("msg_a");
        cache.insert_all(&[a.clone(), json!({"type": "message"})]);
        assert_eq!(
            cache.get_within("msg_a", usize::MAX),
            Ok(Some((a.clone(), size(&a))))
        );
        assert_eq!(cache.get_within("msg_b", usize::MAX), Ok(None));
        assert_eq!(
            cache.inner.lock().map.len(),
            1,
            "an item with no id is skipped"
        );
    }

    #[test]
    fn the_oldest_item_is_evicted_past_the_count_cap() {
        let cache = ItemCache::with_limits(2, 1 << 20);
        cache.insert_all(&[item("a"), item("b")]);
        cache.insert_all(&[item("c")]);
        assert!(!has(&cache, "a"));
        assert!(has(&cache, "b") && has(&cache, "c"));
    }

    #[test]
    fn the_oldest_items_are_evicted_past_the_byte_cap() {
        let one = size(&item("a"));
        let cache = ItemCache::with_limits(100, 2 * one);
        cache.insert_all(&[item("a"), item("b"), item("c")]);
        assert!(!has(&cache, "a"));
        assert!(has(&cache, "b") && has(&cache, "c"));
        assert_eq!(cache.inner.lock().bytes, 2 * one);
    }

    #[test]
    fn an_item_over_the_byte_cap_is_not_kept() {
        let one = size(&item("a"));
        let cache = ItemCache::with_limits(100, one);
        let big = json!({"id": "big", "text": "x".repeat(one)});
        cache.insert_all(&[item("a"), big]);
        assert!(!has(&cache, "big"));
        assert!(has(&cache, "a"), "a skipped item evicts nothing");
    }

    #[test]
    fn the_same_id_is_replaced_without_double_counting() {
        let cache = ItemCache::with_limits(2, 1 << 20);
        let newer = json!({"id": "a", "type": "message", "content": ["newer"]});
        cache.insert_all(&[item("a"), item("b")]);
        cache.insert_all(std::slice::from_ref(&newer));
        let inner = cache.inner.lock();
        assert_eq!(inner.map.len(), 2);
        assert_eq!(inner.order.len(), 2);
        assert_eq!(inner.bytes, size(&item("b")) + size(&newer));
        drop(inner);
        assert_eq!(cache.get_within("a", usize::MAX).unwrap().unwrap().0, newer);
        assert!(has(&cache, "b"), "a replacement is not a new item");
    }

    #[test]
    fn a_lookup_over_its_budget_is_too_large() {
        let cache = ItemCache::default();
        let a = item("a");
        let n = size(&a);
        cache.insert_all(std::slice::from_ref(&a));
        assert_eq!(cache.get_within("a", n - 1), Err(TooLarge { size: n }));
        assert_eq!(cache.get_within("a", n), Ok(Some((a, n))));
    }
}
