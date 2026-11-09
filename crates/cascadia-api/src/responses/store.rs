//! The item cache behind `store` and `item_reference` (spec D10).
//!
//! The output items of every response whose `store` is not `false` are kept
//! here, in memory only, so a later request's `item_reference` resolves: the
//! default `@ai-sdk/openai` multi-step loop sends the previous turn's items
//! back by id. The cache is bounded by item count and by bytes (each item
//! charged its serialized size), evicting the least recently used first: a
//! lookup refreshes an item, so a conversation's earliest items, which that
//! loop references again every turn, stay while they are in use. An item
//! larger than the whole byte cap is not kept.
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
pub struct TooLarge {
    /// The item's stored size.
    pub size: usize,
}

struct Entry {
    /// Shared so a lookup clones a pointer under the lock, not the item.
    item: Arc<Value>,
    /// The item's serialized size.
    size: usize,
    /// The stamp of the item's latest use; its one live `order` entry.
    stamp: u64,
}

#[derive(Default)]
struct Inner {
    map: HashMap<String, Entry>,
    /// (stamp, id) in order of use, least recent first, for eviction. A use
    /// pushes a new entry rather than moving the old one, so an entry whose
    /// stamp is no longer its item's is stale and skipped.
    order: VecDeque<(u64, String)>,
    /// The next stamp to hand out.
    next_stamp: u64,
    /// Sum of the stored sizes.
    bytes: usize,
}

impl Inner {
    /// A fresh stamp for `id`, queued as its most recent use.
    fn stamp(&mut self, id: String) -> u64 {
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        self.order.push_back((stamp, id));
        stamp
    }

    /// Remove the stale `order` entries once they outnumber the live ones,
    /// so the queue stays within a constant factor of the map and each use
    /// costs amortized O(1). Their ids are moved into `stale`, and the old
    /// queue's buffer is returned, for the caller to drop after the lock is
    /// released.
    fn compact(&mut self, stale: &mut Vec<String>) -> Option<VecDeque<(u64, String)>> {
        if self.order.len() <= 2 * self.map.len() + 64 {
            return None;
        }
        let mut queued =
            std::mem::replace(&mut self.order, VecDeque::with_capacity(self.map.len()));
        for (stamp, id) in queued.drain(..) {
            if self.map.get(&id).is_some_and(|e| e.stamp == stamp) {
                self.order.push_back((stamp, id));
            } else {
                stale.push(id);
            }
        }
        Some(queued)
    }
}

/// Output items by id, bounded by [`Self::MAX_ITEMS`] and [`Self::MAX_BYTES`].
pub struct ItemCache {
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
    /// skipped. The least recently used items are evicted until both caps
    /// hold.
    pub fn insert_all(&self, items: &[Value]) {
        // Size and copy the items, and the ids the map and the use queue
        // keep, outside the lock.
        let entries: Vec<(String, String, Arc<Value>, usize)> = items
            .iter()
            .filter_map(|item| {
                let id = item.get("id")?.as_str()?;
                let size = serde_json::to_vec(item).map(|b| b.len()).ok()?;
                (size <= self.max_bytes)
                    .then(|| (id.to_owned(), id.to_owned(), Arc::new(item.clone()), size))
            })
            .collect();
        if entries.is_empty() {
            return;
        }
        // Replaced and evicted items, and stale queue ids, are dropped after
        // the lock is released.
        let mut dropped: Vec<Arc<Value>> = Vec::new();
        let mut stale: Vec<String> = Vec::new();
        let spent;
        {
            let mut guard = self.inner.lock();
            // One borrow of the guard, so its fields borrow apart.
            let inner = &mut *guard;
            for (id, queued, item, size) in entries {
                // A replaced item's queue entry goes stale with its stamp.
                let stamp = inner.stamp(queued);
                let entry = Entry { item, size, stamp };
                match inner.map.get_mut(&id) {
                    Some(e) => {
                        let old = std::mem::replace(e, entry);
                        inner.bytes -= old.size;
                        dropped.push(old.item);
                        // The map keeps its own copy of the key.
                        stale.push(id);
                    }
                    None => {
                        inner.map.insert(id, entry);
                    }
                }
                inner.bytes += size;
            }
            while inner.map.len() > self.max_items || inner.bytes > self.max_bytes {
                let Some((stamp, id)) = inner.order.pop_front() else {
                    break;
                };
                if inner.map.get(&id).is_some_and(|e| e.stamp == stamp) {
                    let old = inner.map.remove(&id).expect("just found");
                    inner.bytes -= old.size;
                    dropped.push(old.item);
                }
                stale.push(id);
            }
            spent = inner.compact(&mut stale);
        }
        drop(dropped);
        drop(stale);
        drop(spent);
    }

    /// The item stored under `id` and its stored size, if that size fits
    /// `budget`, marking it the most recently used; over it, [`TooLarge`]
    /// without cloning or refreshing the item. The item is copied after the
    /// lock is released.
    pub fn get_within(&self, id: &str, budget: usize) -> Result<Option<(Value, usize)>, TooLarge> {
        // The queue's copy of the id is made, and stale ids are dropped,
        // outside the lock.
        let queued = id.to_owned();
        let mut stale: Vec<String> = Vec::new();
        let spent;
        let found = {
            let mut inner = self.inner.lock();
            let (item, size) = match inner.map.get(id) {
                None => return Ok(None),
                Some(e) if e.size > budget => return Err(TooLarge { size: e.size }),
                Some(e) => (Arc::clone(&e.item), e.size),
            };
            let stamp = inner.stamp(queued);
            if let Some(e) = inner.map.get_mut(id) {
                e.stamp = stamp;
            }
            spent = inner.compact(&mut stale);
            (item, size)
        };
        drop(stale);
        drop(spent);
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
    fn a_lookup_keeps_an_item_past_the_count_cap() {
        let cache = ItemCache::with_limits(2, 1 << 20);
        cache.insert_all(&[item("a")]);
        cache.insert_all(&[item("b")]);
        assert!(has(&cache, "a"));
        cache.insert_all(&[item("c")]);
        assert_eq!(cache.get_within("b", usize::MAX), Ok(None));
        assert!(has(&cache, "a") && has(&cache, "c"));
    }

    #[test]
    fn a_lookup_keeps_an_item_past_the_byte_cap() {
        let one = size(&item("a"));
        let cache = ItemCache::with_limits(100, 2 * one);
        cache.insert_all(&[item("a"), item("b")]);
        assert!(has(&cache, "a"));
        cache.insert_all(&[item("c")]);
        assert_eq!(cache.get_within("b", usize::MAX), Ok(None));
        assert!(has(&cache, "a") && has(&cache, "c"));
        assert_eq!(cache.inner.lock().bytes, 2 * one);
    }

    #[test]
    fn repeated_lookups_keep_the_order_queue_bounded() {
        let cache = ItemCache::with_limits(4, 1 << 20);
        cache.insert_all(&[item("a"), item("b")]);
        for _ in 0..1000 {
            assert!(has(&cache, "a"));
            cache.insert_all(&[item("b")]);
        }
        let inner = cache.inner.lock();
        assert_eq!(inner.map.len(), 2);
        assert!(inner.order.len() <= 2 * inner.map.len() + 64);
    }

    /// One batch over the count cap keeps only its newest items.
    #[test]
    fn a_batch_over_the_count_cap_keeps_its_newest() {
        let cache = ItemCache::with_limits(2, 1 << 20);
        cache.insert_all(&[item("a"), item("b"), item("c"), item("d")]);
        assert!(!has(&cache, "a") && !has(&cache, "b"));
        assert!(has(&cache, "c") && has(&cache, "d"));
        assert_eq!(cache.inner.lock().map.len(), 2);
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
        let live = inner
            .order
            .iter()
            .filter(|(stamp, id)| inner.map.get(id).is_some_and(|e| e.stamp == *stamp))
            .count();
        assert_eq!(live, 2, "one live queue entry per item");
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
