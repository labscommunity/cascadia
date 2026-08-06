//! Process-global K3 knob overrides, seeded once from `SparseMoEBuilderConfig`.
//!
//! Every call site keeps its `CASCADIA_K3_*` read as the fallback, so an
//! unseeded process behaves exactly as before. Process-global because what it
//! feeds already is: the hot-path flags cache in `OnceLock`s to stay
//! branch-cheap, and the env vars they replace are process-wide too.

use std::sync::OnceLock;

/// Explicit knob values from the builder config. `None` defers to the env var.
#[derive(Debug, Default, Clone, Copy)]
pub struct Overrides {
    pub max_seq: Option<usize>,
    pub prefix_cache_bytes: Option<u64>,
    pub read: Option<bool>,
    pub prefetch: Option<bool>,
    pub simd: Option<bool>,
    pub autopin: Option<bool>,
    pub pin_bytes: Option<u64>,
}

static OVERRIDES: OnceLock<Overrides> = OnceLock::new();

const NONE: Overrides = Overrides {
    max_seq: None,
    prefix_cache_bytes: None,
    read: None,
    prefetch: None,
    simd: None,
    autopin: None,
    pin_bytes: None,
};

/// Seed the overrides. First call wins; later calls return `false` and change
/// nothing — a second K3 engine in one process inherits the first's knobs,
/// which is also what the env vars would have done.
pub fn seed(o: Overrides) -> bool {
    OVERRIDES.set(o).is_ok()
}

pub fn get() -> &'static Overrides {
    OVERRIDES.get().unwrap_or(&NONE)
}
