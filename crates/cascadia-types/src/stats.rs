//! Live API request/token counters.
//!
//! Lives in `cascadia-types` (the zero-dep value-type crate) rather than
//! `cascadia-api` so both the HTTP server (which writes them) and the
//! dashboard (which reads them via `/api/stats`) can share the handle
//! without the visualization crate pulling in the whole OpenAI-server
//! crate. All fields are `Relaxed` atomics — coarse monotonic gauges,
//! not synchronization points.

use std::sync::atomic::AtomicU64;

#[derive(Default, Debug)]
pub struct ApiStats {
    /// Cumulative chat-completion requests admitted (past the permit gate).
    pub requests_total: AtomicU64,
    /// Requests currently executing (incremented on admit, decremented when
    /// the response/stream is fully drained or dropped).
    pub requests_in_flight: AtomicU64,
    /// Cumulative model tokens emitted across all requests.
    pub tokens_total: AtomicU64,
}
