//! Dashboard HTTP routes for visualizing a Cascadia cluster.
//!
//! What this crate adds on top of `cascadia-api`:
//!
//! * `GET /api/topology` — current nodes + measured edges (JSON)
//! * `GET /api/stats` — coarse runtime counters (in-flight requests,
//!   tokens generated). Filled in by the host process via [`DashboardStats`].
//! * `GET /api/events` — server-sent events for live updates (added in a
//!   follow-up; the placeholder route returns a one-shot welcome event).
//! * `embed-spa` feature — when on, serves the built Vite SPA from
//!   `crates/cascadia-dashboard/web/dist` at `/`, including a fallback to
//!   `index.html` for client-side routes.
//!
//! Why a separate crate rather than expanding `cascadia-api`:
//! `cascadia-api` is the OpenAI-compatible surface and shouldn't grow a
//! dependency on `cascadia-topology` or bundled static assets. Keeping
//! the dashboard separable also leaves room for shipping or hiding it
//! independently of the OpenAI API.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use cascadia_topology::{NodeInfo, Topology};
use serde::Serialize;

#[cfg(feature = "embed-spa")]
mod spa;

/// Shared state for the dashboard routes.
///
/// `topology` is the same `Topology` handle the discovery loop writes
/// to (cheap to clone — it's `Arc`-backed inside). `stats` is a small
/// shared counter bag the host process bumps as requests flow through
/// the runner; see [`DashboardStats`].
#[derive(Clone)]
pub struct DashboardState {
    pub topology: Topology,
    pub stats: Arc<DashboardStats>,
}

/// Coarse runtime counters surfaced by `GET /api/stats`.
///
/// All fields are atomic so the host can update them from request
/// handlers without a lock. The dashboard is best-effort: a stale read
/// or a small drift between counters is acceptable.
#[derive(Default)]
pub struct DashboardStats {
    /// Cumulative chat-completion requests admitted (including failed).
    pub requests_total: AtomicU64,
    /// Currently executing chat-completion requests.
    pub requests_in_flight: AtomicU64,
    /// Cumulative tokens emitted by the engine.
    pub tokens_total: AtomicU64,
    /// Configured concurrency ceiling (from `cascadia-api`'s semaphore).
    pub max_concurrent: AtomicU64,
}

impl DashboardStats {
    pub fn new(max_concurrent: u64) -> Arc<Self> {
        let s = Self::default();
        s.max_concurrent.store(max_concurrent, Ordering::Relaxed);
        Arc::new(s)
    }
}

/// Build the dashboard router. Combine with `cascadia-api`'s router in the
/// host process; see `crates/cascadia-cli` for the canonical composition.
pub fn make_router(state: DashboardState) -> Router {
    // Merge SPA routes (when `embed-spa` is on) before `.with_state` so
    // both sub-routers carry the same `Router<DashboardState>` state type
    // when axum unifies them. After `.with_state(state)` the requirement
    // is fulfilled and we return a plain `Router<()>`.
    let r = Router::new()
        .route("/api/topology", get(get_topology))
        .route("/api/stats", get(get_stats));

    #[cfg(feature = "embed-spa")]
    let r = r.merge(spa::router());

    r.with_state(state)
}

#[derive(Serialize)]
struct EdgeOut {
    src: String,
    dst: String,
    latency_ms: f64,
    bandwidth_mbps: f64,
    last_measured: f64,
}

#[derive(Serialize)]
struct TopologyResponse {
    nodes: Vec<NodeInfo>,
    edges: Vec<EdgeOut>,
}

async fn get_topology(State(state): State<DashboardState>) -> Json<TopologyResponse> {
    let nodes = state.topology.nodes();
    let edges = state
        .topology
        .edges()
        .into_iter()
        .map(|((src, dst), m)| EdgeOut {
            src,
            dst,
            latency_ms: m.latency_ms,
            bandwidth_mbps: m.bandwidth_mbps,
            last_measured: m.last_measured,
        })
        .collect();
    Json(TopologyResponse { nodes, edges })
}

#[derive(Serialize)]
struct StatsResponse {
    requests_total: u64,
    requests_in_flight: u64,
    tokens_total: u64,
    max_concurrent: u64,
}

async fn get_stats(State(state): State<DashboardState>) -> Json<StatsResponse> {
    Json(StatsResponse {
        requests_total: state.stats.requests_total.load(Ordering::Relaxed),
        requests_in_flight: state.stats.requests_in_flight.load(Ordering::Relaxed),
        tokens_total: state.stats.tokens_total.load(Ordering::Relaxed),
        max_concurrent: state.stats.max_concurrent.load(Ordering::Relaxed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use cascadia_topology::NodeInfo;
    use serde_json::Value;
    use tower::ServiceExt;

    fn state_with_two_nodes() -> DashboardState {
        let topology = Topology::new();
        topology.add_node(NodeInfo::new("alpha", "10.0.0.1", 8080));
        topology.add_node(NodeInfo::new("beta", "10.0.0.2", 8080));
        topology.measure("alpha", "beta", 1.5, 900.0);
        DashboardState {
            topology,
            stats: DashboardStats::new(16),
        }
    }

    #[tokio::test]
    async fn topology_endpoint_returns_nodes_and_edges() {
        let app = make_router(state_with_two_nodes());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/topology")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(v["edges"].as_array().unwrap().len(), 1);
        assert_eq!(v["edges"][0]["src"], "alpha");
        assert_eq!(v["edges"][0]["dst"], "beta");
        assert_eq!(v["edges"][0]["latency_ms"], 1.5);
    }

    #[tokio::test]
    async fn stats_endpoint_exposes_atomic_counters() {
        let state = state_with_two_nodes();
        state.stats.requests_total.fetch_add(7, Ordering::Relaxed);
        state.stats.tokens_total.fetch_add(2048, Ordering::Relaxed);
        let app = make_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["requests_total"], 7);
        assert_eq!(v["tokens_total"], 2048);
        assert_eq!(v["max_concurrent"], 16);
    }
}
