//! Dashboard HTTP routes for visualizing a Cascadia cluster.
//!
//! What this crate adds on top of `cascadia-api`:
//!
//! * `GET /api/topology` — current nodes + measured edges (JSON)
//! * `GET /api/stats` — coarse runtime counters (in-flight requests,
//!   tokens generated). Read from the shared [`cascadia_types::ApiStats`]
//!   the OpenAI server bumps on the chat hot path.
//! * `embed-spa` feature — when on, serves the built Vite SPA from
//!   `crates/cascadia-dashboard/web/dist` at `/`, including a fallback to
//!   `index.html` for client-side routes. When off, `/` serves a small
//!   built-in pointer page saying the UI isn't in this build and how to
//!   get it — a bare 404 on `/` reads as "the server is broken" when the
//!   startup log just advertised a dashboard (that's exactly how the
//!   first source-build user experienced it).
//!
//! Why a separate crate rather than expanding `cascadia-api`:
//! `cascadia-api` is the OpenAI-compatible surface and shouldn't grow a
//! dependency on `cascadia-topology` or bundled static assets. Keeping
//! the dashboard separable also leaves room for shipping or hiding it
//! independently of the OpenAI API.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use cascadia_topology::{NodeInfo, Topology};
use cascadia_types::ApiStats;
use serde::Serialize;

#[cfg(feature = "embed-spa")]
mod spa;

/// Whether the SPA route is compiled into this build. Exported so the host
/// process can log what `/` will actually serve — evaluated HERE, in the
/// crate that owns the feature, so the answer stays correct even if a
/// dependency other than the host enabled `embed-spa` through feature
/// unification.
///
/// This is a build fact, not a promise that assets are readable right now:
/// `rust-embed` only bakes them into the binary for release builds, and
/// reads `web/dist` from disk per request in debug ones. `spa::shell` warns
/// when that read comes up empty.
pub const SPA_EMBEDDED: bool = cfg!(feature = "embed-spa");

/// Shared state for the dashboard routes.
///
/// `topology` is the same `Topology` handle the discovery loop writes
/// to (cheap to clone — it's `Arc`-backed inside). `stats` is the SAME
/// [`ApiStats`] handle the OpenAI-compat server bumps on the chat hot
/// path, so `/api/stats` reflects live request/token activity rather
/// than the placeholder zeros it returned before. `max_concurrent` is a
/// static config value (the admission ceiling), so a plain `u64`.
#[derive(Clone)]
pub struct DashboardState {
    pub topology: Topology,
    pub stats: Arc<ApiStats>,
    pub max_concurrent: u64,
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

    // No SPA in this build: `/` gets a pointer page instead of axum's
    // empty-body 404. Only `/` — other unknown paths keep 404ing, since
    // there are no client-side routes to resolve without the SPA.
    #[cfg(not(feature = "embed-spa"))]
    let r = r.route("/", get(placeholder_index));

    r.with_state(state)
}

/// Served at `/` when the SPA is not embedded. Self-contained (inline
/// CSS, no assets) so it renders from any build with zero extra routes.
#[cfg(not(feature = "embed-spa"))]
const PLACEHOLDER_HTML: &str = include_str!("placeholder.html");

#[cfg(not(feature = "embed-spa"))]
async fn placeholder_index() -> axum::response::Html<&'static str> {
    axum::response::Html(PLACEHOLDER_HTML)
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
        max_concurrent: state.max_concurrent,
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
            stats: Arc::new(ApiStats::default()),
            max_concurrent: 16,
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

    // Regression tests for the "downloaded main, saw `API + dashboard
    // serving`, got an empty 404 on /" report: a no-SPA build must say
    // something useful at `/`, and must NOT grow an SPA-style fallback
    // (unknown paths keep 404ing — there are no client routes to serve).

    #[cfg(not(feature = "embed-spa"))]
    #[tokio::test]
    async fn root_serves_pointer_page_when_spa_not_embedded() {
        let app = make_router(state_with_two_nodes());
        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response.headers()[axum::http::header::CONTENT_TYPE].clone();
        assert!(ct.to_str().unwrap().starts_with("text/html"));
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let page = std::str::from_utf8(&body).unwrap();
        // The page's whole job: name the feature and the SPA build step.
        assert!(page.contains("dashboard-embed"));
        assert!(page.contains("npm run build"));
    }

    #[cfg(not(feature = "embed-spa"))]
    #[tokio::test]
    async fn unknown_paths_still_404_without_the_spa() {
        let app = make_router(state_with_two_nodes());
        let response = app
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    // The embedded build's half of the same contract. A missing asset used to
    // come back as 200 + the SPA shell, so a browser asking for a hashed
    // `.js` got `text/html`, refused it on MIME grounds, and rendered a blank
    // dashboard — with every response a 200 and nothing in the log.

    #[cfg(feature = "embed-spa")]
    #[tokio::test]
    async fn missing_asset_404s_instead_of_serving_the_shell() {
        let app = make_router(state_with_two_nodes());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/assets/bogus.js")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "embed-spa")]
    #[tokio::test]
    async fn client_routes_still_resolve_to_the_shell() {
        let app = make_router(state_with_two_nodes());
        let response = app
            .oneshot(Request::builder().uri("/chat").body(Body::empty()).unwrap())
            .await
            .unwrap();
        // `/chat` is a client-side route with no file behind it: the SPA
        // router resolves it, so the shell is the correct answer.
        assert_eq!(response.status(), StatusCode::OK);
        let ct = &response.headers()[axum::http::header::CONTENT_TYPE];
        assert!(ct.to_str().unwrap().starts_with("text/html"));
    }

    #[cfg(feature = "embed-spa")]
    #[tokio::test]
    async fn reserved_api_paths_404_rather_than_masking_as_the_shell() {
        for uri in ["/v1/nope", "/api/nope", "/health"] {
            let app = make_router(state_with_two_nodes());
            let response = app
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{uri} must 404, not return the SPA shell"
            );
        }
    }
}
