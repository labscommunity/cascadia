//! Dashboard HTTP routes for visualizing a Cascadia cluster.
//!
//! What this crate adds on top of `cascadia-api`:
//!
//! * `GET /api/topology` — current nodes + measured edges (JSON)
//! * `GET /api/stats` — coarse runtime counters (in-flight requests,
//!   tokens generated). Read from the shared [`cascadia_types::ApiStats`]
//!   the OpenAI server bumps on the chat hot path.
//! * `GET /api/fleet/telemetry` — a JSON file another process on this box
//!   keeps current (the Inkling fleet's `beacon.py` on rank 0: every box's
//!   load figures and stage profiles), passed through as-is. The API port
//!   is the only way into some fleets, so their telemetry leaves through it.
//!   File named by `CASCADIA_FLEET_TELEMETRY_FILE`; a missing file is a 200
//!   with an `error` field, not a failure — most deployments have none.
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

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
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
    make_router_with_telemetry_file(state, telemetry_path())
}

/// [`make_router`] with the fleet telemetry file named by the caller instead
/// of the environment. The environment is read once, in `make_router`, not
/// per request: it cannot change under a running server, and the route tests
/// (which run in parallel in one process) can each use a file of their own.
fn make_router_with_telemetry_file(state: DashboardState, telemetry_file: PathBuf) -> Router {
    // Merge SPA routes (when `embed-spa` is on) before `.with_state` so
    // both sub-routers carry the same `Router<DashboardState>` state type
    // when axum unifies them. After `.with_state(state)` the requirement
    // is fulfilled and we return a plain `Router<()>`.
    let r = Router::new()
        .route("/api/topology", get(get_topology))
        .route("/api/stats", get(get_stats))
        // An explicit route, so it is matched before the SPA fallback (which
        // 404s every unknown `/api/*`) in an `embed-spa` build as well.
        .route(
            "/api/fleet/telemetry",
            get(move || {
                let path = telemetry_file.clone();
                async move { fleet_telemetry_response(&path).await }
            }),
        );

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

/// Env var naming the fleet telemetry file served at `/api/fleet/telemetry`.
pub const FLEET_TELEMETRY_FILE_ENV: &str = "CASCADIA_FLEET_TELEMETRY_FILE";

/// Where the Inkling fleet's `beacon.py` writes it on rank 0.
pub const DEFAULT_FLEET_TELEMETRY_FILE: &str = "/run/cascadia-inkling/telemetry.json";

/// The writer keeps the file under ~400 KB; anything beyond this is not that
/// file, and is not read into memory on the serving path.
const FLEET_TELEMETRY_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// The fleet telemetry file this process serves: `CASCADIA_FLEET_TELEMETRY_FILE`,
/// else the beacon's default.
pub fn telemetry_path() -> PathBuf {
    telemetry_path_from(std::env::var_os(FLEET_TELEMETRY_FILE_ENV))
}

/// [`telemetry_path`] on an explicit value (unset or empty = the default).
fn telemetry_path_from(value: Option<std::ffi::OsString>) -> PathBuf {
    match value {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => PathBuf::from(DEFAULT_FLEET_TELEMETRY_FILE),
    }
}

/// The file's bytes, provided it is a regular-sized JSON document.
async fn read_fleet_telemetry(path: &Path) -> Result<Vec<u8>, &'static str> {
    use tokio::io::AsyncReadExt;
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|_| "no telemetry file")?;
    let mut bytes = Vec::new();
    // One byte past the cap tells "too large" apart from "exactly the cap"
    // without trusting the file's metadata (it is replaced once a second).
    file.take(FLEET_TELEMETRY_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|_| "no telemetry file")?;
    if bytes.len() as u64 > FLEET_TELEMETRY_MAX_BYTES {
        return Err("telemetry file too large");
    }
    // Served under `application/json`, so it has to be JSON: a path that
    // names some other file must not hand that file's contents out.
    serde_json::from_slice::<serde::de::IgnoredAny>(&bytes)
        .map_err(|_| "telemetry file is not JSON")?;
    Ok(bytes)
}

/// `GET /api/fleet/telemetry`: the file as-is, or `{"error", "path"}` — both
/// 200, so a poller tells "this box has no fleet telemetry" from "this build
/// has no such route" (404) without parsing an error page.
async fn fleet_telemetry_response(path: &Path) -> Response {
    let body = match read_fleet_telemetry(path).await {
        Ok(bytes) => bytes,
        Err(error) => serde_json::json!({
            "error": error,
            "path": path.to_string_lossy(),
        })
        .to_string()
        .into_bytes(),
    };
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            // Rewritten once a second: a cached copy is a wrong answer.
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
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

    // `/api/fleet/telemetry`. The file comes from the router's constructor,
    // never from the environment: env vars are process-global and these
    // tests run in parallel. NOT feature-gated — the route has to answer
    // in the `embed-spa` build too, where the SPA fallback 404s every
    // `/api/*` path it is handed.

    /// A file of this test's own in the temp dir, removed on drop.
    struct TempFile(PathBuf);

    impl TempFile {
        fn new(name: &str, contents: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cascadia-dashboard-test-{}-{name}",
                std::process::id()
            ));
            std::fs::write(&path, contents).unwrap();
            TempFile(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    async fn get_telemetry(file: &Path) -> (StatusCode, String, String, Vec<u8>) {
        let app = make_router_with_telemetry_file(state_with_two_nodes(), file.to_path_buf());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/fleet/telemetry")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let header = |name| {
            response
                .headers()
                .get(name)
                .map(|v: &axum::http::HeaderValue| v.to_str().unwrap().to_string())
                .unwrap_or_default()
        };
        let (ct, cc) = (header(header::CONTENT_TYPE), header(header::CACHE_CONTROL));
        let body = to_bytes(response.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap();
        (status, ct, cc, body.to_vec())
    }

    #[test]
    fn telemetry_path_is_the_env_value_else_the_beacon_default() {
        assert_eq!(
            telemetry_path_from(None),
            PathBuf::from("/run/cascadia-inkling/telemetry.json")
        );
        assert_eq!(
            telemetry_path_from(Some("".into())),
            PathBuf::from(DEFAULT_FLEET_TELEMETRY_FILE)
        );
        assert_eq!(
            telemetry_path_from(Some("/tmp/x/t.json".into())),
            PathBuf::from("/tmp/x/t.json")
        );
    }

    #[tokio::test]
    async fn fleet_telemetry_serves_the_file_as_is() {
        // Byte-for-byte: the odd spacing must survive (no re-serialization).
        let contents =
            br#"{"t": 12.5,"fleet":"inkling",  "ranks":{"3":{"rt":1.0,"sys":{"cpu":0.5}}}}"#;
        let file = TempFile::new("as-is.json", contents);
        let (status, ct, cc, body) = get_telemetry(&file.0).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ct, "application/json");
        assert_eq!(cc, "no-store");
        assert_eq!(body, contents);
    }

    #[tokio::test]
    async fn fleet_telemetry_without_a_file_is_a_200_that_says_so() {
        let missing = std::env::temp_dir().join(format!(
            "cascadia-dashboard-test-{}-there-is-no-such-file.json",
            std::process::id()
        ));
        let (status, ct, _, body) = get_telemetry(&missing).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ct, "application/json");
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"], "no telemetry file");
        assert_eq!(v["path"], missing.to_string_lossy().as_ref());
    }

    #[tokio::test]
    async fn fleet_telemetry_never_hands_out_a_file_that_is_not_json() {
        let file = TempFile::new("not-json.txt", b"root:x:0:0:secret\n");
        let (status, ct, _, body) = get_telemetry(&file.0).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ct, "application/json");
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"], "telemetry file is not JSON");
        assert!(!String::from_utf8_lossy(&body).contains("secret"));
    }

    #[tokio::test]
    async fn fleet_telemetry_is_capped_at_4_mib() {
        // Valid JSON on both sides of the cap, so only the size decides.
        let json_string = |len: usize| {
            let mut s = vec![b'a'; len];
            s[0] = b'"';
            s[len - 1] = b'"';
            s
        };
        let at_cap = TempFile::new("at-cap.json", &json_string(4 * 1024 * 1024));
        let (_, _, _, body) = get_telemetry(&at_cap.0).await;
        assert_eq!(body.len(), 4 * 1024 * 1024);

        let over = TempFile::new("over-cap.json", &json_string(4 * 1024 * 1024 + 1));
        let (status, _, _, body) = get_telemetry(&over.0).await;
        assert_eq!(status, StatusCode::OK);
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"], "telemetry file too large");
    }

    #[tokio::test]
    async fn public_router_has_the_fleet_telemetry_route() {
        // `make_router` is what the host process mounts next to the OpenAI
        // routes. Whatever file the environment of this test run names (on a
        // build box: none), the route answers 200 + JSON — never the 404 an
        // unrouted `/api/*` path gets.
        let app = make_router(state_with_two_nodes());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/fleet/telemetry")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let body = to_bytes(response.into_body(), 8 * 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice::<Value>(&body).unwrap();
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
