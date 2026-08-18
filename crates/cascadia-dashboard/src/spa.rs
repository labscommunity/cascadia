//! Embedded SPA assets, gated behind the `embed-spa` feature.
//!
//! `rust_embed::Embed` derives a virtual filesystem from the literal
//! contents of `web/dist/` at compile time. The `serve` handler matches
//! the request path against the embedded files; unknown *extension-less*
//! paths fall back to `index.html` so client-side routes (e.g. `/chat`)
//! resolve to the SPA shell rather than 404.

use axum::body::Body;
use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;
use tracing::warn;

#[derive(Embed)]
#[folder = "web/dist/"]
struct SpaAssets;

pub fn router() -> Router<crate::DashboardState> {
    // The SPA is a `fallback`, not a `GET /*path` catch-all. A catch-all
    // route is GET-only and method-blind, so a method-mismatched request
    // (e.g. `GET /v1/chat/completions`, a POST endpoint) or any unknown
    // path returned 200 + index.html instead of 405/404. As the fallback,
    // axum still returns 405 for a wrong method on a known route and only
    // hands genuinely-unmatched requests to the SPA.
    Router::new().route("/", get(index)).fallback(serve)
}

async fn index() -> Response {
    shell()
}

async fn serve(uri: Uri) -> Response {
    let path = uri.path();
    // Reserved API namespaces: a miss here is a real 404, not a client-side
    // SPA route — otherwise a typo'd `/v1/*` or `/api/*` GET would be masked
    // as 200 + the SPA shell.
    if path.starts_with("/v1/") || path.starts_with("/api/") || path == "/health" {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let rel = path.trim_start_matches('/');
    if let Some(response) = asset(rel) {
        return response;
    }
    // A miss on a path whose last segment carries an extension is a missing
    // *asset*, not a client route. Serving the shell there answers a
    // `<script src="/assets/index-abc123.js">` with `text/html`; the browser
    // refuses it on MIME grounds and the dashboard renders blank — while
    // every response is a 200 and nothing is logged. 404 says what happened.
    if rel.rsplit('/').next().is_some_and(|seg| seg.contains('.')) {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    // Extension-less miss: a client-side route (e.g. `/chat`) that only the
    // SPA router can resolve, so hand over the shell.
    shell()
}

/// A built asset served under its own MIME type (e.g. `/assets/index-abc123.js`),
/// or `None` when this build embeds no such file.
fn asset(path: &str) -> Option<Response> {
    let file = SpaAssets::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    Some(
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.data.into_owned()))
            .unwrap(),
    )
}

/// The SPA shell (`index.html`), served at `/` and for client-side routes.
fn shell() -> Response {
    match SpaAssets::get("index.html") {
        Some(file) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(Body::from(file.data.into_owned()))
            .unwrap(),
        // Unreachable in a release build: build.rs refuses to compile the
        // feature without `web/dist/index.html`, and the assets are baked in.
        // Debug builds are the live case — `rust-embed` reads `web/dist` from
        // disk per request there, so deleting or moving it after the build
        // empties the dashboard while the startup log still advertises one.
        // Say so, rather than 404ing in silence.
        None => {
            warn!(
                "dashboard SPA assets are not readable — this is a debug build, which reads \
                 crates/cascadia-dashboard/web/dist at request time; re-run `npm run build` \
                 there, or build with --release to embed the assets in the binary"
            );
            (StatusCode::NOT_FOUND, "SPA assets missing").into_response()
        }
    }
}
