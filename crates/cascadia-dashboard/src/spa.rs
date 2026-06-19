//! Embedded SPA assets, gated behind the `embed-spa` feature.
//!
//! `rust_embed::Embed` derives a virtual filesystem from the literal
//! contents of `web/dist/` at compile time. The `serve` handler matches
//! the request path against the embedded files; unknown paths fall back
//! to `index.html` so client-side routes (e.g. `/chat`) resolve to the
//! SPA shell rather than 404.

use axum::body::Body;
use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use rust_embed::Embed;

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
    serve_path("index.html")
}

async fn serve(uri: Uri) -> Response {
    let path = uri.path();
    // Reserved API namespaces: a miss here is a real 404, not a client-side
    // SPA route — otherwise a typo'd `/v1/*` or `/api/*` GET would be masked
    // as 200 + the SPA shell.
    if path.starts_with("/v1/") || path.starts_with("/api/") || path == "/health" {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    serve_path(path.trim_start_matches('/'))
}

fn serve_path(path: &str) -> Response {
    // Direct hit on a built asset (e.g. /assets/index-abc123.js).
    if let Some(file) = SpaAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.data.into_owned()))
            .unwrap();
    }
    // SPA fallback: anything that's not a real file becomes index.html so
    // client-side router routes resolve. /api/* is already claimed by the
    // JSON endpoints above this router in the merge order, so it never
    // reaches this branch.
    match SpaAssets::get("index.html") {
        Some(file) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(Body::from(file.data.into_owned()))
            .unwrap(),
        None => (StatusCode::NOT_FOUND, "SPA assets missing").into_response(),
    }
}
