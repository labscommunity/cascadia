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
    Router::new()
        .route("/", get(index))
        .route("/*path", get(serve))
}

async fn index() -> Response {
    serve_path("index.html")
}

async fn serve(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    serve_path(path)
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
