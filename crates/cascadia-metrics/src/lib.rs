//! Prometheus metric inventory for the Cascadia serving stack (#16).
//!
//! One tiny leaf crate so every layer that emits metrics — the HTTP API,
//! the per-stage runner, the activation transport — can share the same
//! process-global registry without a dependency cycle. All metrics are
//! registered on the `prometheus` default registry the first time they
//! are touched (`LazyLock`), and `/metrics` in `cascadia-api` serves
//! [`encode_text`].
//!
//! Conventions:
//! * `cascadia_` prefix (the issue predates the tahoma→cascadia rename).
//! * Histograms end in `_seconds`, counters in `_total`.
//! * Label cardinality is bounded by construction: `endpoint` is the
//!   matched route template (never the raw URI), `model`/`device` are
//!   fixed per process at engine start, `reason` is a closed enum of
//!   rejection sites.
//!
//! Vec-family metrics (everything with labels) only appear in the
//! exposition output once a child has been observed — that's standard
//! Prometheus behavior, not a registration bug.

use std::sync::LazyLock;

use prometheus::{
    register_gauge_vec, register_histogram, register_histogram_vec, register_int_counter_vec,
    register_int_gauge, Encoder, GaugeVec, Histogram, HistogramVec, IntCounterVec, IntGauge,
    TextEncoder,
};

/// TTFT buckets: sub-100 ms warm-path decode up to cold multi-stage
/// pipeline prefill (tens of seconds).
pub const TTFT_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0];

/// Inter-token latency buckets: ~100 tok/s down to seconds-per-token
/// big-model decode.
pub const INTER_TOKEN_BUCKETS: &[f64] = &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

/// End-to-end duration buckets (whole generations and whole HTTP requests).
pub const DURATION_BUCKETS: &[f64] = &[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0];

/// Transport frame buckets: loopback microseconds up to a relayed WAN hop.
pub const TRANSPORT_BUCKETS: &[f64] = &[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0];

// --- Request-level (bumped by cascadia-api) ------------------------------

/// HTTP requests by matched route template and response status code.
/// Counts the API server's own routes; routes merged into the app AFTER
/// router construction (the CLI's embedded dashboard) are not covered.
/// For streaming (SSE) responses the status is the response HEAD — a
/// mid-stream engine failure still counts as 200 here and surfaces in
/// `cascadia_tasks_failed_total` instead.
pub static HTTP_REQUESTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_http_requests_total",
        "HTTP requests served, by matched route and status code.",
        &["endpoint", "status"]
    )
    .expect("register cascadia_http_requests_total")
});

/// Wall-clock HTTP request latency by matched route. For streaming (SSE)
/// responses this measures time to the response HEAD, not full body drain
/// — full generation time is `cascadia_generation_duration_seconds`.
pub static HTTP_REQUEST_DURATION_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "cascadia_http_request_duration_seconds",
        "HTTP request latency by matched route (time-to-head for streaming responses).",
        &["endpoint"],
        DURATION_BUCKETS.to_vec()
    )
    .expect("register cascadia_http_request_duration_seconds")
});

/// Generation requests currently executing (admitted past the permit gate,
/// response/stream not yet fully drained or dropped).
pub static INFLIGHT_TASKS: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "cascadia_inflight_tasks",
        "Generation requests currently executing."
    )
    .expect("register cascadia_inflight_tasks")
});

/// Requests rejected before generation started, by reason. Each reason
/// names the knob that fixes it, not the status code:
///
/// - `capacity` — permit gate full OR the engine's own pending queue full
///   (both 503). Add workers.
/// - `empty_prompt` (400), `multi_prompt` (unsupported batch form, 400) —
///   malformed request.
/// - `prompt_too_large` (413) — over the API's `max_prompt_bytes` body
///   limit. Raise that limit.
/// - `prompt_over_window` (413) — tokenizes past what the engine can window
///   for one request, e.g. a packed slot's KV region. Resize the slots or
///   widen the context. Split from `prompt_too_large` deliberately: same
///   status, different knob, different owner.
/// - `engine_unavailable` (503) — the engine is not loaded, or a pipeline
///   peer is unreachable. Bring the stage or its peer up.
/// - `engine_error` (500) — the engine itself failed the submit (backend,
///   I/O, or an attributed task failure). Read that node's logs.
/// - `invalid_request` (400/404) — bad config, rejected peer/shard, or an
///   unknown model id.
///
/// The engine-raised reasons matter for a specific reason: no `ChunkStream`
/// exists for a request that fails at submit, so the runner's
/// `tasks_failed_total` never sees it. Without them a node failing every
/// request looks exactly like a healthy idle one.
///
/// Rejections issued by router layers before a handler runs (body over the
/// DefaultBodyLimit, malformed JSON) are NOT counted here — they are
/// visible in `cascadia_http_requests_total` by status code.
pub static API_REJECTED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_api_rejected_total",
        "Requests rejected before generation started, by reason.",
        &["reason"]
    )
    .expect("register cascadia_api_rejected_total")
});

// --- Generation-level (bumped by cascadia-runner) ------------------------

/// Time from task submission to the first TOKEN-BEARING chunk delivered
/// to the consumer. Measured at delivery (chunks have no engine-side
/// timestamps), so under concurrent serving it includes buffer residency;
/// empty final markers never contribute a sample.
pub static GENERATION_TTFT_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "cascadia_generation_ttft_seconds",
        "Task submission to first token-bearing chunk delivered to the consumer.",
        &["model"],
        TTFT_BUCKETS.to_vec()
    )
    .expect("register cascadia_generation_ttft_seconds")
});

/// Gap between consecutive token-bearing chunks of one generation,
/// measured at delivery to the consumer (see the TTFT caveat: buffer
/// residency is included under concurrent serving; empty final markers
/// are excluded).
pub static GENERATION_INTER_TOKEN_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "cascadia_generation_inter_token_seconds",
        "Gap between consecutive token-bearing chunks, at delivery.",
        &["model"],
        INTER_TOKEN_BUCKETS.to_vec()
    )
    .expect("register cascadia_generation_inter_token_seconds")
});

/// Whole-generation duration, labeled with how it ended: `stop`, `length`,
/// `cancelled` (client gone / explicit cancel), or `error`. The `cancelled`
/// samples are taken at abandonment, where there is no final chunk to
/// deliver — hence "terminal outcome" rather than "final chunk".
pub static GENERATION_DURATION_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "cascadia_generation_duration_seconds",
        "Task submission to terminal outcome, by finish reason.",
        &["model", "finish_reason"],
        DURATION_BUCKETS.to_vec()
    )
    .expect("register cascadia_generation_duration_seconds")
});

/// Model tokens DELIVERED to consumers across all generations. Tokens an
/// engine grinds through after its client disconnected (until the cancel
/// lands) are not counted.
pub static TOKENS_GENERATED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_tokens_generated_total",
        "Model tokens delivered to consumers across all generations.",
        &["model"]
    )
    .expect("register cascadia_tokens_generated_total")
});

/// Prompt tokens consumed, as reported by engines that can tell (on the
/// final chunk). An engine that never reports contributes NOTHING rather
/// than 0 — the child is never created, so the family can be absent from
/// the scrape entirely. Absence here means "nobody reported", not "zero
/// prompt tokens".
pub static TOKENS_PROMPT_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_tokens_prompt_total",
        "Prompt tokens consumed, when the engine reports them.",
        &["model"]
    )
    .expect("register cascadia_tokens_prompt_total")
});

/// Generations abandoned before completion: explicit `/v1/cancel`
/// (including an engine acknowledging with a Cancelled final marker), or
/// the client dropped the response stream mid-generation.
///
/// Server teardown (engine slot emptied under in-flight generations) is
/// deliberately not counted — a restart is not a client cancellation — but
/// that suppression is CONDITIONAL, not a guarantee: it fires only when a
/// stream is polled after `Runner::close()` empties the slot. A stream
/// dropped before that poll still books a cancel here, so a restart can
/// leave a small nondeterministic bump. Tracked as a follow-up.
pub static TASKS_CANCELLED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_tasks_cancelled_total",
        "Generations abandoned before completion (cancel or client disconnect).",
        &["model"]
    )
    .expect("register cascadia_tasks_cancelled_total")
});

/// Generations that ended in an engine failure (final error chunk).
pub static TASKS_FAILED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_tasks_failed_total",
        "Generations that terminated with an engine error.",
        &["model"]
    )
    .expect("register cascadia_tasks_failed_total")
});

// --- Engine-level (bumped by cascadia-runner during start) ---------------

/// Weight-load + engine-build duration at stage start (transport connect
/// excluded — it can include waiting for a peer to come up).
pub static ENGINE_LOAD_DURATION_SECONDS: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "cascadia_engine_model_load_duration_seconds",
        "Weight load + engine build duration at startup.",
        &["model", "device"]
    )
    .expect("register cascadia_engine_model_load_duration_seconds")
});

/// Engine warmup duration at stage start.
pub static ENGINE_WARMUP_DURATION_SECONDS: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "cascadia_engine_warmup_duration_seconds",
        "Engine warmup duration at startup.",
        &["model", "device"]
    )
    .expect("register cascadia_engine_warmup_duration_seconds")
});

// --- Transport-level (bumped by cascadia-transport) ----------------------

/// Bytes sent on the inter-stage activation links (tensor frames + raw
/// control bytes), header included. Only fully-transferred frames count —
/// bytes from partial/failed sends are not recorded, so sent-vs-recv
/// across a faulty link will not reconcile exactly.
pub static TRANSPORT_SENT_BYTES_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_transport_sent_bytes_total",
        "Bytes sent on inter-stage activation links.",
        &["kind"]
    )
    .expect("register cascadia_transport_sent_bytes_total")
});

/// Bytes received on the inter-stage activation links, header included.
/// Only fully-received frames count (see the sent-bytes caveat).
pub static TRANSPORT_RECV_BYTES_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "cascadia_transport_recv_bytes_total",
        "Bytes received on inter-stage activation links.",
        &["kind"]
    )
    .expect("register cascadia_transport_recv_bytes_total")
});

/// Full tensor-frame send duration (serialize + kernel send + flush). When
/// `CASCADIA_SEND_BURST_BYTES` is set the timed window also contains the
/// deliberate inter-burst sleeps, which are pacing, not link latency.
pub static TRANSPORT_SEND_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "cascadia_transport_send_seconds",
        "Tensor frame send duration (write + flush).",
        TRANSPORT_BUCKETS.to_vec()
    )
    .expect("register cascadia_transport_send_seconds")
});

/// Tensor payload receive duration, measured from header-complete to
/// payload-complete. The wait for a frame to BEGIN is deliberately
/// excluded — idle pipeline stages legitimately sit in that wait between
/// requests, and including it would swamp the histogram with idle time.
pub static TRANSPORT_RECV_PAYLOAD_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "cascadia_transport_recv_payload_seconds",
        "Tensor payload receive duration (header-complete to payload-complete).",
        TRANSPORT_BUCKETS.to_vec()
    )
    .expect("register cascadia_transport_recv_payload_seconds")
});

/// Force registration of EVERY family at startup. Two reasons, and the
/// label-less ones are only the visible half:
///
/// 1. Exposition: a scrape before any traffic still shows the label-less
///    families (vec families stay absent until a child exists — prometheus
///    drops childless families from `gather`, so forcing them is inert for
///    output).
/// 2. Collision detection: registration is the only thing that can fail
///    here, and forcing it at boot turns a duplicate metric name into an
///    immediate startup panic instead of one raised on a rare path — some
///    of which run inside `Drop`, where a panic during unwind aborts.
///
/// Call once at server startup; idempotent.
pub fn init() {
    LazyLock::force(&INFLIGHT_TASKS);
    LazyLock::force(&TRANSPORT_SEND_SECONDS);
    LazyLock::force(&TRANSPORT_RECV_PAYLOAD_SECONDS);
    LazyLock::force(&HTTP_REQUESTS_TOTAL);
    LazyLock::force(&HTTP_REQUEST_DURATION_SECONDS);
    LazyLock::force(&API_REJECTED_TOTAL);
    LazyLock::force(&GENERATION_TTFT_SECONDS);
    LazyLock::force(&GENERATION_INTER_TOKEN_SECONDS);
    LazyLock::force(&GENERATION_DURATION_SECONDS);
    LazyLock::force(&TOKENS_GENERATED_TOTAL);
    LazyLock::force(&TOKENS_PROMPT_TOTAL);
    LazyLock::force(&TASKS_CANCELLED_TOTAL);
    LazyLock::force(&TASKS_FAILED_TOTAL);
    LazyLock::force(&ENGINE_LOAD_DURATION_SECONDS);
    LazyLock::force(&ENGINE_WARMUP_DURATION_SECONDS);
    LazyLock::force(&TRANSPORT_SENT_BYTES_TOTAL);
    LazyLock::force(&TRANSPORT_RECV_BYTES_TOTAL);
}

/// Gather the default registry into Prometheus text exposition format.
/// Returns `(content_type, body)`.
pub fn encode_text() -> (String, Vec<u8>) {
    let encoder = TextEncoder::new();
    let families = prometheus::gather();
    let mut buf = Vec::new();
    // encode only fails on a malformed writer; a Vec<u8> never errors.
    encoder
        .encode(&families, &mut buf)
        .expect("text-encode metrics into Vec<u8>");
    (encoder.format_type().to_string(), buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_registers_and_encodes() {
        init();
        let (content_type, body) = encode_text();
        assert_eq!(content_type, "text/plain; version=0.0.4");
        let text = String::from_utf8(body).unwrap();
        // Label-less metrics appear immediately after init().
        assert!(text.contains("cascadia_inflight_tasks"), "got:\n{text}");
        assert!(
            text.contains("cascadia_transport_send_seconds"),
            "got:\n{text}"
        );
    }

    #[test]
    fn vec_families_appear_once_observed() {
        init();
        GENERATION_TTFT_SECONDS
            .with_label_values(&["test-model"])
            .observe(0.2);
        TOKENS_GENERATED_TOTAL
            .with_label_values(&["test-model"])
            .inc_by(3);
        let (_, body) = encode_text();
        let text = String::from_utf8(body).unwrap();
        assert!(
            text.contains("cascadia_generation_ttft_seconds_bucket"),
            "got:\n{text}"
        );
        assert!(
            text.contains("cascadia_tokens_generated_total{model=\"test-model\"} 3"),
            "got:\n{text}"
        );
    }

    #[test]
    fn bucket_layouts_match_issue_16() {
        assert_eq!(TTFT_BUCKETS.len(), 10);
        assert_eq!(INTER_TOKEN_BUCKETS.len(), 10);
        assert_eq!(DURATION_BUCKETS.len(), 10);
        assert_eq!(TRANSPORT_BUCKETS.len(), 7);
        // Strictly increasing (prometheus requires it; catch typos here).
        for buckets in [
            TTFT_BUCKETS,
            INTER_TOKEN_BUCKETS,
            DURATION_BUCKETS,
            TRANSPORT_BUCKETS,
        ] {
            assert!(buckets.windows(2).all(|w| w[0] < w[1]));
        }
    }
}
