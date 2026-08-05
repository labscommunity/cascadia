# Prometheus metrics

Every stage started with `--api` serves Prometheus text exposition format at
`GET /metrics` (issue #16). The registry is process-global: request-level,
generation-level, engine-level, and transport-level metrics all land on the
same endpoint. `--api` binds on rank 0 only — it is ignored with a warning on
every other rank, and relay-only stages have no HTTP listener at all. So one
N-stage pipeline has exactly **one** scrape target, its rank-0 node; a fleet
running several independent pipelines has one target per pipeline.

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

Prometheus scrape config — one target per pipeline, each its rank-0 node:

```yaml
scrape_configs:
  - job_name: cascadia
    static_configs:
      - targets: ["pipeline-a-rank0:8000", "pipeline-b-rank0:8000"]
```

Metric families with labels appear in the output once the first sample is
observed (standard Prometheus client behavior); label-less families are
present from startup.

## Request-level (HTTP API)

| Name | Type | Labels | Description |
|---|---|---|---|
| `cascadia_http_requests_total` | counter | `endpoint`, `status` | Requests by **matched route template** (`/v1/cancel/:task_id`, never the raw URI — bounded cardinality) and status code. Streaming (SSE) responses count their response HEAD: a mid-stream engine failure still shows as 200 here (it surfaces in `cascadia_tasks_failed_total`). Covers the API server's own routes; the CLI's embedded dashboard routes are merged in after router construction and are not counted. |
| `cascadia_http_request_duration_seconds` | histogram | `endpoint` | Request latency. For streaming (SSE) responses this is time-to-response-head; full generation time is `cascadia_generation_duration_seconds`. |
| `cascadia_inflight_tasks` | gauge | — | Generation requests currently executing (admitted past the permit gate, response not yet drained/dropped). |
| `cascadia_api_rejected_total` | counter | `reason` | Rejections before generation started: `capacity` (503 — permit gate full, or the engine's own pending queue full), `empty_prompt` (400), `prompt_too_large` (413 — over the API's `max_prompt_bytes`), `prompt_over_window` (413 — tokenizes past what the engine can window for one request, e.g. a packed slot's KV region), `multi_prompt` (400). The two 413s are separate reasons because they are different knobs: one is the API's byte limit, the other an engine sizing decision. Rejections issued by router layers before a handler runs (body over the 64 KiB `DefaultBodyLimit`, malformed JSON) are not counted here — watch them in `cascadia_http_requests_total` by status. |

There is no queued-tasks gauge. Admission at the API is `try_acquire` on the
permit semaphore, so over-capacity requests are rejected with 503 immediately
rather than waiting on it. Past that gate engines **do** keep a bounded
pending queue — that queue filling is what raises `QueueFull`, the second
source of `reason="capacity"` — but its depth is not exported today. Watch
`cascadia_api_rejected_total{reason="capacity"}` for pressure at either
level.

## Generation-level

Recorded at the runner's chunk stream, so they cover streaming and
non-streaming requests on both `/v1/chat/completions` and `/v1/completions`.
The `model` label is the shard's `model_id`.

| Name | Type | Labels | Description |
|---|---|---|---|
| `cascadia_generation_ttft_seconds` | histogram | `model` | Task submission → first token-bearing chunk **delivered to the consumer**. |
| `cascadia_generation_inter_token_seconds` | histogram | `model` | Gap between consecutive token-bearing chunks of one generation, at delivery. |
| `cascadia_generation_duration_seconds` | histogram | `model`, `finish_reason` | Submission → final chunk. `finish_reason`: `stop`, `length`, `cancelled`, `error`. |
| `cascadia_tokens_generated_total` | counter | `model` | Model tokens **delivered to clients** (uses the engine's `n_tokens` when set — spec-decode and ov-genai report multi-token chunks correctly). Tokens ground out after a client disconnected, before the cancel lands, are not counted. |
| `cascadia_tokens_prompt_total` | counter | `model` | Prompt tokens, for engines that report them on the final chunk. |
| `cascadia_tasks_cancelled_total` | counter | `model` | Generations abandoned before completion: explicit `/v1/cancel` (including an engine acknowledging with a `Cancelled` final marker) or client disconnect mid-stream. Server teardown with generations in flight is deliberately not counted — but that suppression is conditional (it needs the stream to be polled after the engine slot empties), so a restart can still leave a small nondeterministic bump. |
| `cascadia_tasks_failed_total` | counter | `model` | Generations that terminated with an engine error. |

Modes that deliver the whole response on a single final chunk — `ov-genai`
**without** `--cb`, which is the default worker configuration — produce one
TTFT sample equal to the full generation and no inter-token samples. Read
`cascadia_generation_ttft_seconds` as generation latency there, not as
time-to-first-token. Per-token producers (`ov-genai --cb`, `ov-runtime`,
sparse-moe, mock) populate both histograms as named.

Timing caveats: chunks carry no engine-side timestamps, so TTFT and
inter-token gaps are measured when a chunk is **delivered** to the consumer.
For the typical single-stream case this tracks engine cadence closely; under
concurrent serving (several streams sharing one engine) a chunk can sit in
the cross-task buffer, so samples include that residency. Empty final
markers never contribute timing samples. For tool-call streaming requests
the API buffers the whole generation before responding, so
`cascadia_http_request_duration_seconds` for those requests spans the full
generation rather than time-to-head.

## Engine-level

Set once at stage startup by the runner.

| Name | Type | Labels | Description |
|---|---|---|---|
| `cascadia_engine_model_load_duration_seconds` | gauge | `model`, `device` | Weight load + engine build (transport connect excluded — it can include waiting for a peer to come up). |
| `cascadia_engine_warmup_duration_seconds` | gauge | `model`, `device` | `Engine::warmup()` duration. |

## Transport-level

Counted inside `cascadia-transport`, so every engine's inter-stage traffic is
covered without per-engine wiring. `kind` is `tensor` (framed activation
tensors, header included) or `raw` (dist-spec control bytes).

| Name | Type | Labels | Description |
|---|---|---|---|
| `cascadia_transport_sent_bytes_total` | counter | `kind` | Bytes sent on activation links (fully-sent frames only). |
| `cascadia_transport_recv_bytes_total` | counter | `kind` | Bytes received on activation links (fully-received frames only). |
| `cascadia_transport_send_seconds` | histogram | — | Tensor frame send duration (write + flush). |
| `cascadia_transport_recv_payload_seconds` | histogram | — | Tensor payload receive, **header-complete → payload-complete**. The wait for a frame to begin is excluded: idle stages legitimately sit in that wait between requests, and including it would swamp the histogram with idle time. |

Byte counters record only fully-transferred frames — bytes on the wire from
partial or failed sends/receives (e.g. a mid-frame timeout on a degraded
link) are not counted. The duration histograms sample on the same condition,
which matters more than it sounds: a link that stalls mid-frame produces
**no** sample rather than a slow one, so `recv_payload_seconds` holds its
healthy-looking percentiles while `recv_bytes_total` quietly stops
advancing. Alert on the counter's rate, not the histogram, to catch a
stalled link.

Reconciling `sent_bytes` on stage N against `recv_bytes` on stage N+1 is not
possible today even in principle: only rank 0 serves `/metrics`, so the
downstream stages' counters cannot be scraped at all (see the follow-ups).

## Histogram buckets

Calibrated for LLM serving (see `crates/cascadia-metrics/src/lib.rs`):

- TTFT: `0.05 0.1 0.25 0.5 1 2 5 10 30 60`
- Inter-token: `0.01 0.025 0.05 0.1 0.25 0.5 1 2 5 10`
- Durations (generation + HTTP): `0.1 0.5 1 5 10 30 60 120 300 600`
- Transport frames: `0.0001 0.001 0.01 0.05 0.1 0.5 1`

## Useful queries

```promql
# Request rate by endpoint
sum by (endpoint) (rate(cascadia_http_requests_total[5m]))

# p95 TTFT
histogram_quantile(0.95, sum by (le) (rate(cascadia_generation_ttft_seconds_bucket[5m])))

# Tokens/second across the fleet
sum(rate(cascadia_tokens_generated_total[1m]))

# Capacity pressure (503s)
rate(cascadia_api_rejected_total{reason="capacity"}[5m])

# Inter-stage link throughput
rate(cascadia_transport_sent_bytes_total[1m])
```

## Not yet covered (follow-ups)

- **Sparse-MoE expert metrics** (`experts_dispatched_total` per layer,
  per-layer compute histograms): the hook point exists
  (`cascadia-engine-sparse-moe/src/runner.rs`, `dispatch_expert`) but wiring
  it deserves its own validated change on real MoE hardware.
- **Metrics endpoint on relay-only stages**: workers without `--api` expose
  nothing; a lightweight metrics-only listener is a possible follow-up.
- **OTLP / trace exemplars**: out of scope for v1, per #16.
