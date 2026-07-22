# Prometheus metrics

Every stage started with `--api` serves Prometheus text exposition format at
`GET /metrics` (issue #16). The registry is process-global: request-level,
generation-level, engine-level, and transport-level metrics all land on the
same endpoint. Relay-only stages (workers started without `--api`) have no
HTTP listener and therefore no scrape endpoint today — put the API rank on
your scrape list, and scrape each API-bearing node in a multi-node fleet.

```bash
curl -s http://127.0.0.1:8416/metrics | head
```

Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: cascadia
    static_configs:
      - targets: ["cascadia-host-1:8416", "cascadia-host-2:8416"]
```

Metric families with labels appear in the output once the first sample is
observed (standard Prometheus client behavior); label-less families are
present from startup.

## Request-level (HTTP API)

| Name | Type | Labels | Description |
|---|---|---|---|
| `cascadia_http_requests_total` | counter | `endpoint`, `status` | Requests by **matched route template** (`/v1/cancel/:task_id`, never the raw URI — bounded cardinality) and status code. |
| `cascadia_http_request_duration_seconds` | histogram | `endpoint` | Request latency. For streaming (SSE) responses this is time-to-response-head; full generation time is `cascadia_generation_duration_seconds`. |
| `cascadia_inflight_tasks` | gauge | — | Generation requests currently executing (admitted past the permit gate, response not yet drained/dropped). |
| `cascadia_api_rejected_total` | counter | `reason` | Rejections before the engine: `capacity` (503, permit gate full), `empty_prompt` (400), `prompt_too_large` (413), `multi_prompt` (400). |

There is no queued-tasks gauge: admission is `try_acquire` on the permit
semaphore — over-capacity requests are rejected with 503 immediately, never
queued. Watch `cascadia_api_rejected_total{reason="capacity"}` instead.

## Generation-level

Recorded at the runner's chunk stream, so they cover streaming and
non-streaming requests on both `/v1/chat/completions` and `/v1/completions`.
The `model` label is the shard's `model_id`.

| Name | Type | Labels | Description |
|---|---|---|---|
| `cascadia_generation_ttft_seconds` | histogram | `model` | Task submission → first engine chunk. |
| `cascadia_generation_inter_token_seconds` | histogram | `model` | Gap between consecutive chunks of one generation. |
| `cascadia_generation_duration_seconds` | histogram | `model`, `finish_reason` | Submission → final chunk. `finish_reason`: `stop`, `length`, `cancelled`, `error`. |
| `cascadia_tokens_generated_total` | counter | `model` | Model tokens emitted (uses the engine's `n_tokens` when set — spec-decode and ov-genai report multi-token chunks correctly). |
| `cascadia_tokens_prompt_total` | counter | `model` | Prompt tokens, for engines that report them on the final chunk. |
| `cascadia_tasks_cancelled_total` | counter | `model` | Generations abandoned before completion: explicit `/v1/cancel` or client disconnect mid-stream. |
| `cascadia_tasks_failed_total` | counter | `model` | Generations that terminated with an engine error. |

Engines that deliver the whole response on a single final chunk (`ov-genai`)
produce one TTFT sample equal to the full generation and no inter-token
samples; per-token engines (`ov-runtime`, sparse-moe, mock) populate both.

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
| `cascadia_transport_sent_bytes_total` | counter | `kind` | Bytes sent on activation links. |
| `cascadia_transport_recv_bytes_total` | counter | `kind` | Bytes received on activation links. |
| `cascadia_transport_send_seconds` | histogram | — | Tensor frame send duration (write + flush). |
| `cascadia_transport_recv_payload_seconds` | histogram | — | Tensor payload receive, **header-complete → payload-complete**. The wait for a frame to begin is excluded: idle stages legitimately sit in that wait between requests, and including it would swamp the histogram with idle time. |

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
