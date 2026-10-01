# Architecture

Cascadia is a Cargo workspace at the repo root. Each crate has a single responsibility and a stable interface; engines and discovery backends are swappable.

## Design decisions

- **Engine plurality: OpenVINO-first, pluggable.** The `Engine` + `Builder` traits live in `cascadia-engine`; seven engines ship behind them (`mock`, `ov-genai`, `ov-runtime`, `ov-dist-spec`, `gemma4`, `sparse-moe`, `qwen35`). Future engines (IPEX, OneAPI direct) plug behind the same trait.
- **Discovery: zero-config peer-to-peer.** Workers find each other over mDNS; no central control plane.
- **Topology stores measured latency + bandwidth.** Latency is the dominant placement signal on Intel fleets — a 50 ms WAN hop drops throughput 65% — so Cascadia's topology graph stores per-link measurements, not just edge types.
- **Rust-only workers.** One static binary per node; no runtime Python dependency, no pip install on workers. Python is only needed at export time (`cascadia shard`).

## `cascadia-api`

OpenAI-compatible HTTP server (axum). Routes: `/health`, `/v1/models`, `/v1/chat/completions` (non-streaming + SSE streaming), `/v1/cancel/<task_id>`. Backpressure via a concurrent-request semaphore (default 16); request body cap and rendered-prompt cap (both `--api-max-body-mb`, default 1 MiB, on every engine) enforce 413 / 503 responses on oversized or over-capacity input.

## `cascadia-runner`

Per-stage `Runner`. Connects upstream + downstream transports, loads weights, builds the engine, warms it up, and exposes `submit` / `generate` / `cancel`. Concurrent-safe — multiple `generate()` callers share one engine through a `Mutex`; chunks for other tasks emitted during one caller's `step()` are buffered for their owners.

## `cascadia-engine`

Two trait definitions — the plugin seam:

- `Engine`: `warmup`, `submit`, `step`, `cancel`, `close`. `submit` returns `EngineError::QueueFull` when the per-engine pending cap is reached.
- `Builder`: `configure_listen`, `connect`, `load`, `build`, `close`.

## `cascadia-engine-openvino`

Five engines:

- `ov-genai` — single-stage `openvino_genai.LLMPipeline` via the C++ FFI shim. FastDraft + Prompt Lookup variants.
- `ov-runtime` — multi-stage stateful KV cache. Pre-exported per-stage v3+ shards; each stage owns its layer range and runs SDPA attention with internal RoPE.
- `ov-dist-spec` — multi-stage spec decode with mask-based KV-cache rewind on rejected drafts. v5 shards (canonical optimum-style inputs).
- `gemma4` — Gemma 4 multi-stage: per-layer-type attention, KV-sharing, per-layer-input embeddings. `gemma4_cached_v1.x` shards.
- `qwen35` (alias `qwen36-moe`) — Qwen3.5-family staged chain (GatedDeltaNet; Qwen3.6 MoE or dense Qwen3.8) from `qwen3_5*` IR-surgery shards; single-box or N-rank pipeline; in-process prefix cache. See [architectures/qwen36-moe-support.md](architectures/qwen36-moe-support.md).

## `cascadia-engine-mock`

Deterministic word-echo engine — splits the prompt and emits one word per `step()`. Used by API / runner / CLI tests.

## `cascadia-engine-sparse-moe`

CPU-targeted sparse mixture-of-experts engine (Kimi K2.6-style models, MiniMax-M2). Runs attention/norm shells natively in Rust (default; OV IR shells are an optional backend) and dispatches only the top-k experts the router selects each step. Experts execute as per-(layer, expert) OV IRs by default, or through the `cascadia-int4-gemm` AVX-512 kernels against packed int4 weight binaries (`int4_bin` backend).

## `cascadia-int4-gemm`

Hand-rolled AVX-512 INT4 GEMM kernels for the sparse-MoE expert path — group-32 symmetric quantization with bf16 scales, matching the compressed-tensors on-disk format.

## `cascadia-dashboard`

Dashboard HTTP routes (`/api/topology`, `/api/stats`) and an embedded Vite SPA (behind the `embed-spa` feature) for visualizing a cluster; without the feature, `/` serves a built-in pointer page explaining how to enable the UI. Kept separate from `cascadia-api` so the OpenAI surface doesn't grow a topology dependency or bundled static assets.

## `cascadia-ov-genai-shim`

C++ FFI shim wrapping `openvino-genai`. `extern "C"` only; every entry point catches `...` so a C++ exception cannot unwind into Rust UB. Stub mode (no link) is the default for dev / CI; `--features openvino` links against the real OV GenAI 2026.2+ SDK.

## `cascadia-types`

Engine-agnostic core types (serde-serializable): generation tasks and chunks, shard descriptions, peer layout. Only `serde` + `thiserror` as dependencies, so downstream crates share vocabulary without version-lockstep.

## `cascadia-transport`

TCP activation relay between pipeline stages. Wire format: 20-byte header (`payload_len`, `dtype`, `dim0`, `dim1`, `dim2`) then row-major payload. dtype codes: `0=f32, 1=f16, 2=i8, 3=i32, 4=i64`. Caps incoming payloads at 256 MiB and applies a 60 s read timeout per recv.

### Injected streams and re-attach (issue #76)

Embedders that carry activations over their own transport (encrypted p2p
streams, in-process pipes) can skip TCP entirely. Enable the
`injected_streams` feature of `cascadia-engine` on every crate that implements
or calls this API (`cascadia-runner` already does); it is off by default so a
plain `Engine`/`Builder` implementer pulls no tokio/socket2/prometheus (CI
checks this). A runnable walk-through is
`cargo run -p cascadia-runner --example injected_duplex`.

**Starting.** `Runner::start_with_streams(links, shard)` starts a stage over
already-connected `ByteStream`s (`Box<dyn InjectedStream>`; `InjectedStream`
is any `AsyncRead + AsyncWrite + Send + Unpin`), passed in a `StreamLinks`:
`upstream` / `downstream` for pipeline stages, `ep_driver` for an inkling
expert worker, `ep_workers` for an inkling expert-parallel driver. Links must
match the stage exactly. Streams must follow the contract on `InjectedStream`
in `cascadia-transport`: connected, cancel-safe reads, prompt failure on every
operation, in order, fresh on every attach. The transport applies no nodelay,
tuning or keepalive to them (call `set_nodelay(true)` on a `TcpStream`
yourself); a peer that dies silently is caught only by the frame-start idle
ceiling (900 s by default) unless the embedder notices first.

**Link death.** There is no link-death event: the embedder watches its own
ends, and treats a link as dead when its end sees EOF or a write error (the
engine may drop its end itself). In stream mode a connection-fatal step does
not end `run_relay_loop`; the relay parks (logging `warn` "relay step hit a
dead peer link; waiting for re-attach" once per attach epoch: a rejected
re-attach stays quiet, and the next death after a successful one warns again)
until `reattach` or `close`. This now holds for every engine: sparse-moe,
OvMoe, pipeline and inkling expert workers return a connection-fatal error on
every step while their link is latched dead, instead of sleeping and
returning empty. A sparse-moe, OvMoe or pipeline worker that latches on a
**protocol** failure while its upstream is still live (a failed prefill
frame, a prefix-restore miss, a bad hidden width, a downstream failure
mid-frame) also closes every injected link it holds (`warn` "worker latched
on a protocol failure; closed its injected links ..."), so its neighbours
and the embedder see EOF at once rather than at the head's reply deadline;
re-attach all of them. A worker whose **upstream died** (EOF, a dead-link
error, the idle ceiling) closes nothing: its healthy downstream stays open,
so the outage does not cascade toward the tail and a re-attach that replaces
only its upstream is accepted. TCP/UDS links are never closed this way.

**Re-attach.** Close your own end of every stream being replaced first: a step
blocked on the dead link holds the engine lock that the re-attach needs, and
only closing that stream releases it. Then call `Runner::reattach(links)`
(`&self`) with fresh streams; `None` keeps a link. The engine returns to a
clean between-requests state; weights stay loaded. Rules:

- **Relay rule:** a stage that has an upstream link must replace it on every
  re-attach, for the same reason (an idle relay's step blocks on its upstream
  read while holding the engine lock). Any re-attach therefore cascades to the
  head, which refreshes head-side session state (prefix index, qwen36
  handshake, RESET). Enforced by `check_reattach_streams`.
- ov-runtime **stateful** stages must also replace their downstream, so the
  reset cascades to the tail and stale KV never survives a re-attach.
- dist_spec workers must replace **both** links: their connection-fatal path
  closes both hops together.
- A kept link must still be live: keeping one this stage closed at its latch,
  one the transport dropped, or one whose peer closed it (once this stage has
  read the EOF) is `PeerRejected` ("the kept ... link is dead"), on every
  engine that can keep a link: sparse-moe, OvMoe and pipeline stages
  (including an EP driver's kept `ep_workers`), ov-runtime static stages,
  gemma4 and qwen36. dist_spec and inkling expert workers replace every link
  anyway. `is_connected()` on `ActivationServer`/`ActivationClient` reports
  the same liveness.
- Issue one re-attach per outage, for all its links at once; concurrent or
  duplicate re-attaches for one outage are an embedder bug. Retrying after a
  dropped `reattach` call is fine (see below).

Outcomes: `PeerRejected` means validation failed and nothing was swapped
(also returned in TCP mode); `NotLoaded` means never started or closed; any
other error means streams may have been swapped, so the runner is fenced
(`submit` and open streams fail with `NotConnected`, a relay parks) until a
later re-attach succeeds. The fence and an **attach epoch** live on the
runner's `EngineSlot`; the epoch moves whenever the engine's
`reattach_streams` runs and does not return `PeerRejected`. Every head
request submitted before it through `generate`/`generate_async`, whether
running or still queued in the engine, is then cancelled in the engine under
the same lock, and its stream ends, without emitting anything new (chunks
buffered before the swap are delivered first), with an error chunk "link
re-attached; in-flight request dropped (...)" (`NotConnected`), logged as
`warn` "link re-attached under an in-flight request; failing its stream".
Resubmit those requests; such a stream always ends with that error, never
with another task's engine failure. (A task given to `Runner::submit`
directly has no stream and is not tracked.) Task ids must be unique among
live requests: a new task reusing an orphaned id while the old stream is
still open gets none of its output. Dropping the `reattach` future before its
blocking half takes the engine lock cancels it (nothing swapped, fenced or
bumped); once the lock is taken it completes, and is still counted and
logged. Retrying after a dropped call is safe: close your ends of the streams
handed to the dropped call, then call `reattach` again with fresh ones; if
the dropped call swapped, the retry replaces those links like any re-attach.
Success logs `info` "link re-attached" (with `dropped_requests`, inside the
caller's tracing span) and
increments `cascadia_link_reattach_total{side}` (`upstream`, `downstream`,
`ep_driver`, `ep_worker`) once per replaced link; a post-swap failure logs
`warn` "re-attach failed after swapping streams; ...".

**Shutdown.** `Runner::close` is mandatory in stream mode: it is the only
thing besides `reattach` that wakes a parked relay, and without it the relay's
`spawn_blocking` thread never returns and tokio runtime shutdown hangs. A
relay idle on a live upstream holds the engine lock inside its read, so close
your ends of its streams before calling `close`. If `close` waits more than
5 s for that lock it logs one `warn` ("close() is still waiting for the engine
lock: a step is blocked on a peer link ...", naming the idle relay, the frame
idle ceiling and closing the injected ends) and keeps waiting; this also
applies in TCP mode, where it can wait up to the idle ceiling, as on main.

**Engine state on reset (TCP-visible where noted).**

- gemma4: `cancel` only drops the task again, as on main (every admission
  scrubs before it runs). Re-attach resets the session but keeps
  `state_restored`, so the next new sequence rebuilds the OV request instead
  of cheap-resetting over warm-restore residue; relays now do the same at
  wire position 0 (TCP: one `recreate_request` on the first cold sequence
  after a restore). A rejected warm verdict (`abort_warm_resume`) rebuilds
  once; the flag stays set only if that rebuild failed, so the next sequence
  does not rebuild a second time.
- ov-runtime and dist_spec: `clear_ov_state` rebuilds the OV request on the
  first cold turn after a warm restore (head admission, relay prefill reset,
  dist_spec driver and worker `Reset`, re-attach), because `reset_state`
  cannot clear `set_state_blob` residue. TCP-visible: one rebuild per stateful
  rank on a cold turn that follows a warm one; at most one per restore (a
  scrub after a failed restore or abort clears the flag when it succeeds). A
  dist_spec worker whose `Reset` rebuild fails still forwards the `Reset`
  downstream, logs `warn`, and retries the rebuild at its next `Reset`.
- ov-runtime stateful multi-stage heads refuse a cold 1-token prompt (prompt
  tokens plus resume ids) at `submit` with `InvalidConfig` ("1-token prompts
  are not supported on a stateful multi-stage chain ..."), which the API
  answers 400 `invalid_request` without marking `/health` degraded: the
  stateful wire carries no position, so a relay cannot tell it from a decode
  step and would run it on the previous request's KV. Use a static or packed
  layout, or a longer prompt. TCP-visible; chat requests are templated and
  never this short, a raw `/v1/completions` prompt or a CLI stdin line can be
  (the CLI stdin loop prints "request refused: ..." and reads the next line).
  The check tokenizes the prompt once more at submit, only on such a head.
- dist_spec: dropping a `TargetSendHandle` cancels its round trip if it has
  not started writing, so an orphan cannot put a stale FORWARD on a
  re-attached stream.

**Worker links in TCP mode.** A dead link still exits the worker
(`RelayExit::ConnectionFatal`, non-zero exit) for a supervisor restart. New:
sparse-moe, OvMoe and pipeline workers latch a dead upstream on every
transport, including a socket the transport dropped after the idle ceiling, a
reset or a mid-frame timeout, and exit instead of logging every 200 ms forever
(OvMoe also exits on upstream EOF now). A TCP worker cannot re-accept and the
head never re-dials, so the chain recovers once the head is restarted too, as
with any worker restart.

**Idle chains.** The frame-start idle ceiling (default 900 s) bounds every
idle upstream read, so a chain that carries no requests for longer than that
loses its worker links: TCP workers exit for a restart, injected ones park
until re-attached. Keep traffic flowing, re-attach, or raise or disable the
ceiling (`CASCADIA_FRAME_IDLE_CEILING_SECS`, `0` = no ceiling).

**Verification status.** CI covers the runner, transport and engine
re-attach logic with stub engines and the example above. Rig-only (needs real
OpenVINO and models): stateful and static ov-runtime chains re-attached idle
and mid-request matching a fresh chain token-for-token; a 3-stage stateful
chain cascading a first-hop re-attach; a sparse-moe chain re-attach plus a
worker restart without a "kv-prefix cache diverged" latch; inkling
expert-parallel re-attach of one worker; gemma4/ov-runtime/dist_spec rebuild
after a warm restore (and its cost); the real-`openvino` build of the stub-
gated tests.

## `cascadia-topology`

Topology graph with per-link latency and bandwidth measurements. This is where Cascadia diverges from exo, whose topology only tracks edge type (Socket vs RDMA). Empirically, latency is the dominant placement signal on Intel fleets — a 50 ms WAN hop drops throughput 65%.

## `cascadia-discovery`

mDNS peer discovery via the `mdns-sd` crate. Advertises `_cascadia._tcp.local.` and browses for siblings in the same namespace (a TXT-record field; peers in other namespaces are ignored). Zero-config: spin up workers on the same LAN and they find each other.

## `cascadia-download`

Model registry plus on-demand HuggingFace pull. **Not wired into the CLI** — no crate depends on it; workers never download (only `cascadia shard` fetches). Registry lives at `~/.cache/cascadia/registry.json`; writes are atomic (`.tmp` + `fsync` + rename). Symlinks at the registry path are rejected to prevent path-substitution attacks.

## `cascadia-cli` + `cascadia`

`cascadia worker --rank N --total M --engine <name> --model <dir> ...` is the core serving subcommand; `run` is its single-machine sugar. Other subcommands: `shard` (bundled exporter), `doctor` (environment checks), `discover` (mDNS browse), `engines`, `completions` (shell completions), `profile-devices` / `profile-stages` / `place` / `run-placement` (placement tooling). The `cascadia` crate is the binary entry point and depends on `cascadia-cli`.
