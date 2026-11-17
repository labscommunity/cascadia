# `ov-policy` — robot policies (VLA)

`ov-policy` serves a vision-language-action (VLA) robot policy on one machine: one observation in
(camera frames, robot state, an instruction), one action chunk `[T, D]` out. It runs OpenVINO
`policy_package` exports on the OpenVINO runtime Cascadia already links, with no Python on the
serving machine. It is not a chat engine: there is no `/v1/chat/completions`, no dashboard, and it
runs only under `cascadia run`.

The full design, the package contract and the measurements are in
[`../design/2026-11-15-vla-policy-engine.md`](../design/2026-11-15-vla-policy-engine.md) (called
"the design doc" below).

## When to use

- You run a robot policy (SmolVLA, π0.5, π0, X-VLA, EVO1, VLA-JEPA, XR0, GR00T N1.7, MolmoAct2) on
  an Intel GPU or CPU and want an openpi-compatible WebSocket, or Intel's physicalai protocol, with
  no Python on the serving machine.
- One policy on one machine. For language models use the chat engines; for more than one machine,
  no policy path exists (`cascadia worker` refuses `ov-policy`).

## Package format

A **`policy_package`** directory: `manifest.json` (`"format": "policy_package"`, version `1.x`,
runner `SinglePass`, artifact `openvino`), the policy IR (`.xml` + `.bin`) and the tokenizer IR.
The engine executes the manifest's pre- and postprocessing steps exactly; anything it does not know
is refused at load with an error, never guessed.

The engine loads nine policy kinds. Where a package for each comes from:

| Kind | Package source |
|---|---|
| `smolvla`, `pi05` | Intel publishes LIBERO packages on Hugging Face (`OpenVINO/smolvla-libero-fp16-ov`, `OpenVINO/pi05-libero-fp16-ov`); `cascadia export-policy --model <hf-model-id or ./local-dir> --output-dir <dir>` exports a LeRobot checkpoint of either kind (design doc, "Exporter"). |
| `gr00t_n1d7` | Intel's edge-ai-suites single-IR exporter, run on NVIDIA's checkpoint. |
| `molmoact2` | Intel's exporter in physical-ai-studio. |
| `pi0`, `xvla`, `evo1`, `vlajepa`, `xr0` | Exported by hand with Intel's physicalai-train exporter, patched or with LeRobot modules per kind. |

`cascadia export-policy` handles only `smolvla` and `pi05`. For the other seven kinds no exporter
ships with Cascadia yet. The design doc's Scope section and
[the policy fixtures README](../../crates/cascadia-engine-openvino/tests/fixtures/policy/README.md)
record how each package was made: which exporter, which revision, which patches.

**Tiers.** A kind is *supported* once it has PyTorch parity, Windows CPU and GPU runs and LIBERO
task success at f16; until then it is *experimental*: served, with metadata, docs and CLI, but not
promised. The current tier of each kind is the table under "Tiers" in the design doc's Scope
section.

A package whose normalization statistics are identity placeholders (an export that never had
stats) is refused: its actions would be wrong for any robot. `--policy-allow-identity-stats` serves it
anyway, for testing only.

## Example

```bash
cascadia export-policy --model lerobot/smolvla_libero --output-dir ./smolvla_libero
cascadia run ./smolvla_libero --engine ov-policy
```

`--engine ov-policy` is required (`run` defaults to `ov-genai`). The server loads the package, runs
three warmup calls, then listens on **`127.0.0.1:8000`**. The compile goes through the same
OpenVINO cache as the chat engines (`cascadia/ov-cache` under the user cache directory), so a
restart skips it; the cache never changes outputs.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--device GPU\|CPU` | `GPU` | Where the policy IR runs: `GPU`, `GPU.N` (one of several GPUs), `CPU` or `CPU.N`; the compound forms (`AUTO`, `MULTI:`, `HETERO:`, `BATCH:`) are refused. The tokenizer always runs on CPU. **NPU is refused** for every policy (SmolVLA and π0.5 fail NPU compilation as exported; X-VLA compiles and returns wrong actions). |
| `--ov-inference-precision f16\|f32` | `f16` on GPU | GPU compute precision (chat's flag; `bf16` is refused here). CPU always runs f32 and refuses `f16`. Some large models (π0, XR0, GR00T, MolmoAct2) fail to compile at f32 on a 16 GB integrated GPU; that is a load error naming the IR and the device. |
| `--api HOST:PORT` | `127.0.0.1:8000` | Where the HTTP and WebSocket API listens. A non-loopback host is refused without `--policy-allow-all-interfaces`; see [Security](#security). |
| `--policy-allow-origin ORIGIN` | none | A browser `Origin` allowed to open the WebSocket (repeatable). |
| `--policy-allow-identity-stats` | off | Serve a package with placeholder statistics (testing only). |
| `--zenoh HOST[:PORT]` | off | Also serve Intel's physicalai remote-inference protocol over Zenoh (below). |
| `--zenoh-name NAME` | the manifest's policy name | The slot name physicalai clients are given. |
| `--policy-allow-all-interfaces` | off | Let the robot front ends (`--api` and `--zenoh`) bind a non-loopback host. |

The robot-only flags carry the engine's prefix, as the sycl-llama ones carry `--llama-`; the Zenoh
ones carry their front end's. Each is refused on a chat engine, and the chat-only `run` flags
(`--prefix-cache-gb`, `--ep-*`, `--api-max-body-mb`, `--elastic*`, `--llama-*`) are refused here,
as each chat engine refuses another engine's flags. Settings shared with chat keep chat's names:
`--ov-inference-precision` above, and `CASCADIA_API_MAX_CONCURRENT`, which caps concurrent
connections across the WebSocket and Zenoh front ends (default 4, since a robot keeps its socket
open); one more WebSocket is refused with HTTP 503.
`--elastic` in particular is refused rather than ignored: its memory interposer has not been
measured with a policy. `cascadia worker` refuses `ov-policy`: a policy is served by one process on
one machine.

## Front ends

Both front ends share one loaded model behind one lock, so calls from either queue on the same
engine. A call holds the model for its inference time (about 100 ms to 1 s on an Arc 140V GPU at
f16, depending on the model; see the design doc's Scope section for each kind).

### openpi WebSocket

- `GET /v1/policy/metadata`: the policy's spec as JSON (camera names, frame size, state length,
  chunk `[T, D]`, noise mode, resize convention, RTC chunk size, package digests, precision,
  device).
- `WS /v1/policy/ws`: openpi's framing (msgpack with `msgpack_numpy` arrays), so
  `openpi_client.WebsocketClientPolicy` drives it:

  ```python
  from openpi_client.websocket_client_policy import WebsocketClientPolicy
  client = WebsocketClientPolicy(host="ws://127.0.0.1:8000/v1/policy/ws", port=None)
  ```

  Pass the whole URL as `host` and `port=None`; a port argument would be appended after the path.
  On connect the server sends the metadata; each request is a map of `images` (camera name to a
  `uint8` `[H, W, 3]` RGB array), `state`, `prompt` and optional `seed` and `noise`; the reply
  carries `actions` `[T, D]` and `server_timing.infer_ms`. Errors come back as a text frame. A
  refused upgrade (a foreign `Origin`, 403; the server full, 503) gets the chat API's
  `{"error": "..."}` JSON body.
- `GET /health`: `{"status":"ok"}`, or 503 `{"status":"degraded"}` after the engine failed a call
  (a refused request does not count), until a call succeeds again, as on the chat engines.
- `GET /metrics`: the Prometheus registry, as on the chat engines: per-route
  `cascadia_http_requests_total` and `cascadia_http_request_duration_seconds`,
  `cascadia_engine_model_load_duration_seconds` and `cascadia_engine_warmup_duration_seconds`, and
  `cascadia_api_rejected_total{reason="capacity"}` for an upgrade refused because the server is
  full, plus `cascadia_policy_infers_total{model,result}` (`ok`, `invalid` or `error`, from both
  front ends; `invalid` includes requests that did not decode),
  `cascadia_policy_infer_duration_seconds{model}` and, for Zenoh calls dropped before the model ran
  them, `cascadia_policy_dropped_total{model,reason}` (`superseded` or `expired`).

The keys are Cascadia's, so openpi clients are transport-compatible, not drop-in: a client written
for another server's keys needs a small mapping.

**Frames are sent at the model's resolution.** The WebSocket does not resize: the client resizes on
the robot with the package's convention, which the metadata states as `resize` (`center`,
`top_left` or `stretch`). Sending a frame resized with the wrong padding gives silently wrong
actions, not an error. `tools/vla_libero_client.py` has a reference resize for each convention.

### physicalai over Zenoh

Intel's `physicalai` runtime runs a policy on another process or machine through
`RemoteInferenceModel`. With `--zenoh`, Cascadia serves that protocol next to the WebSocket, so code
written against `RemoteInferenceModel` drives a Cascadia policy unchanged:

```bash
cascadia run /models/smolvla-libero-ov --engine ov-policy --zenoh 127.0.0.1
```

Without a port, the server listens on physicalai's name-derived port, which the client dials when
it is given none. This front end resizes frames for its clients (physicalai sends frames at the
camera's resolution) where the package's resize is ported exactly, and otherwise refuses a frame of
another size with a message naming the size to send. One physicalai client is served at a time,
with physicalai's budget and superseding rules.

Release bundles include this front end. A source build needs `--features zenoh`:

```bash
cargo build --release -p cascadia --features openvino,zenoh
```

### `cascadia policy-bridge`

`policy-bridge` serves physicalai's protocol on loopback and forwards each call to any openpi
WebSocket server: a local `cascadia run --engine ov-policy`, or an authenticated gateway in front of
a remote one.

```bash
cascadia policy-bridge --zenoh 127.0.0.1:7447 \
    --upstream wss://policy.example.internal/v1/policy/ws \
    --token-file ~/.config/cascadia/policy-token \
    --ca-file ~/.config/cascadia/gateway-ca.pem
```

- `--token-file` sends the file's token as `Authorization: Bearer`, only over `wss://` or to a
  loopback upstream. It is never logged.
- `--ca-file` trusts the `CERTIFICATE` blocks of that PEM file **instead of** the public web roots,
  for a gateway behind a private CA. It refuses a `ws://` upstream and a file with no parseable
  certificate.
- `--zenoh-name` sets the slot name, as on `run`; it defaults to the upstream's policy name.

One bridge serves one physicalai client. It ships in the same builds as `--zenoh`. Its loopback
port has no authentication: any process on the same host can call the remote policy through it,
with the bridge's token. Run it only on a host you trust, such as the robot's own machine.

## Real-time chunking (RTC)

A package exported with RTC (SmolVLA and π0.5 first) takes four more request fields:
`prev_chunk_left_over` (the not-yet-executed part of the previous chunk, `[T, D]`),
`inference_delay`, `execution_horizon` and `max_guidance_weight`, all four or none. The robot keeps
executing while the next chunk is computed, and the new chunk is pinned to what it has not executed
yet, which removes most of the jump between chunks. The server keeps no state: the client sends the
leftover with every call, so any call can go to any server. Without the fields (or at weight 0) an
RTC package behaves like a plain one. The metadata's `rtc_chunk_size` is non-null for an RTC
package, and a package without RTC refuses the fields. Details and measurements: the design doc's
"Real-time chunking" section.

## Limitations

Checked before any work is queued:

- WebSocket messages are capped at the size the spec allows (frames, state, noise, the RTC leftover
  and a 1 KiB prompt, plus 64 KiB of framing); larger ones close the connection.
- Prompts are at most 1 KiB of UTF-8, and a prompt longer than the tokenizer's port is refused (an
  error frame naming the limit), never silently truncated.
- At most `CASCADIA_API_MAX_CONCURRENT` (default 4) connections, shared by the WebSocket and Zenoh
  front ends, one request in flight each. At capacity the upgrade is refused with HTTP 503.
- The server pings every 20 s and closes a connection that misses a pong, so a dropped robot
  frees its slot. HTTP request headers must arrive within 10 s. There is no idle timeout.
- Non-finite actions (possible from f16 overflow on the GPU) are an error, never sent.
- One model per process; NPU, multi-machine sharding and server-side resize on the WebSocket are
  not supported.
- `molmoact2` is refused on an ARM CPU: OpenVINO's ARM CPU plugin computes its graph wrongly (6.1e-1
  off PyTorch on PyTorch's own inputs). Use an x86 CPU or the GPU.

## Doctor

`cascadia doctor` has an `ov-policy` section: it finds the OpenVINO tokenizers extension the engine
tokenizes prompts with (a miss is a warning naming the paths tried), and checks that uv 0.13.6 or
newer is on `PATH` for `cascadia export-policy` (a missing, unreadable or older uv is informational,
with the install or update command: serving needs no uv).

## Security

The policy API has **no authentication and no TLS**, like the chat API (see
[`../../SECURITY.md`](../../SECURITY.md)). A policy drives a robot, so the defaults are narrower:

- **Loopback only, unless you opt in.** `--engine ov-policy` binds `127.0.0.1:8000` unless `--api`
  is given, and both `--api` and `--zenoh` refuse a non-loopback host unless
  `--policy-allow-all-interfaces` is given (they then warn).
- **Bound to a network address, anyone who can reach the port can drive the robot.** With
  `--api 0.0.0.0:8000 --policy-allow-all-interfaces` (or any non-loopback host), every client on that network can read the
  metadata and send observations that become motor commands; the same holds for Zenoh. In this single-box OSS mode there is no way to restrict who calls.
  Do it only on a network you control, or keep the server on loopback and put an authenticated TLS
  hop in front (a reverse proxy, or `policy-bridge` with `--token-file` against a gateway that
  checks the token).
- **Origin allowlist.** A WebSocket upgrade that carries an `Origin` header not listed in
  `--policy-allow-origin` (empty by default) is refused, so a web page in a browser on the same
  machine cannot open the socket. Clients that send no `Origin` (robots, scripts) are always
  allowed: it is a guard against browsers, not authentication.
- **Packages are code-adjacent: load them only from sources you trust.** A package is not
  executable code, but its IRs are parsed and compiled by OpenVINO and the tokenizer extension, which
  are not hardened against hostile files, and its manifest decides how observations are transformed
  and how actions are scaled. A tampered package can make the robot move wrongly with no error.
  The metadata's `package_sha256` covers every file in the package except a top-level `.cache`
  folder (the per-download timestamps `hf download --local-dir` writes, so two servers that
  downloaded one revision match); compare it with the digest of the package you vetted. A manifest
  that names a path under `.cache` is refused at load, so nothing the engine reads is left out.
