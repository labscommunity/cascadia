# VLA policy engine (`ov-policy`) — 2026-11-15

## What this is

Serving vision-language-action (VLA) robot policies from Cascadia OSS: one observation in (camera
images, robot state, an instruction), one action chunk out. v1 runs Intel's `policy_package`
OpenVINO exports of **SmolVLA** and **π0.5** on the OpenVINO runtime OSS already links through
`cascadia-ov-genai-shim`. No new runtime, no Python on workers.

It is a new engine, not a new `Engine` impl: `Engine` streams text tokens (`GenerationTask` in,
`Chunk { token_id, text }` out), while a policy call is tensors in, one `[T, D]` float array out.
So it gets a sibling trait, `PolicyEngine`, its own serve path, and leaves `Engine` untouched.

## Why it belongs in OSS

This engine uses the OpenVINO runtime inside the GenAI SDK we already pin (2026.4.1), through the
shim's existing `cascadia_runtime_*` API that `ov-runtime` uses. Networking across
machines is out of scope.

## Evidence (measured 2026-11-15)

Latency: Core Ultra 7 258V, Arc 140V, Windows 11, OpenVINO 2026.4.1 (Python), GPU with the f16
hint, 20 runs, synthetic inputs, other services running on the same box:

| Package | GPU p50 / p90 | NPU | CPU p50 | Memory |
|---|---|---|---|---|
| `OpenVINO/smolvla-libero-fp16-ov` | 144 / 164 ms | compile fails (`Bucketize` i64 output) | 1868 ms | 2.0 GB |
| `OpenVINO/pi05-libero-fp16-ov` | 344 / 345 ms | compile fails (MLIR) | 11557 ms | 7.7 GB |
| `lerobot/pi05_droid`, exported by us | 312 / 313 ms | not run | not run | 7.6 GB |

Graph parity: the `pi05_droid` export (physicalai-train 0.3.0, `use_random_input_noise=False`, so a
constant zero start) matched PyTorch on the traced input on the OpenVINO **CPU device in fp32**: max
abs diff 0.0047, mean 0.0013, against a mean output magnitude of 0.78. Not yet measured: GPU f16
parity, and any task success.

Step 0 (2026-11-15) settled what the build depends on:
- The pinned GenAI SDK ships the tokenizer extension
  (`runtime\bin\intel64\Release\openvino_tokenizers.dll`, 2026.4.1.0). OpenVINO 2026.4.1 loads it
  and runs Intel's π0.5 and SmolVLA tokenizers on CPU (`[1, 64]` and `[1, 48]`).
- `physical-ai-studio` `8f132416` and runtime `e3dd944` install together only with a uv override:
  the studio names its own runtime revision in `[tool.uv.sources]`, and uv refuses two URLs for one
  package.
- An export of `HuggingFaceVLA/smolvla_libero` there takes `noise` as an input with no
  `RandomUniform` left, and Intel's runtime draws fresh noise per call. In float32, the IR matches
  PyTorch on the same noise to 4.1e-6 (max) on CPU f32.
- The exporter's default dtype (bf16) is unsafe: with a padded prompt mask and bf16 inference
  precision (the CPU default on bf16-capable hosts, here a Ryzen 9950X), every action is NaN (350 of
  350). A float32 export saved with fp16 weights (1,097 MB against 1,293 MB for bf16) is finite under
  both precisions and within 2.8e-3 of float32 PyTorch on CPU f32; bf16 compute on it is off by up to
  0.66 normalized units.

**That DROID package is not usable for a robot.** Its `normalize`/`denormalize` stats are identity
(mean 0, std 1, q01 −1, q99 1 over all 32 dimensions): `lerobot/pi05_droid` ships no normalization
stats (`policy_preprocessor.json` has `"features": {}`), and the export filled in placeholders. Its
latency row stands (stats do not change the graph); its accuracy does not. DROID stays out of v1
claims until a stats source is settled (open question 3).

## Scope

v1:
- Packages with `"format": "policy_package"`, version `1.x`, runner `SinglePass`, artifact
  `openvino`, policies `smolvla` and `pi05`. Anything else is refused at load with a clear error.
- Devices: GPU (default) and CPU. NPU is refused: both models fail NPU compilation as exported.
- A single-box metadata endpoint and WebSocket, served by `cascadia run --engine ov-policy`.
- `cascadia export-policy`: LeRobot checkpoints (Hugging Face) to a `policy_package`.

Not in v1: π0, π0-FAST, GR00T and other policies until each export is checked; NPU; multi-stage
sharding; per-robot key mappings for existing openpi clients; real-time chunking (RTC) server side;
server-side image resize; routing across machines.

## The package contract

The engine executes the manifest (`manifest.json`, mirrored in `config.json`); it does not guess.
Each step is ported from Intel's `physicalai` runtime (`openvinotoolkit/physicalai`,
`src/physicalai/inference/`, read at `e3dd944`) and pinned by fixtures from that code.

Loading rules:
- A component is either `{"type": ..., <params>}` or `{"class_path": ..., "init_args": {...}}`;
  `class_path` wins, mapped to the short names below. An unknown component refuses the load.
- `policy.name` is required: Intel's fallback that derives it from `policy.source.class_path` takes
  the wrong path segment (`policy` for `physicalai.policies.pi05.policy.Pi05`).
- Artifact paths (`model.artifacts.openvino`, `ov_tokenizer.artifact`) must be plain relative names
  inside the package directory: no `..`, no absolute, drive or UNC paths. Anything else refuses the
  load.
- The engine compares the manifest with the compiled model's ports (names via
  `Runtime::input_aliases`, shapes, dtypes) and refuses a mismatch at load, not at the first call.
- Placeholder statistics refuse the load: a `normalize`/`denormalize` whose every feature has
  mean 0 and std 1 (`mean_std`), or q01 −1 and q99 1 (`quantiles`), or min −1 and max 1
  (`min_max`), is an export that never had stats (our DROID package). `--allow-identity-stats`
  overrides it for testing.

| Step | Behaviour |
|---|---|
| `normalize` | per feature on `state`, in f64 like Intel's numpy promotion. `mean_std`: `(x − mean) / (std + eps)`. `min_max`: `2(x − min)/(max − min + eps) − 1`. `quantiles`: `2(x − q01)/d − 1` with `d = q99 − q01`, replaced by eps only where it is 0. `identity`. |
| `pi05` | Images: uint8 → `/255` → `x·2 − 1`, CHW. Slots in `input_features` order (see "Cameras"); `empty_cameras` extra slots filled with `−1`, mask false; stacked `images [n_slots, 1, 3, H, W]` f32 and `image_masks [n_slots, 1]` bool. Prompt: the normalized state is binned with `bin = digitize(x, linspace(−1, 1, 257)[:−1]) − 1`, which gives **−1 for x < −1** and 255 for x ≥ 1 − 2/256; text `"Task: {task}, State: {bins joined by ' '};\nAction: "`, task stripped with `_` and `\n` replaced by spaces. If `noise_shape` is set and the call carries no noise: `noise ~ N(0,1)` of `[1, *noise_shape]` f32. |
| `smolvla_resize` | Images as in `pi05` at `image_resolution` (512). With a non-empty `image_key_reorder_map`, each value is a **slot index**. When `num_cameras` > 0, camera `name` goes to slot `map[name]`, every index below `num_cameras` that no camera takes is a masked dummy of `−1` in that position, and a value ≥ `num_cameras` refuses the load. When `num_cameras` ≤ 0 (the default), cameras are ordered by their slot values with no dummies. A duplicate or negative value always refuses the load. |
| `new_line` | Append `"\n"` to the task if missing. |
| `noise` | `noise ~ N(0,1)` of `[1, *shape]` f32, unless the call carries noise. |
| `ov_tokenizer` | Run `tokenizer.xml` **on CPU, always** (its string ops are CPU-only; only the policy IR follows `--device`) → `tokenized_prompt` i64, `tokenized_prompt_mask` bool. Padding side, `max_length` and truncation live in the tokenizer IR and differ by package: Intel's π0.5 LIBERO pads left to 64 with a `truncation` state variable defaulting on; Intel's SmolVLA pads right to 48 with an always-on `Truncate` op; packages from our exporter (8f132416) carry the same `truncation` state variable, defaulting off, so an overlong prompt comes out wider than the port. See "Prompt length". |
| `denormalize` | On `action`, not an exact inverse: `mean_std`: `x·std + mean`; `min_max`: `(x + 1)/2·(max − min) + min` (no eps in either); `quantiles`: `(x + 1)/2·d + q01`, `d` as in `normalize`. |
| `action_chunk_trimmer` | Keep the first `n_action_steps` actions. |

### Prompt length

A prompt is checked once per distinct task text, at the first request that uses it, so the answer
never changes inside an episode: π0.5 writes the state bins into the prompt and their token count
moves with the state (Gemma splits digits, so a 3-digit bin costs more than a 1-digit one). The
check tokenizes the task with the worst-case state (every bin `255`) and the real token count:
- Where the IR has the `truncation` state variable (Intel's π0.5 LIBERO, and every package from our
  exporter), it is set to false for the check through the infer request's state, and the task is
  refused only when the count exceeds the port length; an exact fit is accepted.
- Where truncation is an always-on `Truncate` op (Intel's SmolVLA), a full mask is the only signal:
  the task is refused when the mask is full.

A refusal is an error frame naming the token limit. Intel's runtime truncates silently, which for
π0.5 drops the `;\nAction: ` suffix; we deliberately do not.

### Cameras

Slot order decides which camera the model sees where, so it is never taken from the client. Both
policies use the first rule that applies:
1. SmolVLA with a non-empty `image_key_reorder_map`: each camera at its mapped slot index, with
   dummies in the unassigned slots (see the `smolvla_resize` row).
2. `input_features` of type VISUAL, in manifest order. A feature `images.<name>` is advertised as
   `<name>`; a single bare `images` feature (single-camera exports) is advertised as `image`.
3. Neither (Intel's published SmolVLA manifest): `camera_0..camera_{n−1}` in IR slot order, where n
   is the IR `images` port's first dimension minus any dummy slots.

An empty map (the default in new SmolVLA exports) falls through to rule 2. New-style SmolVLA
exports name features with a doubled prefix (`images.images.image`) where π0.5 exports of the same
data strip it; the engine strips a second `images.` too, so both advertise `image` and `image2`. The shapes `input_features` declare are not the wire resolution:
Intel's π0.5 LIBERO declares `[3, 256, 256]` while its image step resizes to 224, and clients send
224×224. A request must carry
exactly the advertised camera names; a missing or extra name is an error.

Images must arrive at the model resolution, uint8 HWC RGB, so the engine never has to reproduce
OpenCV's bilinear resize pixel for pixel. The client resizes on the robot, and must use the
package's padding convention, which the metadata states as `resize`: bilinear, pad value 0 before
scaling, padded **centred** for `pi05` and **top and left** for `smolvla_resize` (content at the
bottom right). openpi's `resize_with_pad` centres, so it is correct for π0.5 only; a 640×480 frame
sent to SmolVLA through it lands 64 rows off with no error. `tools/` ships a reference resize for
each convention.

### Noise

Flow-matching policies start denoising from noise. The engine classifies each package by inspecting
the graph at load (the shim records the op types from the `ov::Model` before compiling, since it
keeps only the compiled model afterwards), and reports the mode in the metadata:

| Mode | How it is detected | What the engine does |
|---|---|---|
| `input` | the model has a `noise` input | draws `N(0,1)` per call, unless the call carries noise |
| `in_graph_random` | a `RandomUniform` op | serves it, with a warning: wrong distribution, and identical on every call on the GPU plugin. Intel's `pi05-libero-fp16-ov`. |
| `constant` | neither | serves it; the start is fixed (a deliberate deterministic export, or a 0.3.0 export that baked zeros). Intel's `smolvla-libero-fp16-ov`. |

Intel's fix is `physical-ai-studio` `374d2cb9` (2026-11-11, unreleased): exports take `noise` as an
input. The exporter produces `input`-mode packages only.

## Serving

`cascadia run --engine ov-policy --model <dir>` takes its own path in `cmd_run`, before
`WorkerArgs::single_node`: load and warm the engine, then serve the policy router plus `/health`.
`build_builder` and `cascadia worker` refuse `ov-policy` with an error; the dashboard and the chat
routes are not served for a policy model. There is no auto-selection: `--engine` defaults to
`ov-genai`, so the flag is required.

**Unauthenticated, like the chat API** (SECURITY.md: "No client authentication on the HTTP API").
Because a policy drives a robot, `ov-policy` binds `127.0.0.1:8000` unless `--api` is given.
`RunArgs.api` becomes `Option<String>`, resolved per engine (`:8000` for the chat engines as
today), since clap cannot tell an explicit `--api :8000` from a default. The WebSocket upgrade is
refused when an `Origin` header is present and not in `--policy-allow-origin` (empty by default),
so a browser page cannot open it. Authenticated network access is out of scope.

Limits, all checked before work is queued:
- WebSocket message cap computed from the spec: the cameras' H×W×3 bytes, state and noise
  arrays, the prompt cap, plus 64 KiB of framing. Larger messages close the connection.
- Prompt at most 1 KiB of UTF-8.
- At most 4 concurrent connections (`--policy-max-connections`); one request in flight per
  connection, as openpi's protocol is synchronous anyway.
- Dead connections free their slot: the server pings every 20 s and closes after a missed pong, so a
  robot whose Wi-Fi drops cannot leave a half-open socket holding a slot. The HTTP request headers
  must arrive within 10 s (hyper's `header_read_timeout`). There is deliberately no idle or
  first-request timeout: openpi clients open one connection per session and never reconnect, and
  the DROID client waits on a human (typing the instruction, rating an episode) between calls. A
  live idle client keeps answering pings.
- The handler splits the socket and keeps reading it while a request waits for the model lock and
  while inference runs, so pings are answered and pongs recorded throughout. openpi's client pings
  every 20 s and closes after 20 s without a pong; axum answers pings only while the socket is read,
  and π0.5 on CPU (about 11.5 s a call, four connections queued) leaves a socket unread for over
  40 s. A test runs CPU π0.5 with two or more clients for longer than 40 s.
- The policy serve loop is its own: the existing `serve_with_nodelay` builds hyper with no
  `.with_upgrades()`, so a WebSocket would answer 101 and never upgrade. The policy loop uses
  `serve_connection(...).with_upgrades()` with a timer, and a test opens a real TCP WebSocket
  through it (router-level `oneshot` tests cannot see this).
- At capacity the upgrade is refused with HTTP 503. openpi's client retries only on a refused TCP
  connection, so it exits with that error rather than waiting; the error text says the server is
  full.
- One compiled model per process behind a mutex; a call holds it for about 150–350 ms on the GPU.

## API

`GET /v1/policy/metadata` returns the `PolicySpec` as JSON. Each field has one source:

| Field | Source |
|---|---|
| `cameras` (names) | as in "Cameras" |
| `height`, `width` | the image step's `image_resolution`, else the IR `images` port |
| `state_len` | the IR `state` port |
| `chunk` `[T, D]` | T from `action_chunk_trimmer.n_action_steps` when present, else the IR output; D from the IR output |
| `noise` | the mode above |
| `noise_shape` | the IR `noise` port, `input` mode only (the decode rule and the message cap need it) |
| `resize` | the image step: `center` (`pi05`) or `top_left` (`smolvla_resize`), bilinear, pad 0 before scaling |
| `device`, `model`, `prompt_max_bytes` | the server |

`WS /v1/policy/ws` uses openpi's framing, so `openpi_client.WebsocketClientPolicy` can drive it:
- **Connecting.** openpi uses `host` as the full URI when it starts with `ws`, so the client passes
  `host="ws://<box>:<port>/v1/policy/ws"` and `port=None` (a port argument would be appended after
  the path).
- **Metadata.** On connect the server sends the metadata as one msgpack message.
- **Request.** One msgpack map: `images` (a map of camera name to array), `state` (array), `prompt`
  (string), and optional `seed` (int) and `noise` (array).
- **Reply.** `actions` (array `[T, D]`), plus `server_timing.infer_ms`.
- **Errors.** A text frame, as openpi does (no HTTP status on a socket).

Arrays use openpi's `msgpack_numpy` encoding: a map whose keys are msgpack **bin**
(`b"__ndarray__"`, `b"data"`, `b"dtype"`, `b"shape"`), the dtype as a numpy string, the shape as an
array. Replies must be encoded the same way, with `"<f4"` actions. This is pinned by golden bytes:
`tools/vla_reference.py` packs sample replies and requests with the real `openpi_client` and commits
the bytes as fixtures, and the Rust tests must produce and accept exactly those bytes, so CI needs
no Python. Decoding rules:

- `images`: dtype `|u1`, shape exactly `[H, W, 3]`.
- `state`: dtype `<f4`, or `<f8` cast to f32 (numpy's default); length `state_len`; every value
  finite.
- `noise`: only for a package in `input` mode (an error otherwise); dtype `<f4`, shape exactly
  `[1, *noise_shape]`; every value finite with |x| ≤ 10.
- Before anything is allocated: `len(data) == itemsize · Π shape`, computed with checked arithmetic.
  Any mismatch is an error naming the expected value.
- Replies: if any action is non-finite (possible from f16 overflow on the GPU), the call returns an
  error frame, never the actions.

Keys are ours, so openpi clients are transport-compatible, not drop-in: a DROID client's
`observation/joint_position` needs a mapping (out of scope for v1).

## Code layout

| Where | Change |
|---|---|
| `cascadia-types` | `policy.rs`: `Observation { images: Vec<(String, Image)>, state: Vec<f32>, prompt: String, seed: Option<u64>, noise: Option<Vec<f32>> }`, `ActionChunk { actions: Vec<f32>, steps, dim, infer_ms }`, `PolicySpec` (fields as in "API"). |
| `cascadia-engine` | `PolicyEngine: Send { fn spec(&self) -> &PolicySpec; fn warmup(&mut self); fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk>; fn close(&mut self) {} }` and a deterministic `MockPolicyEngine` for API and CLI tests on macOS. |
| `cascadia-ov-genai-shim` | A `bool` dtype in `dtype_from_code`, `code_from_dtype`, the header defines and Rust `DType` (`from_code`, `bytes_per_element`); a string-tensor input for the tokenizer; a separate CPU runtime handle for the tokenizer, with the `openvino_tokenizers` extension added to that handle's `ov::Core`, and a way to set an infer request's state variable (`truncation`); `cascadia_runtime_has_op_type(handle, type)`, answered from op types recorded off the `ov::Model` before compile. Alias lookup already exists (`input_aliases`). |
| `cascadia-engine-openvino` | `policy.rs`: manifest parsing (both component forms, version check, path confinement), port checks, the steps above, noise-mode detection, `OvPolicyEngine`. |
| `cascadia-api` | Policy router on its own state (the chat `AppState` stays untouched), the limits above, `axum` `ws` feature, msgpack with bin keys. |
| `cascadia-cli` | `EngineKind::OvPolicy`, the policy serve path in `cmd_run`, `RunArgs.api` as `Option<String>`, refusal in `worker`/`build_builder`, `export-policy` with its driver script and hashed requirements file embedded by `include_str!` (as `cascadia shard` embeds `EXPORT_REQUIREMENTS`, since a release bundle carries only the binary), a `doctor` check for the tokenizer extension. |
| `tools/` | `vla_reference.py` (fixture generator), `vla_libero_client.py` (benchmark adapter). Sources only; nothing here is read at run time. |
| `.github/workflows/ai-pc-tests.yml` | A step that reads the repository variable `CASCADIA_POLICY_PACKAGE_DIR`, checks the package hashes, exports `CASCADIA_POLICY_PACKAGE` and runs the policy tests (see "Testing"). |
| docs | `docs/engines/ov-policy.md`; `ARCHITECTURE.md` and `docs/engines/README.md` updated for the sibling trait and the new routes. |

## Exporter: `cascadia export-policy`

`cascadia export-policy <hf-model-id> --out <dir>` runs Intel's `physicalai-train` in **its own
virtual environment**, created with `uv` under the Cascadia cache. (`cascadia shard` has no such
environment: it uses the caller's Python and prints a `pip install` line; the exporter does not
follow that, because its dependencies would collide.) Python is needed only on the exporting
machine.

- **Requires `uv`.** Nothing in the repo ships it; when `uv` is not on PATH the exporter refuses
  with the install command, before doing anything else.
- **Pinned install.** The environment installs from a hashed requirements file (`uv pip compile
  --generate-hashes`) embedded in the binary:
  - `physicalai-train` at `physical-ai-studio` `8f132416` (it contains `374d2cb9`);
  - the `physicalai` runtime at `e3dd944`, the same revision the engine is ported from and the
    fixtures are generated with. That commit's own `uv.lock` names an older runtime (`8381d43`), so
    the lock overrides it: one runtime revision for the port, the fixtures and `--verify`. If the two
    prove incompatible, step 0 finds out before any porting;
  - `openvino==2026.4.1` and a matching `openvino-tokenizers`;
  - torch CPU wheels.

  The runtime pin is a uv override (`--overrides`), not a plain requirement: without it uv refuses
  to resolve (step 0).
- **Precision.** The policy is loaded with `dtype="float32"` and the IR saved with fp16 weights. The
  tool's default (bf16) exports produce all-NaN actions under bf16 inference with a padded prompt
  (step 0), so the exporter never writes one.

  The commit alone pins nothing below it (its `pyproject.toml` says `openvino>=2026.4`). Bump to a
  release when one carries the fix.
- **Chunk length** comes from the checkpoint (`n_action_steps=None`); the loader's default of 50
  rejects 15-step checkpoints.
- **Refusals before anything is downloaded:**
  - **Memory.** Less than 32 GiB of available RAM plus free swap (the main gate), or less than
    15 GiB of physical RAM as the OS reports it (a 16 GiB VM reports about 15.6). Measured: 14.3 GB
    RSS plus up to 10 GB swap for π0.5, completed on a 16 GiB box with 48 GiB swap. A 38.6 GB Mac
    with other work running froze and needed a forced restart.
  - **A serving box.** A cascadia node or OVMS process running on this host. The message suggests a
    disposable Linux machine.
  - **Gated files.** No access to `google/paligemma-3b-pt-224` for π0/π0.5, with a message saying
    how to accept the Gemma licence.
  - **Missing statistics.** No normalization stats in the checkpoint, unless `--stats <file>` is
    given. This catches the identity stats above.
- **Licence.** The package gets the upstream licence and a Gemma-terms notice, and the exporter
  prints which terms apply.
- **`--verify`** runs the traced input through PyTorch and OpenVINO, on CPU fp32 and on GPU f16
  when a GPU is present, and prints the max and mean differences.

## Testing

- **Fixtures.** `tools/vla_reference.py` runs Intel's `physicalai` runtime at `e3dd944` on small
  inputs and writes fixtures to `crates/cascadia-engine-openvino/tests/fixtures/policy/`, under
  100 KB in total, plus one end-to-end reference per staged package (input, noise and expected
  actions, a few KB). Python runs only when fixtures are regenerated, never in CI. Cases:
  - each `normalize`/`denormalize` mode, and the placeholder-stats refusal (our DROID manifest);
  - π0.5 binning, including values below −1 and above 1;
  - prompt text;
  - camera ordering under all three rules, and dummy slots;
  - token ids for a few prompts per tokenizer (ids only, no weights), including one that fills the
    port;
  - golden msgpack bytes for requests and replies, packed by the real `openpi_client`.
- **No OpenVINO (the stub CI lane and macOS).** Unit tests against those fixtures: manifest parsing,
  both component forms, path confinement, version refusal, placeholder stats, request decoding with
  every rejection rule, msgpack bytes against the golden fixtures, and API/CLI against
  `MockPolicyEngine`.
- **With OpenVINO.** On the self-hosted AI-PC lane (`ai-pc-tests.yml`), the only lane that runs
  `cargo test --features openvino`; the Linux lane only type-checks. The tests are ordinary tests
  that print a skip reason and return when `CASCADIA_POLICY_PACKAGE` is unset (`#[ignore]` is fixed
  at compile time, and the lane does not pass `--ignored`).
  - **Staging.** The lane is a pool of runners, so the package is staged on every runner with the
    `ai-pc` label, at the path in the repository variable `CASCADIA_POLICY_PACKAGE_DIR`.
  - **Hash check.** The committed reference records the SHA-256 of the package's `manifest.json`
    and IR files. When the variable is set, a missing directory or a hash mismatch **fails** the
    step; only an unset variable skips. The step prints the hashes it ran against.

  Steps:
  1. Tokenizer ids against the fixtures, and the prompt-length check with the worst-case state.
  2. End to end against the committed reference for that package, with the same `noise` passed in.
     It runs on our `input`-mode export of `HuggingFaceVLA/smolvla_libero` (not gated); Intel's
     published SmolVLA has no noise input. Differences are measured **before `denormalize`**, in
     normalized units, so every action dimension counts the same (a raw 0.05 is more than one
     standard deviation of LIBERO's rotation actions). Tolerances are fixed up front: ≤ 1e-4 for
     our CPU run against Intel's runtime on CPU, and ≤ 0.02 for GPU f16 against that CPU reference.
     The CPU comparison only means "a port bug" when everything else is equal, so both sides set
     `INFERENCE_PRECISION_HINT=f32` (OpenVINO's CPU plugin otherwise picks bf16 or f16 from the
     hardware), the reference is generated on an `ai-pc` runner, and the reference file records the
     generator's CPU and precision next to the package hashes. A first GPU run above
     0.02 is a finding to investigate, not a threshold to adopt.
- **Accuracy.** `tools/vla_libero_client.py` adapts Intel's LIBERO benchmark observations to the
  WebSocket schema. It runs a few episodes of one suite on `HuggingFaceVLA/smolvla_libero` and
  `lerobot/pi05_libero_finetuned_v044`, both exported in `input` mode, against Intel's published
  packages (`constant` and `in_graph_random`). Speed and parity do not show that a robot succeeds at
  a task; this does. Where the simulator runs is open question 2.
  The adapter connects through `openpi_client.WebsocketClientPolicy` exactly as the doc tells users
  to (`host="ws://…/v1/policy/ws"`, `port=None`), so the accuracy run is also the end-to-end
  interoperability test.
- **Rig.** Latency through `cascadia run` on a Core Ultra AI PC for SmolVLA and π0.5 LIBERO, plus one
  scripted check through the real `openpi_client`: the metadata arrives first, one `infer` returns
  actions, a bad request raises the error frame's text, and a fifth connection is refused at
  capacity.

## Open questions

1. **GPU f16 accuracy.** bf16 compute is off by up to 0.66 normalized units on the fp16-weight
   package (step 0, CPU). The Arc GPU computes in f16, which has more mantissa than bf16 but less
   range. It must be measured on a Lunar Lake box with a padded prompt before the 0.02 GPU tolerance
   is trusted; if it fails, the engine runs the policy at f32 on the GPU instead.
2. **Where the LIBERO simulator runs.** It needs MuJoCo and Linux. The engine needs an AI PC GPU.
   The two must reach each
   other over the network.
3. **DROID statistics.** openpi publishes DROID norm stats with its checkpoints, but in its own
   format. Can the exporter take them through `--stats`?
4. **GPU memory.** When an LLM engine and a policy share one box: out of scope for v1 (one engine
   per process), left to the operator.
