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
  `openvino`, policies `smolvla`, `pi05`, `pi0`, `xvla`, `evo1`, `vlajepa`, `xr0`, `gr00t_n1d7` and
  `molmoact2`. Anything else is refused at load with a clear error. The π0, X-VLA, EVO1, VLA-JEPA and XR0 packages are our
  exports (Intel's runtime has no π0, X-VLA or EVO1, its VLA-JEPA preprocessor is a draft, and its
  XR0 export draws noise in the graph, which ours takes as an input); `xvla` and
  `evo1` take their cameras and prompt like `smolvla` (`evo1` strips the task with Python's
  `str.strip()`, as LeRobot's EVO1 does). `vlajepa`, `xr0`, `gr00t_n1d7` and `molmoact2` have their
  own input ports (see "Port layouts"). The `molmoact2` package is Intel's own export, with its own
  runtime steps. The `gr00t_n1d7` package is Intel's edge-ai-suites single-IR export of
  NVIDIA's checkpoint, saved with fp16 weights, static shapes and named ports.
- Devices: GPU (default, f16 unless `--ov-inference-precision f32`) and CPU (always f32). NPU is refused:
  SmolVLA and π0.5 fail NPU compilation as exported, and X-VLA compiles but returns wrong actions.
- X-VLA's LIBERO packages flip `image2` on the host (`image_flip`), not in the graph: on the Arc
  140V GPU plugin (OpenVINO 2026.4.1) a flip inside the graph, which exports as a negative-stride
  `Slice`, gives wrong `image2` features even at f32, while the CPU is exact. An exporter must keep
  flips and negative-stride slices out of a GPU graph.
- π0 on a 16 GB integrated GPU (Arc 140V) runs at f16 only: at `--ov-inference-precision f32` its weights
  (about 12.5 GB in f32) fail GPU compilation. A failed compile is a load error naming the IR and
  the device, with a hint to drop `--ov-inference-precision f32` on the GPU; nothing is special-cased per model.
- EVO1 (`zuoxingdong/evo1_libero`, InternVL3-1B with 32 flow steps in one graph) on the Arc 140V:
  GPU f16 about 250 ms a call, served actions within 6.4e-4 of LeRobot's, no gripper flips. That
  holds only because the exporter splits the action head's sequence-pooling FC (one MatMul over
  50×896 = 44,800 inputs) into a batched MatMul per horizon slot plus a sum: unsplit, the GPU plugin
  (OpenVINO 2026.4.1) at f16 gets every flow step after the first wrong (max 2.19, no NaN) while f32
  and the CPU are right, and `rt_info` f32 pins do not help. An exporter must keep FCs with tens of
  thousands of inputs out of an f16 GPU graph.
- XR0 (`XiaomiRobotics/Xiaomi-Robotics-0-LIBERO`, Qwen3-VL-4B with a DiT, 5 flow steps) on the Arc
  140V: fp16 weights at GPU f16 about 232 ms for a 30-step chunk, within 3.2e-3 of PyTorch f32, no
  NaN, about 11 GB. At `--ov-inference-precision f32` the GPU compile fails (`CL_INVALID_EVENT`), a load error as
  for π0; CPU f32 is about 5.6 s a call and 26 GB.
- VLA-JEPA (`lerobot/VLA-JEPA-LIBERO`, Qwen3-VL-2B with all four flow steps in one graph) on the Arc
  140V: GPU f16 about 99 ms a call and within 8.2e-4 of PyTorch f32, GPU f32 about 610 ms, CPU f32
  about 3 s, no gripper flips. The GPU f16 default holds: LIBERO `libero_spatial` 29/30 at GPU f16,
  as PyTorch through the same client.
- GR00T N1.7 (`nvidia/GR00T-N1.7-LIBERO`, `libero_spatial`; Cosmos-Reason2-2B with a 4-step flow
  head in one graph) on the Arc 140V, the IR alone through OpenVINO's Python API on random frames:
  GPU f16 about 160 ms for a 16-step chunk, within 7.4e-3 (normalized) of PyTorch f32, no NaN,
  about 7 GB; CPU f32 about 3 s, within 3.3e-3 in served units. Through the engine (whole call,
  pre- and postprocessing included) see the tier table.
  OpenVINO matches Intel's PyTorch export wrapper to 3.4e-6; the wrapper itself is 4.7e-3 to
  7.1e-3 off NVIDIA's model, a gap not yet traced; it costs no task success: LIBERO
  `libero_spatial` 27/30 at GPU f16 against 26/30 for NVIDIA's PyTorch model on the same
  episodes. At `--ov-inference-precision f32` the GPU compile fails (`CL_INVALID_EVENT`), a load error as for
  π0. One package per embodiment and camera count.
  NVIDIA's licence notice (the package's `NOTICE`) is served as the metadata's `attribution`.
- MolmoAct2 (`allenai/MolmoAct2-LIBERO`, 5.4B, 10 flow steps from zeros in one graph) on the Arc
  140V, the IR alone through OpenVINO's Python API: GPU f16 about 0.92 s for a 10-step chunk, within
  1e-3 (normalized) of PyTorch f32, no NaN, about 10 GB (through the engine see the tier table);
  at `--ov-inference-precision f32` the GPU compile fails (`CL_INVALID_EVENT`), a load error as for π0. LIBERO
  `libero_spatial` 28/30 at GPU f16 against 27/30 for Intel's PyTorch at f32 on the same
  episodes. The x86 CPU at f32 is within 1e-3 of PyTorch (about 35 s a call on a Core Ultra 7);
  OpenVINO 2026.4.1's ARM CPU plugin computes the graph wrong (the IR alone is 0.61 off PyTorch on
  PyTorch's own inputs on macOS), so MolmoAct2 on an ARM CPU
  is refused at load, before the compile (use an x86 CPU or the GPU). Its noise is
  `constant`: the graph starts from zeros, so calls are deterministic. The weights' licence is unstated on the model card (the code is Apache-2.0).
- A single-box metadata endpoint and WebSocket, served by `cascadia run --engine ov-policy`.
- `cascadia export-policy`: LeRobot checkpoints (Hugging Face) to a `policy_package`.

Tiers: a kind is **supported** once it has PyTorch parity, Windows CPU and GPU runs and LIBERO
success at f16; until then it is **experimental** (served, with metadata, docs and CLI, but not
promised).

LIBERO success is measured the same way for every kind: `libero_spatial`, 10 tasks × 3 episodes
(init states 0–2), seed 7, through `tools/vla_libero_client.py`. The kind passes when its GPU f16
success lies inside the 95% (Wilson) interval of the same checkpoint in PyTorch, served through the
same client on the same episodes. Thirty episodes separate only large gaps, so this bar catches a
broken port, not a few points. Windows runs are on a Core Ultra 7 258V with the Arc 140V; "engine"
means the whole call through `cascadia run` or `policy_openvino`, pre- and postprocessing
included, and differences are against PyTorch f32 unless stated.

| Kind | Ours, GPU f16 | PyTorch, same episodes | Build | Windows CPU and GPU | Tier |
|---|---|---|---|---|---|
| `smolvla` | 23/30 | 19/30 (LeRobot, CPU f32; LeRobot's own `lerobot-eval` 23/30) | `6ef2a88e9` | engine: GPU f16 ~250 ms, within 9.5e-4 of the reference; CPU f32 1.8 s, ~1e-6 | supported |
| `pi05` | 28/30 (Intel's `pi05-libero-fp16-ov`) | 29/30 (LeRobot, CPU f32) | `125dd23d0` | engine: GPU f16 ~620 ms, within 1.03e-3 of CPU f32 on the same box (our export) | supported |
| `pi0` | 15/30 | 19/30 (LeRobot, CPU f32) | `125dd23d0` | engine: GPU f16 only (f32 does not compile); CPU f32 11.3 s, within 1.8e-3 | supported, at the low edge of the interval (0.46–0.78): 8 episodes only PyTorch solved, 4 only ours |
| `xvla` | 26/30 | 25/30 (LeRobot, CPU f32) | `125dd23d0` | engine: GPU f16 position within 4.9e-3, rotation 1.1°, one gripper near its threshold; CPU f32 5.9 s, within 2.9e-4 | supported |
| `vlajepa` | 29/30 | 29/30 (LeRobot, CPU f32) | `125dd23d0` | engine: GPU f16 ~107 ms (the LIBERO server); CPU f32 3.0 s within 1.65e-4 through OpenVINO's Python API, the engine's CPU path within 1.65e-4 on macOS | supported |
| `evo1` | 28/30 | 28/30 (LeRobot, CPU f32) | `125dd23d0` | engine: GPU f16 ~330 ms, within 6.4e-4; CPU f32 6.1 s, within 3.6e-4 | supported |
| `gr00t_n1d7` | 27/30 | 26/30 (NVIDIA's model, CPU bf16) | `125dd23d0` | engine at `7a6b19e5a`: GPU f16 ~240 ms, within 4.8e-3; CPU f32 3.5–4.3 s, within 3.3e-3 | supported |
| `xr0` | 27/30 | 27/30 (Intel's, CPU f32) | `125dd23d0` | engine: GPU f16 ~231 ms (the LIBERO server), within 3.2e-3 through OpenVINO's Python API; CPU f32 5.6 s within 8.3e-4 through the Python API | supported |
| `molmoact2` | 28/30 | 27/30 (Intel's, CPU f32) | `8da6f67c2` | engine at `7a6b19e5a`: GPU f16 ~990 ms, within 4.0e-4; CPU f32 35 s, within 3.6e-4 | supported |

Every v1 kind meets the rule; π0 only just. No run had an error frame or a non-finite action. For
VLA-JEPA and XR0 the Windows CPU run is the package through OpenVINO's Python API, not the engine.
The `gr00t_n1d7` and `molmoact2` engine runs were repeated at `7a6b19e5a`, after RTC and the
package digest landed, and gave the same differences as at their LIBERO builds.

Not in v1: π0-FAST, which no exporter can produce as a package: physical-ai-studio (`8f132416`
and main of 2026-11-16) wraps it as a LeRobot alias whose only export backend is Torch, so
`to_openvino` raises `NotImplementedError`; and LeRobot's π0-FAST decodes its action tokens one at
a time with a KV cache, stopping at EOS, then turns them into actions with a BPE decode and an
inverse DCT on the host, which the single-pass runner cannot run. Other policies until each export
is checked; NPU; multi-stage sharding; per-robot key mappings for existing openpi clients;
server-side image resize; routing
across machines. Real-time chunking (RTC) is served for packages exported with it (see "Real-time
chunking"), SmolVLA and π0.5 first.

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
  (`min_max`), is an export that never had stats (our DROID package). `--policy-allow-identity-stats`
  overrides it for testing.

| Step | Behaviour |
|---|---|
| `normalize` | per feature on `state`, in f64 like Intel's numpy promotion. `mean_std`: `(x − mean) / (std + eps)`. `min_max`: `2(x − min)/(max − min + eps) − 1`. `quantiles`: `2(x − q01)/d − 1` with `d = q99 − q01`, replaced by eps only where it is 0. `identity`. Ours, GR00T's: `min_max_clip` with `bounds` `["min", "max"]` (the default) or `["q01", "q99"]`: `2(x − lo)/(hi − lo) − 1`, 0 where `np.isclose(hi, lo)`, clipped to [−1, 1]; `pad_to` zero-pads the result (GR00T's `max_state_dim`) and `history` must be 1; both only for `gr00t_n1d7`. |
| `pi05` | Images: uint8 → `/255` → `x·2 − 1`, CHW. Slots in `input_features` order (see "Cameras"); `empty_cameras` extra slots filled with `−1`, mask false; stacked `images [n_slots, 1, 3, H, W]` f32 and `image_masks [n_slots, 1]` bool. Prompt: the normalized state is binned with `bin = digitize(x, linspace(−1, 1, 257)[:−1]) − 1`, which gives **−1 for x < −1** and 255 for x ≥ 1 − 2/256; text `"Task: {task}, State: {bins joined by ' '};\nAction: "`, task stripped with `_` and `\n` replaced by spaces. If `noise_shape` is set and the call carries no noise: `noise ~ N(0,1)` of `[1, *noise_shape]` f32. |
| `pi0` | Ours (Intel's runtime dropped π0). The `pi05` parameters and image handling (dummy slots, centred padding, optional `noise_shape`); the prompt is left alone, so with `new_line` the tokenizer sees the task plus `"\n"` and no state bins. |
| `smolvla_resize` | Images as in `pi05` at `image_resolution` (512). Ours: `resize` is `top_left` (the default, Intel's convention) or `stretch` (EVO1: LeRobot stretches frames bicubically, with antialiasing and no padding), and only changes the advertised `resize`. With a non-empty `image_key_reorder_map`, each value is a **slot index**. When `num_cameras` > 0, camera `name` goes to slot `map[name]`, every index below `num_cameras` that no camera takes is a masked dummy of `−1` in that position, and a value ≥ `num_cameras` refuses the load. When `num_cameras` ≤ 0 (the default), cameras are ordered by their slot values with no dummies. A duplicate or negative value always refuses the load. |
| `image_flip` | Ours. `cameras`: advertised camera names whose frames are rotated 180° (both H and W flipped) before the CHW conversion; a name that is not one of the package's cameras, a duplicate or an empty list refuses the load. Clients send frames as the environment gives them. The flip applies to the frame as sent, so it matches a flip-then-resize only when no padding is needed (a square frame to a square model input, as in LIBERO). |
| `new_line` | Append `"\n"` to the task if missing. |
| `noise` | `noise ~ N(0,1)` of `[1, *shape]` f32, unless the call carries noise. `port` (ours, `gr00t_n1d7` only) names the input it feeds, `actions` for GR00T. |
| `ov_tokenizer` | Run `tokenizer.xml` **on CPU, always** (its string ops are CPU-only; only the policy IR follows `--device`) → `tokenized_prompt` i64, `tokenized_prompt_mask` bool. Padding side, `max_length` and truncation live in the tokenizer IR and differ by package: Intel's π0.5 LIBERO pads left to 64 with a `truncation` state variable defaulting on; Intel's SmolVLA pads right to 48 with an always-on `Truncate` op; packages from our exporter (8f132416) carry the same `truncation` state variable, defaulting off, so an overlong prompt comes out wider than the port. Our X-VLA package's BART tokenizer pads right to 50 and also has the variable (defaulting on). See "Prompt length". |
| `denormalize` | On `action`, not an exact inverse: `mean_std`: `x·std + mean`; `min_max`: `(x + 1)/2·(max − min) + min` (no eps in either); `quantiles`: `(x + 1)/2·d + q01`, `d` as in `normalize`; `min_max_clip`: the action clipped to [−1, 1], then `(x + 1)/2·(hi − lo) + lo`. |
| `action_slice` | Ours (`gr00t_n1d7` only). Keep the first `dims` values of each row (GR00T's 132-wide padded actions to LIBERO's 7); wider than the actions at its step refuses the load. |
| `action_chunk_trimmer` | Keep the first `n_action_steps` actions. |
| `gripper_binarize` | Ours (LeRobot EVO1's LIBERO postprocessing). `index`, `threshold`, `above`, `below`: each row's `a[index]` becomes `above` if `a[index] > threshold` (strictly), else `below`. Runs where it is listed (after `denormalize` for EVO1); an `index` outside the width at its step refuses the load. |
| `vla_jepa_prompt` | Ours. `template`: the chat prompt with every `{instruction}` replaced by the task verbatim (no stripping); a template without `{instruction}` refuses the load. Must come before `ov_tokenizer`. |
| `vla_jepa_inputs` | Ours, ported from Intel's draft `VLAJEPAPreprocessor` (runtime `e3dd944`). Frames uint8 HWC at exactly `image_resolution` (224×224; other sizes are refused, so Intel's first area resize is the identity) → `/255` → antialiased bicubic to `vision_resolution` (256×256) with separable matrices built at load like Intel's `_bicubic_weights` (`a = −0.5`, centre `(i + 0.5)·scale`, support `2·max(scale, 1)`, rows normalized in f64, stored f32; `rows @ img @ colsᵀ` summed in f64) → `(x − mean)/std` → tiled `temporal_patch_size` times → reshaped `(1, tp, 3, gh/m, m, p, gw/m, m, p)`, transposed `(0, 3, 6, 4, 7, 2, 1, 5, 8)` → `[gh·gw, 3·tp·p²]` per camera in `image_keys` order, concatenated into `pixel_values`. `embodied_positions`: the indices of `embodied_token_id` in the ids; anything but exactly `num_embodied_tokens` (at load, for the empty task, and per call) is refused. A `vision_resolution` not a multiple of `patch_size · merge_size`, or an unknown parameter, refuses the load. The state goes to the graph **raw** and the graph's actions are served as they are: its normalize and denormalize stats are graph constants, so a `vlajepa` manifest may not carry `normalize`, `denormalize` or any step but these, `ov_tokenizer`, `noise` and `action_chunk_trimmer` (and there is no placeholder-stats check to run). |
| `xr0` | Intel's `XR0Preprocessor` (runtime `e3dd944`). Cameras are the VISUAL `input_features` in order, all one `[3, H, W]`; frames uint8 HWC at exactly that size, and only a size Intel's resize (sides to multiples of `image_factor`, area ≤ `image_max_pixels`) leaves alone, so its OpenCV resize never runs (LIBERO: 256×256). Pixels `(x · rescale_factor − mean)/std` in f32, then the Qwen patchify `vla_jepa_inputs` uses. Prompt: Qwen3-VL's chat template, `# <Title> View` per camera (`base` → Base, `wrist_left` → Left-Wrist, …) with `patches/merge²` image pads, then `Generate robot actions for the task:\n{task} /no_cot`, the task verbatim. State: raw, zero-padded to `max_state_dim`, `[1, 1, 32]`. `normalize_state: true`, a non-empty `image_key_view_map` or an unknown parameter refuses the load. `noise` may follow it. |
| `xr0_denormalize` | Intel's `XR0Postprocessor`: `action · (std + action_eps) + mean` in f32 with per-timestep `[T, D]` stats (which must match the graph's output at load), then the first `action_dim` values of a row. `action_mode: delta` (needs a state passthrough output) and unknown parameters refuse the load; all-identity stats are a placeholder. |
| `gr00t_image` | Ours, NVIDIA's eval transform (Isaac-GR00T `3df8b382`, `image_augmentations.py`): letterbox to a square, shortest edge to `shortest_edge`, a centred `crop_fraction` crop (`max(1, int(side · fraction))` at `(side − crop) // 2`), back to `shortest_edge` with OpenCV `INTER_AREA`. The engine takes only square frames of `shortest_edge` (every VISUAL feature `[3, s, s]`), so the letterbox and first resize do nothing; the 243 → 256 upscale is OpenCV's own arithmetic, ported bit for bit (`INTER_AREA` that does not shrink is a fixed-point bilinear with area-mode weights, the vertical pass in OpenCV's 128-bit SIMD form), checked against `cv2.resize`. `interpolation` other than `area` refuses the load. `image_flip` runs before it. |
| `qwen_vl_patchify` | Ours. Pixels `(x/255 − mean)/std` in f32, then the Qwen patchify `vla_jepa_inputs` uses, cameras in VISUAL order. `grid_thw` must be `[1, s/p, s/p]` per camera. |
| `gr00t_prompt` | Ours. With `formalize_language`, the task lowercased and every character but `\w` and `\s` dropped (Python's `re.sub(r"[^\w\s]", "", task.lower())`; Rust keeps the few combining marks Unicode counts as alphabetic, which Python drops); then `template` with `{images}` replaced by `image` once per camera and `{task}` by the task. |
| `qwen_vl_inputs` | Ours. The tokenizer's unpadded ids, each `image_token_id` repeated `tokens_per_image` (the merged patches of a frame, checked at load) times, right-padded with `pad_id` to `seq_len`, `attention_mask` 1 on the ids; `position_ids [3, 1, seq_len]` are Qwen3-VL's mRoPE (text: one position for all three rows; an image: `t`, `t + row`, `t + col` over the merged grid, the next text at its maximum + 1), 0 on the padding. `padding_side` other than `right` or `rope` other than `qwen3_vl_mrope` refuses the load. Must follow `ov_tokenizer`. |
| `constant` | Ours. An `int64` `value` fed to `port` on every call (GR00T's `embodiment_id`); its length must be the port's. |
| `molmoact2` | Intel's `MolmoAct2Preprocessor` (runtime `e3dd944`), with Intel's f32 arithmetic (its stats are f32 arrays). Cameras are `image_keys` (`images.` optional) in order; frames uint8 HWC at exactly `image_size` (378×378), so its `INTER_LINEAR_EXACT` resize is the identity. State: quantile-normalized with the mask (masked dims pass through), clipped to [−1, 1], each value written as `<state_{rint((s + 1)/2 · (num_state_tokens − 1))}>` between `<state_start>` and `<state_end>` (NaN → 0, ±inf → ±1), one added token per value, so the prompt's length does not move with the state. Task: whitespace runs to one space, one leading `task:` / `instruction:` / `language instruction:` / `goal:` (`:` or `-`, case-insensitive) and then one leading `the task is to` / `your task is to` dropped, trailing `.,!?;:` dropped, lowercased. Prompt: `Image 1<|image|>Image 2<|image|>` (one camera: `<|image|>`), then `<|im_start|>user\nThe task is to {task}. The setup is <setup_start>{setup_type}<setup_end>. The current state of the robot is {state tokens}. The expected control mode is <control_start>{control_mode}<control_end>. Given these, what action should the robot take to complete the task?<|im_end|>\n<|im_start|>assistant\n<action_output>` (setup and control wrapped only when non-empty, flagged and not already wrapped). A `normalization_mode` other than `QUANTILES`, missing `state_stats` or an unknown parameter refuses the load. |
| `molmoact2_inputs` | Intel's `MolmoAct2ModelInputs`, single-crop `resize` mode only (`image_crop_mode` otherwise refuses the load). BOS (`bos_token_id`) prepended to the valid ids unless already first, the width growing by one; each `<|image|>` placeholder becomes `image_start`, `image_patch` × the pooled grid (with `image_col_id` per row when `use_single_crop_col_tokens`), `image_end`; re-padded with `pad_token_id` to the expanded length plus the pads, mask 1 over the ids; `token_type_ids` 1 on `image_token_ids` under the mask. `images [1, cameras, patches, patch²·3]`: `(x/255 − mean)/std` in f32, row-major 14×14 patches each flattened `(py, px, c)`. `token_pooling [1, cameras · pooled, 4]`: the patch-index grid padded with −1 up to whole 2×2 cells (before: half the pad, after: the rest), one row per cell, image `k`'s indices offset by `k · patches`. `action_dim_is_pad [1, max_action_dim]`: false for the first `action_dim`. Must follow `ov_tokenizer`. |
| `molmoact2_postprocess` | Intel's `MolmoAct2Postprocessor`: `action` clipped to [−1, 1], then the f32 quantile denormalize with its mask (masked dims stay clipped). An `action_key` other than `action`, missing `action_stats` or a non-quantile mode refuses the load; identity quantiles in either block are a placeholder. |
| `xvla_rotation_6d_to_axis_angle` | Ours, ported from LeRobot's step of that name. Each action row (at least `max(expected_action_dim, 10)` wide, checked at load) becomes 7 values: position `[0:3]` as is; the 6-D rotation `[3:9]` to axis-angle (Gram-Schmidt with LeRobot's `1e-6` eps, the quaternion as the top eigenvector of Bar-Itzhack's K with `w ≥ 0`, then axis × angle), in f64; the gripper `[9]` as 1 above 0.5, else −1. Near π the axis-angle sign follows `w ≥ 0`, as in LeRobot. |

### Templated prompts

`vla_jepa_prompt`, `xr0`, `gr00t_prompt` and `molmoact2` put the task into a chat template, so a task that spells a special token
(`<|image_pad|>`, `<|im_end|>`, `<|embodied_action|>`, …) would be tokenized as structure: image
features would land in the wrong positions or the template would end early, and the model would
get a silently wrong input. Such a task is refused (an error frame, checked once per task text like
the length). The set is read from the package, not guessed: the strings in the tokenizer IR's
`SpecialTokensSplit` pattern, `added_tokens` of a `tokenizer.json` beside it, and every `<…>`
marker of the template itself (`<cot>`); a package whose tokenizer declares none is refused at
load. Behind that, every call's ids must carry each single-token marker exactly as often as the
template does (XR0: 64 image pads per camera, checked at load; VLA-JEPA: 32 embodied tokens;
GR00T: one image token per camera, checked at load). Where a step rewrites the task first
(`gr00t_prompt`'s `formalize_language`, `molmoact2`'s lowercasing), both the task as sent and as
rewritten are checked. MolmoAct2's state tokens (`<state_0>` … `<state_255>`) are refused in a
task too, but not counted, since they change with the state; its `<|image|>` placeholders are
checked at load, one per camera.

### Port layouts

Each policy kind names one port layout; the layout has its own port check (`bind_*`) and its own
feeds, and the prompt length check, noise, the action output and the postprocessors are shared. A
new kind with new ports adds a layout, not a branch inside another one.

| Layout | Kinds | Inputs |
|---|---|---|
| image slots | `smolvla`, `pi05`, `pi0`, `xvla`, `evo1` | `state` f32 (normalized), `tokenized_prompt` i64, `tokenized_prompt_mask` bool, `images [slots, 1, 3, H, W]` f32, `image_masks [slots, 1]` bool, optional `noise` |
| XR0 | `xr0` | `tokenized_prompt` i64 and `tokenized_prompt_mask` i64 `[1, 256]` (right-padded with `<|endoftext|>`), `pixel_values [512, 1536]` f32, `state [1, 1, 32]` f32 raw, `noise [1, 30, 32]`; `action [1, 30, 32]` normalized, then `xr0_denormalize` |
| GR00T | `gr00t_n1d7` | in this order, each bound by name or else by its index (the CPU plugin renames `state`): `input_ids` and `attention_mask` i64 `[1, 192]`, `pixel_values [512, 1536]` f32, `position_ids [3, 1, 192]` i64, `state [1, 1, 132]` f32 (normalized, zero-padded), the `constant` ports (`embodiment_id [1]` i64), the noise port (`actions [1, 40, 132]`); the one output (`predicted_action [1, 40, 132]`, normalized), whatever its name; then `action_chunk_trimmer` 16, `action_slice` 7, `denormalize` |
| MolmoAct2 | `molmoact2` | `input_ids`, `attention_mask` and `token_type_ids` i64 `[1, 650]` (the tokenizer's 255, BOS, and 2 × 197 for the expanded images), `images [1, 2, 729, 588]` f32, `token_pooling [1, 392, 4]` i64, `action_dim_is_pad [1, 32]` bool; no state or noise input; `action [1, 10, 7]` normalized, then `molmoact2_postprocess` |
| VLA-JEPA | `vlajepa` | `input_ids` i64 and `attention_mask` i64 `[1, L]` (the tokenizer pads right to L = 288 with `<|endoftext|>`, mask 0), `pixel_values [cameras·patches, 3·tp·p²]` f32 (`[512, 1536]`), `embodied_positions [1, 32]` i64, `state` f32 raw, `noise [1, 7, 7]`; `action [1, 7, 7]` is final |

At load the VLA-JEPA binder checks `pixel_values`' width against the step's geometry, its rows
against a whole number of cameras (which must match `image_keys`), and `embodied_positions`' length
against `num_embodied_tokens`.

A postprocessor can change the action width; the served `[T, D]` (spec, metadata, replies) is
the width the postprocessors leave, and `denormalize` stats are checked against the width at
their step.

### Prompt length

A prompt is checked once per distinct task text, at the first request that uses it, so the answer
never changes inside an episode: π0.5 writes the state bins into the prompt and their token count
moves with the state (Gemma splits digits, so a 3-digit bin costs more than a 1-digit one). The
check tokenizes the task with the worst-case state (every bin `255`) and the real token count:
- Where the IR has the `truncation` state variable (Intel's π0.5 LIBERO, every package from our
  exporter, and our π0, X-VLA and VLA-JEPA packages), it is set to false for the check through the infer request's state, and the task is
  refused only when the count exceeds the port length; an exact fit is accepted.
- Where truncation is an always-on `Truncate` op (Intel's SmolVLA), a full mask is the only signal:
  the task is refused when the mask is full.

VLA-JEPA's prompt carries no state, so its count is exact: its tokenizer IR's `truncation`
variable defaults on, the engine holds it off, and a task is served up to exactly 288 tokens and
refused at 289 (LIBERO's tasks take 221–234). XR0's works the same way at 256 (its LIBERO template
alone is 191 tokens); truncating would cut the template. GR00T's tokenizer does not pad: the
image tokens it emits are expanded to 64 each after it, so the prompt may hold `seq_len` less
what the images take, 64 language tokens (template and task) beside two cameras; the count is
exact and a task over it is refused, in language tokens (LIBERO's example uses 28). MolmoAct2's
tokenizer pads right to 255 with a `truncation` variable the engine holds off, so a task is served
up to exactly 255 tokens (LIBERO's prompt takes 95, the state tokens one each whatever the state)
and refused at 256; every call's ids must then expand to exactly the graph's 650.

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

On the WebSocket, images must arrive at the model resolution, uint8 HWC RGB (the physicalai front end resizes for its clients; see "Intel remote-inference front end"), so the engine never has to reproduce
OpenCV's bilinear resize pixel for pixel. The client resizes on the robot, and must use the
package's padding convention, which the metadata states as `resize`: bilinear, pad value 0 before
scaling, padded **centred** for `pi05` and **top and left** for `smolvla_resize` (content at the
bottom right). VLA-JEPA's `resize` is `stretch`: a plain resize to 224×224 with no padding; so is EVO1's
(`smolvla_resize` with `"resize": "stretch"`; LIBERO renders at 448, so the two agree there), and
XR0's (`xr0`: frames at exactly 256×256). GR00T's is `center` (its letterbox pads evenly with 0),
and only square frames at exactly 256×256 are taken, which the letterbox leaves alone. MolmoAct2's
is `stretch`: a plain resize to 378×378 (OpenCV `INTER_LINEAR_EXACT`, no padding). openpi's `resize_with_pad` centres, so it is correct for π0.5 only; a 640×480 frame
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

### Real-time chunking (RTC)

RTC lets a robot keep executing while the next chunk is computed: the new chunk is denoised as an
inpainting problem pinned to the part of the previous chunk the robot has not executed yet (Black
et al., arXiv 2506.07339). The engine uses the identity-Jacobian guidance both shipping
implementations use (LeRobot `policies/rtc`, physical-ai-studio `mixins/rtc.py`), and it is not
the engine's code: Intel's exporter traces it into the single-pass graph when the policy is
exported with `rtc_enabled` (physical-ai-studio `8f132416`, SmolVLA and π0.5). Each denoising step
then adds a few elementwise ops: `x1 = x_t - t·v_t`, `err = (leftover - x1) · w`, with the prefix
weights `w` from `inference_delay` and `execution_horizon`, and `v_t -= min(c·r, β) · err`, where
`β` is `max_guidance_weight`.

- **Ports.** Such a graph has four more inputs: `prev_chunk_left_over` f32 `[1, T, D]` (the served
  chunk's shape), `inference_delay` i64, `max_guidance_weight` f32 and `execution_horizon` i64
  (scalars). The engine binds all four or none; any other subset, a leftover that is not the
  action output's shape, or another scalar shape is a load error. Only the `ImageSlots` layout
  takes them.
- **Package.** An RTC package serves the whole chunk (no `action_chunk_trimmer`, no
  shape-changing postprocessor), since the client resends what it has not executed of that very
  chunk; anything else is a load error. Its `normalize` step carries the action statistics under
  `prev_chunk_left_over`, and the engine normalizes the leftover with them before the graph, as
  physicalai's runtime does; a package without them is refused. Those statistics are read only
  for a model with the RTC ports, so a plain package loads exactly as before.
- **Stateless server.** The client keeps the action queue and sends the leftover on every call, as
  physicalai's `RTCExecution` does (the leftover zero-padded at the end to `[T, D]`, in the action
  space the server returned). Any call can go to any server, so replay and failover need nothing
  more. The metadata states `rtc_chunk_size` (T) for an RTC package and null otherwise.
- **Off by default.** A call without the RTC fields feeds a zero leftover and
  `max_guidance_weight` 0, which makes the graph's correction exactly zero: a call with the fields
  at weight 0 is bit-identical to one without them. Against the package Intel's exporter makes
  from the same checkpoint without RTC the chunk is not bit-identical: the RTC trace folds to a
  smaller graph (SmolVLA: 42,757 IR ops against 67,583; 63 `Cos` against 492), so the two compile
  to different kernels. They agree within 1.1e-5 on CPU f32, and the RTC package matches PyTorch within
  1.1e-6. Robots that do not send the fields, and packages exported without RTC, behave as before.
- **Measured** (2026-11-16, SmolVLA `HuggingFaceVLA/smolvla_libero` and π0.5
  `lerobot/pi05_libero_finetuned_v044` exported with physical-ai-studio `8f132416`, input-mode
  noise, OpenVINO 2026.4.1):

  | | SmolVLA | π0.5 |
  |---|---|---|
  | CPU f32 (x86), weight 0 vs no fields | bit-identical | bit-identical |
  | CPU f32 (x86), no fields vs the export without RTC | 1.07e-5 | 1.73e-6 |
  | CPU f32 (x86), RTC vs PyTorch RTC (f32 twin) | 1.01e-6 | 1.10e-6 |
  | Arc 140V GPU f16 vs CPU f32: plain / RTC | 3.4e-3 / 4.2e-3 | 1.0e-3 / 7.7e-4 |
  | Arc 140V GPU f16 p50, 50-step chunk: plain / RTC | 315 / 313 ms | 618 / 616 ms |

  Delayed LIBERO (`libero_spatial`, 10 tasks × 3 episodes, seed 7, GPU f16; the robot keeps
  executing D old actions while a chunk is computed, horizon 20, `max_guidance_weight` 5), RTC off
  against on with the same schedule: SmolVLA (D 5) 9/30 and 7/30 successes, seam jump mean 0.425
  against 0.044 (p90 0.864 against 0.080); π0.5 (D 10) 26/30 and 25/30, seam jump mean 0.360
  against 0.035 (p90 1.155 against 0.062). RTC removes about 90% of the jump between chunks;
  success is unchanged within the run's noise.
- **Checks** (physicalai's `_validate_rtc_inputs`): `1 <= execution_horizon <= T / 2`,
  `0 <= inference_delay <= execution_horizon`, `max_guidance_weight` finite and non-negative,
  the leftover `T × D` finite values. A package without RTC refuses the fields rather than
  ignoring them.

## Serving

`cascadia run <dir> --engine ov-policy` (the package directory is `run`'s positional) takes its own path in `cmd_run`, before
`WorkerArgs::single_node`: load and warm the engine, then serve the policy router plus `/health`
and `/metrics`, which behave as the chat API's (`/health` turns 503 `degraded` after an engine
failure until a call succeeds again; refused requests do not count). Shutdown closes the engine, as
chat closes its runner. `build_builder` and `cascadia worker` refuse `ov-policy` with an error; the
dashboard and the chat routes are not served for a policy model, and the chat-only `run` flags are
refused. There is no auto-selection: `--engine` defaults to
`ov-genai`, so the flag is required.

**Unauthenticated, like the chat API** (SECURITY.md: "No client authentication on the HTTP API").
Because a policy drives a robot, `ov-policy` binds `127.0.0.1:8000` unless `--api` is given.
`RunArgs.api` becomes `Option<String>`, resolved per engine (`:8000` for the chat engines as
today), since clap cannot tell an explicit `--api :8000` from a default. The WebSocket upgrade is
refused when an `Origin` header is present and not in `--policy-allow-origin` (empty by default),
so a browser page cannot open it. Authenticated network access is out of scope.

Limits, all checked before work is queued:
- WebSocket message cap computed from the spec: the cameras' H×W×3 bytes, state and noise
  arrays, the RTC leftover (`T × D × 4` bytes, RTC packages only), the prompt cap, plus 64 KiB of
  framing. Larger messages close the connection.
- Prompt at most 1 KiB of UTF-8.
- At most 4 concurrent connections (`CASCADIA_API_MAX_CONCURRENT`, as for chat); one request in flight per
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
| `chunk` `[T, D]` | T from `action_chunk_trimmer.n_action_steps` when present, else the IR output; D from the IR output after the postprocessors (7 for `xvla_rotation_6d_to_axis_angle`) |
| `noise` | the mode above |
| `noise_shape` | the IR `noise` port, `input` mode only (the decode rule and the message cap need it) |
| `attribution` | the package's `NOTICE` file, else the manifest's `metadata.attribution`, else `policy.source.attribution`; omitted when there is none (GR00T's NVIDIA Open Model License notice) |
| `resize` | the image step: `center` (`pi05`, `pi0`, `gr00t_image`) or `top_left` (`smolvla_resize`), bilinear, pad 0 before scaling; `stretch` (`vla_jepa_inputs`, `xr0`, `molmoact2`, `smolvla_resize` with `"resize": "stretch"`), no padding |
| `rtc_chunk_size` | T when the IR has the RTC port group, else null |
| `manifest_sha256`, `package_sha256`, `precision` | the load, as in "Metadata identity" |
| `device`, `model`, `prompt_max_bytes` | the server |

`model` is the package directory name unless the embedder sets `OvPolicyOptions.model_id`; then
it is that id, in both the HTTP and the socket metadata. The id must follow the same rule as a
Zenoh slot name: it starts with a letter or digit and holds only letters, digits, `.`, `_` and `-`.
Anything else is a load error.

Two more load options and one reading, for a program that embeds the engine:

- `OvPolicyOptions.cache_dir: Option<PathBuf>`. When set, the policy compile passes it to
  OpenVINO as `CACHE_DIR`, so a restart loads the compiled model from the cache instead of
  compiling again (SmolVLA on an Apple CPU: 9.9 s uncached, 3.3 s from the cache). The cache never
  changes outputs: a test loads the package without a cache, then twice from one cache directory,
  and checks that the CPU actions are bit-identical all three times. The tokenizer is not cached.
  `cascadia run` passes the chat engines' default cache (`cascadia/ov-cache` under the user cache
  directory).
- `OvPolicyEngine::gpu_memory_bytes() -> Option<u64>`. The GPU memory the compiled policy holds
  now: OpenVINO's `GPU_MEMORY_STATISTICS` (`ov::intel_gpu::memory_statistics`), summed over the
  allocation types. Each engine compiles in its own OpenVINO `Core`, so the figure counts only this
  policy. Read it after `warmup` for the serving figure. It is `None` on CPU, when the plugin
  cannot answer, and after `close`.

NPU stays refused for every policy.

`WS /v1/policy/ws` uses openpi's framing, so `openpi_client.WebsocketClientPolicy` can drive it:
- **Connecting.** openpi uses `host` as the full URI when it starts with `ws`, so the client passes
  `host="ws://<box>:<port>/v1/policy/ws"` and `port=None` (a port argument would be appended after
  the path).
- **Metadata.** On connect the server sends the metadata as one msgpack message.
- **Request.** One msgpack map: `images` (a map of camera name to array), `state` (array), `prompt`
  (string), and optional `seed` (int) and `noise` (array). An RTC package also takes
  `prev_chunk_left_over` (array), `inference_delay` (int), `max_guidance_weight` (float) and
  `execution_horizon` (int), all four or none.
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
- RTC fields: only for a package with `rtc_chunk_size` (an error otherwise). `prev_chunk_left_over`
  dtype `<f4`, shape `[T, D]` or `[1, T, D]`; the scalars as plain numbers or numpy scalars; the
  ranges in "Real-time chunking". `None` for any of them counts as absent. A leftover that
  normalizes past 1e4 standard deviations (or to a non-finite value) is refused before the model
  runs, since the graph's f32 and the GPU's f16 arithmetic would overflow into NaN.
- Before anything is allocated: `len(data) == itemsize · Π shape`, computed with checked arithmetic.
  Any mismatch is an error naming the expected value.
- Replies: if any action is non-finite (possible from f16 overflow on the GPU), the call returns an
  error frame, never the actions.

Keys are ours, so openpi clients are transport-compatible, not drop-in: a DROID client's
`observation/joint_position` needs a mapping (out of scope for v1).

## Intel remote-inference front end

Intel's `physicalai` runtime runs a policy on another machine through `RemoteInferenceModel`, which
speaks its own protocol (version 1) over Zenoh. `cascadia run --engine ov-policy --zenoh
<HOST[:PORT]>` serves that protocol next to the WebSocket, so code written against
`RemoteInferenceModel` drives a Cascadia policy with no change. It is behind the `zenoh` cargo
feature (off in a plain `cargo build`; the release bundles build with it, so `--zenoh` and
`cascadia policy-bridge` ship) and uses the Rust `zenoh` crate pinned to 1.9.0, the release the
Python client locks (`eclipse-zenoh==1.9.0`); an end-to-end run of Intel's own client against the
Rust server is the proof of wire compatibility, not the version numbers. Two advisories in that
tree are recorded in `deny.toml` as not reachable: lz4_flex's decompression (RUSTSEC-2026-0041) is
compiled only with zenoh's `transport_compression`, which is off, and unsafe-libyaml
(RUSTSEC-2024-0449) parses only YAML config files, which the server never loads. zenoh 1.10.2 still
carries both, so a bump would not clear them.

**Wire.** Byte for byte what `physicalai/inference/remote` exchanges, pinned by golden bytes that
`crates/cascadia-api/tests/fixtures/physicalai/generate.py` packs with Intel's own `_protocol.py`:
- Keys `physicalai/inference/<name>/metadata`, `/predict` and `/reset`. `<name>` is the slot name the
  client is given (`--zenoh-name`, default the manifest's `policy.name`); without a port the server
  listens on Intel's name-derived port (`20000 + sha256(prefix)[0..4] mod 40000`), which is what the
  client dials when it is given none.
- Handshake: `server_id` (fresh per start), `name`, `policy` (`name`, `manifest_sha256` = SHA-256 of
  the package's `manifest.json` bytes, `backend`, `device`), `chunk_size` (T), `rtc` (`supported: true` and `chunk_size` T for an RTC
  package, else `supported: false` and null), `image_codecs` `["jpeg", "raw"]`, `max_request_bytes` (32 MiB, physicalai's
  default, since frames arrive at the camera's resolution). The client's `expected_policy` and `expected_manifest_sha256` check these.
- Predict: `seq`, `budget_ms` (1 to 60000), `inputs` (tagged `__np__` arrays and string lists) and
  `images` (JPEG or raw RGB, `HxWx3` or `1xHxWx3`), validated in the order Intel's server does.
  The inputs then map onto the same `Observation` as a WebSocket request and pass the same
  `check_observation`: `state` `[D]` or `[1, D]` (float32 or float64), `task` a one-string list
  (the prompt), cameras as `images.<name>` with the advertised names, or a bare `images` for a
  one-camera package (what Intel's `PolicySource` sends for one camera), optional `noise` for
  input-mode packages, and for an RTC package the four inputs `RTCExecution` injects
  (`prev_chunk_left_over` float32 `[1, T, D]`, the others 0-d numpy arrays).
- Frames: physicalai's client sends them at the camera's native resolution (unless it is given
  `max_image_side`) and physicalai's server resizes them in the model's preprocessor, so this front
  end runs the package's own first resize, where it is ported exactly, before the observation is
  checked. Frames are at most 4096 pixels a side, with decoded bytes within `max_request_bytes`.
  The spec names the resize as `server_resize`:
  - `letterbox` (physicalai's `pi05` and `smolvla_resize`, LeRobot's `pi0`): OpenCV `INTER_LINEAR`
    arithmetic in `f32` (torch `align_corners=False`, no antialiasing) to the aspect-kept size,
    padded with 0 on the top and left or centred as `resize` says.
  - `area` (VLA-JEPA): physicalai's first step, average pooling as torch `interpolate(mode="area")`,
    to `image_resolution`; the engine then runs its antialiased bicubic step as before.
  - `smart_resize_cubic` (XR0, AArch64 servers only): Qwen's `smart_resize` target and OpenCV
    `INTER_CUBIC` on `u8`, byte for byte, including OpenCV's 16-byte `f32` vector loop. On x86 the
    opencv-python wheel physicalai pins (5.0.0.93) hands this resize to Intel IPP, whose bytes differ
    from OpenCV's own code by 1 level on about 4% of them and are not public, so an x86 server
    advertises no server resize for XR0 and refuses frames of another size. (With IPP turned off,
    x86 OpenCV gives the same bytes as AArch64, and so does this port on x86.) The export takes one frame size, so only frames that resize lands on it are taken;
    others are refused naming the size they would become (a 480x640 camera becomes 256x320, which
    a 256x256 LIBERO export cannot take, on physicalai's own server too).

  Float results are rounded to the engine's `u8` frame, within 0.5/255 of the preprocessor's own
  float frame. Golden pixels come from the preprocessors themselves: physicalai's
  `ResizeSmolVLA`, `Pi05Preprocessor`, `vla_jepa._area_weights` and `xr0._resize_image`, and
  LeRobot's `resize_with_pad_torch` for pi0. Packages with an `image_flip` before their resize
  (X-VLA) and packages physicalai's runtime does not serve (EVO1's antialiased bicubic stretch,
  GR00T's `INTER_AREA` square, MolmoAct2's processor) refuse a frame of another size with a message
  naming the size to send (and `max_image_side` when the aspect ratio already matches), rather
  than feed the robot an approximation. The WebSocket keeps its exact-size rule: openpi clients
  resize on the robot.
- Reply: `outputs.action` as float32 `[1, T, D]` (the shape Intel's own models return), `seq`,
  `server_id` and the server's queue, compute and decode seconds. Reset replies `{ok: true}`;
  policies keep no state between calls, so it only orders with the queue.
- Errors go back through `reply_err` as `{protocol_version, code, seq}` with Intel's codes:
  `invalid_request` (anything the decoder or `check_observation` refuses; the reason is logged, since
  the protocol carries none), `unsupported_version`, `superseded` (a queued predict replaced by a
  newer one), `expired` (the budget had passed when the model was free), `model_error`,
  `shutting_down`.

JPEG frames are decoded by `zune-jpeg`: pure Rust (no C toolchain or system libjpeg on any target),
maintained, and the JPEG decoder of the `image` crate. Its pixels differ from OpenCV's
libjpeg-turbo decode, which Intel's server uses, by at most 2 levels per channel on the fixture
frames; the test bounds that difference.

**Sharing.** Both front ends run on one engine: a `SharedPolicy` holds the engine mutex and the
connection slots. The Zenoh front end takes one slot for as long as it serves (it is one robot, as in
Intel's server: one worker, one call at a time, newer calls superseding queued ones), and its calls
wait on the same mutex as WebSocket calls.

**Security.** Like Intel's server, there is no authentication: the session is a peer that listens
on the one configured TCP endpoint, never connects out, has multicast and gossip scouting and shared
memory off, and bounds incoming messages to `max_request_bytes`. It binds loopback unless
`--policy-allow-all-interfaces` is given, and then warns; the same opt-in governs the WebSocket's
`--api` host. Authenticated network access is not built
on Zenoh: Zenoh's `usrpwd` check happens inside the transport against a static credential
dictionary, so the queryable never learns who called, and Intel's predict message carries no client
identity; per-client authorization cannot be enforced there. A deployment that needs it keeps Zenoh
on loopback and puts an authenticated hop in front through the bridge below.

**Bridge.** `cascadia policy-bridge --zenoh 127.0.0.1:<port> --upstream <ws(s) URL> [--token-file
<path>] [--ca-file <pem>] [--zenoh-name <slot>]` serves Intel's protocol on loopback only (the same pinned
Zenoh settings) and forwards each predict as one openpi call to any openpi server: Cascadia's own, or
a gateway that takes `Authorization: Bearer`. The token is read from the file, sent only over
`wss://` or to loopback, and never logged. A `wss://` upstream is checked against the public web
roots, or, with `--ca-file`, against the `CERTIFICATE` blocks of that PEM file **instead**: a
gateway behind a private CA is trusted for that CA alone, so a publicly issued certificate for the
same host name is not also accepted. `--ca-file` with a `ws://` upstream, an unreadable file, one
with no certificate or one whose certificate does not parse refuses the start with an error naming
the file; the error never quotes the file, and the certificates are never logged (only their
count). One bridge serves one Intel client. Its handshake has a `server_id` minted
for the bridge's lifetime, so an upstream reconnect is not a server restart to the client, and the
name, `manifest_sha256` and chunk size the upstream's metadata states; a reconnect to an upstream
whose metadata differs is refused. It honours `budget_ms` and superseding like Intel's server: a call
still running when its budget ends is answered `expired`, its late reply is read and dropped (and
counted) before the next call, and the bridge warns once when the upstream is slower than the
client's budget. `reset` closes the upstream connection, so a new episode starts on a fresh one.

**Metadata identity.** So a bridge can answer the handshake and a client can tell whether two
servers serve the same policy, `PolicySpec` carries `manifest_sha256`, `package_sha256` and
`precision` (`f16` or `f32`, as compiled), and the socket metadata adds `policy_name`, `chunk_size`
and `rtc_chunk_size` (null). The socket metadata bytes hold no timing, so two loads of one package
at one precision send identical bytes; the slowest warmup call appears only in
`GET /v1/policy/metadata` as `warmup_ms`.

- `manifest_sha256` is the SHA-256 of `manifest.json` alone, as Intel's handshake defines it. Two
  packages whose manifests are byte-identical but whose other files differ (an int8 and an f16
  weights file under one manifest) share it, so it is not enough to tell two servers apart.
- `package_sha256` (`cascadia_engine_openvino::policy::package_sha256`) covers every file in the
  package directory: `sha256sum` of the `sha256sum` listing (`<hex>  <path>`) of every file, by
  `/`-separated path relative to the package, sorted bytewise, so it can be checked by hand with
  `cd <pkg> && find . -type d -ipath ./.cache -prune -o -type f -print | sed 's|^\./||'
  | LC_ALL=C sort | tr '\n' '\0' | xargs -0 sha256sum | sha256sum` (for a package without symlinks).
  A copied package has the same digest; any added, removed, renamed or changed file changes it. A
  symlink to a file counts as that file's content under the link's path, wherever it points, because
  a Hub snapshot is made of links into its blob store (see `confine_artifact`). A symlink to a
  directory, a dangling symlink, any other non-file, and a name `sha256sum` would escape (not UTF-8,
  `\` or a newline) refuse the load, naming the entry. Each file is opened without blocking (unix)
  and checked to be a regular file on the open handle, so an entry swapped for a FIFO after the walk
  is refused instead of waited on. Nothing else is ignored by default, OS metadata files
  (`.DS_Store`, `Thumbs.db`) included: an extra file can only make two equal packages look
  different, which refuses a failover, never the reverse. Three exceptions. A top-level directory
  named `.cache` (any case; deeper ones and a file of that name still count): `hf download
  --local-dir` writes per-download timestamps under `.cache/huggingface/`, so two servers that
  downloaded one pinned revision would never match. The engine never reads it, because
  `confine_artifact` refuses a manifest path with an empty or `.` component, or whose first
  component is `.cache` (also with NTFS's dropped trailing dots or spaces), and every other file it
  opens sits beside a manifest artifact. As a backstop for names a lexical check cannot see (a
  symlink, a junction, an NTFS 8.3 name such as `CACHE~1`), `load` canonicalizes every file it
  opens before reading it and refuses one that resolves into the package's `.cache`.
  `OvPolicyOptions.cache_dir`, when it lies inside the package: the compile cache never changes
  outputs and is written after the digest is taken. And `OvPolicyOptions.digest_ignore`
  (`package_sha256_with(dir, ignore)`), empty by default: top-level file names, no globs and no
  paths, matched without case (as APFS and NTFS open them), for files a deployer writes next to the
  package that never reach the engine (a staging checksum, say), so two nodes staged by one
  deployer with the same files still match. The same name in a subdirectory still counts.
  `package_sha256_with` does not know which files the engine reads; `check_digest_ignore(manifest,
  ignore)` does, and `load` runs it: it refuses, without case, `manifest.json`, `NOTICE`, every
  manifest-named top-level artifact and any file sharing its stem (`.bin` weights, a `.json` beside
  it), and the `tokenizer.json` read beside the tokenizer IR for its added special tokens. The digest is taken
  at load after the cheap refusals (device, manifest) and before the compile, reading every file
  once in 1 MiB blocks. A refusal (an entry above, a bad ignore name) is a configuration error; any
  other I/O error while hashing (a file shared or replaced mid-walk, an interrupted read) is a
  backend error, which a caller may retry. The key is left out of
  the metadata when empty, so a spec built without a package (the mock engine, tests) sends the
  bytes it sent before; a loaded package now sends one more key, so its socket metadata bytes, and
  any hash of them, change once with this release. That is the point: the old bytes were the same
  for packages that compute different actions.
- `OvPolicyEngine::engine_build()` names the engine that computes the actions:
  `cascadia-ov-policy/<crate version>+<git sha> openvino/<runtime build>`, where the sha is the
  commit the crate was built from (`unknown` outside a git checkout) and the runtime build is
  `ov::get_version().buildNumber` of the OpenVINO runtime actually loaded, not the one compiled
  against. The sha is the build environment's `CASCADIA_GIT_SHA` when set (pass it as a build
  argument in a container or tarball build, which has no `.git`; one left set in a shell
  overrides git too), else the checkout's `HEAD`, else `unknown`. It has no time or host in it, so two builds of one commit on one runtime give the same
  string, and `engine_build(runtime)` gives it for any runtime string. It is not in the metadata
  bytes. A build whose tracked files differ from `HEAD`, staged or not, reports `<HEAD>-dirty`;
  untracked files do not count, since a cargo git checkout always holds an untracked `.cargo-ok`. An `engine_build` with `+unknown`, a `-dirty` sha, or
  otherwise not of this form never matches anything, itself included, since it cannot tell two
  engines apart; `engine_build_is_known(&str)` applies that rule.
- Two servers compute the same actions for a request when `package_sha256`, `precision`, `device`
  and `engine_build` agree and `engine_build_is_known` holds; a client that fails a session over
  from one server to another should require all of that (the socket metadata carries the first
  three).

## Code layout

| Where | Change |
|---|---|
| `cascadia-types` | `policy.rs`: `Observation { images: Vec<(String, Image)>, state: Vec<f32>, prompt: String, seed: Option<u64>, noise: Option<Vec<f32>>, rtc: Option<RtcInputs> }`, `ActionChunk { actions: Vec<f32>, steps, dim, infer_ms }`, `PolicySpec` (fields as in "API"). |
| `cascadia-engine` | `PolicyEngine: Send { fn spec(&self) -> &PolicySpec; fn warmup(&mut self) -> EngineResult<Vec<u32>>; fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk>; fn close(&mut self) {} }` and a deterministic `MockPolicyEngine` for API and CLI tests on macOS. `warmup` makes `WARMUP_CALLS` (3) calls on zero frames, zero state, an empty prompt and seed 0, and returns each call's `infer_ms` rounded up to whole milliseconds; any failed call is returned as the error, so a package that cannot infer fails its load instead of serving errors. A caller can size a per-call timeout from the slowest. |
| `cascadia-ov-genai-shim` | A `bool` dtype in `dtype_from_code`, `code_from_dtype`, the header defines and Rust `DType` (`from_code`, `bytes_per_element`); a string-tensor input for the tokenizer; a separate CPU runtime handle for the tokenizer, with the `openvino_tokenizers` extension added to that handle's `ov::Core`, and a way to set an infer request's state variable (`truncation`); `cascadia_runtime_has_op_type(handle, type)`, answered from op types recorded off the `ov::Model` before compile. Alias lookup already exists (`input_aliases`). |
| `cascadia-engine-openvino` | `policy/`: manifest parsing (both component forms, version check, path confinement), port checks, the steps above (one module per family: `gr00t`, `molmoact2`, `qwen_vl`, `vla_jepa`, `xr0`), noise-mode detection, package identity (`package.rs`), `OvPolicyEngine` (`ov.rs`). `build.rs` sets the sha `engine_build` reports. |
| `cascadia-api` | Policy router on its own state (the chat `AppState` stays untouched), the limits above, `axum` `ws` feature, msgpack with bin keys. The codec is a public module, `cascadia_api::policy`, so other serving layers can reuse it: `decode_request(&[u8], &PolicySpec) -> Result<Observation, RequestError>` (every decoding rule above; `RequestError` carries the client-facing message), `encode_reply(&ActionChunk) -> Vec<u8>`, `encode_metadata(&PolicySpec) -> Vec<u8>`, `metadata_json(&PolicySpec) -> String`, and `max_payload_bytes(&PolicySpec) -> u32` (the WebSocket cap formula). `physicalai` (Intel's wire codec, built without Zenoh so its tests run everywhere), and behind the `zenoh` feature `zenoh_server` and `bridge`. |
| `cascadia-cli` | `EngineKind::OvPolicy`, the policy serve path in `cmd_run`, `RunArgs.api` as `Option<String>`, refusal in `worker`/`build_builder`, `--zenoh` and `policy-bridge` (feature `zenoh`), `export-policy` with its driver script and hashed requirements file embedded by `include_str!` (as `cascadia shard` embeds `EXPORT_REQUIREMENTS`, since a release bundle carries only the binary), a `doctor` `ov-policy` section (the tokenizer extension, and uv for `export-policy`). |
| `tools/` | `vla_reference.py` (the codec's golden msgpack bytes), `vla_libero_client.py` (benchmark adapter, with a reference resize per convention). Sources only; nothing here is read at run time. |
| `.github/workflows/ai-pc-tests.yml` | Two steps that read the repository variable `CASCADIA_POLICY_PACKAGE_DIR`, check the package's `manifest.json` against the committed fixture and print the hashes of its top-level `.json`, `.xml` and `.bin` files, export `CASCADIA_POLICY_PACKAGE` and run `policy_openvino` with `CASCADIA_POLICY_GPU=1`; both skip when the variable is unset (see "Testing"). |
| `.github/workflows/ci.yml`, `security.yml`, `release.yml` | A `zenoh` lane: clippy and tests of `cascadia-api` and `cascadia-cli` with `--features zenoh` on Linux, a Windows compile of `cascadia-api --features zenoh`. cargo-deny runs with `--all-features` (stated explicitly). Release bundles build with `openvino,dashboard-embed,zenoh`. |
| docs | `docs/engines/ov-policy.md`; `docs/ARCHITECTURE.md` and `docs/engines/README.md` updated for the sibling trait and the new routes; a policy-endpoint note in `SECURITY.md`. |

## Exporter: `cascadia export-policy`

`cascadia export-policy --model <hf-model-id> --output-dir <dir>` (`-o <dir>`; the same argument
names as `cascadia shard`) runs Intel's `physicalai-train` in **its own virtual environment**,
created with `uv` under the Cascadia cache. (`cascadia shard` has no such
environment: it uses the caller's Python and prints a `pip install` line; the exporter does not
follow that, because its dependencies would collide.) Python is needed only on the exporting
machine.

- **Requires `uv` 0.13.6 or newer.** Nothing in the repo ships it; when `uv` is missing or older
  the exporter refuses with the install or `uv self update` command, before doing anything else.
  0.13.0 brought `--require-build-hashes`, 0.13.6 the pinned CPython 3.12.16. It also needs `git`,
  which uv uses for the two git-pinned packages. Every uv call runs with `--no-config` and without
  `UV_*` variables other than `UV_CACHE_DIR`, so a user's uv settings cannot change the install.
- **Files.** `tools/export_policy/`: the driver (`export_policy.py`), the inputs
  (`pyproject.toml`), and `lock.sh`, which compiles them into `requirements.txt`,
  `requirements-preflight.txt` and `build-constraints.txt` (with `--exclude-newer` at a fixed date,
  so a relock with the same inputs gives the same files). The CLI embeds the driver and the three
  files; each environment is a uv-managed CPython 3.12.16 under `<cache>/cascadia/export-policy/`,
  named by a hash of the Python version and its pins, so a new pin set gets a new environment.
- **Pinned install.** The environment installs from a hashed requirements file (`uv pip compile
  --generate-hashes`) embedded in the binary:
  - `physicalai-train` at `physical-ai-studio` `8f132416` (it contains `374d2cb9`);
  - the `physicalai` runtime at `e3dd944`, the same revision the engine is ported from and the
    fixtures are generated with. That commit's own `uv.lock` names an older runtime (`8381d43`), so
    the lock overrides it: one runtime revision for the port, the fixtures and `--verify`. If the two
    prove incompatible, step 0 finds out before any porting;
  - `openvino==2026.4.1` and a matching `openvino-tokenizers`;
  - torch CPU wheels.

  The runtime pin is a uv override (`override-dependencies` in the inputs, the same mechanism as
  `--overrides`), not a plain requirement: without it uv refuses to resolve (step 0). torch and
  torchvision come from the PyTorch CPU index. The preflight environment (wheels only) installs
  with `--require-hashes`. The export environment cannot: uv refuses `--require-hashes` for git
  requirements (checked with uv 0.13.8), and the two packages cannot be hashed archives because
  both take their version from git. It installs with `--verify-hashes`, so every wheel is checked
  against its hash and the git packages are pinned by full commit, plus `--build-constraints
  build-constraints.txt --require-build-hashes` (a uv preview feature), so the build backends that
  turn those commits and the two sdist-only packages (`docopt`, `pyturbojpeg`) into wheels must
  match their hashes too.
- **Precision.** The policy is loaded with `dtype="float32"` and the IR saved with fp16 weights. The
  tool's default (bf16) exports produce all-NaN actions under bf16 inference with a padded prompt
  (step 0), so the exporter never writes one.

  The commit alone pins nothing below it (its `pyproject.toml` says `openvino>=2026.4`). Bump to a
  release when one carries the fix.
- **Chunk length** comes from the checkpoint (`n_action_steps=None`, for SmolVLA too); the
  loader's default of 50 rejects 15-step checkpoints. The CLI compares the exported manifest with
  the checkpoint's `chunk_size` and `n_action_steps` (a trimmer exactly when they differ, none with
  `--rtc`) and refuses a package that disagrees.
- **Revision.** The preflight resolves the checkpoint's commit; every later download (config,
  weights, statistics, licence files) is bound to that commit, and `NOTICE` names it.
- **No half-made package.** The export, `--stats`, the licence files, `NOTICE` and `--verify` all
  happen in a staging directory beside `--output-dir` (`.<name>.partial-<pid>`, refused if it
  exists), which is renamed to `--output-dir` only when everything succeeded and removed on any
  failure.
- **`--rtc`** exports the RTC port group (SmolVLA and π0.5): the policy's `rtc_enabled` is set and
  `n_action_steps` is the full chunk, so the package has no trimmer.
- **Refusals before the checkpoint is downloaded.** Memory is checked before any environment
  exists; the gated files and statistics are read from Hugging Face metadata only,
  in a small environment holding `huggingface_hub` at the export environment's versions (a few MB),
  before the export environment is installed:
  - **Memory.** Less than 32 GiB of available RAM plus free swap (the main gate), or less than
    15 GiB of physical RAM as the OS reports it (a 16 GiB VM reports about 15.6). Measured: 14.3 GB
    RSS plus up to 10 GB swap for π0.5, completed on a 16 GiB box with 48 GiB swap. A 38.6 GB Mac
    with other work running froze and needed a forced restart.
  - **A serving box is a warning, not a refusal**, as for `cascadia shard`: a cascadia node or
    OVMS process, or `cascadia run`, `worker` or `run-placement` (found after any global flags),
    on this host.
  - **Bad arguments.** A model id that is not `owner/name` under Hugging Face's rules, or an
    `--output-dir` with no directory name (`.`, `/`).
  - **Gated files.** No access to `google/paligemma-3b-pt-224` for π0.5, with a message saying
    how to accept the Gemma licence. (`8f132416` exports SmolVLA and π0.5 only; any other policy
    type is refused.)
  - **Missing statistics.** No normalization stats in the checkpoint (no `policy_preprocessor.json`,
    a normalizer with no STATE feature, an unnormalizer with no ACTION feature, or a `state_file`
    missing from the repository), unless `--stats <file>` is given. This catches the identity
    stats above. The file is JSON, `{"state": {...}, "action": {...}}`, with `mean`/`std`,
    `q01`/`q99` or `min`/`max`; it replaces the values in every `normalize` and `denormalize` step
    (the RTC leftover takes the action block; a step without a mode is `mean_std`), and each
    step's mode must find its keys. Widths and keys are checked against the checkpoint's config
    before the export environment is installed. Stats narrower than the package (DROID: 8-wide
    stats, 32-wide state and action) are padded with the mode's identity (mean 0 and std 1, q01 −1
    and q99 1, min −1 and max 1): LeRobot normalizes at the robot's width and then zero-pads
    (lerobot v0.6.0 `NormalizerProcessorStep`, then `pad_vector` in `modeling_pi05.py`), and an
    identity stat maps that zero pad to zero.
- **Licence.** The package gets a `NOTICE` (the checkpoint, its revision, the model card's
  licence, and for π0.5 the Gemma Terms of Use notice), which the engine serves as `attribution`,
  plus any licence files at the checkpoint's repository root. The exporter prints which terms
  apply before and after the export.
- **`--verify`** runs the traced input through PyTorch and the package's IR, on CPU fp32 and on
  GPU f16 when a GPU is present, and prints the max and mean differences; a non-finite output
  fails the export.
- **`cascadia doctor`** reports whether the `openvino_tokenizers` extension the engine tokenizes
  with is found beside the loaded OpenVINO core or under `INTEL_OPENVINO_DIR`.

## Testing

What runs where, as shipped. Items marked **not shipped** are planned and not in the repo.

- **Fixtures.** Committed, all small (about 115 KB under
  `crates/cascadia-engine-openvino/tests/fixtures/policy/`; its README names each file's source):
  - the manifests of Intel's published packages and of our exports, copied unmodified, including
    our DROID manifest for the placeholder-stats refusal;
  - generated references: OpenCV's `INTER_AREA` resize for GR00T, LeRobot's X-VLA rotation
    conversion, and the SmolVLA end-to-end reference (served actions for the fixed inputs of
    `tests/policy_openvino.rs`, from an independent numpy + OpenVINO Python pipeline on CPU with
    `INFERENCE_PRECISION_HINT=f32`; the file records that source);
  - golden msgpack bytes for openpi requests and replies, packed by the real `openpi_client`
    (`tools/vla_reference.py msgpack` into `crates/cascadia-api/tests/fixtures/policy/`), and
    physicalai's wire bytes and resize pixels, packed by Intel's own code
    (`crates/cascadia-api/tests/fixtures/physicalai/generate.py`).

  Python runs only when fixtures are regenerated, never in CI. **Not shipped:** a generator that
  runs Intel's `physicalai` runtime for per-step fixtures and token ids; the step tests carry their
  expected values in the Rust tests.
- **No OpenVINO (the stub CI lane and macOS).** Unit tests against those fixtures: manifest parsing,
  both component forms, path confinement, version refusal, placeholder stats, request decoding with
  every rejection rule, msgpack bytes against the golden fixtures, and API/CLI against
  `MockPolicyEngine`. The physicalai front end and the bridge are behind `zenoh`, which the stub
  lane does not build: `ci.yml`'s `zenoh` lane runs clippy and the tests of `cascadia-api` and
  `cascadia-cli` with `--features zenoh` on Linux and compiles `cascadia-api --features zenoh` on
  Windows. `security.yml` runs cargo-deny with `--all-features`, so the zenoh tree is audited too.
- **With OpenVINO.** On the self-hosted AI-PC lane (`ai-pc-tests.yml`), the only lane that runs
  `cargo test --features openvino`; the Linux lane only type-checks. The tests are ordinary tests
  that print a skip reason and return when `CASCADIA_POLICY_PACKAGE` is unset (`#[ignore]` is fixed
  at compile time, and the lane does not pass `--ignored`).
  - **Staging.** The lane is a pool of runners, so the package (our `input`-mode export of
    `HuggingFaceVLA/smolvla_libero`) is staged on every runner with the `ai-pc` label, at the path
    in the repository variable `CASCADIA_POLICY_PACKAGE_DIR`.
  - **Hash check.** When the variable is unset, the check step prints a skip line and the test step
    is skipped. When it is set, a missing directory, a missing `manifest.json`, or a
    `manifest.json` whose SHA-256 differs from the committed fixture
    (`smolvla-libero-noise-manifest.json`) **fails** the step. It prints the SHA-256 of every
    top-level `.json`, `.xml` and `.bin` file it ran against. **Not shipped:** committed hashes of
    the IR files; until they are recorded, a package with the right manifest and different weights
    passes the check and is caught only by the reference comparison below.
  - The tests then run with `CASCADIA_POLICY_PACKAGE` set and `CASCADIA_POLICY_GPU=1`.

  Steps (`tests/policy_openvino.rs`, the `smolvla_*` tests):
  1. The package spec, tokenizer ids for a known prompt, and the prompt-length refusal past the
     port.
  2. End to end against the committed reference with the same `noise` passed in: the served
     actions on CPU within 1e-4, and two CPU calls within 1e-5 of each other. The comparison is on
     the served (denormalized) actions, not before `denormalize`; the reference is the independent
     Python pipeline above, not Intel's runtime. The GPU at `--ov-inference-precision f32` must stay within 1e-3
     of the reference; the GPU at the default f16 must be finite and within 0.5, with its
     differences printed. Whether those matter is decided by the LIBERO comparison below, not by a
     number chosen in advance (see "GPU precision").
  3. The compile cache gives bit-identical CPU actions; `model_id` replaces the directory name;
     `gpu_memory_bytes` is `None` on CPU and measured on the GPU.

  The other kinds' tests (`CASCADIA_POLICY_PACKAGE_<KIND>`, listed at the top of
  `tests/policy_openvino.rs`) take packages and references no runner stages; they run by hand.
- **Accuracy (by hand, not in CI).** `tools/vla_libero_client.py` adapts LIBERO observations to the
  WebSocket schema and connects through `openpi_client.WebsocketClientPolicy` exactly as
  `docs/engines/ov-policy.md` tells users to (`host="ws://…/v1/policy/ws"`, `port=None`), so an
  accuracy run is also the end-to-end interoperability test. It runs episodes of one suite against
  our `input`-mode exports and Intel's published packages, at `--ov-inference-precision f16` and `f32` on the
  GPU. Speed and parity do not show that a robot succeeds at a task; this does. Where the simulator
  runs is open question 2. Run for every kind (2026-11-15 and 16); the results are the tier table.
- **RTC (by hand).** `policy_openvino`'s RTC tests take SmolVLA and π0.5 LIBERO exported with and
  without RTC from one checkpoint, the RTC export's float32 twin, and raw inputs with Intel's
  PyTorch outputs for them (`CASCADIA_POLICY_PACKAGE_<M>_RTC`, `_NORTC`, `_RTC_F32`,
  `CASCADIA_POLICY_RTC_REF_<M>`): guidance off is bit-identical to no fields and within 1e-4 of the
  plain export; with guidance on the CPU f32 chunk is within 1e-4 of PyTorch's RTC path; on the GPU
  at f16 it is finite and deviates from CPU f32 by at most twice the plain chunk's deviation plus
  1e-3. A delayed-LIBERO A/B (`tools/vla_libero_client.py --async-delay D --execution-horizon S
  [--rtc W]`) compares success and seam jumps with RTC off and on under the same schedule.
- **Rig (by hand).** Latency through `cascadia run` on a Core Ultra AI PC for SmolVLA and π0.5
  LIBERO, plus a check through the real `openpi_client`: the metadata arrives first, one `infer`
  returns actions, a bad request raises the error frame's text, and a fifth connection is refused
  at capacity. **Not shipped:** a script for this check; `tests/policy_ws.rs` covers the same
  behaviour against `MockPolicyEngine` over a real TCP socket.

## Open questions

1. **GPU precision (decided 2026-11-16 after LIBERO: f16 stays the default).** Measured on an
   Arc 140V with our SmolVLA noise package and a real, padded prompt, five input draws against CPU
   f32:

   | GPU mode | worst max abs (normalized) | mean abs | NaN | p50 latency |
   |---|---|---|---|---|
   | default f16 | 0.009 – 0.122 | 0.0013 – 0.0021 | 0 | 143 ms |
   | `INFERENCE_PRECISION_HINT=f32` | ≤ 3.2e-4 | ~1e-6 | 0 | 346 ms |

   The engine defaults to the GPU's f16 and takes `--ov-inference-precision f16|f32`. LIBERO compared task
   success at both: SmolVLA 23/30 at f16 and at f32 (one episode differs each way), X-VLA 26/30 at
   f16 against 27/30 at f32 (one episode) with f32 4.6 times slower. f16 succeeds as often, so it
   stays the default. CPU is always f32, because bf16 compute is off by up to 0.66
   (step 0).
2. **Where the LIBERO simulator runs (settled).** The simulator on a Linux x86 machine (MuJoCo with
   CPU rendering), the engine on an AI PC's GPU (the f16-versus-f32 comparison needs the GPU), joined
   by an SSH tunnel. The added network delay doesn't affect an accuracy run,
   since the simulator waits for each reply. Every LIBERO run in the tier table ran this way.
3. **DROID statistics.** openpi publishes DROID norm stats with its checkpoints, but in its own
   format. Can the exporter take them through `--stats`?
4. **GPU memory.** When an LLM engine and a policy share one box: out of scope for v1 (one engine
   per process), left to the operator.
