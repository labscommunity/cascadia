# CLI reference

Every command and flag `cascadia` accepts. This mirrors `cascadia <command> --help`,
which is always authoritative — run it if this page and the binary disagree.

```
cascadia [--log-level <LEVEL>] <COMMAND>
```

`--log-level` is global (default `info`, overrides `RUST_LOG`) and `--help` works
on every subcommand. `--version` is only accepted on the top-level command.

> **Security.** The HTTP API and the inter-stage TCP relay are plaintext and
> unauthenticated. Bind them to trusted networks only, or put TLS + auth on a
> reverse proxy in front of `--api`. See [SECURITY.md](../SECURITY.md).

## Commands

| Command | What it does |
|---|---|
| [`doctor`](#cascadia-doctor) | Check environment + hardware. **Run this first.** |
| [`run`](#cascadia-run) | Serve a model on one machine (OpenAI API). |
| [`worker`](#cascadia-worker) | Run one stage of a distributed pipeline. |
| [`shard`](#cascadia-shard) | Export a model into per-stage OpenVINO IRs. |
| [`engines`](#cascadia-engines) | List registered inference engines. |
| [`discover`](#cascadia-discover) | List Cascadia peers on the LAN. |
| [`profile-devices`](#cascadia-profile-devices) | Benchmark each OV device against one model. |
| [`profile-stages`](#cascadia-profile-stages) | Cost table: each shard stage on each device. |
| [`place`](#cascadia-place) | Solve device placement from that cost table. |
| [`run-placement`](#cascadia-run-placement) | Launch a pipeline from a solved placement. |
| [`completions`](#cascadia-completions) | Generate a shell completion script. |

**Models are never downloaded at run time.** `run` and `worker` take a *local
directory* — either a whole-model OpenVINO IR or a `cascadia shard` tree. Only
`cascadia shard` fetches from HuggingFace. Passing an HF repo id to `run` or
`worker` is an error.

---

## `cascadia doctor`

Environment + hardware self-check. Catches the silent OpenVINO CPU-only
fallback, which otherwise shows up much later as unexplained slowness.

```
cascadia doctor [--strict]
```

| Flag | Default | Meaning |
|---|---|---|
| `--strict` | off | Exit non-zero if any check is WARN or FAIL. For CI / provisioning gates. |

Build-only checks (Rust, C++, export-time Python) report as informational, so
`--strict` still passes on a release bundle, which ships no toolchain.

---

## `cascadia run`

Single-machine serving. Sugar for a one-stage `worker` with the OpenAI API on.

```
cascadia run [OPTIONS] <MODEL>
```

`<MODEL>` — a local model directory. Either a whole-model OpenVINO IR (from
`optimum-cli export openvino`, or a pre-exported `*-int4-ov` download) or a
`cascadia shard` tree. **Not** an HF repo id.

| Flag | Default | Meaning |
|---|---|---|
| `--device <DEVICE>` | `GPU` | `GPU` / `CPU` / `NPU`. Also the indexed and compound OpenVINO forms — see [`--device` forms](#device-forms). |
| `--engine <ENGINE>` | `ov-genai` | Inference engine. Use `ov-runtime` for a `cascadia shard` tree. |
| `--api <API>` | `:8000` | API bind address. `127.0.0.1:8000` binds loopback only. |
| `--prefix-cache-gb <GB>` | min(16, RAM/4) | `qwen35` only: budget of the in-process prefix cache (system-block + chat-boundary state snapshots, restored for prompts that extend a cached prefix). `0` disables. |
| `--api-max-body-mb <MB>` | `1` | Largest chat-completions request body (MiB); the rendered prompt is capped at the same size (1 MiB ≈ 250K tokens of English). Only `qwen35` also bounds the prompt in tokens against the model window; on other engines this byte cap is the whole guard, so lower it on an exposed endpoint. |

Note the defaults differ from `worker` (`GPU`/`ov-genai` here, `CPU`/`mock`
there): `run` assumes you want real inference on the accelerator.

```bash
cascadia run ~/models/qwen3-1.7b-int4-ov --device GPU
```

---

## `cascadia worker`

One stage of a pipeline. Multi-stage inference is several `worker` processes
chained by `--listen` / `--next`.

```
cascadia worker --rank <N> --total <N> --model <DIR> [OPTIONS]
```

### Topology

| Flag | Default | Meaning |
|---|---|---|
| `--rank <RANK>` | *required* | 0-based stage index. |
| `--total <TOTAL>` | *required* | Total number of stages. |
| `--model <MODEL>` | *required* | Local model directory. Not an HF repo id. |
| `--listen <LISTEN>` | `:9100` | Bind address for the upstream-receiving socket: `host:port`, `:port`, or a [unix socket](#unix-domain-sockets) `unix:/path.sock`. |
| `--next <NEXT>` | — | Downstream peer: `host:port` or `unix:/path.sock`. Required for every stage but the last. |
| `--api <API>` | — | API bind address (TCP only). **Rank 0 only** — other ranks enter the relay loop and never bind it. Passing it there logs a warning and is otherwise ignored. |
| `--engine <ENGINE>` | `mock` | Inference engine. `mock` runs without OpenVINO. |
| `--device <DEVICE>` | `CPU` | OpenVINO device target — see [below](#device-forms). |

Rank 0 serves the API and holds the first layers; the last rank produces tokens
and returns them up the chain. Start the *downstream* worker first — engines
wait 60 s for a downstream peer, then give up.

### Unix domain sockets

When two adjacent stages run on the SAME host (iGPU stage 0 → dGPU stage 1 on
one workstation, or single-box stage testing), link them over a Unix domain
socket instead of loopback TCP ([#17](https://github.com/labscommunity/cascadia/issues/17)):

```bash
# stage 1 (start first)
cascadia worker --rank 1 --total 2 --model <DIR> --listen unix:/tmp/cascadia-1.sock

# stage 0
cascadia worker --rank 0 --total 2 --model <DIR> --next unix:/tmp/cascadia-1.sock --api :8416
```

Address forms recognized for `--listen`/`--next` (and for each entry of
`--ep-workers`, the Inkling expert-parallel driver's worker list):
`unix:/path.sock` (explicit), an absolute path, or any `.sock`-suffixed path.
The wire format is identical to TCP; the two stages must simply agree on the
path.

`/tmp` is used above for brevity. For real deployments put the socket in a
private directory — `$XDG_RUNTIME_DIR` or a `0700` directory you own — so
other local users can't plant or race a file at that path. The path must fit
in `sun_path` (104 bytes on macOS, 108 on Linux); a longer one is rejected at
startup with a clear error.

Measured round-trip latency vs `127.0.0.1` TCP (Apple Silicon macOS, release
build, `cargo test -p cascadia-transport --release --test uds_vs_tcp_bench --
--ignored --nocapture`):

| Frame | TCP p50 | UDS p50 | Win |
|---|---|---|---|
| 14 KiB (decode-step hidden state) | ~32–55 µs | ~4–5 µs | ~85–92 % |
| 1 MiB (prefill/logits class) | ~145–180 µs | ~108–111 µs | ~24–40 % |

Notes:

- Unix only (Linux/macOS). On Windows a `unix:` address fails fast with a
  clear error; use TCP there.
- The socket file is set to mode `0600` (owner-only) before the listener
  accepts any connection.
- Ownership of the `--listen` path is an flock on `<path>.lock`, held for
  the worker's lifetime and released by the kernel when it exits (however it
  exits). While another worker holds it, a new worker refuses to start with
  an address-in-use error and leaves the live socket alone. With the lock
  free, a socket file left at the path (by a crash) is stale and is reclaimed
  — unlinked and re-bound. A **non-socket** file is never deleted — the
  worker refuses to start instead. The empty `.lock` file stays next to the
  socket and is reused by the next worker on that path.
- On a clean close (or when the server is dropped) the worker unlinks the
  socket file, but only if the path still refers to the socket it bound
  (inode check), so it never removes a successor's socket. Relay ranks
  (every rank without `--api`) install no SIGTERM/SIGINT handler, so `kill`
  or Ctrl-C ends them without that cleanup and the file is left behind; it
  is reclaimed by the stale-socket path above on the next start.
- `--api` stays TCP; unix addresses are for the inter-stage activation
  relay only.
- Cross-host topology probing doesn't apply: a UDS-listening stage
  advertises no TCP relay port over mDNS (port `0`), peers don't
  latency-probe it, and `cascadia discover` labels it as an in-host unix
  link rather than suggesting a `--next host:port` for it.

### Layer split

| Flag | Default | Meaning |
|---|---|---|
| `--layer-start <N>` | `0` | First transformer layer this stage holds (inclusive). |
| `--layer-end <N>` | `0` | One-past-last layer (exclusive). `0` = unset. |

Both at `0` means an even split across ranks. Setting them pins an asymmetric
split — e.g. a high-RAM CPU node holding most layers while small iGPU nodes hold
a few each. MiniMax-M2 `sparse-moe` only.

### OpenVINO tuning

| Flag | Default | Meaning |
|---|---|---|
| `--ov-cache-dir <DIR>` | platform cache | Compiled-blob cache (plugin `CACHE_DIR`). See note below. |
| `--ov-kv-precision <P>` | optimal | GPU KV-cache precision (`u8` / `f16`). |
| `--ov-dyn-quant-group <N>` | — | GPU dynamic-quantization group size. |
| `--ov-performance-mode <MODE>` | — | `LATENCY` / `THROUGHPUT` / `CUMULATIVE_THROUGHPUT`. |
| `--ov-inference-precision <PREC>` | — | `f16` / `bf16` / `f32`. |
| `--ov-num-streams <N>` | — | Parallel inference streams (`NUM_STREAMS`). |
| `--ov-num-threads <N>` | — | Host CPU thread cap (`INFERENCE_NUM_THREADS`). CPU plugin only. |
| `--ov-allow-auto-batching` | off | Allow GPU-plugin internal auto-batching. |
| `--ov-execution-mode <MODE>` | — | `ACCURACY` / `PERFORMANCE`. |
| `--prefix-cache-gb <GB>` | min(16, RAM/4) | `qwen35` only, single-process (`--total 1`): byte budget of the chain-state prefix cache; a Qwen3.8-27B snapshot is ~130 KB per context token as serialised (1.2 GB at 8 K, 4.45 GB at 32 K). `0` disables. See [qwen3.8.md](architectures/qwen3.8.md). |
| `--api-max-body-mb <MB>` | `1` | Largest `/v1/chat/completions` body (MiB); the rendered prompt is capped alike. Was a fixed 64 KiB / 32 KiB (~8K tokens) before. `qwen35` additionally rejects prompts at or past `max_position_embeddings` (413). A chat request is measured across **every** turn of `messages` plus the tool fields, for every engine (not just the last turn), so a long history can 413 where a last-turn-only check would have passed. |
| `--ep-workers <host:port,...>` | — | `sparse-moe` (Inkling) expert-parallel **driver**: dispatch each MoE layer's selected experts to these running workers (each entry may also be a [unix socket](#unix-domain-sockets) `unix:/path.sock`); this rank runs every layer's attention/router locally and holds no expert weights. Implies `--total 1`. Start the workers first. |
| `--ep-worker-index <N>` / `--ep-worker-count <W>` | — | `sparse-moe` (Inkling) expert-parallel **worker**: serve expert shard N of W (experts with `id % W == N`, shared experts included) for every MoE layer on `--listen`; no API, no attention, no sequence state. |
| `--ov-config <KEY=VALUE>` | — | Raw OV plugin property passthrough, repeatable. See below. |
| `--elastic` | off | Elastic memory posture (Linux; file-backed big allocations). On `--engine sycl-llama` it instead enables `GGML_STREAM_WEIGHTS=1` on the spawned `llama-server` (device-side weight streaming; with the default `--elastic-vram auto` a model that fits runs at stock speed) — see below and [engines/sycl-llama.md](engines/sycl-llama.md). |
| `--elastic-min-mb <MB>` | 1 | Elastic threshold; 16 = weights-only, zero speed cost. |
| `--elastic-pool-mb <MB>` | 8192 | Elastic retained-mapping pool cap (0 = off). |

**`--ov-config KEY=VALUE`** forwards any plugin property to OpenVINO verbatim,
alongside the typed `--ov-*` flags — the escape hatch for knobs without a
dedicated flag (`--ov-config KV_CACHE_PRECISION=u8`,
`--ov-config DYNAMIC_QUANTIZATION_GROUP_SIZE=0`). Repeatable; applied last, so it
overrides a typed flag setting the same key. The value may contain `=` (split on
the first only). Malformed entries (no `=`, empty key) are rejected at parse
time; OV itself validates the key/value. NPU-prefixed keys are gated to an NPU
device + `--engine ov-genai`, like the typed NPU flags. Cross-platform.

**`--elastic`** serves large allocations from file-backed mappings so the
worker's weight copies, KV state and scratch are kernel-reclaimable rather than
committed. Measured on unmodified OpenVINO CPU workers (ramlab exp 199): Linux
2064→506 MB committed at −1% decode (`--elastic-min-mb 16` → 1549 MB at parity);
Windows 1329→223 MB private commit at −6% decode. Same file-backed mechanism,
different injection vector: **Linux** re-execs the worker once with an allocator
interposer `LD_PRELOAD`ed (serving PID unchanged via execv; logs `elastic
posture active`); **Windows** inline-hooks the UCRT allocation family
(`malloc`/`free`/`realloc`/`calloc`/`_msize` + `_aligned_*`) in-process with
Microsoft Detours before the OV engine loads (logs `elastic posture active
(in-process)`). The OV-native knobs cannot substitute — they cannot disable
oneDNN's dirty repacked copies (D-004) — so `--elastic` still asserts
`ENABLE_MMAP=YES` to keep the weight blob clean but relies on the interposer for
the cut. The Windows hook is compiled in only when `cascadia-elastic` was built
with `DETOURS_DIR` set; otherwise `--elastic` reports inactive there.

**`--ov-cache-dir` is on by default and matters.** For `ov-genai`, `ov-runtime`,
`gemma4` and `sparse-moe`, leaving it unset defaults to
`<user-cache>/cascadia/ov-cache` (`~/.cache` on Linux, `~/Library/Caches` on
macOS, `%LOCALAPPDATA%` on Windows). This turns a ~20 s cold GPU compile into a
~1 s warm load on every later run — the single biggest latency win on the
`ov-genai` path. Pass `--ov-cache-dir ""` to disable.

Two engines don't get that default: `ov-dist-spec` uses the flag verbatim (so it
is off unless you pass a path), and `qwen35` ignores it entirely.

On **Xe2 / Battlemage** GPUs, set `--ov-inference-precision f16` explicitly: f16
and bf16 share XMX throughput, but the default can silently fall back to f32.

### NPU

| Flag | Meaning |
|---|---|
| `--npu-prefill-chunk-size <TOKS>` | `NPUW_LLM_PREFILL_CHUNK_SIZE` (OV 2025.3+). |
| `--npu-max-prompt-len <TOKS>` | `MAX_PROMPT_LEN` — static-shape constraint. |
| `--npu-min-response-len <TOKS>` | `MIN_RESPONSE_LEN` — static-shape constraint. |

All three apply **only** with `--engine ov-genai` on an NPU device — only that
engine routes them through an `ov::genai::LLMPipeline`. Anywhere else they are
dropped, with a warning in the log.

### Speculative decode

| Flag | Default | Meaning |
|---|---|---|
| `--draft-model <DIR>` | — | Draft model path (FastDraft companion). A local IR dir, not an HF id. |
| `--draft-device <DEVICE>` | same as `--device` | Device for the draft model. |
| `--spec-k <K>` | `5` | Draft length per round. |
| `--prompt-lookup <N>` | `0` | Prompt Lookup decoding with n-gram size N. Mutually exclusive with `--draft-model`. |
| `--cb` | off | Continuous batching (#20, ov-genai only): concurrent requests share one `ContinuousBatchingPipeline` (paged attention, CPU/GPU plugins) instead of serializing one generation at a time. Incompatible with `--draft-model` / `--prompt-lookup`. NPU serves the static NPUW pipeline and cannot continuous-batch. |
| `--cb-cache-size <GB>` | `0` | KV-cache GB for `--cb` (0 = ov-genai dynamic allocation). |
| `--cb-max-num-seqs <N>` | `0` | Max sequences per batch iteration (0 = ov-genai default 256). |
| `--cb-max-batched-tokens <N>` | `0` | Max tokens per batch iteration (0 = ov-genai default 256). |
| `--cb-dynamic-split-fuse <BOOL>` | ov-genai default (on) | Dynamic-split-fuse scheduler toggle for `--cb`. |
| `--cb-prefix-caching <BOOL>` | ov-genai default (off) | KV-block prefix reuse across requests for `--cb`. |
| `--packed-slots <N>` | `0` | Continuous batching on the **NPU** (`--engine ov-runtime`, static `--target npu` exports): serve N concurrent requests in ONE inference by packing them into the sequence dimension with a per-row mask. Needs a packed variant beside the decode IR (`tools/packed_variant.py --slots N`). A different mechanism to `--cb` — the NPU has no paged attention. Works single- and multi-stage; every stage must run the same `--packed-slots` value (baked into the packed IR shape). See [docs/perf/NPU_PACKED_SLOTS.md](perf/NPU_PACKED_SLOTS.md). |
| `--packed-prefix <N>` | `0` | Reserve N KV slots as a read-only shared prefix every packed slot may attend to — prefix caching without paging. Costs per-slot context. Requires `--packed-slots`. Single-stage only (`--total 1`): only the first stage admits requests, so a relay stage never populates the shared prefix. |

For distributed speculative decode, every rank needs `--engine ov-dist-spec` —
they share a wire protocol. See [engines/ov-dist-spec.md](engines/ov-dist-spec.md).

### Sparse-MoE tuning

`sparse-moe` engine only. Deep-dives: [perf/A3_TOPK_REDUCTION.md](perf/A3_TOPK_REDUCTION.md),
[perf/CHESS_PER_CHANNEL.md](perf/CHESS_PER_CHANNEL.md).

| Flag | Default | Env var | Meaning |
|---|---|---|---|
| `--top-k-override <K>` | manifest | — | Dispatch only the first K experts per token. K2.6 default is 8; K=4 gives +146% tok/s at matched quality. |
| `--routing-threshold <T>` | `0.0` | — | Skip experts whose router weight is below T. Applied after `--top-k-override`. |
| `--kv-prefix-cache-size <N>` | `0` | — | Cache post-prefill KV per prompt prefix. Single-stage only. ~150 MiB per snapshot, so practical caps are 1–8. |
| `--max-cached-experts <N>` | `0` (unbounded) | `CASCADIA_MAX_EXPERTS_CACHED` | LRU bound on resident experts. ~25 MiB each (int4_bin) or ~75 MiB (ov_ir). |
| `--ffn-sparsity-threshold <T>` | `0.0` (dense) | `CASCADIA_FFN_SPARSITY_THRESHOLD` | Skip FFN lanes below `T · max|silu(gate)|`. Useful range 0.05–0.15. |
| `--ffn-axpy-down` | off | `CASCADIA_FFN_AXPY_DOWN` | AXPY-form down kernel. Only meaningful with `--ffn-sparsity-threshold > 0`. |
| `--ffn-axpy-prebuild` | off | `CASCADIA_FFN_AXPY_PREBUILD` | Prebuild the AXPY cache for every (layer, expert). ~20 s once; ~190 GiB disk at K2.6. |
| `--ffn-sparsity-thresholds-file <F>` | — | `CASCADIA_FFN_SPARSITY_THRESHOLDS_FILE` | Per-channel thresholds (CHESS). Takes precedence over the scalar. |
| `--ffn-sparsity-capture-dir <D>` | — | `CASCADIA_FFN_SPARSITY_CAPTURE_DIR` | Dump `silu(gate)` histograms for calibration. Needs `--ffn-axpy-down`. |

Raising sparsity trades quality for speed. Validate against dense before you
deploy.

### sycl-llama

`sycl-llama` engine only. Deep-dive: [engines/sycl-llama.md](engines/sycl-llama.md).
All flags exist on both `cascadia run` and `cascadia worker`.

| Flag | Default | Env var | Meaning |
|---|---|---|---|
| `--llama-bin <PATH>` | auto | `CASCADIA_LLAMA_BIN` | Path to `llama-server`. Resolution: flag > env > `llama-server` on `PATH`. |
| `--llama-ctx <N>` | `4096` | — | Context size (`-c`). |
| `--llama-ngl <N>` | `99` | — | GPU layers (`-ngl`). Forced to 0 when `--device CPU`. |
| `--llama-args <ARGS>` | — | — | Extra raw args, one value per occurrence split on spaces, repeatable: `--llama-args "-ctk q8_0 -fa on"`. |
| `--llama-load-timeout <SECS>` | auto | — | Per-attempt health deadline: `60 + 8 per GiB` of model file. |
| `--llama-load-retries <N>` | `1` | — | Extra load attempts after a child exit or health timeout. |
| `--elastic-vram <auto\|GiB>` | `auto` | `GGML_STREAM_VRAM_MB` | Resident-weight budget for `--elastic`: `auto` = free VRAM − non-streamed weights − 2× largest layer − reserve; `0` = stream every layer. |
| `--elastic-share <N>` | — | `GGML_STREAM_VRAM_SHARE` | Expected co-tenant count for `--elastic-vram auto` (sycl-llama): caps automatic resident weights at 1/N of the card so N instances load equal shares. Ignored with an explicit `--elastic-vram` or `GGML_STREAM_RESIDENT_LAYERS`. |
| `--llama-mtp` | off | — | Speculative decoding with the model's own MTP head (`--spec-type draft-mtp`; GGUF needs nextn layers, e.g. Qwen3.8-27B). Qwen3.8-27B on the B70: 19.0 → 26.5 t/s resident, 0.80 → 2.39 t/s at `--elastic-vram 8`, 0.67 → 2.02 t/s fully streamed. |
| — | `0` | `CASCADIA_EXPERT_CACHE_MB` → `GGML_STREAM_EXPERT_CACHE_MB` | MiB of hot-expert residency on MoE models under `--elastic`. Default 0: only the experts the router selects are read per token (automatic, any MoE); a budget pins the hottest expert slices on device. |

### Other

| Flag | Default | Meaning |
|---|---|---|
| `--max-tokens <N>` | `64` | Max new tokens in stdin mode. |
| `--advertise-engines <LIST>` | from `--engine` | Override the engines list in the mDNS record. Cosmetic — dashboard only. |
| `--advertise-device <LABEL>` | from `--device` | Override the device label in the mDNS record. Cosmetic. |

---

## `cascadia shard`

Export a HuggingFace causal-LM into per-stage OpenVINO IRs. **The only command
that downloads from HuggingFace.** Needs export-time Python deps (~3 GB); run
`cascadia doctor` to see the exact pinned `pip install` line for your host.

```
cascadia shard --model <MODEL> -o <DIR> --num-stages <N> [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <MODEL>` | *required* | HF repo id, a local dir with safetensors + `config.json`, or — for the Gemma-4 / Qwen3.5-family (3.6 MoE, 3.8 dense) surgery paths — an already-exported OpenVINO IR dir. |
| `-o, --output-dir <DIR>` | *required* | Output shard tree (created for you). |
| `--num-stages <N>` | *required* | Pipeline stages to split into. |
| `--quantization <Q>` | `int4` | `fp16` / `int4` / `int4-asym` / `int8`. INT4 is the typical choice on Intel; FP16 if NNCF is unavailable or you want max quality. |
| `--target <T>` | `cpu-gpu` | `npu` emits a stateless static-shape shard the NPU compiler accepts; `cpu-gpu` emits the stateful dynamic-shape shard. |
| `--layer-split <LIST>` | even | Explicit per-stage boundaries. `--num-stages 3 --layer-split 16,24` on 32 layers → `[0,16) [16,24) [24,32)`. |
| `--default-dtype <D>` | `fp16` | torch dtype during export. FP16 is **required** for `--target npu`. |
| `--static-seq <N>` | `1` | NPU only: fixed query-window length. Must be 1. |
| `--static-context <N>` | `1024` | NPU only: fixed total context length. |
| `--stage <N>` | all | Export only this stage (debug) — re-export one stage without redoing the rest. |
| `--python <PATH>` | auto | Override the interpreter running the bundled exporter. |
| `--skip-check` | off | Skip interpreter/dependency detection. Faster start if you know the env is good. |

```bash
cascadia shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
  -o ~/shards/llama-3.1-8b --num-stages 2 --quantization int4
```

Model-family dispatch happens inside the exporter. A Gemma-4 OpenVINO IR dir
(with `openvino_language_model.xml`) routes to the text-surgery path; Gemma-4
safetensors still use the torch exporter. Unsupported families are rejected up
front — see [SHARDING.md](SHARDING.md) for the support table.

---

## `cascadia engines`

Lists the registered inference engines. No flags.

---

## `cascadia discover`

Browse mDNS for Cascadia peers on the LAN.

```
cascadia discover [--namespace <NS>] [--timeout <SECS>]
```

| Flag | Default | Meaning |
|---|---|---|
| `--namespace <NS>` | `default` | Discovery namespace. Peers in another namespace are ignored. |
| `--timeout <SECS>` | `5` | How long to listen. Announces land in ~2.5 s in practice. |

Discovery is informational: workers still need explicit `--rank` / `--total` /
`--next`. Auto-ring formation is not wired up yet ([#89](https://github.com/labscommunity/cascadia/issues/89)).

---

## Heterogeneous placement

A four-step pipeline that measures your hardware, then solves which device each
stage should run on. Only the measuring steps (`profile-stages`, `profile-devices`)
need a real OpenVINO build — a stub binary refuses them. `place` is a pure solver
and `run-placement --dry-run` only prints commands, so both run anywhere.
Background: [perf/THREE_TIER_PLACEMENT.md](perf/THREE_TIER_PLACEMENT.md).

```
shard --target npu  →  profile-stages  →  place  →  run-placement
                       placement_profile.json      placement.json
```

`profile-devices` is a separate, simpler tool: it benchmarks *whole* models per
device and is not part of this chain.

### `cascadia profile-devices`

Benchmark each OV device on this host against one whole model. Writes
`device_profile.json`. See [perf/DEVICE_PROFILE.md](perf/DEVICE_PROFILE.md).

```
cascadia profile-devices --model <DIR> [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <DIR>` | *required* | Exported OV-GenAI model dir. Same path you'd give `worker --engine ov-genai`. |
| `--output <FILE>` | `device_profile.json` | Where to write the profile. |
| `--devices <LIST>` | `auto` | `auto` enumerates every plugin OV sees. A comma list pins the set (`CPU,GPU` to skip a flaky NPU). `HETERO:` strings pass through verbatim. |
| `--prompt <TEXT>` | `"Explain Intel Lunar Lake in three sentences."` | Measurement prompt. Keep it short — the bench measures decode. |
| `--max-tokens <N>` | `32` | Tokens generated per run. |
| `--runs <N>` | `3` | Measured runs per device; best is reported as tok/s. |
| `--warmup <N>` | `1` | Warmup runs, not counted. Bump to 2 on large first-vs-second-run variance. |
| `--include-hetero-permutations` | off | Also profile every `HETERO:` priority order. Count grows factorially — 6 extra runs on a 3-device host. |
| `--ov-cache-dir <DIR>` | off | Off by default so cold-compile time is measured honestly. |
| `--no-summary` | off | Suppress the stdout table. JSON is still written. |

### `cascadia profile-stages`

Cost table: every shard stage measured on every device (latency + memory +
op-support). Writes the `placement_profile.json` that `place` consumes.

```
cascadia profile-stages --shard <DIR> [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--shard <DIR>` | *required* | Multi-stage shard dir (`stage_0/`, `stage_1/`, …). **Must be a static export** — `cascadia shard --target npu`. |
| `--output <FILE>` | `placement_profile.json` | The input to `cascadia place`. |
| `--devices <LIST>` | `auto` | `auto`, or a comma list like `GPU,NPU,CPU`. |
| `--runs <N>` | `5` | Timed forward passes per (stage, device); the min is recorded. |
| `--warmup <N>` | `2` | Warmup passes, not counted. |
| `--mem-headroom <F>` | `0.9` | Fraction of each device's memory treated as usable — leaves room for KV cache and activations. |
| `--pool-gb <N>` | — | Usable shared UMA pool in GiB. Sets the solver's global memory gate *and* the CPU tier's budget. Omit to skip the global gate. |
| `--force` | off | Re-measure even when a cached profile with a matching fingerprint exists. |

Results are cached by fingerprint (shard, devices, pool) — NPU compiles are slow,
so a matching profile is reused unless you pass `--force`.

### `cascadia place`

Solve the three-tier {iGPU, NPU, CPU} placement ILP. Writes `placement.json`.

```
cascadia place --profile <FILE> [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--profile <FILE>` | *required* | The `placement_profile.json` from `profile-stages`. |
| `--output <FILE>` | `placement.json` | Solved placement. |
| `--worker-overhead-gb <N>` | `1` | Resident overhead *per worker process* in GiB, beyond its weights. Used only to warn when the total footprint would exhaust the pool and swap. ~1 GiB/worker measured for a 12-stage fp16 run on Lunar Lake. |

### `cascadia run-placement`

Launch the solved pipeline: one worker per stage, each pinned to its device.

```
cascadia run-placement --shard <DIR> --placement <FILE> [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--shard <DIR>` | *required* | The same shard dir you profiled. |
| `--placement <FILE>` | *required* | `placement.json` from `cascadia place`. |
| `--api <API>` | `127.0.0.1:8000` | API bind address for the pipeline head (rank 0). |
| `--relay-host <HOST>` | `127.0.0.1` | Host the relay sockets bind/dial. |
| `--relay-base <PORT>` | `9100` | Base relay port. Stage `r` (r≥1) listens on `relay_base + r`. Concurrent runs on one host must not overlap. |
| `--ov-cache-dir <DIR>` | — | Forwarded to every worker. |
| `--dry-run` | off | Print the worker command lines and exit without spawning. |

This launcher spawns all stages as **local** child processes. To span machines,
run `cascadia worker` per box by hand.

---

## `cascadia completions`

```
cascadia completions <SHELL>
```

`<SHELL>` is one of `bash`, `zsh`, `fish`, `elvish`, `powershell`. The script
goes to stdout.

```bash
cascadia completions zsh  > ~/.zfunc/_cascadia
cascadia completions bash > /etc/bash_completion.d/cascadia
```

---

## Device forms

`--device` is forwarded verbatim to `ov::Core::compile_model`, so it accepts
every form OpenVINO does:

| Form | Meaning |
|---|---|
| `CPU` | Host CPU. |
| `GPU`, `GPU.0`, `GPU.1`, … | A specific GPU. The iGPU is `.0` by convention. |
| `NPU`, `NPU.0`, … | Neural Processing Unit (Lunar Lake and later). |
| `AUTO` | Let OpenVINO pick. |
| `MULTI:GPU.1,GPU.0,CPU` | Round-robin across devices. |
| `HETERO:GPU.1,CPU` | Split the graph by op affinity. |
| `BATCH:GPU` | Auto-batch (throughput-favored). |

Run `cascadia doctor` to see which devices OpenVINO can actually reach on this
host. A device that `clinfo` reports healthy is **not** necessarily one the
OpenVINO GPU plugin can see.

## Engines

| Engine | Input layout | Notes |
|---|---|---|
| `mock` | none | No OpenVINO needed. Dev / discovery testing. |
| `ov-genai` | whole-model OpenVINO IR | Default for `run`. Adds FastDraft speculative decode + prompt lookup. |
| `ov-runtime` | `cascadia shard` tree | The staged pipeline engine. |
| `ov-dist-spec` | `cascadia shard` tree | Distributed speculative decode. Every rank must use it. |
| `gemma4` | `gemma4_cached_v1.x` shards | Per-layer-type asymmetric attention, KV-sharing, baked softcap; v1.1 masks sliding layers to their window (v1 trees warn at load). |
| `sparse-moe` | `manifest.json` + expert tree | Top-k expert routing through an AVX-512 int4 GEMM. CPU-targeted; single-stage or pipeline-parallel (`--total >= 2`). |
| `qwen35` (alias `qwen36-moe`) | Qwen3.5-family surgery output (Qwen3.6 MoE, dense Qwen3.8) | Greedy-only, batch=1. CPU or iGPU decode. In-process prefix cache (`--prefix-cache-gb`, default min(16 GiB, RAM/4), 0 = off): system-block + chat-boundary state snapshots restored for prompts that extend a cached prefix. |

Per-engine deep dives: [engines/](engines/). Per-family export notes:
[architectures/](architectures/).

## Structured output (`response_format`)

`/v1/chat/completions` accepts OpenAI's `response_format` field:

| `type` | Output |
|---|---|
| `text` (or no field) | Free text, as before. |
| `json_object` | Any JSON object. |
| `json_schema` | JSON matching `response_format.json_schema.schema`. A missing, `null`, `true` or `{}` schema means any JSON object; `false` is rejected. Object properties come out in the schema's order. |

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name a city and its country."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"]}}}
}'
```

`/v1/completions` ignores `response_format` (OpenAI does not define it there).

Every engine enforces `json_object` and `json_schema`:

| Engine | How |
|---|---|
| `ov-genai` | OpenVINO GenAI's structured output (xgrammar), stateful and `--cb`. Needs OpenVINO GenAI 2026.4 or newer. At startup the engine runs one short structured-output check on the loaded model; if it fails, startup continues and JSON requests get 400 `invalid configuration: structured output is not available for this model: <reason>`. With `--draft-model` (speculative decoding) the check is skipped and JSON requests get that 400 with `not supported with a draft model (speculative decoding); run without --draft-model for response_format`: on OpenVINO GenAI 2026.4.1 the draft's proposals break the constraint. `--prompt-lookup` is fine. Not available on NPU either (`--device NPU`, `NPU.0`, or a `HETERO:`/`AUTO:` list that includes NPU): the check is skipped and JSON requests get that 400 with `not reliable on NPU with this OpenVINO GenAI version (the output constraint is not fully applied); use GPU or CPU for response_format`, because on GenAI 2026.4.1 the NPU pipeline does not fully apply the constraint. A schema xgrammar would take minutes to compile (a long run of optional properties, `minProperties`/`maxProperties`, nested `type` lists, many distinct string length bounds; see the errors below), or would reject, is refused before it reaches GenAI. GenAI compiles each distinct schema once and keeps the result for the pipeline's life (a repeat is served in milliseconds); the first request with a new schema waits for its compile, which a string `maxLength` above 128 makes take a few seconds (3–5 s each on a Core Ultra 258V CPU with Qwen2.5's tokenizer). Each kept grammar holds about 3–30 MB (more for long strings; measured on an Arc GPU), and GenAI drops them only with the pipeline, so a pipeline lets 64 distinct schemas through; past that a new one gets `structured output is not available for this model: this worker has reached its limit of 64 distinct response_format schemas per pipeline …` (a schema already served still works), and that new schema makes the worker rebuild the pipeline, which clears the limit. Without `--cb` the rebuild runs at once, between two answers: requests waiting their turn carry on on the new pipeline. With `--cb` it runs at once when nothing is in flight; otherwise the worker first admits nothing (503 `temporarily unavailable: this worker is rebuilding its structured-output pipeline; retry the request shortly`) until the requests in flight finish, fails those still running after 15 s with 503 `this request was still running when the worker began rebuilding its structured-output pipeline; retry the request shortly`, and rebuilds on the next request. Neither 503 marks the worker unready on `/health`. A rebuild reloads the model (about 6.5 s for a 1.5B model on a laptop CPU) and runs the startup check again; if that check now fails, a request with a schema still waiting its turn gets the check's 400 when its turn comes. A pipeline is rebuilt at most once every 10 minutes (`CASCADIA_OV_GENAI_REBUILD_FLOOR_SECS` sets this in whole seconds, from 60 to 604800, a week) plus up to 60 s of random jitter, counted from when it was built, so the first rebuild can come no sooner than that after the worker loads its model; until then a new schema keeps getting the 400, which says how long is left, and the first new schema after that starts the rebuild. The request that starts an immediate rebuild is then served on the new pipeline (it counts toward the new limit), unless the new pipeline's check fails (it gets the check's 400) or the schemas still waiting their turn fill the new limit (it gets the 400). With `--cb` and requests in flight, the request that starts the drain gets the 400. Each rebuild is logged at WARN when it starts and when it ends (with its duration and how many schemas it cleared). If a rebuild fails, the worker serves nothing until it is restarted: every new request gets 500 `ov-genai pipeline rebuild failed: <reason>; this worker serves no requests until it is restarted`, and `/health` reports `degraded`. The first refusal on each pipeline is logged once at WARN (`the response_format schema budget is full`); each one after is logged at DEBUG with the running count. An `integer` the schema leaves unbounded is sent to GenAI bounded to the `i64` range, 19 digits (`minimum` -9223372036854775807, `maximum` 9223372036854775807 where the schema sets none; xgrammar refuses an integer bound past `i64`, so a non-negative integer gets the same maximum), so a model can't repeat digits until `max_tokens`; a schema with more than 256 integers to bound goes as written (logged: `integer length not bounded`); `number` is not bounded (xgrammar already caps a fraction at 6 digits). The 64-schema count goes by the client's schema; `json_object` (one fixed grammar) is not counted and never refused for it. |
| `ov-runtime` | In-engine token mask (llguidance), on the stateful path, static NPU exports and `--packed-slots` (one mask per slot). |
| `ov-dist-spec` | In-engine token mask (llguidance). Drafts are proposed unmasked; the target's verify rows are masked, so the output is the same as non-speculative masked decoding. |
| `gemma4` | In-engine token mask (llguidance). |
| `qwen35` | In-engine token mask (llguidance). |
| `sparse-moe` | In-engine token mask (llguidance), applied before sampling (temperature, top-p). n-gram speculative decoding (`--prompt-lookup`) is skipped for a JSON request; it decodes plainly. |
| `sycl-llama` | Forwarded to llama-server, which compiles the schema to its own grammar, with every `$ref` replaced by its target first (see the refs paragraph below). A `pattern` (or `patternProperties` key) the llguidance regex engine can't parse, such as zod's email regex with its look-aheads, is dropped first and logged (`response_format pattern not enforced`), as on the llguidance engines; llama-server would reject it with a 400. With thinking off (every JSON request without callable tools) the request also sets `reasoning_budget_tokens: 0`: llama-server's grammar for some chat templates (Gemma 4) allows a thought block before the JSON, and the budget closes one as soon as it opens. A schema llama-server would take too long to convert (too many optional properties, see the errors below) is refused before it is sent. |
| `mock` | Answers with a minimal instance of the schema, best effort: not every keyword is honoured. |

The llguidance engines (`ov-runtime`, `ov-dist-spec`, `gemma4`, `qwen35`,
`sparse-moe`):

- Emit JSON with Python `json.dumps` separators: `": "` after each key and
  `", "` between items, and no other whitespace between tokens.
- Treat `oneOf` as `anyOf` (pydantic's discriminated unions compile).
- Accept the schema keywords llguidance does not enforce (`not`,
  `if`/`then`/`else`, `dependentRequired`, `dependentSchemas`,
  `uniqueItems`, `contains`, `propertyNames`, unknown `format`s, and
  `minProperties`/`maxProperties` where llguidance can't enforce them, such
  as on objects with optional properties) and log them per request
  (`response_format keywords accepted but not enforced`). A `pattern` (or
  `patternProperties` key) the regex engine can't parse, such as one with
  look-around or a backreference, is dropped and logged the same way; the
  rest of the schema is still enforced.
- Bound an `integer` the schema leaves unbounded to the `i64` range, 19
  digits at most (`minimum: -9223372036854775807`,
  `maximum: 9223372036854775807` where the schema sets none), so a model
  can't repeat digits until `max_tokens`; every `i64` id (a Twitter/X or
  Discord snowflake, `i64::MAX`) still fits. A non-negative integer gets
  the same maximum: neither llguidance nor xgrammar builds an integer
  bound past `i64`, so a `u64` above `i64::MAX` is not admitted where the
  schema sets no maximum. A schema with too many integers for that
  (several hundred) is built without the added bounds, and logged
  (`integer length not bounded`); so is one the added bounds leave no
  value for (an `allOf` with a `const` past them, say). `number` is not
  bounded. A lower bound at or below -2^63 (C#'s `long.MinValue`, and the
  added minimum), which llguidance can't build, is raised to
  -9223372036854774784. A schema with an integer at or below -2^63 as a
  `const` or `enum` value or an upper bound is a 400
  (`` an integer `const`, `enum` value or upper bound at or below about -9.2e18 (-2^63, compared as a 64-bit float) is not supported ``): llguidance's build can't take it.
  The value is compared as a 64-bit float, so -9223372036854775807 counts
  too. A value is an integer here when it is written as one, or when the
  schema has an `integer` type anywhere (llguidance can intersect it with
  that type through `allOf`, a `$ref` or merged properties); a `number`
  such as `{"type": "number", "const": -1e20}` in a schema with no
  `integer` builds. Every object in the schema is checked for this, data
  (`default`, `examples`) included, so a schema with an `integer` type
  and a data object such as `{"const": -1e20}` gets this 400 too.
- Bound and clamp the integer a `#/...` ref reaches, read against the
  root (the API accepts only plain pointers into objects; see the limits
  below). A target inside `default` or `examples` reached only
  through a `$id` below the root is not bounded or clamped: its integers
  may take more than 19 digits, or llguidance refuses a lower bound it
  can't build with its own 400. Schemas in `$defs` and properties are
  bounded where they sit.
- Compile the schema when the request is submitted: a schema llguidance
  cannot build (for example `{"type": 42}` or an unresolvable `$ref`) is a
  400 before the request queues. A client's own top-level `x-guidance`
  object is replaced by the server's settings. Before the request queues,
  the server also builds each new schema once off the engine, at most 2 at
  a time and within 2 seconds (see the errors below), and feeds that
  grammar the first token of an answer (`{`, `[` and a few others); a schema that passed
  is remembered (up to 256) and `json_object` is never built there.
- Build their grammar factory from the model's tokenizer on the first JSON
  request, not at startup: that request pays for it once (about half a
  second). The tokenizer needs an end-of-sequence token; when the factory
  can't be built, the engine logs why then (`structured output unavailable
  for this model`) and JSON requests get 400 `invalid configuration:
  structured output is not available for this model: <reason>`.
- Requests without `response_format` take the same decode path as before:
  no mask, no grammar.

Multi-stage pipelines (`--total >= 2`) enforce the format too: the first
stage, which holds the tokenizer, computes each step's mask and sends it with
the step to the stage that picks the token. Only steps of a JSON request
carry a mask; every other request's stage traffic is unchanged. A mask that
repeats the one sent just before on the same link (most steps inside a JSON
string or number) is sent as a short marker instead of the full mask. Such a
step costs, per hop, the plain step plus: 8 bytes on `ov-runtime`, 4 bytes per
hidden row on `gemma4` (one row on a decode step), a 4-byte mask frame (24
bytes with its header) on `qwen35`, and on `sparse-moe` 4·r·⌈4/r⌉ bytes for a
frame of r rows (the marker's 4 words, padded to a whole column on every
row): 16 bytes on a one-row step, 24 on a 3-row window, 4·r from 4 rows up,
so a decode batch of 8 streams with one JSON row costs 32 bytes. A full mask costs about vocab/8 bytes
(16 KB for a 128k vocabulary). Every stage
must run a build with structured-output support. If a later stage runs an
older build, a JSON request fails (5xx) with `a later pipeline stage runs a
build without structured-output support; upgrade every stage`; normal
requests are unaffected. On `ov-runtime`, `gemma4` and `sparse-moe` the
first stage checks the pipeline with a short probe before the first JSON
request on it (an older stage reads the probe, ignores it and stays in
step), so that request fails within about 10 s (a few seconds more on a
`gemma4` pipeline of more than two stages), not at its token deadline.
After that, JSON requests on that pipeline fail at once; the check runs
again after 1 minute, or after 10 minutes when a JSON request's own step
went unanswered. A stage that was only slow and answers the probe late
clears the check on the next JSON request. `qwen35` knows from its
connection handshake and fails JSON requests at once. When deploying, upgrade every
stage before relying on `response_format`.

A malformed `response_format` is a 400 on every engine (it used to be
ignored).

Behaviour:

- Thinking is turned off for a JSON request: a JSON answer has no room for
  a reasoning preamble.
- When the request also offers tools the model may call (`tools` non-empty
  and `tool_choice` `"auto"` or absent), the answer is constrained to either
  a tool call or JSON matching the format; see
  [Tool calling](#tool-calling-tools-tool_choice). Under `"required"` or a
  named tool the format plays no part: the answer is the call. Where that
  constraint can't be applied, the format is **not enforced** and the server
  logs `response_format not enforced`; thinking then stays as requested,
  except on `ov-genai`, where it is turned off (thinking with tools is
  otherwise refused there). Such a request is answered (200) even on a node
  where `response_format` alone gets a 400 (NPU, `--draft-model`).
- A token limit can cut the JSON short: the response then has
  `finish_reason: "length"` and the content does not parse. Leave room in
  `max_tokens`.
- On `ov-genai`, small models may emit long runs of whitespace inside the
  JSON: still valid, but it spends tokens and can run into the token limit.
- On `ov-genai`, some JSON Schema keywords are accepted but **not
  enforced**: `not`,
  `if`/`then`/`else`, `dependentRequired`, `dependentSchemas`,
  `uniqueItems`, `contains`, `minContains`, `maxContains`, `multipleOf`, an
  `allOf` with more than one member other than the plain-object shape below,
  and constraints placed beside
  `$ref`/`anyOf`/`oneOf`/`allOf`/`const`/`enum` (annotations such as
  `description` are fine there). The server logs the keywords it found
  (`response_format schema uses keywords the output constraint does not
  enforce`). The llguidance engines log their own list (above), and
  llama-server's schema converter may differ again. Validate the answer
  client-side if they matter.
- An `allOf` of two or more plain object schemas (each member holds only
  `type: "object"`, `properties`, `required`, a boolean
  `additionalProperties`, `title` and `description`), as zod's `.and()`
  emits, is merged into one object before the grammar is built: properties
  in member order, `required` joined, and `additionalProperties: false` if
  any member has it. A property defined in more than one member must have
  the same schema in each (compared as compact text); otherwise the `allOf`
  is not merged and is reported as not enforced. A node that holds
  such an `allOf` beside its own `properties`, `required`,
  `additionalProperties`, `$ref`/`anyOf`/`oneOf`/`const`/`enum` or a
  non-object `type` is left as is, and so is an `allOf` with any other
  member (a `$ref`, other keywords).
- Tool schemas (`tools[].function.parameters`) reach the chat template with
  their keys in the client's order, as transformers and vLLM render them.
  Earlier builds sorted the keys alphabetically, so a tool prompt can differ
  from an older node's render (and miss a prefix cache built by it).

Schema limits (each is a 400 `invalid response_format: …`):

| Limit | Value |
|---|---|
| Schema size (JSON without whitespace) | 64 KiB (65 536 bytes) |
| Nesting depth of objects and arrays | 64 |
| Nesting depth counted the way the grammar converter (`ov-genai`'s, xgrammar) walks the schema, through `$ref`: each subschema it descends into (a property, `items`, an `anyOf` case, a `$ref`'s target) is one level below its parent. It enters each `$ref` (by its text) once: a `$ref` already entered, such as one back into a recursive definition, stops there, but a schema reached by plain nesting is walked again each time it is reached | 64 |
| A coarse bound on the same depth whatever order a backend walks the schema in: each definition a `$ref` points to counts its own nesting, and definitions that refer to each other in a loop count together. A backstop to the limit above (a UI tree whose node is a union of 30 component types bounds at 126); over it the request gets `… may nest deeper than 512 levels through $ref` | 512 |
| Any one `minItems`, `maxItems`, `minLength`, `maxLength`, `minProperties`, `maxProperties`, or number in a `pattern` / `patternProperties` repetition quantifier (`{n}`, `{n,}`, `{n,m}`, `{,m}`) | 10 000 |
| Sum of all those counts in one schema; a `minX`/`maxX` pair on one object counts once, at the larger value | 200 000 |
| `patternProperties` keys on one object | 64 |
| `patternProperties` keys in one schema, all objects together, counted again each time the grammar builder visits or merges an object anew (each entry of a `type` list, each `allOf` that merges it) | 128 |
| Sum of the lengths of every `pattern` and `patternProperties` key in one schema | 16 KiB (16 384 bytes) |

Counts are taken everywhere in the schema, including inside `default` and
`examples` (a `$ref` can point there). Each count keyword, and
`minContains`/`maxContains`, must be written as a plain non-negative
integer (`0`, `10`; not `0.0`, `0e0`, `-0`, `1.5` or `"0"`), or the request
gets `… <keyword> must be a non-negative integer`. An object that repeats a key is
rejected too. A local `$ref` must be `#` or a plain JSON pointer into
objects, `#/key/key…`: no empty segment (`//`, a trailing `/`), no `~`
escape, no `%` encoding, no array index, and it must name an existing
schema (a JSON object or `true`/`false`, not a string, number or list such
as `#/required`); otherwise the request gets `unsupported $ref "<ref>"`. A `$ref`
that names a document outside the schema (`https://…`, `other.json`, a
`urn:` no `$id` in the schema has) gets `` $ref "<uri>" is not supported: it
names a document outside the schema … ``: no backend fetches one. The
check is the one llguidance makes first, so it refuses only what
llguidance can't build; under a `urn:` base `$id`, and inside a ref target
in `default`/`examples`, such a ref is left to the backend, which on the
llguidance engines answers `invalid configuration: response_format
schema: Resource '<uri>' is not present in a registry …`. A ref the
schema itself answers is fine, also through a `$id` (a root `$id`, a
TypeBox-style `$id` in `$defs` named by `"$ref": "T"`, or the fragment
`$id`s schema generators put on every subschema), as are properties named
`id` or `$id`. That last applies to the llguidance engines. On `ov-genai`
(xgrammar resolves only `#` and `#/...`, and makes any other ref it
reaches match any JSON) every such ref the grammar reaches (a TypeBox
`"$ref": "T"`, a `urn:`, an `https://` document, also inside a ref target
in `default`) gets `` $ref "<ref>" is not supported by this server's engine
(<engine>): it enforces only references inside the schema written as `#`
or `#/...` pointers … ``. On `sycl-llama` llama-server is never sent a
`$ref` the client wrote: the engine replaces every `#` / `#/...` ref in
the schema (and in each tool's `parameters`) with a copy of its target
before forwarding (llama.cpp's own ref handling can give a ref the wrong
shape without failing). A schema or tool without a `$ref` is sent exactly
as written. Pointers are read as RFC 6901 (`~1` is `/`, `~0` is `~`). A
`$ref`'s `description` and `title` are carried onto the copy and its other
siblings dropped, as llama-server ignores them; `default` and `examples`
are kept, unless a `$ref` sits inside one (then that keyword is left out);
`const` and `enum` are sent as written. Every `$ref` left in what is sent
(in `const` or `enum`, or carried in a `description` or `title`) must be one
llama-server's ref scan accepts (a `#/...` pointer that resolves, or an
`https://` document without a fragment), else `` $ref "<ref>" at "<where>"
is not supported: llama-server fails on it `` (`<where>` a JSON pointer
into the schema as sent). A recursive
schema (a linked list, a tree, `"$ref": "#"`) is sent with each schema on
the cycle copied to a reserved `$defs` entry (`__cascadia_r<N>x`,
`__cascadia_t<i>r<N>x` for tool `i`) that the refs on the cycle point to
(the root stays the root); names starting `__cascadia_` are reserved, so a
`$defs` key or `$ref` using them gets `… names starting __cascadia_ are
reserved`. Hoisting needs llama-server's chat endpoint, so with
`--llama-args "--no-jinja"` a recursive schema gets
`` $ref "<ref>" makes the schema recursive, which this engine can't enforce
when llama-server runs without --jinja ``. (The engine reads only `--jinja`
and `--no-jinja`, also spelled `--no_jinja`, in `--llama-args`, the last
one winning; a `--no-jinja` given as another flag's value also counts,
which can refuse more than needed but never less.) A ref that
can't be inlined (a `$ref` that is not `#` or `#/...`, or one that names
nothing) gets `` $ref "<ref>" is not supported: only a reference inside the
schema (`#` or `#/...`) to an existing schema is `` (for a tool, `invalid
tools: tools[<i>].function.parameters: …`). Only what llama-server
receives is checked: no `response_format` while tools are callable, and
only the tools `tool_choice` leaves (none for `"none"`, the named one for
a named choice). One budget covers all of it: over 1 MiB once inlined (definitions
reused many times over) gets `` … with every $ref replaced by its target
are over 1048576 bytes ``, and nesting deeper than 256 levels `` … nests
deeper than 256 levels ``. The 64 KiB cap applies to what the client
sends; the llama-server cost guard (below) applies to the inlined
`response_format` schema and to each tool's parameters it receives. An unknown `$schema`
(`http://json-schema.org/schema#`, an OpenAPI dialect) is not a ref:
`ov-genai` and `sycl-llama` ignore it, and llguidance refuses to build
the schema (`invalid configuration: response_format schema: Resource
'<uri>' is not present in a registry …`). A schema whose walk
would take the converter too long gets `… schema is too complex`.

Errors (`{"error": "<message>"}`; on a stream, after the SSE stream has
started, a failure is an error event `{"object":"error","error":{"message":
"<message>","type":"<type>"}}` whose type is `invalid_request_error` for
the 400s below and `engine_error` for anything else):

| Status | Message | Cause |
|---|---|---|
| 400 | `invalid response_format: <reason>` | Malformed field (unknown `type`, non-object schema, `false` schema, a repeated key) or a schema over the limits above. Every engine. |
| 400 | `invalid response_format: response_format.json_schema.schema: minLength (5) is greater than maxLength (2) at #/properties/a; no value can satisfy both` | A bound pair no value meets (`minLength`/`maxLength`, `minItems`/`maxItems`, `minProperties`/`maxProperties`, `minimum`/`maximum`, the exclusive forms), or an integer-only range with no integer in it (`no integer meets …`), wherever it sits in the schema, with the JSON pointer of the object that holds it; also split across an `allOf` (its holder, its members and their own `allOf` members, and a property or `items` several of them define), then naming the pointer of each bound (`minLength (5) at #/allOf/0 is greater than maxLength (2) at #/allOf/1; …`). A member that is a `$ref` is left out. Values under `enum`, `const`, `default` and `examples` are data and are not checked. Every engine. |
| 400 | `invalid configuration: response_format json_schema: <reason>` | `ov-genai`: the backend could not build a constraint from the schema. |
| 400 | `response_format json_schema: <reason>` | `ov-genai` without `--cb`: generation failed for a request with a schema, and the backend's message `<reason>` names the grammar or JSON schema. Other generation failures stay 503. On a streamed request this message arrives as the stream's error event, with `"type":"invalid_request_error"`. |
| 400 | `invalid configuration: structured output is not available for this model: <reason>` | `ov-genai`: the startup check found structured output does not work for the loaded model or runtime. Also `ov-genai` once a pipeline has let 64 distinct schemas through to GenAI, for a new schema: `structured output is not available for this model: this worker has reached its limit of 64 distinct response_format schemas per pipeline (GenAI keeps each compiled grammar for the pipeline's life); schemas it has already served keep working, and <when>`. Past the floor between rebuilds the worker rebuilds the pipeline to clear the limit, and serves the request on the new one when it can; otherwise `<when>` says how that went: `the worker has just rebuilt its pipeline to clear the limit, and the requests waiting their turn filled it again; retry the request once they finish`, `the worker rebuilds its pipeline to clear the limit once the requests in flight finish (within 15 s); retry the request after that` (`--cb` with requests in flight), or, while the 10-minute floor between rebuilds holds, `the worker rebuilds its pipeline to clear the limit at most once every 10 minutes; the next rebuild can start in <n> s, on the first new schema after that` (see the engine table). The llguidance engines: the grammar could not be built from the model's tokenizer (logged at startup). |
| 503 | `temporarily unavailable: this worker is rebuilding its structured-output pipeline; retry the request shortly` | `ov-genai --cb`: the worker is waiting (at most 15 s) for the requests in flight to finish before rebuilding its pipeline to clear a full schema budget (above). Retry. `/health` stays ready. |
| 503 | `this request was still running when the worker began rebuilding its structured-output pipeline; retry the request shortly` | `ov-genai --cb`: the request was still running 15 s after the worker stopped admitting work to rebuild its pipeline. Retry. `/health` stays ready. |
| 500 | `backend error: ov-genai pipeline rebuild failed: <reason>; this worker serves no requests until it is restarted` | `ov-genai`: rebuilding the pipeline to clear a full schema budget failed. Every new request gets this until the worker is restarted, and `/health` reports `degraded`; a request that was waiting its turn (without `--cb`) gets 503 with the text after `backend error: `. |
| 400 | `invalid configuration: response_format schema: <reason>` | The llguidance engines: llguidance could not build a grammar from the schema; The message is cut to one line of at most 300 characters. |
| 5xx | `… a later pipeline stage runs a build without structured-output support; upgrade every stage` | Multi-stage `ov-runtime`, `gemma4`, `qwen35` or `sparse-moe`: a later stage runs an older build (see above). |
| 5xx | `structured output: the downstream pipeline link failed before the request started; retry it` | Multi-stage `sparse-moe`: the link to the next stage was found dead (or out of step) when a JSON request was about to start. Nothing is latched; on a multi-stream pipeline the head re-dials on its next step. Retry. |
| 5xx | `structured output: a pipeline stage lost its token-mask state; retry the request` | Multi-stage `ov-runtime`, `gemma4`, `qwen35` or `sparse-moe`: a stage got a repeated-mask marker without the mask it repeats. Not expected in normal operation; only that request fails, and the next one sends full masks again. Retry. On a multi-stream `sparse-moe` pipeline, other JSON streams whose markers were already on their way when the last stage missed fail the same way, and a miss at a middle stage closes that link and fails every stream on it (neither is expected to happen). |
| 400 | `invalid response_format: response_format schema is too complex to compile` | The llguidance engines: building the schema's grammar took longer than 2 seconds in the server's check before the request queues, or the grammar it built failed on the first token of an answer (a parser limit). |
| 503 | `busy compiling response_format schemas; retry shortly` | The llguidance engines: the server's schema check had no free slot (2 builds at once) within half a second. Retry. |
| 5xx | `… grammar: no token of the … is allowed` | The llguidance engines: decoding hit a state the grammar allows no token from. The request ends with this error, never with unconstrained output. |
| 5xx | `… structured output generation failed` | The llguidance engines: the grammar failed while generating, or a resumed request's earlier output does not match the grammar. The grammar's own reason is logged, not sent. The request ends with this error, never with unconstrained output. |
| 400 | `invalid configuration: response_format json_schema: llama-server would take too long to build a grammar for this schema: …` | `sycl-llama`: llama-server's schema converter runs inside the request with no time limit, and some schemas inside the limits above take it seconds to minutes: an object with many properties not in `required` (cubic in their count: 2000 take about 5 s, 6000 minutes), long property names over many properties, many `allOf` merges of the same properties, a long `type` list, many distinct `$ref`s, long `enum`/`const` values merged many times. The bridge estimates the converter's time from the schema (each object's optional count cubed, each rule name's bytes, each JSON value it copies or prints) and refuses the schema before sending it when the estimate is over about 0.13 s of converter time; one object of about 430 optional properties with short names is at the limit. The message names the object with the most optional properties. A schema the converter would loop on (an `allOf` member whose `$ref`s form a cycle) gets `… llama-server can't build a grammar for this schema: …`, and one that nests deeper than 256 visits `… this schema nests deeper than 256 levels …`. |
| 400 | `invalid configuration: response_format json_schema: xgrammar would take too long to compile a grammar for this schema: …` | `ov-genai`: OpenVINO GenAI compiles the schema with xgrammar inside the request with no time limit, and a worker stuck in a long compile answers no other request. Some schemas inside the limits above take it minutes or never finish (times with a 151,665-token vocabulary on one CPU thread): an object's run of properties not in `required` (quadratic in its length: 300 take 1.7 s, 1000 take 18 s, 6000 over 15 minutes); `minProperties`/`maxProperties` over optional properties (cubic: 100 take 4.8 s, 200 take 45 s); `type` lists that name one type twice, nested (`["object","object"]` 12 deep takes 1.9 s there and 12.8 s in OpenVINO GenAI on an Intel GPU, 20 deep never finishes); many distinct string length bounds (xgrammar unrolls `minLength`/`maxLength` one character at a time up to 128: `maxLength` 32 takes 0.5 s, 64 1.1 s, 255 or more 3.5 s, once per distinct pair of bounds) or long `pattern` repetitions; many `number` ranges with fractional bounds in one object. The engine estimates the compile time from the schema and refuses it before compiling when the estimate is over about 10 s; a first compile of a few seconds (cached per schema after) is allowed, so three strings with `maxLength` 100, 254 and 1000 (9.6 s) pass. The message names the largest part of the estimate. A schema xgrammar would reject gets `… xgrammar would reject this schema: <what>` before it is compiled, and one the estimate can't cover (schemas nested more than 256 deep, a count past 2^31) gets `… xgrammar's compile time for this schema can't be estimated (<what>), so it is refused`. |
| 400 | `llama-server rejected the request: <reason>` | `sycl-llama`: llama-server answered 4xx (other than 408/429), for example a schema its grammar converter rejects; `<reason>` is llama-server's `error.message`, or the start of its reply. On a streamed request the SSE stream has already started, so this message arrives as the stream's error event instead, with `"type":"invalid_request_error"`. |

On `ov-genai --cb`, a schema whose compile fails while a batch runs fails that
batch step. Every request in a failed step gets 503 (on a stream, the error
event). When the failure is blamed on the step's new schemas (below: while
fewer than 8 such failures have run since the last successful step, or when
the health check after them succeeds), the messages depend on the request.
When the step held one new schema, its requests get `cb step failed: <backend
reason>` if the backend's reason reads as a grammar or schema error (the
reason is cut to one line without source paths or timestamps, at most 300
characters), and the next
message otherwise. When it held several, their requests get `cb step failed:
a schema in this batch may have caused it; retry the request`, since the
reason may come from any of them. Every other request gets `cb step failed:
the batch this request ran in failed; retry the request`; so does every
request in a step not blamed on a schema, including one whose health check
failed. When the failed step held exactly one new schema-constrained
request (or several with the same schema), that schema is then refused for 10
minutes with `invalid configuration: response_format json_schema: this schema
failed to compile on this server; retry later or change the schema`. When it
held several different new schemas, none is refused at once, but a schema in
two such failed steps within 10 minutes is, unless a request with it produced
output in between. A schema that produced output, for a request still in the
batch or in the last 10 minutes, compiled, so it is never blamed (and output
lifts a refusal of it); the worker remembers up to 256 such schemas,
forgetting the oldest first. Steps failed by resource
exhaustion, and `json_object`, are never blamed on a schema. A failed step
that held a new request with a schema of its own (one that had not produced
output yet) is not counted toward the worker's failure limit, whatever else
was in the batch. Client requests never decide whether the device is
healthy: after 8 such failures since the last successful step the worker
runs a short internal health check (a plain request of a few tokens, alone
on the pipeline). If it succeeds, the schemas keep the blame and the count
starts over; if it fails, the step counts. A failed step with no such
request counts as usual. After 3 counted failures since the last successful
step the worker refuses every new request (`cb pipeline unusable: …`). The
health check runs again when a request arrives at least 30 seconds after the
last one; its first success clears the limit and that request is admitted.
Schemas refused during a run of failures (since the last successful step)
in which the health check failed, or that ended with the limit clearing, are
released when the run ends; in any other run they stay refused.

## Tool calling (`tools`, `tool_choice`)

`/v1/chat/completions` accepts OpenAI's `tools`, `tool_choice` and
`parallel_tool_calls`. Tools reach the model through its chat template; the
answer is read back into `message.tool_calls` (streamed: one `tool_calls`
delta before the final chunk).

`tool_choice`:

| Value | What the model may do |
|---|---|
| absent or `"auto"` | Call any tool or answer in text. |
| `"none"` | Answer in text; the tools may still be rendered, and nothing is read as a call. |
| `"required"` | Call at least one tool. |
| `{"type":"function","function":{"name":N}}` | Call function `N`. |
| `{"type":"custom","custom":{"name":N}}` | Call custom tool `N` (see below). |
| `{"type":"allowed_tools","allowed_tools":{"mode":"auto"\|"required","tools":[…]}}` | Only the listed tools (each `{"type":"function","function":{"name":N}}` or `{"type":"custom","custom":{"name":N}}`) exist for this turn: the prompt, the constraint and the answer use that subset. `mode` then acts as `"auto"` or `"required"`. A call to an unlisted tool is dropped from the answer. |

A malformed `tool_choice` is a 400 (`{"error": "<message>"}`) before the
request reaches the engine: an unknown string or `type`, a named choice
without a name, a name not in `tools` or of the other kind (a function
choice naming a custom tool, or the reverse), and for `allowed_tools` a
`mode` other than `"auto"`/`"required"`, an empty or missing `tools` list,
or an entry that is malformed or not in `tools`. A custom tool without a
name, or whose name another tool already uses, is a 400 too. As before,
`"required"` with no `tools` is accepted and answered as text.

### Enforcement

A turn that must call a tool (`"required"`, a named function or custom
tool, `allowed_tools` with mode `"required"`) is enforced by constrained
decoding, the same machinery as `response_format`: the output is a JSON
array of `{"name": …, "parameters": …}` objects, each `name` one of the
callable tools and `parameters` that tool's JSON Schema (a named choice: one
item, that tool). It does not depend on the model's own tool-call syntax.
This runs on every engine with structured output: `ov-genai`, `ov-runtime`,
`ov-dist-spec`, `gemma4`, `qwen35`, `sparse-moe`, and `sycl-llama` (sent to
llama-server as a `json_schema` `response_format`; llama-server's own
`tool_choice` handling does not force a call). Each tool's `$defs` /
`definitions` are kept (moved to the root of the combined schema, and its
`$ref`s rewritten to match).

A tool whose `parameters` hold a bound pair no value can meet, such as
`minLength` above `maxLength`, `minItems` above `maxItems`, or an integer
range holding no integer, is accepted with that pair dropped: the API
removes exactly those two keywords at that place, logs a warning naming the
tool and the pair (the first time the tool's name is seen; later at debug),
and uses the tool without them everywhere (the prompt, the tools sent to
the engine, and the constraint), in every `tool_choice` mode. The rest of
the call shape is still enforced; a forced call stays forced. When the
constraint is sent over such a tool and the engine applies it,
`cascadia_constraint_unenforced_total{reason="schema_rejected"}` counts it
once, when the answer is in (not when the engine dropped the constraint and
counted its own reason, and not for a request that ends on an error or is
cancelled before its final chunk). Left in, no call could match the schema and a backend that
builds its own grammar from the tools (llama-server on `sycl-llama`) would
decode until the request is killed. A `response_format` schema with such a
pair is still a 400.

- Thinking is turned off for an enforced turn: the constraint starts at the
  first token. On `ov-genai` it is also off for a forced turn that could not
  be constrained. On `sycl-llama` an enforced turn also sets
  `reasoning_budget_tokens: 0`, as a JSON request with thinking off does,
  and keeps it when the request is sent again without the constraint.
- `finish_reason` is `"stop"` for a named function or custom tool (as OpenAI
  reports it) and `"tool_calls"` for every other answer with calls. A
  forced answer cut by `max_tokens` before it parses is returned as content
  with `finish_reason: "length"`.
- `parallel_tool_calls: false` gives at most one call: enforced in the
  constraint when one is sent, otherwise the first parsed call is kept
  (streamed and not). Only an explicit `false` counts; a non-boolean value is
  ignored.
- `response_format` `json_object` / `json_schema` together with callable
  tools under `"auto"`: the answer is either a tool call or JSON matching the
  format. A JSON answer whose root is itself an array of
  `{"name", "parameters"}` objects naming declared tools is read as tool
  calls.

**Best effort.** The tool constraint never turns a request into an error.
Where it can't be applied, the request is still answered (200) without it:
the answer is read as the model wrote it (the array form first, then the
model's own tool-call syntax), the server logs `output constraint not
enforced; answering unconstrained` (or `tool-call output constraint not
enforced` when the API could not build it) with the reason, and
`cascadia_constraint_unenforced_total{reason}` counts it
([METRICS.md](METRICS.md)). When the engine drops the constraint, its final
chunk says so and the answer is read exactly as one sent without it: under
`"auto"` beside a JSON `response_format`, the model's own bare JSON call
(`{"name": …, "parameters": …}`) is then a tool call, as before the
constraint existed, streamed and not. A reason that holds for every request
on the server (`npu`, `draft_model`, `genai_version`, `probe_failed`,
`no_tokenizer`, `stage_unsupported`) is logged at warn the first time in
the process and at debug after; the counter counts every request. Reasons:

| `reason` | When |
|---|---|
| `npu` | `ov-genai` on NPU (`--device NPU`, or a `HETERO:`/`AUTO:` list with NPU). |
| `draft_model` | `ov-genai` with `--draft-model`. |
| `genai_version` | `ov-genai` on OpenVINO GenAI older than 2026.4. |
| `probe_failed` | `ov-genai`: the startup structured-output check failed for this model. |
| `schema_rejected` | The backend refused the combined schema (`ov-genai`, the llguidance engines, or llama-server on `sycl-llama`, which is then sent the request again once without it), or `ov-genai --cb` refuses the schema after a failed compile, or (`ov-genai`) xgrammar would reject it. Also counted, once the answer is in, when the constraint was applied without a tool's bound pair no value can meet (see above): the rest is enforced, that pair is not. A request that ends on an error or is cancelled before its final chunk is not counted. |
| `too_complex` | The combined schema is over the [schema limits](#structured-output-response_format), or took longer than the 2 s build budget (llguidance engines). A schema over the build budget is remembered for 10 minutes (up to 256), so a later turn resending it runs unconstrained at once instead of waiting out the budget again; after that it is built again, in case the machine was only busy. On `ov-genai`, also a schema xgrammar would take too long to compile (the estimate behind that engine's 400 for `response_format`, made again for each request), and a new schema once the pipeline has let its 64 distinct schemas through to GenAI while the 10-minute floor between pipeline rebuilds holds (where a strict schema gets the budget-full 400); a dropped schema is not counted against those 64, nor as a refusal (it raises no budget-full warning). Past the floor, a forced tool's new schema rebuilds the pipeline as a strict one would and then runs constrained on it; with `--cb` and requests in flight it starts the drain before the rebuild and gets that drain's retryable 503, like any request during it. On the llguidance engines, also a grammar that fails while the answer is generated (a token or a mask runs past the parser's limits, such as an object with hundreds of optional properties): the constraint is dropped there and the rest of the answer is generated unconstrained, keeping what was already produced. On `sycl-llama`, also a constraint whose grammar cost, added to the tools' own (see below), is over llama.cpp's grammar cost limit. |
| `busy` | The llguidance engines' schema check had no free slot. A tool constraint does not wait for one, so it never delays a `response_format` request queued for a slot. |
| `unsupported_ref` | A tool schema (or the `response_format` schema beside it) has a `$ref` that can't be kept once nested: anything other than `#` or `#/$defs/…` / `#/definitions/…` naming its own definitions, or a `$ref` beside a nested `$id`. |
| `no_tokenizer` | A llguidance engine could not build its grammar factory from the model's tokenizer. |
| `stage_unsupported` | A later pipeline stage can't apply token masks (see below). |

**`sycl-llama`: tools too costly for llama-server are a 400.** llama-server
builds a grammar from every forwarded tool's `parameters` itself (a lazy
tool-call grammar), under any `tool_choice` including `"auto"`, with the
same JSON Schema converter it uses for `response_format`, inside the
request and with no time limit. That converter's work on an object is cubic
in its count of properties not in `required`, so a tool with a long run of
optional properties would stall llama-server for seconds to minutes. The
bridge estimates the converter's time from the tools it forwards (all of
them; only the named one under a named choice; none under `"none"`, which
sends no tools), with the same estimate and limit as a `response_format`
schema (cost 134217728, 512³, about 0.13 s of converter time; one object of
about 430 optional properties with short names is at the limit), sums it
over the tools and, when the sum is over the limit, refuses the request
before sending it: 400 `{"error": "invalid configuration: tools:
llama-server would take too long to build a grammar from the tools'
parameters: their cost is over the limit of 134217728; the costliest is
tools[<i>] (<name>)…"}`. When that one tool is over the limit alone, the
message goes on with the reason for its schema (`: llama-server would take
too long to build a grammar for this schema: …`, naming the object with the
most optional properties). This is a refusal of the tools, not of the
constraint, so it is not best effort: the request is not answered without
the tools. Split such a tool, or mark more of its properties `required`.

Multi-stage pipelines: enforcement needs every stage to run a build with
structured-output support, so upgrade every stage before relying on it.
With an older later stage, the first enforced request on a link waits for
the first stage's mask check (about 10 s on `ov-runtime`, `gemma4` and
`sparse-moe`; `qwen35` knows from its handshake) and then runs
unconstrained; later enforced requests on that link run unconstrained at
once (`stage_unsupported`) until the check runs again. A link that fails
outright, or a masked step that fails mid-generation, still fails that
request (5xx) as for a JSON request.

### Custom tools

`{"type":"custom","custom":{"name":N,"description"?:D,"format"?:F}}` tools
are accepted. The model sees each one as a function with a single required
string parameter, `input`, described by `D`. When `F` is a grammar
(`{"type":"grammar","grammar":{"syntax","definition"}}`), its syntax and
definition are added to that description so the model can follow it; the
grammar is **not enforced** inside `input`.

A call to a custom tool comes back as
`{"id","type":"custom","custom":{"name":N,"input":"…"}}`, in the message and
streamed alike (the streamed delta adds `index`; OpenAI's chunk schema has
no custom shape). `input` is the call's `input` string, else its only string
argument, else the raw arguments text. An assistant message replaying a
custom call in `tool_calls` is accepted and rendered to the template as the
function form with arguments `{"input": …}`.

### Gemma 4

Gemma 4's own tool-call syntax
(`<|tool_call>call:NAME{key:<|"|>text<|"|>,…}<tool_call|>`) is parsed into
`tool_calls`. `ov-runtime`, `gemma4`, `ov-dist-spec` and `ov-genai` keep
those three marker tokens when decoding a model whose vocabulary has them
(`ov-genai` in continuous batching and for stateful and VLM-layout exports
such as `OpenVINO/gemma-4-E2B-it-int4-ov`, checked end to end on Linux), and
`sycl-llama` gets Gemma 4 calls back from llama-server already parsed. The
markers never appear in `content`, with or without tools: the API removes
them when the chat template writes Gemma 4 calls or a `tokenizer.json` the
engines read (`tokenizer/tokenizer.json` or `tokenizer.json` in the model
directory) has `<|tool_call>`.

### Unchanged for existing requests

A request without `tool_choice`, or with `"auto"` or `"none"`, and without
`response_format`, is decoded and read as before: no constraint, and an
answer with calls reports `finish_reason: "tool_calls"` as before. What is
new: `"required"` and named choices are now enforced; `parallel_tool_calls:
false` now limits the answer to one call; `response_format` beside callable
tools is now enforced as "a call or the JSON"; custom tools and
`allowed_tools` (previously a 422 / 400) are accepted; Gemma 4 calls are
parsed on the engines above. A tool whose `parameters` hold a bound pair
no value can meet is sent without that pair in every mode, `"auto"` and
`"none"` included (see Enforcement). Under a named choice, a call to
another tool is dropped from the answer; when no call is left, the text is
the content.
