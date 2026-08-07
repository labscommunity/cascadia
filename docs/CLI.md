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
| `--listen <LISTEN>` | `:9100` | Bind address for the upstream-receiving socket. |
| `--next <NEXT>` | — | Downstream peer `host:port`. Required for every stage but the last. |
| `--api <API>` | — | API bind address. **Rank 0 only** — other ranks enter the relay loop and never bind it. Passing it there logs a warning and is otherwise ignored. |
| `--engine <ENGINE>` | `mock` | Inference engine. `mock` runs without OpenVINO. |
| `--device <DEVICE>` | `CPU` | OpenVINO device target — see [below](#device-forms). |

Rank 0 serves the API and holds the first layers; the last rank produces tokens
and returns them up the chain. Start the *downstream* worker first — engines
wait 60 s for a downstream peer, then give up.

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

**`--ov-cache-dir` is on by default and matters.** For `ov-genai`, `ov-runtime`,
`gemma4` and `sparse-moe`, leaving it unset defaults to
`<user-cache>/cascadia/ov-cache` (`~/.cache` on Linux, `~/Library/Caches` on
macOS, `%LOCALAPPDATA%` on Windows). This turns a ~20 s cold GPU compile into a
~1 s warm load on every later run — the single biggest latency win on the
`ov-genai` path. Pass `--ov-cache-dir ""` to disable.

Two engines don't get that default: `ov-dist-spec` uses the flag verbatim (so it
is off unless you pass a path), and `qwen36-moe` ignores it entirely.

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
| `--packed-prefix <N>` | `0` | Reserve N KV slots as a read-only shared prefix every packed slot may attend to — prefix caching without paging. Costs per-slot context. Requires `--packed-slots`. |

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
| `--model <MODEL>` | *required* | HF repo id, a local dir with safetensors + `config.json`, or — for the Gemma-4 / Qwen3.6 surgery paths — an already-exported OpenVINO IR dir. |
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
| `gemma4` | `gemma4_cached_v1` shards | Per-layer-type asymmetric attention, KV-sharing, baked softcap. |
| `sparse-moe` | `manifest.json` + expert tree | Top-k expert routing through an AVX-512 int4 GEMM. CPU-targeted; single-stage or pipeline-parallel (`--total >= 2`). |
| `qwen36-moe` | Qwen3.6 surgery output | Greedy-only, batch=1. CPU-targeted decode. |

Per-engine deep dives: [engines/](engines/). Per-family export notes:
[architectures/](architectures/).
