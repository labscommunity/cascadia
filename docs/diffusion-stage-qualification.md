# Diffusion stage qualification

## Local OpenVINO comparison — 2026-11-13

The fixed 512×512 SDXL profile was exported and run on an idle Intel Tiber
Arrow Lake machine (`cascadia-tate-02`, PDX88-HP0756, Core Ultra 7 265T,
32 GB RAM, Windows). Existing fleet/monitoring services were left running.
No tests used alpha, beta, charlie, delta, tate-08, or tate-07.

Source: [OpenVINO SDXL INT8](https://huggingface.co/OpenVINO/stable-diffusion-xl-base-1.0-int8-ov), revision:
`d54e5b43fcb6b4b358f3a134269115d3fcb3eb25`.
Export manifest:
`c6dae31741b7816562a8bf65d27888814ab9a393321681cc606afda58b328243`.
The manifest and selected PNGs/reports are in
[`evidence/diffusion-20261113`](evidence/diffusion-20261113).

The test used the committed `tools/diffusion/tests/hardware_sdxl.py`: a black
and white manga shrine prompt, seed 4242, 20 steps, guidance 7.5, GPU, explicit
FP32 accumulation, one OpenVINO stream, and three consecutive runs per mode.
Each mode loaded only its selected model components in a fresh process.
The partitioned process loaded both U-Net halves, never the full U-Net.

| Measurement | Full denoiser | Two local partitions |
|---|---:|---:|
| Cold generation | 69.989 s | 73.346 s |
| Warm generation | 68.891 / 69.498 s | 69.638 / 69.991 s |
| Peak process working set | 13,582,823,424 B | 10,478,194,688 B |
| Peak committed process bytes | 14,192,881,664 B | 11,081,297,920 B |
| Steady committed bytes | about 9.13 GB | about 9.88 GB |
| PNG bytes | 366,576 | 367,150 |

All three full-mode images had SHA-256
`8dc0024aaca6e425e20ec7bc735beb228f09d39f6ad46977a36b966947ae3785`.
All three partition-mode images had SHA-256
`c0cb7f93fad147bfb5bd8d3e1328ae09c1b338848594087cf2784ba1289ba680`.
Cold and warm output matched within each mode. Between modes, pixel mean
absolute error was **0.153879/255**, RMS error **0.596760/255**, and maximum
channel difference **37**. Visual review found the same composition and
linework without a visible structural change. This is one prompt/seed;
it does not establish parity across a manga evaluation set.

The first partition contains 1,282,357,988 weight bytes and the second
1,294,340,856. The full U-Net contains 2,576,698,520. Splitting reduced compile
peak memory here, but loading both halves locally did not reduce steady
process memory or speed up generation. Per-stage call timings and transferred
tensor sizes are recorded in each report.

Runtime versions: OpenVINO 2026.4.1, NumPy 2.4.6, Optimum Intel 2.2.0,
Diffusers 0.39.0, Transformers 5.5.4, Torch 2.15.0, Pillow 13.0.0.
The qualified environment's dependency constraints are checked in separately.
The serving runtime rejects mismatched critical dependency versions.

## Local 1024×1024 comparison

The same host, checkpoint, prompt, seed, and 20-step settings were tested at
1024×1024. The export manifest is
`c76fec7fab018d17e59998f237006b189026bacf3815a2db436d7bcd8824e3fb`.
The full graph used upstream `1cfc23be`; the partitioned run used `a634089e`,
which adds rejection of unsupported pipeline metadata without changing inference.
All three images within each mode were byte-identical.

| Measurement | Full denoiser | Two local partitions |
|---|---:|---:|
| First generation | 355.309 s | 340.778 s |
| Repeated generation | 345.891 / 341.267 s | 339.894 / 341.055 s |
| Peak process working set | 17,577,885,696 B | 19,237,355,520 B |
| Peak committed process bytes | 18,197,520,384 B | 20,544,475,136 B |
| PNG bytes | 2,328,240 | 2,328,629 |

Full-mode PNG SHA-256:
`8631646be219c967b61bb4ae82bf30adf9270cf4c6dac5430f05412e96e62d25`.
Partition-mode PNG SHA-256:
`080d184d8282f9299cbb021b4709f058db9eb7226c502f655771772cb2b7a0de`.
Pixel mean absolute error was **0.033209/255**, RMS **0.194450/255**, and maximum
channel difference **12**. Visual review found the same composition and linework.
These timings do not establish a speedup: there are only three runs per mode,
and the first includes warm-up effects. Both partitions were resident on one
host, and their combined memory peak was higher than the full graph at this size.

Reports and selected PNGs are under `full-1024` and `layers-1024` in the evidence
folder. Compilation caches persist between processes; a first generation means
the first request in that process, not necessarily an empty compilation cache.
Process working set, committed bytes, and shared iGPU memory describe overlapping
resources and must not be added together as independent physical RAM.

The documented `python -m cascadia_diffusion generate` command also passed on
the same GPU with the reviewed `a634089e` package: 512×512, component mode,
four steps, seed 120, and an adult manga portrait prompt. The decoded PNG had
the requested dimensions and nonzero image variance; its digest is recorded
in `cli-512.json`.

## Portable gates

The initial upstream workspace gate passed 1,190 tests across 134 suites
(six existing ignored tests), workspace Clippy, and formatting. The new crate
covers artifact tampering and path traversal, binary truncation/length/NaN
rejection, owner/boot/sequence fencing, cancellation and expiry during active
execution, drain, absolute lease lifetime, and uncertain-execution quarantine.
Python tests share the Rust golden manifest/frame and exercise actual loopback
HTTP authentication, framing, and retaining capacity after a client disconnect.

## Distributed release qualification — 2026-11-13

Report date uses America/Chicago; evidence timestamps use UTC.

The same SDXL artifacts were served through the enterprise admitted QUIC mesh on two
idle older Tiber Arrow Lake PCs, each Core Ultra 7 265T / 32 GB RAM, Windows, Intel GPU
driver `32.0.101.8132`. The final production-feature release binary used enterprise
`c3da10f98b6c84dd12f1478b651a7cc989a24e6b` and upstream
`450520bbc7c2ee69c174ffbbdc32b80c261a95ea`. Its SHA-256 was
`3e39f0b7e235bb69f026324ac9acfdd4821d23be5a072de9aaacaf353e5397fd`. The packaged Python
source was verified against that upstream revision.

| Placement | Resolution | First 20-step image | Result |
|---|---|---:|---|
| layers | 512x512 | 484.360 s | Exact local reference PNG |
| layers | 1024x1024 | 1929.589 s | Exact local reference PNG |
| components | 512x512 | 76.174 s | Exact local reference PNG |

The complete 512 layer suite also passed two seeded repeats, two additional manga
prompts, capacity exclusion, disconnected-client ownership, signed operator
drain/restart, worker boot rediscovery, worker-loss quarantine, explicit recovery,
stage-placement checks, and owned-process cleanup. The 1024 layer run was a single-image
smoke test. Component placement also accepted a maximum 64 KiB public request and
rejected a request one byte larger without quarantining healthy workers.

A legal public request can grow when normalized and wrapped for the internal sidecar.
The internal `/generate` bound is 256 KiB while the public limit remains 64 KiB; an
actual loopback HTTP regression verifies this boundary. Seven portable Python
contract/HTTP tests and the upstream workspace CI passed.

The two hosts used a Tailscale relay. A layer request sends about 55 MB per denoising
step at 512 pixels and about 217 MB at 1024 pixels. These results establish the tested
placement and lifecycle behavior, not a network speedup. Whole-image replicas remain the
preferred throughput strategy when the pipeline fits. No tested model exceeded one
host's usable memory.

Sanitized per-run summaries and first-image metrics are under
[`evidence/diffusion-20261113/distributed`](evidence/diffusion-20261113/distributed).
The complete [enterprise qualification
report](https://github.com/labscommunity/cascadia-enterprise/blob/7c787b16271374a77dd6f68624ba558eea3e9fa7/docs/operator/diffusion-stage-qualification.md)
records images, memory, failure injection, and two earlier interrupted attempts caused
by a scheduled demo process kill. Final tests used a hash-identical executable with an
isolated filename, leaving that host-management job unchanged.

This is a bounded SDXL deployment profile. Sustained multi-user load, 64 GB hardware,
larger-than-one-host capacity, other diffusion architectures, CFG parallelism, and
tensor/sequence parallelism remain separate qualification work. The prompt samples do
not establish specialized manga quality or panel consistency.
