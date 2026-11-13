# Diffusion stage qualification

## Local OpenVINO comparison — 2026-11-13

The fixed 512×512 SDXL profile was exported and run on an idle Intel Tiber
Arrow Lake machine (`cascadia-tate-02`, PDX88-HP0756, Core Ultra 7 265T,
32 GB RAM, Windows). Existing fleet/monitoring services were left running.
No tests used alpha, beta, charlie, delta, tate-08, or tate-07.

Source OpenVINO SDXL INT8 revision:
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

## Portable gates

The initial upstream workspace gate passed 1,190 tests across 134 suites
(six existing ignored tests), workspace Clippy, and formatting. The new crate
covers artifact tampering and path traversal, binary truncation/length/NaN
rejection, owner/boot/sequence fencing, cancellation and expiry during active
execution, drain, absolute lease lifetime, and uncertain-execution quarantine.
Python tests share the Rust golden manifest/frame and exercise actual loopback
HTTP authentication, framing, and retaining capacity after a client disconnect.

Multi-host placement, network fault injection, 1024-pixel operation, sustained
load, and a model exceeding one host's usable memory require additional
qualification through the enterprise integration. This local comparison alone
must not be advertised as distributed production qualification, a speedup, or
64 GB machine coverage.
