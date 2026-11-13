# Cascadia diffusion runtime

This package exports, verifies, and executes fixed-shape SDXL OpenVINO stages.
The companion `cascadia-diffusion` Rust crate owns admission leases, execution
lifetime, and the bounded binary tensor contract. Fleet authentication, peer
placement, and the public image API live in `labscommunity/cascadia-enterprise`.

The initial profile supports one image at a time, 512 or 1024 pixels square,
1–50 inference steps, and classifier-free guidance in `(1, 30]`. A 256-pixel
export is available for diagnostics. The scheduler, tokenizer, model configs,
and weights are pinned together by a SHA-256 manifest. Models are local files;
serving never downloads model code or executes Python from a model repository.

## Install and export

Use a dedicated Python environment. Install the package from the same upstream
commit as the Rust integration. The runtime extra pins the inference stack;
the worker extra installs only the dependencies needed to execute stages.

```sh
python -m venv .venv
.venv/bin/python -m pip install './tools/diffusion[runtime]'
# For the qualified Windows/Python 3.14 environment, also pass:
# -c tools/diffusion/constraints-windows-py314.txt
cascadia-diffusion export-sdxl --source /models/sdxl-openvino \
  --revision MODEL_REPOSITORY_COMMIT --size 512 --output /models/sdxl-stages-512
```

The source must already contain `text_encoder`, `text_encoder_2`, `unet`, and
`vae_decoder` OpenVINO IRs, their configs, and the SDXL tokenizer/scheduler
metadata. Record the source repository and its license alongside the artifact;
`--revision` must be its immutable 40-character commit. Fused adapters and
quantized weights produce different artifact hashes. Runtime adapter injection
and alternate schedulers are unsupported. Export and coordinator startup reject
non-Euler schedulers, alternate conditioning flags, and tokenizers outside the
77-token CLIP profile; they never silently substitute pipeline settings.

Export prints a manifest digest. Pin that digest in deployment configuration;
never accept the expected digest from a requesting peer. An export refuses to
replace an existing output directory and only publishes a complete verified
bundle. The two denoiser partitions preserve every live tensor crossing the
cut, including residual connections. The exporter minimizes transfer within a
45–55% weight-balance window. This is a memory partition, not a speedup promise.

```sh
cascadia-diffusion verify --source /models/sdxl-stages-512 \
  --manifest DIGEST --components text_encoder text_encoder_2 unet vae_decoder
cascadia-diffusion pack --source /models/sdxl-stages-512 \
  --manifest DIGEST --components unet_second --output /models/worker-b
```

A worker bundle contains only the selected IR weights and shared metadata.
It retains the complete manifest so peers can agree on one pipeline identity.
Verification rejects modified, missing, unlisted, or symlinked files. Keep the
bundle immutable and writable only by the deployment owner for the lifetime
of the serving process; caches and runtime configuration go outside it.

## Execute locally

```sh
cascadia-diffusion generate --source /models/sdxl-stages-512 \
  --manifest DIGEST --mode layers --device GPU --cache /var/cache/cascadia/images \
  --request request.json --output image.png
```

`request.json` contains a `prompt` and optional `negative_prompt`, `rng_seed`,
`num_inference_steps`, and `guidance_scale`. `mode=components` uses the complete
local denoiser; `mode=layers` executes both partitions locally. The latter is a
useful correctness baseline before distributing the same artifact. Both use
explicit FP32 accumulation, one OpenVINO stream, and copied stage outputs.
Quantized weights retain their exported representation.

## Supervised serving contract

`cascadia-diffusion serve --config CONFIG` binds only `127.0.0.1`. Its random
per-process bearer token comes from `CASCADIA_DIFFUSION_TOKEN`; it is never
accepted on the command line. The JSON config has exactly these fields:

```json
{
  "artifact": "/models/worker-b",
  "manifest": "PINNED_SHA256",
  "components": ["unet_second"],
  "device": "GPU",
  "cache": "/var/cache/cascadia/images",
  "port": 19044,
  "coordinator": null
}
```

Authenticated `/health` reports readiness only after verification and compile.
`/execute` accepts and returns the Rust crate's binary tensor frames. A
coordinator uses `/generate` and an authenticated loopback `/invoke` callback
configured by its supervisor. Metadata-only pipeline graphs have no fallback
executor. The sidecar is not an unauthenticated network service: enterprise
wraps it in admitted QUIC, reserves all stages before dispatch, and binds every
execution to the caller, manifest, worker boot, lease, and sequence.

A dropped caller does not release a running execution. Expired/cancelled leases
retain running work until confirmed completion. An ambiguous engine failure
quarantines the worker until restart; restarting changes the boot identity.
The supervisor must kill only its own process group or Windows Job Object.

## Validation and limits

Portable contract tests need no inference dependencies:

```sh
PYTHONPATH=tools/diffusion/src python -m unittest discover -s tools/diffusion/tests
cargo test -p cascadia-diffusion
```

`tests/hardware_sdxl.py` is an explicitly invoked real-hardware qualification
harness. It records three seeded runs, component calls, images, runtime
versions, latency, and process memory. Run it only on authorized idle hosts.
Full vs partitioned graph compilation can produce small numerical differences;
qualify the actual checkpoint, precision, resolution, and devices before
rollout. A verified graph cut alone does not establish image-quality parity,
a multi-host speedup, or support for a model that exceeds one host's memory.

The first exporter supports SDXL only. FLUX, Qwen-Image, diffusion transformer
partitioning, CFG parallelism, runtime LoRAs, and fine-tuning are separate
capabilities and must not be inferred from this contract. Size and dtype limits
are admission bounds, not a guarantee that arbitrary graphs fit a device.

Measured results and their limits are in [the qualification report](../../docs/diffusion-stage-qualification.md).
