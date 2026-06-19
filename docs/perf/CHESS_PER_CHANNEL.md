# Per-channel FFN sparsity thresholds (CHESS) — workflow & file format

**Status:** landed alongside PR for issue [#38](https://github.com/labscommunity/cascadia/issues/38).
**Builds on:** [#34](https://github.com/labscommunity/cascadia/issues/34) (global-τ infra),
[#35](https://github.com/labscommunity/cascadia/issues/35) (AXPY-form down kernel).

## What this is

PR #34 landed global-τ FFN sparsity: skip any intermediate lane `i` where
`|silu(gate[i])| < τ · max_j|silu(gate[j])|` for a single scalar `τ`. The
quality cost rises steeply past τ ≈ 0.05 on K2.6 because *some* channels are
reliably small-magnitude contributors (safe to drop) while *others* are
reliably large-magnitude (must keep). A global τ can't separate them.

This PR adds **per-channel thresholds**: τ becomes a vector `τ[c] ∈ R^{intermediate}`
calibrated offline from a representative corpus, following the
[CHESS](https://arxiv.org/abs/2409.01366) approach (Liu et al. 2024, MIT).
Same dispatch shape, same kernels — just substitute a vector cutoff for the
scalar one.

## The math (one screen)

Runtime mask construction, per (layer, expert) call:

```text
silu_gate[c] = silu(gate_out[c])
max_abs      = max_j |silu_gate[j]|
active[c]    = ( |silu_gate[c]| >= τ[layer, c] * max_abs )
```

Calibration target, per (layer, channel):

```text
τ[layer, c] = quantile_{1 - target_active_frac}(
    |silu(gate[layer, c])| / max_j |silu(gate[layer, j])|
)
```

`target_active_frac = 0.5` is the standard CHESS setting (~50% lanes active,
<1 pp accuracy drop at iso-sparsity vs. CATS in published results).

The per-token `max_abs` normalisation is identical to PR #34's global-τ path,
which means a uniform threshold vector (`τ[c] = τ0` for all c) is
**bit-identical** to the global-τ path at `τ = τ0`. That invariant is locked
by `crates/cascadia-int4-gemm/src/ffn_sparsity.rs::per_channel_uniform_matches_global`
and equivalents for the AXPY-form and bf16-boundary entry points.

## File format

JSON, schema `version: 1`. See `crates/cascadia-int4-gemm/src/ffn_thresholds.rs`.

```json
{
  "version": 1,
  "model_id": "kimi-k2.6-instruct",
  "n_intermediate": 2048,
  "calibration_n_tokens": 12345,
  "target_active_frac": 0.5,
  "notes": "optional free-form provenance",
  "layers": [
    { "layer_id": 0, "thresholds": [0.123, 0.456, ...] },
    ...
  ]
}
```

- **Size at K2.6**: 60 layers × 2048 channels × 4 B ≈ 480 KiB binary,
  ~1.5 MB pretty-printed JSON.
- **Layer coverage is optional**: layers absent from `layers` fall back to
  the scalar `--ffn-sparsity-threshold` value (or dense if that's `0.0`).
  This is the right semantic for partial calibration runs.
- **Per-(layer, channel) only**: per-(layer, expert, channel) is a future
  extension (would be 60 × 384 × 2048 ≈ 189 MiB), gated on whether
  calibration data is dense enough per-expert to estimate the quantile
  reliably. For K=8 active experts per token × 1k calibration tokens you
  see ~21 samples per expert per layer — too few. Per-(layer, channel)
  aggregates across the K=8 active experts and gets ~21k samples per
  layer-channel.

## Workflow

### Phase 1 — capture

Start a worker with the capture flag pointing at an empty directory.
Capture requires `--ffn-axpy-down` (the only path that surfaces `silu(gate)`
to the runner; the bf16-boundary path doesn't).

```bash
cascadia worker \
  --model /path/to/k26 \
  --device CPU \
  --ffn-axpy-down \
  --ffn-axpy-prebuild \
  --ffn-sparsity-capture-dir /tmp/k26_gate_caps
```

Then exercise the model with a representative prompt corpus via the
`/v1/chat/completions` endpoint. 500–1000 prompts × ~50 tokens each is
the CATS / CHESS reference budget. On K2.6 single-stage that's roughly
half an hour of wall time on miner.

On clean shutdown (`SIGTERM`), the engine drains the in-memory histograms
to `/tmp/k26_gate_caps/layer_<lid>.bin` — one file per covered layer,
~1 MiB each.

### Phase 2 — calibrate

```bash
cargo run --release --bin calibrate_ffn_thresholds -- \
  --capture-dir /tmp/k26_gate_caps \
  --target-active-frac 0.5 \
  --model-id kimi-k2.6-instruct \
  --output /tmp/k26_thresholds_50.json
```

The tool reads the histograms, computes per-channel quantiles, and writes
the threshold JSON file. Wall time on miner: seconds.

### Phase 3 — serve

```bash
cascadia worker \
  --model /path/to/k26 \
  --device CPU \
  --ffn-axpy-down \
  --ffn-axpy-prebuild \
  --ffn-sparsity-thresholds-file /tmp/k26_thresholds_50.json
```

Per-channel thresholds take precedence over `--ffn-sparsity-threshold` for
any layer present in the file. Layers absent from the file fall back to the
scalar value (so you can mix: per-channel for layers you calibrated against,
global-τ as a conservative fallback for the rest).

## Quality checks before deploy

1. **Smoke test** on the canonical K2.6 prompts:
   ```bash
   K26_MODEL_DIR=/path/to/k26 \
   CASCADIA_FFN_SPARSITY_THRESHOLDS_FILE=/tmp/k26_thresholds_50.json \
     cargo test -p cascadia-engine-sparse-moe --test k26_layer0_eval -- --nocapture
   ```
   Expect the canonical prompts ("Paris", "Pacific", "four") to still produce
   sensible first tokens. A regression here means the calibration corpus
   wasn't representative of the test distribution.

2. **Active-fraction sanity check**: the worker logs the average active
   fraction every N expert calls (`ffn_active_fraction`). With a 0.5
   `target_active_frac` calibration, the runtime average should land in
   the 0.45 – 0.55 range. Wide deviation means the calibration corpus
   doesn't match the served distribution.

3. **End-to-end tok/s**: PR #34 reported no net wall-time win at quality-
   preserving τ on K2.6 (the global-τ path's safe τ ≤ 0.05 leaves too many
   lanes active for the down kernel to recoup the overhead). Per-channel
   at 0.5 active_frac, paired with `--ffn-axpy-down`, should put us into
   the regime where the kernel speedup ceiling (1/active_frac = 2×) is
   actually realisable on the down projection.

## Why per-(layer, channel) and not per-(layer, expert, channel)

K2.6 has 60 MoE layers × 384 experts × 2048 intermediate channels. Per-
(layer, expert, channel) would be 189 MiB on disk (still fine) but the
calibration data budget doesn't support it: a 1k-token corpus × top-K=8
routed experts produces ~21 samples per (layer, expert), nowhere near
enough for a stable quantile estimate. Per-(layer, channel) pools across
experts and gets ~21k samples per layer-channel, which is plenty for the
1/128 quantile precision the histogram-based estimator achieves.

If/when calibration data scales to ~50k+ prompts (per the EAC-MoE /
PreMoE recipes which find per-expert specialisation matters), the file
format can be bumped to v2 with an inner per-expert dimension and the
runtime can route to that. The current `PerChannelThresholds::get(layer_id)`
API leaves room for this extension.

## Attribution

- **CHESS** — Liu et al. 2024, "CHESS: Optimizing LLM Inference via Channel-
  wise Thresholding and Selective Sparsification" (arxiv:2409.01366,
  EMNLP 2024, MIT licence). The per-channel formulation we ported.
- **CATS** — Lee et al. 2024, "CATS: Contextually-Aware Thresholding for
  Sparsity in LLMs" (arxiv:2404.08763, Apache-2.0). The global-τ baseline
  PR #34 implemented.
- **PowerInfer** — Song et al. 2024, the two-phase Gate-then-Up/Down skip
  pattern (MIT). See rainier `docs/POWERINFER_PORT.md` for the full
  technique map.

Cascadia's CHESS port is an independent Rust implementation — referenced,
not copied — under Apache-2.0.
