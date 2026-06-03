# MiniMax-M2 support

Cascadia can export and run **MiniMax-M2** (`MiniMaxAI/MiniMax-M2`) through
the sparse-MoE engine. M2 is a 62-layer, **all-MoE** transformer — 256
experts per layer, top-8 routing, 230B total / ~10B active params.

## Scope & constraints (read this first)

- **M2 only.** The other open MiniMax text models (MiniMax-Text-01, M1,
  VL-01) use a *lightning (linear) attention* hybrid whose chunked
  recurrence does not trace cleanly into OpenVINO IR. M2 dropped lightning
  attention for plain full-softmax GQA, which exports like a normal MoE
  transformer. M2 is the only tractable MiniMax text model for this engine.
- **Single-stage only.** M2 runs on one high-RAM node (the experts stream
  from NVMe via a bounded LRU, same approach as Kimi K2.6). It does **not**
  fit the Intel AI-PC fleet: even at INT4 the weights are ~115 GB before
  KV/activations, far beyond a 16–32 GB Lunar/Arrow Lake box, and the
  "10B active" figure is a compute property, not a memory one — all 256
  experts/layer must be resident across the ring. Pipeline-parallel MoE is
  not implemented (the K2.6 transport path assumes the Rust MLA shell), so
  the OV-IR backend rejects `total > 1`.

## Design: architecture-in-graph

Unlike the K2.6 path — which runs a hand-written MLA attention kernel in
Rust (`cascadia-int4-gemm`) — the M2 backend keeps the Rust runtime
**architecture-agnostic**. Every M2-specific detail lives in the exported
OpenVINO graphs:

- **GQA** (48 query / 8 KV heads, head_dim 128),
- **full-width QK-norm** (RMSNorm over the whole q/k projection *before*
  the head reshape — not per-head),
- **partial RoPE** (rotary applied to the first `head_dim *
  partial_rotary_factor` dims; the rest pass through),
- **sigmoid routing** with an additive `e_score_correction_bias` used for
  top-k *selection* only, then the raw sigmoid weights are gathered and
  renormalized (DeepSeek-style aux-loss-free routing),
- **SwiGLU experts** (`down(silu(gate(x)) · up(x))`), no shared expert.

The runtime (`crates/cascadia-engine-sparse-moe/src/ov_moe.rs`) only
embeds the token, runs each layer's shell graph (threading a per-layer KV
cache through the 7-tensor contract), dispatches the top-k experts the
shell selected, combines `attn_residual + Σ wₖ·expertₖ(attn_out_post_norm)`,
and runs the head. Because the graphs carry their own shapes, the same code
runs both the tiny synthetic model used by the test and the full 230B model.

The shell graph's I/O contract (decode, seq=1):

| inputs | outputs |
| --- | --- |
| `x` `[1,1,H]`, `past_k`/`past_v` `[1,KV,P,D]`, `past_seq_len` (i64 scalar) | `attn_out_post_norm`, `attn_residual`, `shared_expert_out` (zeros), `routing_ids` `[K]`, `routing_weights` `[K]`, `present_k`/`present_v` `[1,KV,1,D]` |

## Export

```bash
# tiny synthetic model (for the correctness test; no download):
python tools/export_minimax_m2.py --tiny --no-quant --out /tmp/m2_tiny

# full model from a local FP8 checkout (dequant FP8 -> INT4):
python tools/export_minimax_m2.py --model /path/to/MiniMax-M2 --out /path/to/m2_export
```

The exporter writes Cascadia's sparse-MoE layout (`manifest.json`,
`layer0/` embed, `head/`, `shells/layer_NN/`, `experts/layer_NN/expert_EEE/`)
with `shell_backend: "ov_ir"` in the manifest. FP8 block weights
(`float8_e4m3fn`, `[128,128]` `weight_scale_inv`) are dequantized to fp32
then re-quantized to INT4 via NNCF; the router gate / `e_score_correction_bias`
are kept full precision.

Needs a Python env with `torch`, `transformers>=4.57` (for `MiniMaxM2`),
`openvino` 2026.x, and `nncf`.

## Run

```bash
cascadia worker --engine sparse-moe --model /path/to/m2_export --device CPU
```

The builder reads `manifest.json`; `shell_backend: "ov_ir"` automatically
selects the M2 engine (no extra flag). Generation is greedy in v1
(sampling config is ignored). Bound expert RAM with
`--max-cached-experts N` / `CASCADIA_MAX_EXPERTS_CACHED` so experts stream
from disk rather than all staying resident.

## Tests

- **Python pipeline harness** — `tools/test_minimax_m2_pipeline.py
  --model-dir <tiny>` runs the exported OV graphs as a full pipeline and
  checks the greedy token stream against the canonical HF model
  (`reference.json`, written by `--tiny`).
- **Rust e2e** — `cargo test -p cascadia-engine-sparse-moe --test
  minimax_m2_eval` drives [`OvMoeRunner`] over the same export and asserts
  the same match. Gated on `M2_MODEL_DIR`; skips when unset (needs
  OpenVINO + the fixture). Both validate against a `--no-quant` tiny
  export, since INT4 on a 2-layer random model is too lossy to preserve
  argmax — INT4 is exercised on the learned full-model weights.
