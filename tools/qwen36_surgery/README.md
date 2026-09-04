# qwen36_surgery

IR-surgery toolkit for the Qwen3.5-family staged engine (`--engine qwen35`):
slices an official int4 IR — `OpenVINO/Qwen3.6-35B-A3B-int4-ov`
(`qwen3_5_moe`) or `OpenVINO/Qwen3.8-27B-int4-ov` (dense `qwen3_5`) — into
per-stage shards (no re-quantization). The exporter reads hidden size,
layer count and per-layer attention type from the model dir's
`config.json`. Design: `docs/architectures/qwen36-moe-support.md`;
Qwen3.8 specifics: `docs/architectures/qwen3.8.md`.

## Producer (this is what you run)
- `export_qwen36_moe.py` — cuts the official IR into stage shards + manifest.

## Fixtures
- `golden/qwen36_parity_64.json` — token-parity golden consumed by
  `crates/cascadia-engine-openvino/tests/qwen36_parity.rs`.
- `golden/promptset_*.json`, `promptset.json` — prompt sets for the
  diagnostics below.

## Diagnostics / provenance
`probe_*.py`, `proto_m3_decode.py`, `m4_gate_*.py` are the one-off spikes
that established the surgery, hand-off, and parity claims; several are
cited by `//` provenance comments in `crates/cascadia-engine-openvino/src/qwen36.rs`.
Kept as the recorded validation recipe, not part of the shipped binary.
