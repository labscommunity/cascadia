# qwen36_surgery

IR-surgery toolkit for the Qwen3.6-35B-A3B (`qwen3_5_moe`) staged engine —
slices the official `OpenVINO/Qwen3.6-35B-A3B-int4-ov` IR into per-stage
shards (no re-quantization). Design: `docs/architectures/qwen36-moe-support.md`.

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
