"""OpenVINO engine plugin.

Four engines:

- `OpenVINOEngine` / `OpenVINOBuilder` — PyTorch-eager fp16 with manual KV cache.
  Supports distributed pipeline-parallel (multi-stage). Imported by default.
- `OptimumOVEngine` / `OptimumOVBuilder` — single-stage OV Runtime via
  optimum-intel. Auto-exports from HF id. Optional `assistant_model` for
  spec decode (incompatible with optimum-intel 1.27 — falls back to plain).
- `OVRuntimeEngine` / `OVRuntimeBuilder` — multi-stage OV Runtime with
  stateful KV cache. Loads pre-exported per-stage IR shards.
- `OVSpecDecodeEngine` / `OVSpecDecodeBuilder` — single-stage OV Runtime
  with mask-based-rewind speculative decoding. ~1.5x single-user speedup.
"""

from tahoma.worker.engines.openvino.engine import OpenVINOBuilder, OpenVINOEngine
from tahoma.worker.engines.openvino.loader import ModelShard

__all__ = ["ModelShard", "OpenVINOBuilder", "OpenVINOEngine"]
