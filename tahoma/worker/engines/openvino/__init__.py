"""OpenVINO engine plugin.

Two paths:

- `OpenVINOEngine` / `OpenVINOBuilder` — PyTorch-eager fp16 with manual KV cache.
  Supports distributed pipeline-parallel (multi-stage). Imported by default.
- `OptimumOVEngine` / `OptimumOVBuilder` — single-stage OV Runtime via
  optimum-intel. Loads a pre-exported INT4 OV IR. Optional dep.
"""

from tahoma.worker.engines.openvino.engine import OpenVINOBuilder, OpenVINOEngine
from tahoma.worker.engines.openvino.loader import ModelShard

__all__ = ["ModelShard", "OpenVINOBuilder", "OpenVINOEngine"]
