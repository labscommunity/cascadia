"""OpenVINO engine plugin."""

from tahoma.worker.engines.openvino.engine import OpenVINOBuilder, OpenVINOEngine
from tahoma.worker.engines.openvino.loader import ModelShard

__all__ = ["ModelShard", "OpenVINOBuilder", "OpenVINOEngine"]
