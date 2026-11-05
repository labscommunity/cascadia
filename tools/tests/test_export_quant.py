"""The exporter's weight compression either delivers what was asked for or
stops the export: an INT4 tree must never turn out to hold FP16 weights."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ov = pytest.importorskip("openvino")
nncf = pytest.importorskip("nncf")
import openvino.opset13 as ops  # noqa: E402

import export_quant  # noqa: E402


def _matmul_model(channels: int):
    """x -> MatMul(x, w), w of shape (channels, channels), like a linear layer."""
    x = ops.parameter([1, channels], ov.Type.f32, name="x")
    rng = np.random.default_rng(0)
    w = ops.constant(rng.standard_normal((channels, channels)), ov.Type.f32)
    return ov.Model([ops.matmul(x, w, False, False)], [x], "m")


@pytest.mark.parametrize("channels", [576, 256])
def test_int4_compresses_any_channel_size(channels):
    # 576 does not divide by the group size of 128: SmolLM2's hidden size.
    out = export_quant.compress_weights(_matmul_model(channels), "int4", print)
    assert "i4" in export_quant.constant_types(out)


def test_int4_asym_and_int8_compress():
    assert "u4" in export_quant.constant_types(
        export_quant.compress_weights(_matmul_model(256), "int4_asym", print)
    )
    types = export_quant.constant_types(
        export_quant.compress_weights(_matmul_model(256), "int8", print)
    )
    assert types & {"i8", "u8"}


@pytest.mark.parametrize("quant", ["fp16", "fp32"])
def test_uncompressed_formats_are_left_alone(quant):
    model = _matmul_model(256)
    assert export_quant.compress_weights(model, quant, print) is model


def test_missing_nncf_stops_the_export(monkeypatch):
    monkeypatch.setitem(sys.modules, "nncf", None)
    with pytest.raises(SystemExit, match="--quantization fp16"):
        export_quant.compress_weights(_matmul_model(256), "int4", print)


def test_require_compressor_only_refuses_formats_that_need_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "nncf", None)
    export_quant.require_compressor("fp16")
    with pytest.raises(SystemExit, match="nncf is not installed"):
        export_quant.require_compressor("int8")


def test_a_compression_error_stops_the_export(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no luck")

    monkeypatch.setattr(nncf, "compress_weights", boom)
    with pytest.raises(SystemExit, match="no luck"):
        export_quant.compress_weights(_matmul_model(256), "int4", print)


def test_compression_that_changes_nothing_stops_the_export(monkeypatch):
    monkeypatch.setattr(nncf, "compress_weights", lambda model, **_k: model)
    with pytest.raises(SystemExit, match="no int4 weights"):
        export_quant.compress_weights(_matmul_model(256), "int4", print)
