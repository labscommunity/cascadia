"""Clef joint head: TensorHead == reference loop head; export names match decision.rs.

    python -m pytest tools/tests/test_clef_head.py -v
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from decision import clef_head as ch  # noqa: E402


def _record(hidden_size, t=20):
    torch.manual_seed(0)
    head = ch.JointSchemaHead(hidden_size=hidden_size, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).eval()
    hidden = torch.randn(t, hidden_size)
    q_spans = [(2, 4), (10, 12)]
    o_spans = [[(5, 6), (7, 9)], [(13, 15), (16, 18)]]
    lm_head = torch.randn(50, hidden_size)
    ids = torch.randint(0, 50, (t,))
    return head, hidden, q_spans, o_spans, lm_head, ids


def test_tensor_head_matches_reference():
    hs = 8
    head, hidden, q_spans, o_spans, lm_head, ids = _record(hs)
    ref = ch.reference_logits(head, hidden, ids, q_spans, o_spans, [1, 0], lm_head)
    spans = q_spans + [s for q in o_spans for s in q]
    mask = torch.zeros(len(spans), hidden.shape[0])
    for i, (a, b) in enumerate(spans):
        mask[i, a:b] = 1.0 / (b - a)
    lexical = torch.stack([lm_head[ids[a:b]].mean(0) for q in o_spans for (a, b) in q])
    option_q = torch.tensor([0, 0, 1, 1])
    got = ch.TensorHead(head)(hidden, mask, lexical, option_q, torch.tensor([1, 0]))
    assert torch.allclose(got, ref, atol=1e-5)


def test_export_uses_contract_input_names(tmp_path):
    ov = pytest.importorskip("openvino")
    head, *_ = _record(8)
    xml = tmp_path / "joint_head.xml"
    ch.export_module(ch.TensorHead(head), 8, str(xml))
    m = ov.Core().read_model(str(xml))
    assert [i.get_any_name() for i in m.inputs] == ["hidden", "span_mask", "lexical", "option_q", "type_ids"]


def test_export_dtypes_and_output_match_decision_rs(tmp_path):
    """decision.rs feeds hidden/span_mask/lexical as f32, option_q/type_ids as i64, reads output(0) as f32."""
    ov = pytest.importorskip("openvino")
    head, *_ = _record(8)
    xml = tmp_path / "joint_head.xml"
    ch.export_module(ch.TensorHead(head), 8, str(xml))
    m = ov.Core().read_model(str(xml))
    f32, i64 = ov.Type.f32, ov.Type.i64
    assert [i.get_element_type() for i in m.inputs] == [f32, f32, f32, i64, i64]
    assert len(m.outputs) == 1 and m.outputs[0].get_element_type() == f32
    # Dynamic T/Q/O: the export example was T=12, Q=2, O=4; run T=30, Q=3, O=7 against the reference head.
    torch.manual_seed(1)
    t = 30
    hidden = torch.randn(t, 8)
    lm_head = torch.randn(50, 8)
    ids = torch.randint(0, 50, (t,))
    q_spans = [(1, 3), (8, 10), (20, 22)]
    o_spans = [[(4, 5), (5, 7)], [(11, 12), (12, 14), (14, 16)], [(23, 24), (24, 26)]]
    qtypes = [2, 1, 0]
    ref = ch.reference_logits(head, hidden, ids, q_spans, o_spans, qtypes, lm_head)
    spans = q_spans + [s for q in o_spans for s in q]
    mask = torch.zeros(len(spans), t)
    for i, (a, b) in enumerate(spans):
        mask[i, a:b] = 1.0 / (b - a)
    lexical = torch.stack([lm_head[ids[a:b]].mean(0) for q in o_spans for (a, b) in q])
    option_q = torch.tensor([qi for qi, q in enumerate(o_spans) for _ in q])
    compiled = ov.Core().compile_model(m, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    got = compiled([hidden.numpy(), mask.numpy(), lexical.numpy(), option_q.numpy(), torch.tensor(qtypes).numpy()])[0]
    assert got.shape == (7,)
    assert abs(got - ref.numpy()).max() < 1e-5


def test_export_joint_head_from_real_dir(tmp_path):
    """Real Clef weights: only when CLEF_HEAD_DIR points at joint_head.safetensors + config."""
    ov = pytest.importorskip("openvino")
    pytest.importorskip("safetensors")
    head_dir = os.environ.get("CLEF_HEAD_DIR")
    if not head_dir or not os.path.isfile(os.path.join(head_dir, "joint_head.safetensors")):
        pytest.skip("CLEF_HEAD_DIR with joint_head.safetensors not set")
    xml = tmp_path / "joint_head.xml"
    ch.export_joint_head(head_dir, str(xml))
    m = ov.Core().read_model(str(xml))
    assert [i.get_any_name() for i in m.inputs] == ["hidden", "span_mask", "lexical", "option_q", "type_ids"]
