"""Low-memory weight swap + LoRA graft for Qwen3.5-family stage IRs.

Intel's stock IRs (OpenVINO/Qwen3.5-9B-int8-ov, OpenVINO/Qwen3.8-27B-int8-ov) have the exact graph of any
fine-tune or sibling of the same architecture. For one stage (cut by export_qwen36_moe.extract_stage) every
weight constant is replaced with the target checkpoint's tensor quantized in the stock constant's scheme:
  u8 [out, in] + u8 zp + f16 scale  -> int8 asymmetric per output channel (bit-exact vs NNCF)
  u4 [out, G, 128] + u4 zp + f16    -> int4 asymmetric group 128 (NNCF data-free except .5 ties)
  float constants (norms, dt_bias, A)-> via FLOAT_KIND_TRANSFORM, calibrated against the stock pair
Embeddings (i8 symmetric per row) are swapped in row chunks. Peak memory: one stage.
"""
import json
import re
from pathlib import Path

import numpy as np


def q_asym(w, bits, axis):
    # NNCF's asymmetric range always contains zero: a single-signed row (common in the 4-tap
    # linear-attention conv1d) would otherwise clamp its zero point and saturate.
    hi = 2 ** bits - 1
    lo_v = np.minimum(w.min(axis=axis, keepdims=True), 0)
    hi_v = np.maximum(w.max(axis=axis, keepdims=True), 0)
    scale = (hi_v - lo_v) / hi
    scale = np.where(scale == 0, np.finfo(np.float32).eps, scale)
    zp = np.clip(np.round(-lo_v / scale), 0, hi)
    q = np.clip(np.round(w / scale) + zp, 0, hi)
    return q.astype(np.uint8), zp.astype(np.uint8), scale.astype(np.float16)


def q_sym8(w):
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, np.finfo(np.float32).eps, scale)
    return np.clip(np.round(w / scale), -128, 127).astype(np.int8), scale.astype(np.float16)


TRANSFORMS = {"w": lambda w: w, "1+w": lambda w: 1.0 + w, "exp": np.exp}
FLOAT_KIND_TRANSFORM = {
    "input_layernorm.weight": "1+w",
    "post_attention_layernorm.weight": "1+w",
    "self_attn.q_norm.weight": "1+w",
    "self_attn.k_norm.weight": "1+w",
    "norm.weight": "1+w",
    "linear_attn.A_log": "exp",
    "linear_attn.dt_bias": "w",
    "linear_attn.norm.weight": "w",
}
FLOAT_KINDS = {
    r"layers\.(\d+)\.input_layernorm/aten::mul/Multiply_1$": "input_layernorm.weight",
    r"layers\.(\d+)\.post_attention_layernorm/aten::mul/Multiply_1$": "post_attention_layernorm.weight",
    r"layers\.(\d+)\.linear_attn\.norm/aten::mul/Multiply_1$": "linear_attn.norm.weight",
    r"layers\.(\d+)\.linear_attn/aten::add/Add_2$": "linear_attn.dt_bias",
    r"layers\.(\d+)\.linear_attn/aten::neg/Multiply_1$": "linear_attn.A_log",
    r"layers\.(\d+)\.self_attn\.q_norm/aten::mul/Multiply_1$": "self_attn.q_norm.weight",
    r"layers\.(\d+)\.self_attn\.k_norm/aten::mul/Multiply_1$": "self_attn.k_norm.weight",
    r"language_model\.norm/aten::mul/Multiply_1$": "norm.weight",
}
PROJ = re.compile(r"language_model\.layers\.(\d+)\.([\w.]+)\._openvino_orig_weight(/zero_point|/scale)?$")
CONV = re.compile(r"language_model\.layers\.(\d+)\.linear_attn/aten::_convolution/Reshape(/zero_point|/scale)?$")
SLOT = {None: "w", "/zero_point": "zp", "/scale": "s"}


class Weights:
    """HF safetensors checkpoint, read tensor by tensor (text-model names relative to the layer stack)."""

    def __init__(self, d):
        from safetensors import safe_open  # noqa: F401  (import check)
        self.d = Path(d)
        self.idx = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in self.idx) else "model."
        self._f = {}

    def _file(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self._f:
            self._f[f] = safe_open(self.d / f, "pt")
        return self._f[f]

    def get(self, name):
        n = self.prefix + name
        return self._file(n).get_tensor(n).float().numpy()

    def rows(self, name, a, b):
        n = self.prefix + name
        return self._file(n).get_slice(n)[a:b].float().numpy()


def _classify(model):
    import openvino as ov
    proj, conv, floats = {}, {}, []
    for op in model.get_ops():
        if op.get_type_name() != "Constant":
            continue
        nm = op.get_friendly_name()
        m = PROJ.search(nm)
        if m and "lm_head" not in nm:
            proj.setdefault((int(m.group(1)), m.group(2)), {})[SLOT[m.group(3)]] = op
            continue
        m = CONV.search(nm)
        if m:
            conv.setdefault(int(m.group(1)), {})[SLOT[m.group(2)]] = op
            continue
        if op.get_element_type() not in (ov.Type.f32, ov.Type.f16) or int(np.prod(list(op.get_output_shape(0)) or [1])) <= 1:
            continue
        for ti in op.output(0).get_target_inputs():
            cn = ti.get_node().get_friendly_name()
            for pat, hf in FLOAT_KINDS.items():
                mm = re.search(pat, cn)
                if mm:
                    floats.append((op, ti, (f"layers.{mm.group(1)}." if mm.groups() else "") + hf))
                    break
    return proj, conv, floats


def _replace(op, arr, et, targets=None):
    """Feed `arr` to `targets` (target inputs of op's output; None = every consumer of op)."""
    from openvino import opset13 as ops
    new = ops.constant(arr, dtype=et)
    new.set_friendly_name(op.get_friendly_name())
    for t in list(op.output(0).get_target_inputs()) if targets is None else targets:
        t.replace_source_output(new.output(0))


def swap_stage(model, weights):
    """Replace every weight constant of a cut stage with `weights` (a Weights). Returns a report
    of what was swapped per layer, for `check_swap`: {"proj": {layer: sorted [paths]},
    "conv": {layer: n}, "float": {layer_or_"final": [kinds]}}."""
    import openvino as ov
    proj, conv, floats = _classify(model)
    report = {"proj": {}, "conv": {}, "float": {}}
    for (layer, path), c in proj.items():
        w = weights.get(f"layers.{layer}.{path}.weight")
        shp = list(c["w"].get_output_shape(0))
        if c["w"].get_element_type() == ov.Type.u4:
            q, zp, s = q_asym(w.reshape(shp), 4, axis=-1)
            _replace(c["w"], q, ov.Type.u4)
            _replace(c["zp"], zp, ov.Type.u4)
        else:
            q, zp, s = q_asym(w, 8, axis=-1)
            _replace(c["w"], q, ov.Type.u8)
            _replace(c["zp"], zp, ov.Type.u8)
        _replace(c["s"], s, ov.Type.f16)
        report["proj"].setdefault(layer, set()).add(path)
    for layer, c in conv.items():
        shp = list(c["w"].get_output_shape(0))
        w = weights.get(f"layers.{layer}.linear_attn.conv1d.weight").reshape(shp[0], -1)
        q, zp, s = q_asym(w, 8, axis=-1)
        _replace(c["w"], q.reshape(shp), ov.Type.u8)
        _replace(c["zp"], zp.reshape(list(c["zp"].get_output_shape(0))), ov.Type.u8)
        _replace(c["s"], s.reshape(list(c["s"].get_output_shape(0))), ov.Type.f16)
        report["conv"][layer] = report["conv"].get(layer, 0) + 1
    for op, ti, hf in floats:
        kind = re.sub(r"layers\.\d+\.", "", hf)
        v = TRANSFORMS[FLOAT_KIND_TRANSFORM[kind]](weights.get(hf)).astype(np.float32)
        # only the matched consumer: a float constant shared with another op (or another kind)
        # must keep feeding that op the stock value
        _replace(op, v.reshape(list(op.get_output_shape(0))), op.get_element_type(), targets=[ti])
        m = re.match(r"layers\.(\d+)\.", hf)
        report["float"].setdefault(int(m.group(1)) if m else "final", []).append(kind)
    report["proj"] = {k: sorted(v) for k, v in report["proj"].items()}
    return report


# Projection paths per layer type (PROJ group 2), checked against the stock Qwen3.5-9B int8 IR's
# `_openvino_orig_weight` constants: 8 for linear-attention layers, 7 for full-attention layers.
LAYER_PROJ = {
    "linear_attention": {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                         "linear_attn.in_proj_a", "linear_attn.out_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
    "full_attention": {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
}
LAYER_FLOATS = {
    "linear_attention": {"input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.A_log",
                         "linear_attn.dt_bias", "linear_attn.norm.weight"},
    "full_attention": {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                       "self_attn.k_norm.weight"},
}


def check_swap(report, layers, layer_types, last):
    """Every layer of the stage swapped every projection its type carries (LAYER_PROJ), its conv
    (linear layers) and every float kind its type carries; the last stage also swapped the final norm."""
    for i in layers:
        t = layer_types[i]
        missing = LAYER_PROJ[t] - set(report["proj"].get(i, ()))
        if missing:
            raise ValueError(f"layer {i} ({t}): projections not swapped: {sorted(missing)}")
        if t == "linear_attention" and not report["conv"].get(i):
            raise ValueError(f"layer {i}: conv1d not swapped")
        missing = LAYER_FLOATS[t] - set(report["float"].get(i, []))
        if missing:
            raise ValueError(f"layer {i} ({t}): float constants not swapped: {sorted(missing)}")
    if last and "norm.weight" not in report["float"].get("final", []):
        raise ValueError("final norm not swapped")


# A PEFT LoRA key -> (module path below the decoder: "layers.3.mlp.down_proj" or "lm_head", A | B).
LORA_KEY = re.compile(r"base_model\.model\.(?:model\.)?(?:language_model\.)?(.+)\.lora_([AB])\.weight$")


def lora_modules(keys, src="adapter"):
    """{module: {"A": key, "B": key}} for an adapter's tensor names. The graft and the fp32 reference
    apply y += scale * x A^T B^T only, so refuse anything else, naming the keys: embedding LoRA
    (lora_embedding_A/B), any non-LoRA tensor (a modules_to_save copy such as model.norm.weight,
    trainable token rows), a lora_A without its lora_B or the reverse, and one module named twice."""
    keys = sorted(keys)
    emb = [k for k in keys if ".lora_embedding_" in k]
    if emb:
        raise ValueError(f"{src}: embedding LoRA keys {emb[:4]}: the graft and the reference apply "
                         "linear LoRA only, so the embedding delta would be dropped")
    other = [k for k in keys if not LORA_KEY.match(k)]
    if other:
        raise ValueError(f"{src}: non-LoRA tensors {other[:4]} (modules_to_save copies or other trained "
                         "weights): the graft and the reference would drop them")
    mods = {}
    for k in keys:
        m = LORA_KEY.match(k)
        side = mods.setdefault(m.group(1), {})
        if m.group(2) in side:
            raise ValueError(f"{src}: module {m.group(1)} is named twice ({side[m.group(2)]}, {k})")
        side[m.group(2)] = k
    for mod, side in mods.items():
        for need in ("A", "B"):
            if need not in side:
                have = next(iter(side.values()))
                raise ValueError(f"{src}: {have} has no matching lora_{need} for module {mod}")
    return mods


def lora_targets(adapter_file):
    """Sorted module paths the adapter targets (lora_modules' refusals apply)."""
    from safetensors import safe_open
    with safe_open(adapter_file, "pt") as f:
        return sorted(lora_modules(f.keys(), adapter_file))


def lora_scale_from_config(adapter_file):
    """PEFT scale from adapter_config.json beside the adapter: alpha/r (alpha/sqrt(r) with rsLoRA).
    Refuses PEFT variants a plain y += scale * x A^T B^T graft would get wrong."""
    cfg = json.loads((Path(adapter_file).parent / "adapter_config.json").read_text())
    for k in ("rank_pattern", "alpha_pattern"):
        if cfg.get(k):
            raise ValueError(f"adapter_config.json has a non-empty {k}; one global LoRA scale would be wrong")
    if cfg.get("use_dora"):
        raise ValueError("adapter_config.json has use_dora: true; the graft would drop the DoRA magnitude vectors")
    if cfg.get("lora_bias") not in (None, False):
        raise ValueError(f"adapter_config.json has lora_bias={cfg['lora_bias']!r}; the graft does not add LoRA biases")
    if cfg.get("modules_to_save"):
        raise ValueError(f"adapter_config.json has modules_to_save={cfg['modules_to_save']!r}; the graft and the "
                         "reference keep the base weights of those modules, not the trained copies")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r


def _feeds_subtract(const):
    """True when the constant's decompression chain (Constant -> Convert -> ...) subtracts a zero point."""
    for t in const.output(0).get_target_inputs():
        node = t.get_node()
        if node.get_type_name() == "Subtract":
            return True
        if node.get_type_name() == "Convert" and any(
                u.get_node().get_type_name() == "Subtract" for u in node.output(0).get_target_inputs()):
            return True
    return False


def check_embedding_layout(src, elem_type, has_zero_point, scale_shape, rows):
    """swap_embeddings writes i8 symmetric rows with one f16 scale per row, so the stock constant
    must already be laid out that way (verified on the Qwen3.5-9B IR only). A u8 asymmetric layout
    would keep subtracting the stock zero point from the new symmetric values: every embedding
    shifted, with no error until parity. Refuse anything else, naming what was found."""
    problems = []
    if elem_type != "i8":
        problems.append(f"weight element type {elem_type} (expected i8)")
    if has_zero_point:
        problems.append("a zero point is subtracted (asymmetric)")
    if list(scale_shape) != [rows, 1]:
        problems.append(f"scale shape {list(scale_shape)} (expected [{rows}, 1])")
    if problems:
        raise ValueError(f"{src}: embeddings are not i8 symmetric with one scale per row: {'; '.join(problems)}. "
                         "swap_embeddings only re-quantizes that layout; inspect this IR's embeddings first")


def swap_embeddings(src_xml, weights, out_dir, chunk=16384):
    import openvino as ov
    model = ov.Core().read_model(src_xml)
    consts = {op.get_friendly_name(): op for op in model.get_ops() if op.get_type_name() == "Constant"}
    wop = next((op for nm, op in consts.items() if nm.endswith("_openvino_orig_weight")), None)
    sop = next((op for nm, op in consts.items() if nm.endswith("_openvino_orig_weight/scale")), None)
    if wop is None or sop is None:
        raise ValueError(f"{src_xml}: no embedding Constant ending in '_openvino_orig_weight' "
                         "(and '_openvino_orig_weight/scale')")
    rows, hid = list(wop.get_output_shape(0))
    zp = any(nm.endswith("_openvino_orig_weight/zero_point") for nm in consts) or _feeds_subtract(wop)
    check_embedding_layout(src_xml, wop.get_element_type().get_type_name(), zp,
                           list(sop.get_output_shape(0)), rows)
    q = np.empty((rows, hid), dtype=np.int8)
    s = np.empty((rows, 1), dtype=np.float16)
    for a in range(0, rows, chunk):
        q[a:a + chunk], s[a:a + chunk] = q_sym8(weights.rows("embed_tokens.weight", a, min(rows, a + chunk)))
    _replace(wop, q, ov.Type.i8)
    _replace(sop, s, ov.Type.f16)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(Path(out_dir) / "openvino_text_embeddings_model.xml"), compress_to_fp16=False)


def graft_lora(model, adapter_file, scale, layers=None, lm_head=True):
    """y += scale * (x A^T) B^T on every LoRA-targeted MatMul in `model` (fp32 constants, LoRA term
    first in the Add: OpenVINO's ARM CPU plugin produced NaN with fp16 constants / base-first adds).
    Returns the module paths grafted, one per branch added. Adapter modules with no matching MatMul
    in this stage are skipped silently (a stage holds only some layers), so the exporter MUST assert
    that over all stages every module of lora_targets(adapter_file) was grafted exactly once.

    `layers` (a range) limits grafting to the stage's own decoder layers and `lm_head` to the last
    stage: a stage > 0 also keeps shape-only copies of earlier layers (the stock IR's mask/position
    bookkeeping reads their output shapes), whose values never reach the stage output. Grafting those
    would be dead work and would break the coverage check. None = every match (a whole model)."""
    from openvino import opset13 as ops
    from safetensors.torch import load_file
    tensors = load_file(adapter_file)
    pairs = {mod: {ab: tensors[k].float().numpy() for ab, k in side.items()}
             for mod, side in lora_modules(tensors, adapter_file).items()}
    grafted = []
    for op in list(model.get_ordered_ops()):
        if op.get_type_name() != "MatMul":
            continue
        m = re.search(r"__module\.(?:model\.)*(?:language_model\.)?((?:layers\.\d+\.[\w.]+)|lm_head)/(?:aten|ov_ext)::linear/MatMul$",
                      op.get_friendly_name())
        if not m or m.group(1) not in pairs:
            continue
        if m.group(1) == "lm_head":
            if not lm_head:
                continue
        elif layers is not None and int(m.group(1).split(".")[1]) not in layers:
            continue
        a, b = pairs[m.group(1)]["A"], pairs[m.group(1)]["B"]
        x = op.input_value(0)
        xa = ops.matmul(x, ops.constant(np.ascontiguousarray(a.T, dtype=np.float32)), False, False)
        xab = ops.matmul(xa, ops.constant(np.ascontiguousarray(b.T * scale, dtype=np.float32)), False, False)
        consumers = list(op.output(0).get_target_inputs())
        added = ops.add(xab, op.output(0))
        for c in consumers:
            c.replace_source_output(added.output(0))
        grafted.append(m.group(1))
    model.validate_nodes_and_infer_types()
    return grafted


def add_final_hidden_output(model):
    """Expose the lm_head MatMul's input (= final-norm hidden, all positions) as output 1 `final_hidden`."""
    from openvino import opset13 as ops
    mm = next((op for op in model.get_ops()
               if op.get_type_name() == "MatMul" and op.get_friendly_name().endswith("lm_head/ov_ext::linear/MatMul")),
              None)
    if mm is None:
        raise ValueError("no MatMul named '...lm_head/ov_ext::linear/MatMul' (is this the last stage?)")
    if len(model.outputs) != 1:
        raise ValueError(f"expected 1 output (logits) before adding final_hidden, got {len(model.outputs)}")
    res = ops.result(mm.input_value(0))
    res.output(0).set_names({"final_hidden"})
    model.add_results([res])
    model.validate_nodes_and_infer_types()
    if len(model.outputs) != 2 or "final_hidden" not in model.outputs[1].get_names():
        raise ValueError("final_hidden did not land at output index 1")
