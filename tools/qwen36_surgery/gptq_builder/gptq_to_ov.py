#!/usr/bin/env python
"""Build an OpenVINO VLM-layout IR for Qwen3.5-MoE from a GPTQ-int4 checkpoint WITHOUT materializing a float model.

Inputs
  --template  fp16 IR exported by optimum-intel from the same config with tiny experts (template_export.py):
              provides the exact graph structure; every constant gets replaced here.
  --ckpt      GPTQ-Int4 checkpoint dir (safetensors + config.json + tokenizer).
  --out       output IR dir (openvino_language_model.xml/.bin, text_embeddings, vision IRs, tokenizer IRs, configs).

Expert weights: GPTQ (sym, group 128, zp=8, trivial g_idx) nibbles are repacked into Intel's layout
  u4 [E, out, groups, 128]  ->  Convert(f16) -> Subtract(zp u4=8 -> f16) -> Multiply(scales f16 [E,out,groups,1]) -> Reshape [E,out,in]
exactly the decompression pattern of the official OpenVINO/Qwen3.5-*-int4-ov IRs. No re-quantization.
Non-expert weights (attention, linear-attn, norms, router, shared expert, embeddings, lm_head, vision) are copied bf16->f16 1:1 and,
with --compress int4|int8, compressed afterwards by nncf (per Intel's recipe) while the already-integer expert constants are left alone.
"""
import argparse, json, os, re, shutil, struct, sys, time
import numpy as np
import openvino as ov
from openvino import opset13 as ops

# ----------------------------------------------------------------------------- safetensors (no torch)
DT = {"F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64, "U8": np.uint8, "I8": np.int8}


class Checkpoint:
    def __init__(self, d):
        self.d = d
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wmap = json.load(open(idx))["weight_map"]
        else:
            self.wmap = None
        self.hdr = {}
        self.mm = {}

    def _file(self, key):
        fn = self.wmap[key] if self.wmap else "model.safetensors"
        if fn not in self.hdr:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.hdr[fn] = (json.loads(f.read(n)), 8 + n)
            self.mm[fn] = np.memmap(os.path.join(self.d, fn), dtype=np.uint8, mode="r")
        return fn

    def has(self, key):
        return key in self.wmap if self.wmap else False

    def raw(self, key):
        fn = self._file(key)
        h, base = self.hdr[fn]
        e = h[key]
        a, b = e["data_offsets"]
        arr = np.frombuffer(self.mm[fn][base + a: base + b], dtype=DT[e["dtype"]]).reshape(e["shape"])
        return arr, e["dtype"]

    def f16(self, key):
        arr, dt = self.raw(key)
        if dt == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        if dt == "F32":
            return arr.astype(np.float16)
        if dt == "F16":
            return np.ascontiguousarray(arr)
        raise ValueError(f"{key}: unexpected dtype {dt}")

    def f32(self, key):
        arr, dt = self.raw(key)
        if dt == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)


# ----------------------------------------------------------------------------- GPTQ -> OV u4 repack
def unpack_u4_words(q):
    """q: uint32 [..., R, C] packed along R (8 values per word, low nibble first) -> uint8 [..., 8R, C]."""
    sh = q.shape
    out = np.empty(sh[:-2] + (sh[-2] * 8, sh[-1]), dtype=np.uint8)
    for j in range(8):
        out[..., j::8, :] = ((q >> (4 * j)) & 0xF).astype(np.uint8)
    return out


def pack_u4_pairs(a):
    """a: uint8 nibbles [..., N] with N even -> uint8 [..., N/2], low nibble = first element (OpenVINO u4 order)."""
    return (a[..., 0::2] | (a[..., 1::2] << 4)).astype(np.uint8)


def repack_expert_stack(qweights, scales, group=128):
    """qweights: list of E uint32 arrays [in/8, out] (GPTQ); scales: list of E f16 [in/group, out].
    Returns (u4_bytes [E,out,G,64], scales_f16 [E,out,G,1], shape (E,out,in))."""
    E = len(qweights)
    q = np.stack(qweights)                      # [E, in/8, out] uint32
    nib = unpack_u4_words(q)                    # [E, in, out] uint8
    nib = np.ascontiguousarray(nib.transpose(0, 2, 1))  # [E, out, in]
    E_, out, inn = nib.shape
    G = inn // group
    nib = nib.reshape(E_, out, G, group)
    packed = pack_u4_pairs(nib)                 # [E, out, G, group/2]
    s = np.stack(scales).astype(np.float16)     # [E, G, out]
    s = np.ascontiguousarray(s.transpose(0, 2, 1))[..., None]  # [E, out, G, 1]
    return packed, s, (E_, out, inn)


_KEEP = []  # numpy buffers backing u4 constants must outlive the model (ov.Tensor over packed bytes may share memory)


def u4_constant(packed_bytes, shape, name):
    arr = np.ascontiguousarray(packed_bytes)
    _KEEP.append(arr)  # the Constant references this buffer; it must outlive the model
    t = ov.Tensor(arr, ov.Shape(list(shape)), ov.Type.u4)
    c = ov.op.Constant(t, shared_memory=True)
    c.set_friendly_name(name)
    return c


def decompression_chain(packed, scales, shape, name, zp=8):
    E, out, inn = shape
    G = scales.shape[2]
    w = u4_constant(packed, (E, out, G, 128), name)
    zpb = np.full((E * out * G + 1) // 2, (zp << 4) | zp, dtype=np.uint8)
    z = u4_constant(zpb, (E, out, G, 1), name + "/zero_point")
    scales = np.ascontiguousarray(scales)
    _KEEP.append(scales)
    s = ov.op.Constant(scales, shared_memory=True)
    s.set_friendly_name(name + "/scale")
    cw = ops.convert(w, ov.Type.f16)
    cz = ops.convert(z, ov.Type.f16)
    sub = ops.subtract(cw, cz)
    sub.set_friendly_name(name + "/zero_point/subtract")
    mul = ops.multiply(sub, s)
    mul.set_friendly_name(name + "/fq_weights_1")
    rs = ops.reshape(mul, ops.constant(np.array([E, out, inn], dtype=np.int64)), special_zero=False)
    rs.set_friendly_name(name + "/fq_weights_1/reshape")
    return rs, [w, z, s]


# ----------------------------------------------------------------------------- name mapping
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def hf_key_candidates(friendly):
    n = friendly
    if n.endswith("_compressed"):
        n = n[: -len("_compressed")]
    for p in ("self.model.", "__module.", "self."):
        if n.startswith(p):
            n = n[len(p):]
    cands = [n]
    if n.startswith("model.model."):
        cands.append(n[len("model."):])
    if n.startswith("model.lm_head."):
        cands.append(n[len("model."):])
    return cands


def classify(friendly):
    """-> ('gate'|'up'|'down', layer) for expert constants, else None."""
    m = LAYER_RE.search(friendly)
    if m is None:
        return None
    L = int(m.group(1))
    if "VariadicSplit.0" in friendly:
        return ("gate", L)
    if "VariadicSplit.1" in friendly:
        return ("up", L)
    if "experts.down_proj" in friendly:
        return ("down", L)
    if "experts.gate_up_proj" in friendly:
        return ("gate_up", L)
    return None


CONV_RE = re.compile(r"layers\.(\d+)\.linear_attn/aten::_convolution/Reshape")
EXP_RE = re.compile(r"layers\.(\d+)\.linear_attn/aten::exp/Exp")


def consumer_names(c, depth=2):
    out = []
    frontier = [c.output(0)]
    for _ in range(depth):
        nxt = []
        for o in frontier:
            for t in o.get_target_inputs():
                n = t.get_node(); out.append((n.get_type_name(), n.get_friendly_name())); nxt.append(n.output(0))
        frontier = nxt
    return out


def derived_value(c, name, shape, ck, text_prefix, et):
    """Constants the fp16 export folded/reshaped: returns a numpy array of `shape` or None."""
    conv = lambda a: (a.astype(np.float16) if et == ov.Type.f16 else a.astype(np.float32))
    m = CONV_RE.search(name)
    if m:
        w = ck.f32(f"{text_prefix}layers.{m.group(1)}.linear_attn.conv1d.weight")
        return conv(w.reshape(shape))
    m = EXP_RE.search(name)
    if m:
        a = ck.f32(f"{text_prefix}layers.{m.group(1)}.linear_attn.A_log")
        return conv(np.exp(a).reshape(shape))
    cons = consumer_names(c)
    for typ, cn in cons:
        lm = LAYER_RE.search(cn)
        if lm is None:
            continue
        L = lm.group(1)
        if "linear_attn/aten::add" in cn and typ == "Add":
            b = ck.f32(f"{text_prefix}layers.{L}.linear_attn.dt_bias")
            if b.size == int(np.prod(shape)):
                return conv(b.reshape(shape))
        if typ == "Multiply" and "/aten::mul/" in cn:
            # consumer friendly name: __module.model.model.language_model.layers.L.<module path>/aten::mul/Multiply*
            mod = cn.split("/aten::mul/")[0]
            mod = mod[mod.index("layers."):] if "layers." in mod else mod
            for norm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm", "linear_attn.norm"):
                if mod.endswith(norm):
                    w = ck.f32(f"{text_prefix}layers.{L}.{norm}.weight")
                    if w.size != int(np.prod(shape)):
                        break
                    gated = norm == "linear_attn.norm"  # RMSNormGated: weight * x ; the others: (1 + weight) * x
                    return conv((w if gated else 1.0 + w).reshape(shape))
    # final norm (no layer index): consumer __module.model.model.language_model.norm/aten::mul/Multiply
    for typ, cn in cons:
        if typ == "Multiply" and "language_model.norm/aten::mul" in cn:
            w = ck.f32(f"{text_prefix}norm.weight")
            if w.size == int(np.prod(shape)):
                return conv((1.0 + w).reshape(shape))
    return None


def replace_output(old_node, new_output):
    for tgt in list(old_node.output(0).get_target_inputs()):
        tgt.replace_source_output(new_output)


def chain_output_3d(node):
    """From an expert weight Constant walk single-consumer chains (Convert/Subtract/Multiply/Reshape) to the
    first output whose shape is the 3-D float weight [E, out, in]; that is the port to rewire."""
    cur = node.output(0)
    for _ in range(8):
        ps = cur.get_partial_shape()
        if cur.get_element_type() in (ov.Type.f16, ov.Type.f32, ov.Type.bf16) and ps.rank.get_length() == 3:
            return cur
        tgts = list(cur.get_target_inputs())
        if len(tgts) != 1:
            break
        cur = tgts[0].get_node().output(0)
    raise RuntimeError(f"could not find 3-D float output below {node.get_friendly_name()}")


def rebuild_language_model(tpl_xml, ck, text_prefix, E, inter, hidden, layers, compress, keep, skip_nonexpert=False, dry_map=False):
    core = ov.Core()
    model = core.read_model(tpl_xml)
    consts = [o for o in model.get_ops() if o.get_type_name() == "Constant"]
    print(f"[lm] template ops={len(model.get_ops())} constants={len(consts)}", flush=True)
    expert_nodes = {}
    mapped = unmapped = 0
    unmapped_list = []
    for c in consts:
        et = c.get_output_element_type(0)
        name = c.get_friendly_name()
        cls = classify(name)
        if cls is not None and ("zero_point" not in name) and ("/scale" not in name):
            expert_nodes.setdefault(cls[1], {})[cls[0]] = c
            continue
        if cls is not None:
            continue  # expert zp/scale constants: replaced together with the weight
        if et not in (ov.Type.f16, ov.Type.f32, ov.Type.bf16):
            continue
        if skip_nonexpert:
            continue
        shape = tuple(c.get_output_shape(0)) if c.get_output_partial_shape(0).is_static else None
        if shape is None or int(np.prod(shape)) < 16:
            continue
        key = next((k for k in hf_key_candidates(name) if ck.has(k)), None)
        if key is not None and tuple((ck.raw(key)[0]).shape) != shape:
            key = None  # same name but folded/reshaped: treat as derived
        if key is None:
            arr = derived_value(c, name, shape, ck, text_prefix, et)
            if arr is None:
                unmapped += 1
                d = c.get_data().astype(np.float32)
                unmapped_list.append((name, shape, str(et), f"min={d.min():.3g} max={d.max():.3g} mean={d.mean():.3g}", [f"{t}:{n[-60:]}" for t, n in consumer_names(c)[:2]]))
                continue
        else:
            arr = ck.f16(key) if et == ov.Type.f16 else ck.f32(key)
        if tuple(arr.shape) != shape:
            raise RuntimeError(f"shape mismatch {name}: template {shape} vs ckpt {arr.shape} ({key})")
        arr = np.ascontiguousarray(arr)
        new = ov.op.Constant(arr, shared_memory=True)
        new.set_friendly_name(name)
        replace_output(c, new.output(0))
        keep.append(arr)
        mapped += 1
    print(f"[lm] non-expert constants mapped={mapped} unmapped={unmapped}", flush=True)
    for u in unmapped_list[:60]:
        print("   UNMAPPED:", u, flush=True)
    if dry_map:
        return None
    if layers is None:
        layers = sorted(expert_nodes)
    print(f"[lm] expert layers in template: {len(expert_nodes)} -> rebuilding {len(layers)}", flush=True)
    for L in layers:
        t0 = time.time()
        nodes = expert_nodes[L]
        pre = f"{text_prefix}layers.{L}.mlp.experts."
        gate_q, gate_s, up_q, up_s, down_q, down_s = [], [], [], [], [], []
        for e in range(E):
            gate_q.append(ck.raw(f"{pre}{e}.gate_proj.qweight")[0].view(np.uint32))
            gate_s.append(ck.raw(f"{pre}{e}.gate_proj.scales")[0])
            up_q.append(ck.raw(f"{pre}{e}.up_proj.qweight")[0].view(np.uint32))
            up_s.append(ck.raw(f"{pre}{e}.up_proj.scales")[0])
            down_q.append(ck.raw(f"{pre}{e}.down_proj.qweight")[0].view(np.uint32))
            down_s.append(ck.raw(f"{pre}{e}.down_proj.scales")[0])
        if "gate_up" in nodes:
            raise RuntimeError("template keeps fused gate_up constant; split handling not implemented")
        for kind, qs, ss in (("gate", gate_q, gate_s), ("up", up_q, up_s), ("down", down_q, down_s)):
            packed, scales, shape = repack_expert_stack(qs, ss)
            exp_out = inter if kind != "down" else hidden
            exp_in = hidden if kind != "down" else inter
            if shape != (E, exp_out, exp_in):
                raise RuntimeError(f"layer {L} {kind}: repacked shape {shape} != expected {(E, exp_out, exp_in)}")
            node = nodes[kind]
            port = chain_output_3d(node)
            if tuple(d.get_length() for d in port.get_partial_shape()) != shape:
                print(f"   note: template expert port shape {port.get_partial_shape()} -> {shape}", flush=True)
            chain, cs = decompression_chain(packed, scales, shape, node.get_friendly_name().replace("/prim::ListUnpack/VariadicSplit.0", ".experts.gate_proj").replace("/prim::ListUnpack/VariadicSplit.1", ".experts.up_proj"))
            for tgt in list(port.get_target_inputs()):
                tgt.replace_source_output(chain.output(0))
        print(f"[lm] layer {L}: experts rebuilt in {time.time()-t0:.1f}s", flush=True)
    model.validate_nodes_and_infer_types()
    if compress != "none":
        import nncf
        mode = nncf.CompressWeightsMode.INT4_ASYM if compress == "int4" else nncf.CompressWeightsMode.INT8_ASYM
        kw = dict(mode=mode, ratio=1.0, all_layers=True)
        if compress == "int4":
            kw["group_size"] = 128
        t0 = time.time()
        model = nncf.compress_weights(model, **kw)
        print(f"[lm] nncf {compress} compression of remaining float weights done in {time.time()-t0:.0f}s", flush=True)
    return model


def aux_candidates(name, tag, text_prefix):
    base = name[:-len("_compressed")] if name.endswith("_compressed") else name
    for pfx in ("self.model.", "__module.", "self."):
        if base.startswith(pfx):
            base = base[len(pfx):]
    c = list(hf_key_candidates(name)) + [base]
    if tag == "emb":
        c += [f"{text_prefix}embed_tokens.{base}", f"{text_prefix}embed_tokens.weight"]
    elif tag == "vis":
        c += [f"model.visual.patch_embed.{base}", f"model.visual.{base}"]
    elif tag == "vpos":
        c += [f"model.visual.pos_embed.{base}", f"model.visual.{base}"]
    else:
        c += [f"model.visual.{base}"]
    return c


VIS_MOD_RE = re.compile(r"__module\.([A-Za-z0-9_.]+)/aten::([a-z_]+)/([A-Za-z]+)")


def derived_vision_value(c, shape, ck, et, tag):
    """Anonymous vision constants: linear biases, LayerNorm weight/bias, patch-embed conv bias — resolved from the consumer's module path."""
    conv = lambda a: (a.astype(np.float16) if et == ov.Type.f16 else a.astype(np.float32))
    for typ, cn in consumer_names(c):
        m = VIS_MOD_RE.search(cn)
        if not m:
            continue
        mod, aten, op = m.group(1), m.group(2), m.group(3)
        root = {"vis": "model.visual.patch_embed.", "vpos": "model.visual.pos_embed.", "merger": "model.visual."}.get(tag, "model.visual.")
        if aten == "linear" and op.startswith("Add"):
            key = f"{root}{mod}.bias"
        elif aten == "layer_norm" and op.startswith("Multiply"):
            key = f"{root}{mod}.weight"
        elif aten == "layer_norm" and op.startswith("Add"):
            key = f"{root}{mod}.bias"
        elif aten == "_convolution" and (op.startswith("Add") or op.startswith("Reshape")):
            key = f"{root}{mod}.bias"
        else:
            continue
        if ck.has(key):
            a = ck.f32(key)
            if a.size == int(np.prod(shape)):
                return conv(a.reshape(shape))
    return None


def rebuild_simple(tpl_xml, ck, keep, tag, text_prefix="model.language_model."):
    """1:1 constant replacement for the text-embeddings and vision IRs."""
    core = ov.Core()
    model = core.read_model(tpl_xml)
    mapped = unmapped = 0
    for c in [o for o in model.get_ops() if o.get_type_name() == "Constant"]:
        et = c.get_output_element_type(0)
        if et not in (ov.Type.f16, ov.Type.f32):
            continue
        if not c.get_output_partial_shape(0).is_static:
            continue
        shape = tuple(c.get_output_shape(0))
        if int(np.prod(shape)) < 16:
            continue
        name = c.get_friendly_name()
        key = next((k for k in aux_candidates(name, tag, text_prefix) if ck.has(k)), None)
        if key is not None and ck.raw(key)[0].size != int(np.prod(shape)):
            key = None
        if key is None:
            arr = derived_vision_value(c, shape, ck, et, tag) if tag != "emb" else None
            if arr is None:
                unmapped += 1
                if unmapped <= 12:
                    print(f"   [{tag}] UNMAPPED: {name} {shape} {et} consumers={[f'{t}:{n[-50:]}' for t, n in consumer_names(c)[:2]]}", flush=True)
                continue
        else:
            arr = ck.f16(key) if et == ov.Type.f16 else ck.f32(key)
        if tuple(arr.shape) != shape:
            if arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape)
            else:
                raise RuntimeError(f"[{tag}] shape mismatch {name}: {shape} vs {arr.shape}")
        arr = np.ascontiguousarray(arr)
        new = ov.op.Constant(arr, shared_memory=True)
        new.set_friendly_name(name)
        replace_output(c, new.output(0))
        keep.append(arr)
        mapped += 1
    model.validate_nodes_and_infer_types()
    print(f"[{tag}] mapped={mapped} unmapped={unmapped}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compress", choices=["none", "int8", "int4"], default="int4", help="nncf mode for the non-expert weights")
    ap.add_argument("--layers", default=None, help="comma list of layer indices to rebuild (debug); default all")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-nonexpert", action="store_true", help="keep the template's non-expert constants (use when the template is an official IR); experts still rebuilt from GPTQ")
    ap.add_argument("--only-aux", action="store_true", help="skip the language model (already built in --out); rebuild embeddings/vision/tokenizer only")
    ap.add_argument("--dry-map", action="store_true", help="only report the non-expert constant mapping and exit")
    ap.add_argument("--skip-aux", action="store_true", help="do not rebuild embeddings/vision/tokenizer, just copy them from the template dir")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = Checkpoint(a.ckpt)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    E, inter, hidden, nl = tc["num_experts"], tc["moe_intermediate_size"], tc["hidden_size"], tc["num_hidden_layers"]
    text_prefix = "model.language_model." if ck.has("model.language_model.layers.0.input_layernorm.weight") else "model."
    print(f"[cfg] E={E} inter={inter} hidden={hidden} layers={nl} prefix={text_prefix}", flush=True)
    keep = []
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None

    t0 = time.time()
    if a.only_aux:
        lm = None
    else:
      lm = rebuild_language_model(os.path.join(a.template, "openvino_language_model.xml"), ck, text_prefix, E, inter, hidden, layers, "none" if a.skip_nonexpert else a.compress, keep, a.skip_nonexpert, a.dry_map)
      if a.dry_map:
          return
      ov.save_model(lm, os.path.join(a.out, "openvino_language_model.xml"), compress_to_fp16=False)
      print(f"[lm] saved in total {time.time()-t0:.0f}s", flush=True)
      del lm
      keep.clear()

    if a.skip_aux:
        for f in os.listdir(a.template):
            if f.startswith("openvino_") and not f.startswith("openvino_language_model") and f.endswith((".xml", ".bin")):
                shutil.copy2(os.path.join(a.template, f), a.out)
        for aux in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
            src = os.path.join(a.template, aux)
            if os.path.exists(src):
                shutil.copy2(src, a.out)
        print("[aux] copied from template:", sorted(os.listdir(a.out)), flush=True)
        return
    for sub, tag in (("openvino_text_embeddings_model.xml", "emb"),) + (() if a.skip_vision else (("openvino_vision_embeddings_model.xml", "vis"), ("openvino_vision_embeddings_pos_model.xml", "vpos"), ("openvino_vision_embeddings_merger_model.xml", "merger"))):
        p = os.path.join(a.template, sub)
        if not os.path.exists(p):
            print(f"[{tag}] template {sub} missing, skipped", flush=True)
            continue
        m = rebuild_simple(p, ck, keep, tag, text_prefix)
        if tag == "emb" and a.compress != "none":
            import nncf
            m = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(m, os.path.join(a.out, sub), compress_to_fp16=False)
        keep.clear()

    # configs + tokenizer
    cfg2 = json.loads(json.dumps(cfg))
    cfg2.pop("quantization_config", None)
    cfg2.get("text_config", cfg2).pop("quantization_config", None)
    cfg2.get("text_config", cfg2)["mtp_num_hidden_layers"] = 0
    json.dump(cfg2, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for aux in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
        src = os.path.join(a.ckpt, aux)
        if os.path.exists(src):
            shutil.copy2(src, a.out)
    try:
        from transformers import AutoTokenizer
        from openvino_tokenizers import convert_tokenizer
        tok = AutoTokenizer.from_pretrained(a.ckpt)
        ov_tok, ov_detok = convert_tokenizer(tok, with_detokenizer=True)
        ov.save_model(ov_tok, os.path.join(a.out, "openvino_tokenizer.xml"))
        ov.save_model(ov_detok, os.path.join(a.out, "openvino_detokenizer.xml"))
        print("[tok] tokenizer IRs written", flush=True)
    except Exception as e:
        print(f"[tok] tokenizer conversion FAILED: {type(e).__name__}: {e}", flush=True)
    print("[done]", sorted(os.listdir(a.out)), flush=True)


if __name__ == "__main__":
    main()
