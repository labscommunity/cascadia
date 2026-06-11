# M2' brick 2: semantic parity — reconstruct layer-0's FULL MoE block
# (router + top-8 renorm + 8 routed experts + shared expert) from byte
# slices, compare against the real model's MoE output captured via
# add_outputs on the official IR. Conventions read from the graph:
# softmax->top8->renorm; silu; shared = silu-MLP * sigmoid(gate).
import re
import numpy as np
import openvino as ov

DIR = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov"
XML, BIN = DIR + r"\openvino_language_model.xml", DIR + r"\openvino_language_model.bin"
HIDDEN, INTER, TOPK = 2048, 512, 8
L = "layers.0.mlp"

xml = open(XML).read()

def const_info(name_sub, shape_prefix):
    pat = re.compile(
        r'<layer id="\d+" name="([^"]*' + re.escape(name_sub) + r'[^"]*)" type="Const"[^>]*>\s*<data ([^>]*shape="'
        + re.escape(shape_prefix) + r'[^"]*"[^>]*)/>')
    out = []
    for name, d in pat.findall(xml):
        if L in name:
            et = re.search(r'element_type="([^"]*)"', d).group(1)
            off = int(re.search(r'offset="(\d+)"', d).group(1))
            size = int(re.search(r'size="(\d+)"', d).group(1))
            shape = [int(x) for x in re.search(r'shape="([^"]*)"', d).group(1).split(",")]
            out.append((name, et, off, size, shape))
    return out

def read(off, n):
    with open(BIN, "rb") as f:
        f.seek(off); return f.read(n)

def unpack_u4(buf):
    b = np.frombuffer(buf, dtype=np.uint8)
    return np.stack([b & 0x0F, b >> 4], axis=1).reshape(-1)

def dequant(w_off, zp_off, sc_off, rows, g, ge, k=None):
    # k = expert index into leading 256 axis (None = whole tensor is 3D)
    wb, zb, sb = rows*g*ge//2, rows*g//2, rows*g*2
    base_w, base_z, base_s = (w_off + k*wb, zp_off + k*zb, sc_off + k*sb) if k is not None else (w_off, zp_off, sc_off)
    w = unpack_u4(read(base_w, wb)).reshape(rows, g, ge).astype(np.float32)
    zp = unpack_u4(read(base_z, zb)).reshape(rows, g, 1).astype(np.float32)
    sc = np.frombuffer(read(base_s, sb), dtype=np.float16).reshape(rows, g, 1).astype(np.float32)
    return ((w - zp) * sc).reshape(rows, g*ge)

def triple(name_sub, shape_prefix):
    rows = {}
    for name, et, off, size, shape in const_info(name_sub, shape_prefix):
        key = "zp" if "zero_point" in name else ("sc" if "scale" in name else "w")
        rows[key] = (off, shape)
    return rows

# --- gather layer-0 constants ---
gate_t = triple("ListUnpack/VariadicSplit.0", "256, 512")
up_t   = triple("ListUnpack/VariadicSplit.1", "256, 512")
down_t = triple("experts.down_proj", "256, 2048")
router_t = triple("mlp.gate", "256, 32")           # router [256, 2048] quant
sh_gate_t = triple("shared_expert.gate_proj", "512, ")
sh_up_t   = triple("shared_expert.up_proj", "512, ")
sh_down_t = triple("shared_expert.down_proj", "2048, 8")
print("consts found:", {k: bool(v) for k, v in
      dict(g=gate_t, u=up_t, d=down_t, r=router_t, sg=sh_gate_t, su=sh_up_t, sd=sh_down_t).items()}, flush=True)

# shared_expert_gate (the sigmoid gate): small linear [1,2048] — maybe f16/f32 const
seg = const_info("shared_expert_gate", "1, ")
print("shared_expert_gate consts:", [(n[-60:], et, sh) for n, et, off, sz, sh in seg], flush=True)

router = dequant(router_t["w"][0], router_t["zp"][0], router_t["sc"][0], 256, 32, 64)         # [256,2048]
sh_gate = dequant(sh_gate_t["w"][0], sh_gate_t["zp"][0], sh_gate_t["sc"][0], INTER, sh_gate_t["w"][1][1], sh_gate_t["w"][1][2])
sh_up   = dequant(sh_up_t["w"][0], sh_up_t["zp"][0], sh_up_t["sc"][0], INTER, sh_up_t["w"][1][1], sh_up_t["w"][1][2])
sh_down = dequant(sh_down_t["w"][0], sh_down_t["zp"][0], sh_down_t["sc"][0], HIDDEN, 8, 64)

# --- capture real MoE in/out ---
core = ov.Core()
model = core.read_model(XML)
taps = {}
for op in model.get_ops():
    n = op.get_friendly_name()
    if L not in n:
        continue
    if n.endswith("aten::add/Add"):
        taps["moe_out"] = op.output(0)
    elif n.endswith(f"{L}.gate/aten::linear/MatMul"):
        taps["moe_in"] = op.input_value(0)
        taps["router_logits"] = op.output(0)
    elif n.endswith("aten::softmax/Softmax"):
        taps["softmax"] = op.output(0)
    elif n.endswith("aten::topk/TopK"):
        taps["topk_v"] = op.output(0)
        taps["topk_i"] = op.output(1)
    elif n.endswith("aten::div_/Divide"):
        taps["renorm"] = op.output(0)
    elif n.endswith("aten::sum/ReduceSum") and ".gate/" not in n:
        taps["routed_sum"] = op.output(0)
    elif n.endswith("shared_expert.down_proj/ov_ext::linear/MatMul"):
        taps["shared_raw"] = op.output(0)
    elif n.endswith("aten::sigmoid/Sigmoid"):
        taps["shared_gate_sig"] = op.output(0)
order = list(taps.keys())
model.add_outputs([taps[k] for k in order])
print("inputs:", [(i.get_any_name(), str(i.get_partial_shape())) for i in model.inputs][:6], flush=True)
comp = core.compile_model(model, "CPU")
req = comp.create_infer_request()

feeds = {}
for inp in model.inputs:
    nm = inp.get_any_name()
    ps = inp.get_partial_shape()
    dims = [(d.get_length() if d.is_static else 1) for d in ps]
    et = inp.get_element_type().to_dtype()
    if "embed" in nm:
        dims = [d if i != len(dims)-1 else HIDDEN for i, d in enumerate(dims)]
        feeds[nm] = ((np.random.rand(*dims) - 0.5) * 0.05).astype(et)
    elif "attention_mask" in nm:
        feeds[nm] = np.ones(dims, dtype=et)
    else:
        feeds[nm] = np.zeros(dims, dtype=et)
    print("feed:", nm, dims, et, flush=True)
res = req.infer(feeds)
n_taps = len(order)
cap = {k: res[comp.outputs[-(n_taps - i)]] for i, k in enumerate(order)}
x = cap["moe_in"].reshape(-1, HIDDEN)[-1:].astype(np.float32)
y_real = cap["moe_out"].reshape(-1, HIDDEN)[-1:].astype(np.float32)
print("real topk_i:", cap["topk_i"].reshape(-1)[:8], flush=True)
print("real renorm w:", cap["renorm"].reshape(-1)[:8], flush=True)
print("captured x/y norms:", float(np.abs(x).max()), float(np.abs(y_real).max()), flush=True)

# --- reconstruct ---
def silu(v): return v / (1.0 + np.exp(-v))
logits = x @ router.T                                # [1,256]
p = np.exp(logits - logits.max()); p /= p.sum()      # softmax over 256
idx = np.argsort(-p[0])[:TOPK]
w8 = p[0][idx]; w8 = w8 / w8.sum()                   # renormalize top-8
y_routed = np.zeros((1, HIDDEN), dtype=np.float32)
for k, wt in zip(idx, w8):
    g = dequant(gate_t["w"][0], gate_t["zp"][0], gate_t["sc"][0], INTER, 32, 64, k=int(k))
    u = dequant(up_t["w"][0], up_t["zp"][0], up_t["sc"][0], INTER, 32, 64, k=int(k))
    d = dequant(down_t["w"][0], down_t["zp"][0], down_t["sc"][0], HIDDEN, 8, 64, k=int(k))
    y_routed += wt * ((silu(x @ g.T) * (x @ u.T)) @ d.T)
# shared expert with sigmoid gate
seg_t = triple("shared_expert_gate", "1, ")
seg_w = dequant(seg_t["w"][0], seg_t["zp"][0], seg_t["sc"][0], 1,
                seg_t["w"][1][1], seg_t["w"][1][2])  # [1, 2048]
sh_out = (silu(x @ sh_gate.T) * (x @ sh_up.T)) @ sh_down.T
gate_scalar = 1.0 / (1.0 + np.exp(-(x @ seg_w.reshape(-1, HIDDEN).T))) if seg_w is not None else 1.0
y_rec = y_routed + gate_scalar * sh_out

def cmp(name, mine, theirs):
    theirs = theirs.reshape(mine.shape).astype(np.float32)
    d = float(np.abs(mine - theirs).max()); n = float(np.abs(theirs).max()) + 1e-9
    print(f"  CMP {name:16s} max_abs={d:.3e} rel={d/n:.3e}", flush=True)

print("--- stage comparison ---", flush=True)
cmp("router_logits", logits, cap["router_logits"][..., -1, :] if cap["router_logits"].ndim > 2 else cap["router_logits"])
cmp("softmax", p, cap["softmax"][..., -1, :] if cap["softmax"].ndim > 2 else cap["softmax"])
print("  my topk idx:", idx, flush=True)
cmp("renorm_w", np.sort(w8)[::-1].copy(), np.sort(cap["renorm"].reshape(-1)[:TOPK])[::-1].copy())
cmp("routed_sum", y_routed, cap["routed_sum"].reshape(-1, HIDDEN)[-1:])
cmp("shared_raw", sh_out, cap["shared_raw"].reshape(-1, HIDDEN)[-1:])
cmp("shared_sig", np.array(gate_scalar).reshape(-1), cap["shared_gate_sig"].reshape(-1)[-1:])
diff = np.abs(y_rec - y_real)
rel = float(diff.max() / (np.abs(y_real).max() + 1e-9))
print(f"PARITY routed+shared vs real: max_abs={diff.max():.3e} max_rel={rel:.3e}", flush=True)
print("MOE_SEMANTIC_PARITY_OK" if rel < 5e-2 else "MOE_SEMANTIC_PARITY_FAIL", flush=True)
