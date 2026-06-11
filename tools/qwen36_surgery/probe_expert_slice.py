# M2' brick 1: slice ONE expert (layer 0, expert k) from the official IR's
# .bin, dequant in numpy, build an OV model from the same slices, verify
# numpy-vs-OV parity. Proves: offsets, u4 unpack, group-dequant algebra,
# graph construction.
import numpy as np
import openvino as ov
from openvino import opset13 as ops

BIN = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov\openvino_language_model.bin"
K_EXPERT = 3  # arbitrary expert index
HIDDEN, INTER = 2048, 512

# layer-0 offsets from the XML (bytes), full stacked tensors:
TENSORS = {
    "gate": dict(w=19922225,  zp=154139953, sc=156237105, rows=INTER, gdim=(32, 64)),
    "up":   dict(w=164625737, zp=298843465, sc=300940617, rows=INTER, gdim=(32, 64)),
    "down": dict(w=309329225, zp=443546953, sc=445644105, rows=HIDDEN, gdim=(8, 64)),
}

def read(off, nbytes):
    with open(BIN, "rb") as f:
        f.seek(off)
        return f.read(nbytes)

def unpack_u4(buf):
    b = np.frombuffer(buf, dtype=np.uint8)
    lo = b & 0x0F
    hi = b >> 4
    return np.stack([lo, hi], axis=1).reshape(-1)  # low nibble first

def slice_expert(t, k):
    rows, (g, ge) = t["rows"], t["gdim"]
    w_bytes_per_exp = rows * g * ge // 2
    zp_bytes_per_exp = rows * g // 2
    sc_bytes_per_exp = rows * g * 2
    w = unpack_u4(read(t["w"] + k * w_bytes_per_exp, w_bytes_per_exp)).reshape(rows, g, ge)
    zp = unpack_u4(read(t["zp"] + k * zp_bytes_per_exp, zp_bytes_per_exp)).reshape(rows, g, 1)
    sc = np.frombuffer(read(t["sc"] + k * sc_bytes_per_exp, sc_bytes_per_exp), dtype=np.float16).reshape(rows, g, 1)
    deq = (w.astype(np.float32) - zp.astype(np.float32)) * sc.astype(np.float32)
    return deq.reshape(rows, g * ge)  # [rows, K]

gate = slice_expert(TENSORS["gate"], K_EXPERT)   # [512, 2048]
up = slice_expert(TENSORS["up"], K_EXPERT)       # [512, 2048]
down = slice_expert(TENSORS["down"], K_EXPERT)   # [2048, 512]
print("slices:", gate.shape, up.shape, down.shape,
      "ranges:", float(gate.min()), float(gate.max()), flush=True)

x = (np.random.rand(1, HIDDEN).astype(np.float32) - 0.5) * 0.1

# numpy reference: down( silu(gate@x) * (up@x) )
def silu(v): return v / (1.0 + np.exp(-v))
h = silu(x @ gate.T) * (x @ up.T)        # [1, 512]
y_ref = h @ down.T                        # [1, 2048]

# OV model from the SAME dequantized slices
xp = ops.parameter([1, HIDDEN], np.float32, name="x")
g_c = ops.constant(gate.T.copy()); u_c = ops.constant(up.T.copy()); d_c = ops.constant(down.T.copy())
gx = ops.matmul(xp, g_c, False, False)
ux = ops.matmul(xp, u_c, False, False)
hv = ops.multiply(ops.multiply(gx, ops.sigmoid(gx)), ux)
yv = ops.matmul(hv, d_c, False, False)
model = ov.Model([yv], [xp], "expert_l0_k")
req = ov.Core().compile_model(model, "CPU").create_infer_request()
y_ov = req.infer({"x": x})[0]

diff = np.abs(y_ov - y_ref)
rel = diff.max() / (np.abs(y_ref).max() + 1e-9)
print(f"PARITY max_abs_diff={diff.max():.3e} max_rel={rel:.3e} ref_norm={np.abs(y_ref).max():.3e}", flush=True)
print("EXPERT_SLICE_PARITY_OK" if rel < 1e-3 else "EXPERT_SLICE_PARITY_FAIL", flush=True)
