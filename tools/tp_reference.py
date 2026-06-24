"""First-principles Megatron 2-way tensor-parallel reference for Llama-3.2-1B.

Validates:
  (A) our hand-rolled SDPA forward (tp=1) reproduces HF greedy generation token-for-token
  (B) the TP-sliced forward (tp=2, partials summed = all-reduce) == the monolithic forward

If both hold, the column/row-parallel slicing math is correct and ready to port to the
OpenVINO exporter (export_shards.py: cached_layer_forward_sdpa) and the TP engine.
"""
import sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\tatef\.cache\cascadia\models\unsloth--Llama-3.2-1B-Instruct"
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NGEN = int(sys.argv[3]) if len(sys.argv) > 3 else 20

torch.set_grad_enabled(False)
print(f"loading {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
cfg = model.config
H, NH, NKV = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
HD = getattr(cfg, "head_dim", None) or (H // NH)
INT = cfg.intermediate_size
L = cfg.num_hidden_layers
print(f"H={H} NH={NH} NKV={NKV} HD={HD} INT={INT} L={L} vocab={cfg.vocab_size}")
for nm, val, t in [("NH", NH, TP), ("NKV", NKV, TP), ("INT", INT, TP)]:
    assert val % t == 0, f"{nm}={val} not divisible by TP={t}"

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)   # [1,seq,HD] -> [1,1,seq,HD]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

def get_cos_sin(m, hidden, pos):
    out = m.rotary_emb(hidden, pos)
    # HF returns (cos, sin), each [1, seq, HD]
    return out[0], out[1]

def my_forward(model, input_ids, tp, capture=None):
    m = model.model
    nh_r, nkv_r, int_r = NH // tp, NKV // tp, INT // tp
    ng = nh_r // nkv_r
    seq = input_ids.shape[1]
    pos = torch.arange(seq).unsqueeze(0)
    hidden = m.embed_tokens(input_ids)
    cos, sin = get_cos_sin(m, hidden, pos)
    for li, layer in enumerate(m.layers):
        residual = hidden
        x = layer.input_layernorm(hidden)
        attn_partial = torch.zeros_like(hidden)
        for r in range(tp):
            qs = slice(r * nh_r * HD, (r + 1) * nh_r * HD)
            kvs = slice(r * nkv_r * HD, (r + 1) * nkv_r * HD)
            q = F.linear(x, layer.self_attn.q_proj.weight[qs]).view(1, seq, nh_r, HD).transpose(1, 2)
            k = F.linear(x, layer.self_attn.k_proj.weight[kvs]).view(1, seq, nkv_r, HD).transpose(1, 2)
            v = F.linear(x, layer.self_attn.v_proj.weight[kvs]).view(1, seq, nkv_r, HD).transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)
            k = k.repeat_interleave(ng, dim=1); v = v.repeat_interleave(ng, dim=1)
            ao = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ao = ao.transpose(1, 2).reshape(1, seq, nh_r * HD)
            attn_partial = attn_partial + F.linear(ao, layer.self_attn.o_proj.weight[:, qs])
        hidden = residual + attn_partial           # all-reduce(attn) then residual
        residual = hidden
        x = layer.post_attention_layernorm(hidden)
        mlp_partial = torch.zeros_like(hidden)
        for r in range(tp):
            ms = slice(r * int_r, (r + 1) * int_r)
            g = F.linear(x, layer.mlp.gate_proj.weight[ms])
            u = F.linear(x, layer.mlp.up_proj.weight[ms])
            mlp_partial = mlp_partial + F.linear(F.silu(g) * u, layer.mlp.down_proj.weight[:, ms])
        hidden = residual + mlp_partial            # all-reduce(mlp) then residual
        if capture is not None:
            capture.append(hidden.clone())
    hidden = m.norm(hidden)
    return model.lm_head(hidden)

# ---- per-layer numeric check: tp=1 vs tp=TP on a fixed prompt ----
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids
cap1, capN = [], []
lg1 = my_forward(model, ids, 1, cap1)
lgN = my_forward(model, ids, TP, capN)
maxdiff = max((a - b).abs().max().item() for a, b in zip(cap1, capN))
print(f"[B] per-layer max|hidden(tp=1)-hidden(tp={TP})| = {maxdiff:.3e}  (expect ~1e-4 fp32 noise)")
print(f"[B] final logits max diff = {(lg1 - lgN).abs().max().item():.3e}")

# ---- (A) my tp=1 forward vs HF model, greedy generation ----
def greedy(fwd, ids, n):
    out = ids.clone()
    for _ in range(n):
        nxt = fwd(out)[:, -1, :].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
    return out[0, ids.shape[1]:].tolist()

hf = model.generate(ids, max_new_tokens=NGEN, do_sample=False)[0, ids.shape[1]:].tolist()
mine1 = greedy(lambda x: my_forward(model, x, 1), ids, NGEN)
mineN = greedy(lambda x: my_forward(model, x, TP), ids, NGEN)
print(f"[A] HF greedy:      {hf}")
print(f"[A] my tp=1 greedy: {mine1}  match_HF={mine1 == hf}")
print(f"[B] my tp={TP} greedy: {mineN}  match_tp1={mineN == mine1}")
print(f"TEXT tp={TP}: {tok.decode(mineN)!r}")
ok = (maxdiff < 1e-2) and (mine1 == hf) and (mineN == mine1)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
