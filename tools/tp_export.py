"""Tensor-parallel (Megatron-style) shard exporter for cascadia.

Produces, per TP rank, a set of segmented OpenVINO IR sub-models that the
`tp-runtime` engine chains with an all-reduce between segments:

  embed            input_ids   -> hidden                 (replicated, every rank)
  attn_{i}         hidden,mask,pos (+KV) -> partial_attn  (column/row-parallel; stateful KV = this rank's heads)
  mlp_{i}          hidden      -> partial_mlp             (column/row-parallel; stateless)
  head             hidden      -> logits                 (rank 0 only)

The engine runs, per token, per layer i:
  partial_a = attn_i(hidden);  hidden += all_reduce(partial_a)
  partial_m = mlp_i(hidden);   hidden += all_reduce(partial_m)
i.e. the residual adds are done by the engine in f32; each OV segment is a
clean partial producer. Megatron slicing (validated in tp_reference.py):
  q/k/v_proj, gate/up_proj : column-parallel (slice output rows by rank)
  o_proj, down_proj        : row-parallel    (slice input cols; partials summed = all-reduce)

Llama-family only (the model we target). INT4 via nncf, same as export_shards.py.
"""
import argparse, gc, json, math, os, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- rotary (copied from export_shards.py, full-rotary path used for Llama) ---
class TracedRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta=500000.0, partial_rotary_factor=1.0):
        super().__init__()
        self.head_dim = head_dim
        rotary_dim = head_dim if partial_rotary_factor >= 1.0 else int(partial_rotary_factor * head_dim)
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, target_dtype):
        bsz, seq_len = position_ids.shape
        inv = self.inv_freq[None, None, :].expand(bsz, seq_len, -1)
        freqs = position_ids[:, :, None].float() * inv
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(target_dtype), emb.sin().to(target_dtype)


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    half = cos.shape[-1] // 2
    def rot(x):
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (q * cos) + (rot(q) * sin), (k * cos) + (rot(k) * sin)


def build_causal_mask(attention_mask, seq_len, past_kv_len, dtype):
    full = past_kv_len + seq_len
    q_pos = torch.arange(seq_len).unsqueeze(-1) + past_kv_len
    k_pos = torch.arange(full).unsqueeze(0)
    causal = (k_pos <= q_pos).to(dtype)
    pad = attention_mask.unsqueeze(1).to(dtype)
    allow = causal.unsqueeze(0) * pad
    return ((1.0 - allow) * torch.finfo(dtype).min).unsqueeze(1)


def lin(out_f, in_f, weight):
    m = nn.Linear(in_f, out_f, bias=False)
    m.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
    return m


# --- segment modules ---
class EmbedSeg(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__(); self.embed_tokens = embed_tokens
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class AttnSeg(nn.Module):
    """One layer's attention for this TP rank. Outputs the row-parallel PARTIAL
    o_proj output (engine sums across ranks). Stateful: this rank's KV heads."""
    def __init__(self, layer, rank, tp, nh, nkv, hd, rope_theta):
        super().__init__()
        self.nh_r = nh // tp; self.nkv_r = nkv // tp; self.hd = hd
        self.ng = self.nh_r // self.nkv_r
        qs = slice(rank * self.nh_r * hd, (rank + 1) * self.nh_r * hd)
        kvs = slice(rank * self.nkv_r * hd, (rank + 1) * self.nkv_r * hd)
        a = layer.self_attn
        self.input_layernorm = layer.input_layernorm
        self.q_proj = lin(self.nh_r * hd, a.q_proj.in_features, a.q_proj.weight[qs])
        self.k_proj = lin(self.nkv_r * hd, a.k_proj.in_features, a.k_proj.weight[kvs])
        self.v_proj = lin(self.nkv_r * hd, a.v_proj.in_features, a.v_proj.weight[kvs])
        self.o_proj = lin(a.o_proj.out_features, self.nh_r * hd, a.o_proj.weight[:, qs])
        self.rotary = TracedRotaryEmbedding(hd, rope_theta)

    def forward(self, hidden, attention_mask, position_ids, past_k, past_v):
        bsz, seq, _ = hidden.shape
        x = self.input_layernorm(hidden)
        q = self.q_proj(x).view(bsz, seq, self.nh_r, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.nkv_r, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.nkv_r, self.hd).transpose(1, 2)
        cos, sin = self.rotary(position_ids, hidden.dtype)
        q, k = apply_rotary(q, k, cos, sin)
        k = torch.cat([past_k, k], dim=2); v = torch.cat([past_v, v], dim=2)
        past_kv_len = past_k.shape[2]
        ke = k[:, :, None, :, :].expand(bsz, self.nkv_r, self.ng, -1, self.hd).reshape(bsz, self.nh_r, -1, self.hd)
        ve = v[:, :, None, :, :].expand(bsz, self.nkv_r, self.ng, -1, self.hd).reshape(bsz, self.nh_r, -1, self.hd)
        mask = build_causal_mask(attention_mask, seq, past_kv_len, hidden.dtype)
        ao = F.scaled_dot_product_attention(q, ke, ve, attn_mask=mask, dropout_p=0.0,
                                            is_causal=False, scale=1.0 / math.sqrt(self.hd))
        ao = ao.transpose(1, 2).contiguous().reshape(bsz, seq, self.nh_r * self.hd)
        partial = self.o_proj(ao)
        return partial, k, v


class MlpSeg(nn.Module):
    def __init__(self, layer, rank, tp, intermediate):
        super().__init__()
        int_r = intermediate // tp
        ms = slice(rank * int_r, (rank + 1) * int_r)
        m = layer.mlp
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.gate_proj = lin(int_r, m.gate_proj.in_features, m.gate_proj.weight[ms])
        self.up_proj = lin(int_r, m.up_proj.in_features, m.up_proj.weight[ms])
        self.down_proj = lin(m.down_proj.out_features, int_r, m.down_proj.weight[:, ms])
    def forward(self, hidden):
        x = self.post_attention_layernorm(hidden)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadSeg(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__(); self.norm = norm; self.lm_head = lm_head
    def forward(self, hidden):
        return self.lm_head(self.norm(hidden))


# --- OV conversion helpers ---
def make_stateful_with_init(ov_model, add_reorder=True):
    import numpy as np, openvino as ov, openvino.opset13 as ops
    from openvino import PartialShape, Type, Tensor
    from openvino.op.util import Variable, VariableInfo
    beam_idx = axis0 = None
    if add_reorder:
        beam_idx = ops.parameter(PartialShape([-1]), Type.i32)
        beam_idx.set_friendly_name("beam_idx"); beam_idx.output(0).set_names({"beam_idx"})
        axis0 = ops.constant(np.array(0, dtype=np.int32))
    present_src = {}; keep_results = []
    for i, r in enumerate(ov_model.get_results()):
        if i == 0:
            keep_results.append(r); continue
        kv_idx = i - 1
        kv = "value" if kv_idx % 2 == 1 else "key"
        present_src[f"present.{kv_idx // 2}.{kv}"] = r.input_value(0)
    sinks = []; keep_params = []
    for p in ov_model.get_parameters():
        nm = p.output(0).get_any_name()
        if not nm.startswith("past_key_values."):
            keep_params.append(p); continue
        src = present_src[nm.replace("past_key_values.", "present.", 1)]
        et = p.get_element_type(); ps = p.get_partial_shape()
        dims = [(d.get_length() if d.is_static else (0 if idx >= 2 else 1)) for idx, d in enumerate(ps)]
        vi = VariableInfo(); vi.data_shape = ps; vi.data_type = et; vi.variable_id = nm
        var = Variable(vi)
        rv = ops.read_value(ops.constant(Tensor(et, dims)), var)
        feed = rv.output(0)
        if add_reorder:
            feed = ops.gather(rv.output(0), beam_idx, axis0).output(0)
        p.output(0).replace(feed)
        sinks.append(ops.assign(src.get_node(), var))
    new_params = keep_params + ([beam_idx] if add_reorder else [])
    nm2 = ov.Model(keep_results, sinks, new_params)
    nm2.validate_nodes_and_infer_types()
    return nm2


def convert_and_save(module, example_inputs, in_names, out_names, out_dir, quant, stateful):
    import openvino as ov
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs)
    ov_model = ov.convert_model(traced, example_input=example_inputs)
    del traced; gc.collect()
    for i, inp in enumerate(ov_model.inputs):
        shape = inp.partial_shape
        nm = in_names[i]
        if nm in ("input_ids", "hidden_states", "attention_mask", "position_ids"):
            if len(shape) >= 2:
                shape[1] = -1
        elif nm.startswith("past_key_values."):
            if len(shape) >= 3:
                shape[0] = 1; shape[2] = -1
        inp.node.set_partial_shape(shape)
        inp.set_names({nm})
    for i, out in enumerate(ov_model.outputs):
        out.set_names({out_names[i]})
    if stateful:
        ov_model = make_stateful_with_init(ov_model, add_reorder=True)
    if quant in ("int4", "int4_asym"):
        try:
            import nncf
            mode = nncf.CompressWeightsMode.INT4_SYM if quant == "int4" else nncf.CompressWeightsMode.INT4_ASYM
            ov_model = nncf.compress_weights(ov_model, mode=mode, group_size=128, ratio=1.0, all_layers=True)
        except Exception as e:
            print(f"  WARN int4 failed ({e}); fp16")
    ov.save_model(ov_model, os.path.join(out_dir, "openvino_model.xml"), compress_to_fp16=True)
    del ov_model; gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--quantization", default="int4")
    args = ap.parse_args()
    TP = args.tp_size
    torch.set_grad_enabled(False)

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    cfg = model.config
    H, NH, NKV = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
    HD = getattr(cfg, "head_dim", None) or (H // NH)
    INT, L = cfg.intermediate_size, cfg.num_hidden_layers
    theta = float(getattr(cfg, "rope_theta", 500000.0))
    for nm, val in [("NH", NH), ("NKV", NKV), ("INT", INT)]:
        assert val % TP == 0, f"{nm}={val} not divisible by tp={TP}"
    print(f"H={H} NH={NH} NKV={NKV} HD={HD} INT={INT} L={L} tp={TP}")
    nkv_r = NKV // TP

    os.makedirs(args.output_dir, exist_ok=True)
    # tokenizer + config
    tdir = os.path.join(args.output_dir, "tokenizer"); os.makedirs(tdir, exist_ok=True)
    tok.save_pretrained(tdir)
    try:
        cfg.to_json_file(os.path.join(tdir, "config.json"))
    except Exception:
        pass
    json.dump({"model_id": args.model, "tp_size": TP, "num_layers": L, "hidden_size": H,
               "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
               "intermediate_size": INT, "vocab_size": cfg.vocab_size, "rope_theta": theta,
               "quantization": args.quantization, "export_version": "tp_v1"},
              open(os.path.join(args.output_dir, "pipeline_config.json"), "w"), indent=2)

    def ex_attn():
        seq, past = 4, 1
        return (torch.randn(1, seq, H), torch.ones(1, seq + past, dtype=torch.long),
                torch.arange(past, past + seq).unsqueeze(0),
                torch.randn(1, nkv_r, past, HD), torch.randn(1, nkv_r, past, HD))

    for rank in range(TP):
        rdir = os.path.join(args.output_dir, f"rank_{rank}")
        os.makedirs(rdir, exist_ok=True)
        segs = []
        print(f"=== rank {rank} ===")
        # embed (replicated)
        convert_and_save(EmbedSeg(model.model.embed_tokens),
                         (torch.randint(0, cfg.vocab_size, (1, 4)),),
                         ["input_ids"], ["hidden_states"],
                         os.path.join(rdir, "embed"), args.quantization, stateful=False)
        segs.append({"name": "embed", "type": "embed"})
        for i in range(L):
            layer = model.model.layers[i]
            convert_and_save(AttnSeg(layer, rank, TP, NH, NKV, HD, theta), ex_attn(),
                             ["hidden_states", "attention_mask", "position_ids",
                              "past_key_values.0.key", "past_key_values.0.value"],
                             ["hidden_states", "present.0.key", "present.0.value"],
                             os.path.join(rdir, f"attn_{i}"), args.quantization, stateful=True)
            segs.append({"name": f"attn_{i}", "type": "attn", "kv_heads": nkv_r, "head_dim": HD})
            convert_and_save(MlpSeg(layer, rank, TP, INT), (torch.randn(1, 4, H),),
                             ["hidden_states"], ["hidden_states"],
                             os.path.join(rdir, f"mlp_{i}"), args.quantization, stateful=False)
            segs.append({"name": f"mlp_{i}", "type": "mlp"})
            print(f"  layer {i} done")
        if rank == 0:
            convert_and_save(HeadSeg(model.model.norm, model.lm_head), (torch.randn(1, 4, H),),
                             ["hidden_states"], ["logits"],
                             os.path.join(rdir, "head"), args.quantization, stateful=False)
            segs.append({"name": "head", "type": "head"})
        json.dump({"rank": rank, "tp_size": TP, "segments": segs}, open(os.path.join(rdir, "rank_config.json"), "w"), indent=2)
        print(f"rank {rank}: {len(segs)} segments written to {rdir}")
    print("DONE")


if __name__ == "__main__":
    main()
