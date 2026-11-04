#!/usr/bin/env python
"""Structural template IR for Qwen3.5-MoE: real config with moe_intermediate_size shrunk to TINY, random weights, fp16 constants, no compression.
usage: template_export.py <src_ckpt_dir> <out_dir> [tiny=64]"""
import sys, os, time, json, torch, shutil
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))  # gentle on the HVF host
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from optimum.exporters.openvino import export_from_model
from optimum.intel import OVConfig
src, out = sys.argv[1], sys.argv[2]; tiny = int(sys.argv[3]) if len(sys.argv) > 3 else 64
cfg = AutoConfig.from_pretrained(src)
for c in (cfg, getattr(cfg, "text_config", None)):
    if c is not None and getattr(c, "quantization_config", None) is not None:
        c.quantization_config = None
        try: delattr(c, "quantization_config")
        except Exception: pass
cfg.text_config.moe_intermediate_size = tiny
cfg.text_config.mtp_num_hidden_layers = 0
for c in (cfg, cfg.text_config, getattr(cfg, "vision_config", None)):
    if c is not None:
        for attr in ("dtype", "torch_dtype"):
            if hasattr(c, attr): setattr(c, attr, "float32")
print(f"[tpl] layers={cfg.text_config.num_hidden_layers} hidden={cfg.text_config.hidden_size} experts={cfg.text_config.num_experts} tiny_I={tiny}", flush=True)
t0 = time.time(); torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
model = Qwen3_5MoeForConditionalGeneration(cfg).to(torch.float32).eval()
with torch.no_grad():
    for prm in model.parameters():
        prm.normal_(0, 0.02)  # NO literal 0/1 parameters: keeps every norm/bias multiply in the graph (OpenVINO folds x*1 / x+0 away)
print("[tpl] all params randomized; param dtypes:", {str(d) for d in {p.dtype for p in model.parameters()}}, flush=True)
n = sum(p.numel() for p in model.parameters()); print(f"[tpl] random model built: {n/1e9:.2f} B params in {time.time()-t0:.0f}s", flush=True)
proc = AutoProcessor.from_pretrained(src)
t0 = time.time()
export_from_model(model, out, task="image-text-to-text", ov_config=OVConfig(dtype="fp16"), stateful=True, preprocessors=[proc])
print(f"[tpl] export done in {time.time()-t0:.0f}s", flush=True)
cfg.save_pretrained(out); proc.save_pretrained(out)
for aux in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"):
    p = os.path.join(src, aux)
    if os.path.exists(p): shutil.copy2(p, out)
print("[tpl] files:", sorted(os.listdir(out)), flush=True)
