#!/usr/bin/env python3
"""Export a shipped MTP module for Cascadia's existing dense Inkling block.

Run on the build host in scratch space. The source checkpoint/export is read
only. Attention/input projection and the 65k draft unembed use int8 IRs;
the MLP uses the fleet's int4 group-32 layout and fused all-active expert IR.
The small norms remain f32. The raw block files also permit CPU parity work.
No output norm is added: the MTP block's output goes straight to the unembed.
"""
import argparse
import json
from pathlib import Path
import sys

import openvino as ov
from openvino import Model, PartialShape, Type, opset13 as ops
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tools', type=Path, required=True, help='repository tools directory')
    ap.add_argument('--module', type=int, default=0, choices=range(8))
    ap.add_argument('--vocab', type=int, default=65536)
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    if a.out.exists():
        ap.error('output already exists; choose an empty scratch location')
    if a.out.resolve().is_relative_to(a.weights.resolve()):
        ap.error('output must not be inside the source weights')
    sys.path.insert(0, str(a.tools))
    from export_inkling import pack_sections, shell_spec
    from inkling_attn_ov import layer_models, weight_node
    from inkling_moe_layer_ov import dense_as_moe_model

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    man = json.loads((a.weights / 'manifest.json').read_text())
    local = a.module in {0, 2, 4, 5, 6, 7}
    man.update(num_layers=1, dense_layers=[0], dense_intermediate=24576,
               moe_intermediate=3072, num_attention_heads=64, num_kv_heads=8,
               head_dim=128, swa_num_attention_heads=64, swa_num_kv_heads=16,
               swa_head_dim=128, layer_types=['sliding' if local else 'global'],
               d_rel=16, sliding_window=512, rel_extent=1024,
               log_scaling_n_floor=None, has_mtp=False)
    a.out.mkdir(parents=True)
    (a.out / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    p = 'model.mtp.layers.%d.' % a.module
    with safe_open(str(a.weights / 'mtp.safetensors'), framework='pt') as src:
        block = {}
        for key, kind, shape in shell_spec(man, 0):
            t = src.get_tensor(p + 'transformer_block.' + key)
            assert tuple(t.shape) == shape, (key, t.shape, shape)
            block[key] = (t.bfloat16() if kind == 'bf16' else
                          t.squeeze(1).float() if kind == 'conv' else t.float()).contiguous()
        (a.out / 'shells').mkdir()
        save_file(block, str(a.out / 'shells/layer_00.safetensors'))
        w13 = src.get_tensor(p + 'transformer_block.mlp.w13_dn.weight')
        w2 = src.get_tensor(p + 'transformer_block.mlp.w2_md.weight')
        assert tuple(w13.shape) == (49152, 6144) and tuple(w2.shape) == (6144, 24576)
        dense = a.out / 'experts/layer_00'
        dense.mkdir(parents=True)
        with (dense / 'dense.bin').open('wb') as f:
            for chunk in pack_sections(w13[0::2], w13[1::2], w2):
                f.write(chunk)
        norms = {key: src.get_tensor(p + key).float().contiguous()
                 for key in ('hidden_norm.weight', 'embed_norm.weight')}
        input_proj = src.get_tensor(p + 'input_proj.weight').float().numpy()
        assert input_proj.shape == (6144, 12288)
    with safe_open(str(a.weights / 'head.safetensors'), framework='pt') as src:
        norms['final_norm.weight'] = src.get_tensor('norm.weight').float().contiguous()
        unembed = src.get_slice('unembed.weight')[:a.vocab].float().numpy()
    assert 0 < a.vocab <= man['unpadded_vocab_size']
    save_file(norms, str(a.out / 'mtp_norms.safetensors'))

    def save(model, relative):
        path = a.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(model, str(path), compress_to_fp16=False)
        print(relative, flush=True)

    def fc(w, name):
        x = ops.parameter(PartialShape([1, -1, w.shape[1]]), Type.f32, name='x')
        x.get_output_tensor(0).set_names({'x'})
        y = ops.matmul(x, weight_node(w, 'int8'), False, True)
        y.get_output_tensor(0).set_names({'logits'})
        return Model([y], [x], name)

    q, o, _, _ = layer_models(str(a.out), 0, 'int8')
    save(q, 'attn_ov/layer_00/qkvr/openvino_model.xml')
    save(o, 'attn_ov/layer_00/o/openvino_model.xml')
    save(dense_as_moe_model(str(a.out), 0, man), 'dense_moe_ov/layer_00/openvino_model.xml')
    save(fc(input_proj, 'mtp_input_projection'), 'input_ov/openvino_model.xml')
    save(fc(unembed, 'mtp_draft_unembed'), 'head_ov/openvino_model.xml')
    (a.out / 'mtp.json').write_text(json.dumps(dict(
        module=a.module, draft_vocab=a.vocab, hidden_first=True,
        backbone_final_norm=True, backbone_embed_norm=True,
        output_norm=False, source='shipped MTP checkpoint',
        mlp='int4 group32', attention='int8', input_projection='int8', head='int8'), indent=2) + '\n')
    print('export complete', sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file()), 'bytes', flush=True)


if __name__ == '__main__':
    main()
