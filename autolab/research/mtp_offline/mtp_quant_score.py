#!/usr/bin/env python3
"""Price module-0 draft quantization on fleet captures before building the runtime.

Uses the export's weight grids, with f32 arithmetic (not a GPU parity claim).
The full-vocabulary bf16 head is the baseline; int8 65k head is the candidate.
"""
import argparse
import json
from pathlib import Path
import time

from safetensors import safe_open
import torch
from mtp_score import Depth, rms


def int8(w):
    scale = w.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).bfloat16().float()
    q = (w / scale).round().clamp(-127, 127)
    return (q * scale.half().float()).half().float()


def int4(w):
    # pack_int4 rounds using the unrounded scale and stores its bf16 value.
    groups = w.reshape(w.shape[0], -1, 32)
    scale = groups.abs().amax(-1, keepdim=True) / 7
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (groups / scale).round().clamp(-8, 7)
    return (q * scale.bfloat16().half().float()).half().float().reshape(w.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dump', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    man = json.loads((a.weights / 'manifest.json').read_text())
    with safe_open(str(a.weights / 'head.safetensors'), 'pt') as f:
        head = f.get_tensor('unembed.weight').float()[:65536]
        head = int8(head)
        norm = f.get_tensor('norm.weight').float()
    with safe_open(str(a.weights / 'mtp.safetensors'), 'pt') as f:
        depth = Depth(f, 0, ())
    for name in ('wq', 'wk', 'wv', 'wr', 'wo', 'input_proj'):
        setattr(depth, name, int8(getattr(depth, name)))
    for name in ('gate_i', 'up_i', 'w2'):
        setattr(depth, name, int4(getattr(depth, name)))
    rows = []
    for path in sorted(a.dump.glob('p*.safetensors')):
        start = time.monotonic()
        with safe_open(str(path), 'pt') as f:
            md = f.metadata(); tokens = f.get_tensor('tokens')
            final = f.get_tensor('final_out').float(); embed = f.get_tensor('embed_out').float()
        prompt = int(md['prompt_len'])
        hidden = depth.forward(rms(final, norm), embed[1:])
        logits = (hidden / man['logits_mup_width_multiplier']) @ head.t()
        predictions = logits.argmax(-1)[prompt - 1:-1]
        targets = tokens[prompt + 1:]
        hits = (predictions == targets).tolist()
        row = dict(i=int(md['i']), family=int(md['family']), hits=hits,
                   preds=predictions.tolist(), quantization='int4 MLP; int8 attention/input/65k head; f32 arithmetic')
        rows.append(row)
        a.out.write_text(json.dumps(rows))
        print('i=%d a1=%.3f n=%d %.1fs' % (row['i'], sum(hits)/max(1,len(hits)), len(hits), time.monotonic()-start), flush=True)


if __name__ == '__main__':
    main()
