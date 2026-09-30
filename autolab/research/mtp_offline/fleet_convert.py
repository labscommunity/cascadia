#!/usr/bin/env python3
"""Match captured samples to API replies and create the original MTP study format.

Runs on the build host. Inputs are capture_collect.py's directory, the
original rendered prompt token IDs, and read-only export weights.
"""
import argparse
import glob
import json
from pathlib import Path
import struct
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer


def api_text(raw):
    return (raw.replace('<|content_thinking|>', '<think>')
            .replace('<|end_message|><|content_text|>', '</think>')
            .replace('<|end_message|>', '</think>')
            .replace('<|content_text|>', '')
            .replace('<|end_model|>', ''))


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--captures', required=True); ap.add_argument('--prompts', required=True)
    ap.add_argument('--weights', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--api-mode', choices=['completion', 'chat'], default='completion',
                    help='raw completions skip structural tokens; chat translates markers')
    a = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    weights, out = Path(a.weights), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = {r['i']: r['ids'] for r in (json.loads(l) for l in Path(a.prompts).read_text().splitlines())}
    tok = Tokenizer.from_file(str(weights / 'tokenizer.json'))
    with safe_open(weights / 'embed.safetensors', 'pt') as f:
        embed = f.get_tensor('embed.weight')
        norm = f.get_tensor('embed_norm.weight').float()
    eps = json.loads((weights / 'manifest.json').read_text()).get('rms_norm_eps', 1e-5)
    matched = []
    for batch in sorted(Path(a.captures).glob('batch-*')):
        responses = json.loads((batch / 'responses.json').read_text())
        files = json.loads((batch / 'complete.json').read_text())
        used = set()
        for record in files:
            data = bytearray((batch / record['file']).read_bytes())
            if data[:8] not in (b'INKCAP01', b'INKCAP02'):
                raise ValueError('bad capture magic')
            dtype, itemsize = (torch.float16, 2) if data[:8] == b'INKCAP01' else (torch.float32, 4)
            hidden, rank, slot, prompt_rows, sequence = struct.unpack_from('<IIIIQ', data, 8)
            row_bytes = 8 + itemsize * hidden
            if (len(data) - 32) % row_bytes:
                raise ValueError('incomplete captured record')
            n = (len(data) - 32) // row_bytes
            samples = torch.tensor([struct.unpack_from('<q', data, 32 + p * row_bytes)[0] for p in range(n)], dtype=torch.int64)
            generated = samples[prompt_rows - 1:]
            if not prompt_rows or (generated < 0).any():
                raise ValueError('missing prompt boundary or sampled tokens')
            decoded = tok.decode(generated.tolist(), skip_special_tokens=a.api_mode == 'completion')
            if a.api_mode == 'chat':
                decoded = api_text(decoded)
            response_text = lambda r: (api_text(r['response']['text']) if a.api_mode == 'chat' else r['response']['text'])
            candidates = [(common_prefix(decoded, response_text(r)), r) for r in responses
                          if r['i'] not in used and len(prompts[r['i']]) == prompt_rows]
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            if not candidates or candidates[0][0] < min(120, len(decoded)):
                raise ValueError('cannot match capture to API reply: ' + repr(decoded[:140]))
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                raise ValueError('ambiguous capture/API match')
            agreement, response = candidates[0]
            if decoded != response_text(response):
                raise ValueError('capture/API text differs after %d characters for request %d' % (agreement, response['i']))
            i = response['i']; used.add(i)
            ids = torch.tensor(prompts[i] + generated.tolist(), dtype=torch.int64)
            residuals = torch.stack([torch.frombuffer(data, dtype=dtype, count=hidden,
                                      offset=32 + p * row_bytes + 8).clone() for p in range(n)])
            if not torch.isfinite(residuals).all():
                raise ValueError('nonfinite captured residual')
            e = embed[ids].float()
            e = e * torch.rsqrt(e.square().mean(-1, keepdim=True) + eps) * norm
            if not torch.isfinite(e).all():
                raise ValueError('nonfinite normalized embedding')
            save_file({'tokens': ids, 'embed_out': e, 'final_out': residuals, 'argmax': samples},
                      str(out / ('p%04d.safetensors' % i)),
                      metadata={'i': str(i), 'family': str(response['family']), 'prompt_len': str(prompt_rows),
                                'source': 'fleet', 'capture': record['file']})
            matched.append(dict(i=i, rank=rank, slot=slot, positions=n, prompt_rows=prompt_rows,
                                generated=len(generated), text_prefix_match=agreement))
            print('matched', matched[-1], flush=True)
        if len(used) != len(responses):
            raise ValueError('not every API request has a capture')
    (out / 'matches.json').write_text(json.dumps(matched, indent=1))
    print('converted', len(matched), 'fleet sequences')


if __name__ == '__main__':
    main()
