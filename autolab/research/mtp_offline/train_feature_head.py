#!/usr/bin/env python3
"""Bounded final-state draft feature predictor; no target-model weights change.

Use a disjoint training directory and held-out test directory. Validation is
stratified by family inside training; test is scored only after selection.
Frozen output-head grids match the 65k int8 export (arithmetic uses PyTorch).
"""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from safetensors import safe_open
from mtp_score import rms
from mtp_quant_score import int8


class FeatureHead(torch.nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.input = torch.nn.Linear(2 * hidden, width, bias=False)
        self.output = torch.nn.Linear(width, hidden, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    def forward(self, h, e):
        return h + self.output(torch.nn.functional.silu(self.input(torch.cat((h, e), -1))))


def split_sequences(records, seed=42):
    rng = random.Random(seed)
    train, valid = [], []
    for family in sorted({r['family'] for r in records}):
        group = sorted((r for r in records if r['family'] == family), key=lambda r: r['i'])
        rng.shuffle(group)
        n = max(1, round(len(group) * 0.2))
        if n >= len(group):
            raise ValueError('each family needs at least two distinct sequences')
        valid.extend(group[:n]); train.extend(group[n:])
    return train, valid


def load_sequences(directory, norm):
    records = []
    for path in sorted(Path(directory).glob('p*.safetensors')):
        with safe_open(str(path), 'pt') as f:
            md = f.metadata(); ids = f.get_tensor('tokens')
            raw = f.get_tensor('final_out').float(); emb = f.get_tensor('embed_out').float()
            actual = f.get_tensor('argmax')
        p = int(md['prompt_len'])
        if raw.shape[0] != len(ids)-1 or not torch.equal(actual[p-1:], ids[p:]):
            raise ValueError('state/token alignment failed')
        if not torch.isfinite(raw).all() or not torch.isfinite(emb).all():
            raise ValueError('nonfinite input')
        h = rms(raw, norm)
        # Anchor t predicts x[t+2], given h[t] and already verified x[t+1].
        records.append(dict(i=int(md['i']), family=int(md['family']), prompt=tuple(ids[:p].tolist()),
                            h=h[p-1:-1], e=emb[p:-1], target=h[p:], ids=ids[p+1:]))
        assert len(records[-1]['h']) == len(records[-1]['ids'])
    if not records:
        raise ValueError('empty dataset')
    return records


def validate_disjoint(train, test):
    if {r['i'] for r in train} & {r['i'] for r in test}:
        raise ValueError('train/test sequence ID overlap')
    if {r['prompt'] for r in train} & {r['prompt'] for r in test}:
        raise ValueError('train/test exact prompt overlap')
    if len({r['prompt'] for r in train}) != len(train):
        raise ValueError('duplicate training prompt')


def packed(records):
    return {k: torch.cat([r[k] for r in records]) for k in ('h', 'e', 'target', 'ids')}


@torch.no_grad()
def evaluate(model, records, head, mup, device, batch=128):
    rows = []
    for record in records:
        hits, loss, total = 0, 0.0, 0
        for lo in range(0, len(record['ids']), batch):
            h, e, target = (record[k][lo:lo+batch].to(device) for k in ('h', 'e', 'target'))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                pred = model(h, e)
                logits = (pred / mup) @ head.t()
            ids = record['ids'][lo:lo+batch].to(device)
            hits += int((logits.argmax(-1) == ids).sum())
            loss += float((pred.float()-target).square().sum())
            total += len(ids)
        rows.append(dict(i=record['i'], family=record['family'], hits=hits, n=total,
                         feature_mse=loss/(total*record['h'].shape[1])))
    return dict(a1=sum(r['hits'] for r in rows)/sum(r['n'] for r in rows), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True); ap.add_argument('--test', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--lr', type=float, default=0.0003)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(8)
    device = torch.device(a.device); a.out.mkdir(parents=True, exist_ok=True)
    man = json.loads((a.weights/'manifest.json').read_text()); mup = man['logits_mup_width_multiplier']
    with safe_open(str(a.weights/'head.safetensors'), 'pt') as f:
        norm = f.get_tensor('norm.weight').float()
        head = int8(f.get_tensor('unembed.weight')[:65536].float()).to(device)
    records = load_sequences(a.train, norm); heldout = load_sequences(a.test, norm)
    validate_disjoint(records, heldout)
    train, valid = split_sequences(records, a.seed)
    data = packed(train); n = len(data['ids'])
    model = FeatureHead(len(norm), a.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    result = dict(recipe='h + Linear(SiLU(Linear([h, next-token embedding]))); width=512; MSE + 0.1*CE; int8 65k head',
                  train_ids=[r['i'] for r in train], validation_ids=[r['i'] for r in valid],
                  test_ids=[r['i'] for r in heldout], train_positions=n, seed=a.seed, epochs=[],
                  parameters=sum(p.numel() for p in model.parameters()),
                  limitations='bf16-autocast PyTorch evaluation; no device parity or serving speed claim')
    start = time.monotonic()
    result['baseline_validation'] = evaluate(model, valid, head, mup, device)
    best = result['baseline_validation']['a1']
    torch.save(model.state_dict(),a.out/'best.pt'); result['selected_epoch'] = 0
    result['hyperparameters'] = dict(width=a.width, epochs=a.epochs, batch=a.batch, lr=a.lr)
    for epoch in range(a.epochs):
        model.train(); order = torch.randperm(n); summed = 0.0
        for indices in order.split(a.batch):
            h,e,target,ids = (data[k][indices].to(device) for k in ('h','e','target','ids'))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                pred = model(h,e); logits = (pred/mup) @ head.t()
                mse = (pred.float()-target).square().mean()
                mask = ids < len(head)
                ce = torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask]) if mask.any() else mse*0
                loss = mse + 0.1*ce
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            summed += float(loss.detach())*len(indices)
        model.eval(); val = evaluate(model,valid,head,mup,device)
        row = dict(epoch=epoch+1,loss=summed/n,validation_a1=val['a1'],elapsed_s=time.monotonic()-start)
        result['epochs'].append(row)
        if val['a1'] > best:
            best = val['a1']; torch.save(model.state_dict(),a.out/'best.pt'); result['selected_epoch']=epoch+1
        (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(row,flush=True)
    model.load_state_dict(torch.load(a.out/'best.pt',weights_only=True,map_location=device))
    model.eval(); result['test'] = evaluate(model,heldout,head,mup,device)
    result['validation'] = evaluate(model,valid,head,mup,device)
    result['elapsed_s'] = time.monotonic()-start
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('final held-out a1',result['test']['a1'],flush=True)


if __name__ == '__main__':
    main()
