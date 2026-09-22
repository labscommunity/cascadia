#!/usr/bin/env python3
"""Adapt only shipped module-0 input projection on disjoint fleet sequences.

Frozen int4/int8 grids, rank-32 adapter, true-token temporal context. Select
one of 12 epochs on validation; only then score the original held-out 36.
"""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from safetensors import safe_open
from mtp_score import Depth, rms
from mtp_quant_score import int4, int8
from train_feature_head import split_sequences, validate_disjoint


class AdaptedDepth(torch.nn.Module):
    def __init__(self, base, rank=32):
        super().__init__()
        self.base = base
        hidden, inputs = base.input_proj.shape
        self.down = torch.nn.Linear(inputs,rank,bias=False)
        self.up = torch.nn.Linear(rank,hidden,bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self,h,e):
        cat = torch.cat((rms(h,self.base.hidden_norm),rms(e,self.base.embed_norm)),-1)
        return self.base.block(cat @ self.base.input_proj.t() + self.up(self.down(cat)),())


def load_records(directory,norm):
    rows=[]
    for path in sorted(Path(directory).glob('p*.safetensors')):
        with safe_open(str(path),'pt') as f:
            md=f.metadata();ids=f.get_tensor('tokens');raw=f.get_tensor('final_out').float()
            emb=f.get_tensor('embed_out').float();actual=f.get_tensor('argmax')
        p=int(md['prompt_len'])
        if raw.shape[0]!=len(ids)-1 or not torch.equal(actual[p-1:],ids[p:]):
            raise ValueError('state/token alignment failed')
        if not torch.isfinite(raw).all() or not torch.isfinite(emb).all():
            raise ValueError('nonfinite training input')
        rows.append(dict(i=int(md['i']),family=int(md['family']),prompt=tuple(ids[:p].tolist()),
                         h=rms(raw,norm),e=emb[1:],ids=ids[p+1:],start=p-1))
    if not rows: raise ValueError('empty input')
    return rows


@torch.no_grad()
def evaluate(model,records,head,mup,device):
    rows=[]
    model.eval()
    for r in records:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h=model(r['h'].to(device),r['e'].to(device))
            logits=(h[r['start']:-1]/mup) @ head.t()
        ids=r['ids'].to(device)
        hits=int((logits.argmax(-1)==ids).sum())
        rows.append(dict(i=r['i'],family=r['family'],hits=hits,n=len(ids)))
    return dict(a1=sum(r['hits'] for r in rows)/sum(r['n'] for r in rows),rows=rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',type=Path,required=True);ap.add_argument('--test',type=Path,required=True)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--lr',type=float,default=0.0002)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--rank',type=int,default=32)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed);torch.set_num_threads(8);rng=random.Random(a.seed)
    device=torch.device('cuda');man=json.loads((a.weights/'manifest.json').read_text())
    mup=man['logits_mup_width_multiplier']
    with safe_open(str(a.weights/'head.safetensors'),'pt') as f:
        norm=f.get_tensor('norm.weight').float()
        head=int8(f.get_tensor('unembed.weight')[:65536].float()).to(device,dtype=torch.bfloat16)
    with safe_open(str(a.weights/'mtp.safetensors'),'pt') as f: base=Depth(f,0,())
    del base.gate_h,base.up_h
    for name in ('wq','wk','wv','wr','wo','input_proj'):setattr(base,name,int8(getattr(base,name)))
    for name in ('gate_i','up_i','w2'):setattr(base,name,int4(getattr(base,name)))
    for name,value in vars(base).copy().items():
        if isinstance(value,torch.Tensor):
            # Linear grids are exactly representable in bf16 only approximately;
            # this is an explicitly measured bf16-autocast study, not GPU parity.
            setattr(base,name,value.to(device,dtype=torch.bfloat16 if value.ndim==2 and name!='rel_proj' else torch.float32))
    records=load_records(a.train,norm);heldout=load_records(a.test,norm)
    validate_disjoint(records,heldout);train,valid=split_sequences(records,a.seed)
    model=AdaptedDepth(base,a.rank).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    result=dict(recipe='rank-32 input projection adapter; frozen quantized module 0; true-token context; CE; bf16 autocast',
                train_ids=[r['i'] for r in train],validation_ids=[r['i'] for r in valid],test_ids=[r['i'] for r in heldout],
                parameters=sum(p.numel() for p in model.parameters()),epochs=[],seed=a.seed)
    baseline=evaluate(model,valid,head,mup,device);result['baseline_validation']=baseline
    # Keep the zero adapter eligible: training is allowed to be harmful.
    best=baseline['a1'];torch.save(model.state_dict(),a.out/'best.pt');result['selected_epoch']=0
    start=time.monotonic()
    for epoch in range(a.epochs):
        model.train();order=train.copy();rng.shuffle(order);total=0;loss_sum=0.0
        for r in order:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h=model(r['h'].to(device),r['e'].to(device))
                logits=(h[r['start']:-1]/mup) @ head.t();ids=r['ids'].to(device);mask=ids<len(head)
                if not mask.any():continue
                loss=torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=int(mask.sum());loss_sum+=float(loss.detach())*int(mask.sum())
        val=evaluate(model,valid,head,mup,device)
        row=dict(epoch=epoch+1,loss=loss_sum/total,validation_a1=val['a1'],elapsed_s=time.monotonic()-start)
        result['epochs'].append(row)
        if val['a1']>best:
            best=val['a1'];torch.save(model.state_dict(),a.out/'best.pt');result['selected_epoch']=epoch+1
        (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(row,flush=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location=device,weights_only=True))
    result['validation']=evaluate(model,valid,head,mup,device)
    result['test']=evaluate(model,heldout,head,mup,device)
    with torch.no_grad():model.up.weight.zero_()
    result['baseline_test']=evaluate(model,heldout,head,mup,device)
    result['elapsed_s']=time.monotonic()-start
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('final held-out a1',result['test']['a1'],'baseline',result['baseline_test']['a1'],flush=True)


if __name__=='__main__':main()
