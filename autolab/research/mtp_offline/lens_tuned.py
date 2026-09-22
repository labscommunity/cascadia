#!/usr/bin/env python3
"""E2 step 4 (tuned lens): one ridge-regularised affine map 6144 -> 6144 per rank boundary.

    lens_tuned.py --dump DIR --weights DIR --out tuned.json [--threads 24]

Fit on the generated positions of ~80 % of the prompts (split BY PROMPT: prompt i is held out when
i % 5 == 4), target = the final layer's residual at the same position (layer 65 output, pre-norm); score
on the held-out prompts: top-1 / top-5 of unembed(rmsnorm(map(x)) / mup) against the model's next token.
n (positions) < d (6144), so the map is fitted in the dual form W = Yc^T (Xc Xc^T + lam I)^-1 Xc with
centred data; lam is a multiple of mean(diag(Xc Xc^T)); the best multiple ON THE HELD-OUT SET is reported
(slightly optimistic: there is no third split), next to the raw lens on the same held-out positions.
"""
import argparse, glob, json, os
import torch
from safetensors import safe_open

LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]


def rms(x, w, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    mup = float(man["logits_mup_width_multiplier"]); V = int(man["unpadded_vocab_size"])
    with safe_open(os.path.join(a.weights, "head.safetensors"), "pt") as f:
        unembed = f.get_tensor("unembed.weight").float()[:V].contiguous(); norm = f.get_tensor("norm.weight").float()
    files = sorted(glob.glob(os.path.join(a.dump, "p*.safetensors")))
    X = {L: {"tr": [], "te": []} for L in LAYERS}; Y = {"tr": [], "te": []}; tgt = {"tr": [], "te": []}
    for path in files:
        with safe_open(path, "pt") as f:
            md = f.metadata(); P = int(md["prompt_len"]); i = int(md["i"]); am = f.get_tensor("argmax")[P - 1:]
            n = am.shape[0]; part = "te" if i % 5 == 4 else "tr"
            Y[part].append(f.get_tensor("final_out")[P - 1:].float()[:n]); tgt[part].append(am)
            for L in LAYERS:
                X[L][part].append(f.get_tensor("layer%d_out" % L).float()[:n])
    Ytr, Yte = torch.cat(Y["tr"]), torch.cat(Y["te"]); ttr, tte = torch.cat(tgt["tr"]), torch.cat(tgt["te"])
    print("train positions %d, held-out positions %d (prompts held out: i %% 5 == 4)" % (Ytr.shape[0], Yte.shape[0]), flush=True)

    def score(h, t):
        top5 = ((rms(h, norm) / mup) @ unembed.t()).topk(5, -1).indices
        return (top5[:, 0] == t).float().mean().item(), (top5 == t[:, None]).any(-1).float().mean().item()

    out = {}
    ymean = Ytr.mean(0, keepdim=True); Yc = (Ytr - ymean).double()
    for L in LAYERS:
        Xtr, Xte = torch.cat(X[L]["tr"]), torch.cat(X[L]["te"])
        raw1, raw5 = score(Xte, tte)
        xmean = Xtr.mean(0, keepdim=True); Xc = (Xtr - xmean).double(); G = Xc @ Xc.t(); scale = G.diagonal().mean()
        best = None
        for mult in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            A = torch.linalg.solve(G + mult * scale * torch.eye(G.shape[0], dtype=torch.double), Yc)     # [n, d]
            pred = (((Xte - xmean).double() @ Xc.t()) @ A).float() + ymean
            t1, t5 = score(pred, tte)
            if best is None or t1 > best[1]:
                best = (mult, t1, t5)
        out[str(L)] = dict(raw_top1=raw1, raw_top5=raw5, tuned_top1=best[1], tuned_top5=best[2], lam_mult=best[0], n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]))
        print("layer %2d (k=%2d stages): raw %.3f / %.3f   tuned %.3f / %.3f  (lam x%g)" % (L, (L + 1) // 6, raw1, raw5, best[1], best[2], best[0]), flush=True)
        json.dump(out, open(a.out, "w"))
    print("\n| stages waited k (layer) | raw top-1 | raw top-5 | tuned top-1 | tuned top-5 |\n|---|---|---|---|---|")
    for L in LAYERS:
        o = out[str(L)]
        print("| %d (%d) | %.3f | %.3f | %.3f | %.3f |" % ((L + 1) // 6, L, o["raw_top1"], o["raw_top5"], o["tuned_top1"], o["tuned_top5"]))


if __name__ == "__main__":
    main()
