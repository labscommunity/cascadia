#!/usr/bin/env python3
"""E2 (raw logit lens): how well does the residual leaving rank boundary layer L name the model's next token?

    lens_score.py --dump DIR --weights DIR --out lens.json [--threads 32]

For each boundary layer L in {5,11,...,59} (+65 = the final layer as a self-check) and each generated
position t (P-1 .. T-2): logits = unembed(rmsnorm(layer_L_out[t], norm.weight) / mup)[:unpadded];
top-1 / top-5 against the final layer's own argmax at t (= the model's next token).
"""
import argparse, glob, json, os, time
import torch
from safetensors import safe_open

FAMILIES = ["explain", "code", "arithmetic", "story", "lists", "tables", "rewriting", "facts", "poem", "how-to",
            "translation", "true/false"]
LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]


def rms(x, w, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=24); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    mup = float(man["logits_mup_width_multiplier"]); V = int(man["unpadded_vocab_size"])
    with safe_open(os.path.join(a.weights, "head.safetensors"), "pt") as f:
        unembed = f.get_tensor("unembed.weight").float()[:V].contiguous(); norm = f.get_tensor("norm.weight").float()
    rows = json.load(open(a.out)) if (a.resume and os.path.exists(a.out)) else []
    done = {r["i"] for r in rows}
    for path in sorted(glob.glob(os.path.join(a.dump, "p*.safetensors"))):
        if int(os.path.basename(path)[1:5]) in done:
            continue
        t0 = time.time()
        with safe_open(path, "pt") as f:
            md = f.metadata(); P = int(md["prompt_len"]); am = f.get_tensor("argmax")
            tgt = am[P - 1:]                                         # the model's next token at each generated position
            rec = {"i": int(md["i"]), "family": int(md["family"]), "n": int(tgt.shape[0]), "top1": {}, "top5": {}}
            for L in LAYERS:
                x = f.get_tensor("final_out")[P - 1:] if L == 65 else f.get_tensor("layer%d_out" % L)
                x = x.float()[:tgt.shape[0]]
                lg = (rms(x, norm) / mup) @ unembed.t()
                top5 = lg.topk(5, dim=-1).indices
                rec["top1"][str(L)] = (top5[:, 0] == tgt).tolist()
                rec["top5"][str(L)] = (top5 == tgt[:, None]).any(-1).tolist()
        rows.append(rec)
        print("i=%d fam=%s n=%d  top1: %s  (%.1fs)" % (rec["i"], FAMILIES[rec["family"]], rec["n"], " ".join(
            "%d:%.2f" % (L, sum(rec["top1"][str(L)]) / rec["n"]) for L in LAYERS), time.time() - t0), flush=True)
        json.dump(rows, open(a.out, "w"))
    # tables
    def rate(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    print("\n| stages waited k (state leaving rank k-1 = layer) | top-1 | top-5 | n |"); print("|---|---|---|---|")
    for L in LAYERS:
        t1 = [h for r in rows for h in r["top1"][str(L)]]; t5 = [h for r in rows for h in r["top5"][str(L)]]
        print("| %d (rank %d, layer %d) | %.3f | %.3f | %d |" % ((L + 1) // 6, (L + 1) // 6 - 1, L, rate(t1), rate(t5), len(t1)))
    fams = sorted({r["family"] for r in rows})
    print("\ntop-1 by family:\n\n| family | " + " | ".join("L%d" % L for L in LAYERS) + " |"); print("|" + "---|" * (len(LAYERS) + 1))
    for fam in fams:
        rr = [r for r in rows if r["family"] == fam]
        print("| %s | " % FAMILIES[fam] + " | ".join("%.2f" % rate([h for r in rr for h in r["top1"][str(L)]]) for L in LAYERS) + " |")


if __name__ == "__main__":
    main()
