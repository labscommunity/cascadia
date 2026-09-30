#!/usr/bin/env python3
"""E4 add-on: simulate how the MTP chain would really run on rank 0, with SPARSE upkeep of the deeper modules.

    mtp_prodsim.py --dump DIR --weights DIR --out prodsim.json [--threads 24] [--selftest]

Why: mtp_score.py's protocols A/B keep every module's attention/conv context current at EVERY position
(cost: one module pass per depth per token). Protocol C showed the deeper modules collapse without any
context. A cheap implementation runs module 0 at every real token (it has the real state each time) but
modules 1..7 only when a chain is drafted, i.e. at the anchor after each miss. This script measures that:

  prefill: module k runs over the prompt with true tokens (positions 0..P-1-k), one batched pass;
  anchor t (first P-1, then the position after each miss): d1 = module 0 (full context, real states);
      d_{k+1} = module k fed (h_{k-1,t}, embed(d_k)), appended to module k's OWN cache: prompt entries +
      earlier anchors only (true positions for the relative bias, the short convs run over the module's
      own call sequence), entries made from wrong drafts stay in the cache (production cannot know);
  run r = leading right drafts; next anchor = t + r + 1.

Reports per-depth conditional acceptance, P(prefix), E[accepted drafts per chain] and the renewal-model
ms/token, next to protocol B's numbers restricted to the same anchors.
--selftest: the cached/chunked forward must reproduce mtp_score.Depth.block() (max |diff|, argmax equality).
"""
import argparse, glob, json, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_score import Depth, rms, FAMILIES, N_HEADS, HEAD_DIM, D_REL, WINDOW  # noqa: E402


class Cache:
    def __init__(self, d):
        self.K = torch.zeros(d.kv_heads, 0, HEAD_DIM); self.V = torch.zeros(d.kv_heads, 0, HEAD_DIM)
        self.pos = torch.zeros(0, dtype=torch.long)
        kv = d.kv_heads * HEAD_DIM
        self.buf = {"k": torch.zeros(3, kv), "v": torch.zeros(3, kv), "attn": torch.zeros(3, 6144), "mlp": torch.zeros(3, 6144)}


def sconv_step(y, w, cache, name):
    x = torch.cat([cache.buf[name], y], 0)                              # [3 + m, C]
    out = F.conv1d(x.t().unsqueeze(0), w, groups=y.shape[1]).squeeze(0).t()   # valid conv -> [m, C]
    cache.buf[name] = x[-3:].clone()
    return out + y


def chunk(d, cache, prev, e, pos):
    """Run module `d` on m new rows (prev [m,H], e [m,H], true positions pos [m]) on top of its cache."""
    m = prev.shape[0]
    x = torch.cat([rms(prev, d.hidden_norm), rms(e, d.embed_norm)], -1) @ d.input_proj.t()
    h = rms(x, d.attn_norm)
    q = rms((h @ d.wq.t()).view(m, N_HEADS, HEAD_DIM), d.q_norm).transpose(0, 1)
    k_ = rms(sconv_step(h @ d.wk.t(), d.k_sconv, cache, "k").view(m, d.kv_heads, HEAD_DIM), d.k_norm).transpose(0, 1)
    v_ = sconv_step(h @ d.wv.t(), d.v_sconv, cache, "v").view(m, d.kv_heads, HEAD_DIM).transpose(0, 1)
    r = (h @ d.wr.t()).view(m, N_HEADS, D_REL)
    cache.K = torch.cat([cache.K, k_], 1); cache.V = torch.cat([cache.V, v_], 1); cache.pos = torch.cat([cache.pos, pos])
    rep = N_HEADS // d.kv_heads
    kk = cache.K.repeat_interleave(rep, 0); vv = cache.V.repeat_interleave(rep, 0)
    scores = q @ kk.transpose(1, 2) / HEAD_DIM
    dist = pos[:, None] - cache.pos[None, :]
    rel = (r @ d.rel_proj).transpose(0, 1)
    idx = dist.clamp(0, d.extent - 1).unsqueeze(0).expand(N_HEADS, -1, -1)
    scores = scores + rel.gather(-1, idx).masked_fill(((dist < 0) | (dist >= d.extent)).unsqueeze(0), 0.0)
    mask = dist < 0
    if d.local:
        mask = mask | (dist >= WINDOW)
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    a = (torch.softmax(scores, -1) @ vv).transpose(0, 1).reshape(m, N_HEADS * HEAD_DIM) @ d.wo.t()
    x = x + sconv_step(a, d.attn_sconv, cache, "attn")
    h = rms(x, d.mlp_norm)
    mm = (F.silu(h @ d.gate_i.t()) * (h @ d.up_i.t())) @ d.w2.t() * d.global_scale
    return x + sconv_step(mm, d.mlp_sconv, cache, "mlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=24); ap.add_argument("--depths", type=int, default=8)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    mup = float(man["logits_mup_width_multiplier"]); V = int(man["unpadded_vocab_size"])
    with safe_open(os.path.join(a.weights, "head.safetensors"), "pt") as f:
        unembed = f.get_tensor("unembed.weight").float()[:V].contiguous(); final_norm = f.get_tensor("norm.weight").float()
    with safe_open(os.path.join(a.weights, "embed.safetensors"), "pt") as f:
        embed = f.get_tensor("embed.weight"); embed_norm = f.get_tensor("embed_norm.weight").float()
    with safe_open(os.path.join(a.weights, "mtp.safetensors"), "pt") as f:
        depths = [Depth(f, k, ()) for k in range(a.depths)]
    top = lambda h: ((h / mup) @ unembed.t()).argmax(-1)
    emb_of = lambda ids: rms(embed[ids].float(), embed_norm)
    files = sorted(glob.glob(os.path.join(a.dump, "p*.safetensors")))
    if a.limit:
        files = files[:a.limit]
    rows = json.load(open(a.out)) if (a.resume and os.path.exists(a.out)) else []
    done = {r["i"] for r in rows}
    for path in files:
        if int(os.path.basename(path)[1:5]) in done:
            continue
        t0 = time.time()
        with safe_open(path, "pt") as f:
            md = f.metadata(); toks = f.get_tensor("tokens"); eo = f.get_tensor("embed_out").float(); fo = f.get_tensor("final_out").float()
        P = int(md["prompt_len"]); T = toks.shape[0]; g = rms(fo, final_norm)
        if a.selftest:
            for k in (0, 1):
                d = depths[k]; L = T - 1
                ref = d.forward(g, eo[1:T])
                c = Cache(d); one = chunk(d, c, g, eo[1:T], torch.arange(L))
                c = Cache(d); n0 = 40
                parts = [chunk(d, c, g[:n0], eo[1:1 + n0], torch.arange(n0))]
                for t in range(n0, L):
                    parts.append(chunk(d, c, g[t:t + 1], eo[t + 1:t + 2], torch.tensor([t])))
                inc = torch.cat(parts, 0)
                print("selftest depth %d: |one-chunk - batch| max %.3e, |prefill+steps - batch| max %.3e (ref absmax %.1f); "
                      "argmax equal: %s / %s" % (k, (one - ref).abs().max().item(), (inc - ref).abs().max().item(),
                                                  ref.abs().max().item(), bool((top(one) == top(ref)).all()), bool((top(inc) == top(ref)).all())))
            return
        # module 0: full context at every position (true inputs); protocol-A hidden states for the prompt prefill of deeper modules
        hA = [depths[0].forward(g, eo[1:T])]
        p0 = top(hA[0])
        caches = [None]
        for k in range(1, a.depths):
            n = max(P - k, 0)                                          # positions 0..P-1-k: x_{t+k+1} is a prompt token or x_P
            c = Cache(depths[k])
            hk = chunk(depths[k], c, hA[k - 1][:n], eo[k + 1:k + 1 + n], torch.arange(n))
            hA.append(hk); caches.append(c)
        anchors = []; t = P - 1
        while t <= T - 3:
            drafts = [int(p0[t])]; h = hA[0][t:t + 1]
            for k in range(1, a.depths):
                h = chunk(depths[k], caches[k], h, emb_of(torch.tensor([drafts[-1]])), torch.tensor([t]))
                drafts.append(int(top(h)[0]))
            hits = [(t + j + 2 <= T - 1) and (drafts[j] == int(toks[t + j + 2])) for j in range(a.depths)]
            avail = [t + j + 2 <= T - 1 for j in range(a.depths)]
            r = 0
            while r < a.depths and hits[r]:
                r += 1
            anchors.append({"t": t, "hits": hits, "avail": avail, "run": r})
            t = t + r + 1
        rows.append({"i": int(md["i"]), "family": int(md["family"]), "P": P, "T": T, "anchors": anchors})
        runs = [x["run"] for x in anchors]
        print("i=%d fam=%s anchors=%d mean run %.2f  (%.0fs)" % (int(md["i"]), FAMILIES[int(md["family"])], len(anchors),
                                                                  sum(runs) / len(runs), time.time() - t0), flush=True)
        json.dump(rows, open(a.out, "w"))
    # summary
    def summarize(rs, label):
        anc = [x for r in rs for x in r["anchors"]]
        print("\n%s: %d sequences, %d anchors (chains)" % (label, len(rs), len(anc)))
        print("| draft k | n (chain right so far, target exists) | a_k given drafts 1..k-1 right | P(drafts 1..k right) |"); print("|---|---|---|---|")
        e = 0.0
        for k in range(a.depths):
            cond = [x["hits"][k] for x in anc if x["avail"][k] and all(x["hits"][:k])]
            full = [all(x["hits"][:k + 1]) for x in anc if x["avail"][k]]
            pk = sum(full) / max(len(full), 1); e += pk
            print("| %d | %d | %.3f | %.3f |" % (k + 1, len(cond), sum(cond) / max(len(cond), 1), pk))
        print("E[accepted drafts per chain] = %.2f -> (466 + E*45)/(1+E) = %.0f ms/token = %.1f tok/s" % (
            e, (466 + e * 45) / (1 + e), 1000 * (1 + e) / (466 + e * 45)))
    summarize(rows, "all")
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        anc = [x for r in rs for x in r["anchors"]]
        e = sum(sum(all(x["hits"][:k + 1]) for x in anc if x["avail"][k]) / max(sum(1 for x in anc if x["avail"][k]), 1) for k in range(a.depths))
        print("  %-12s chains %4d  a1 %.3f  E %.2f  -> %.1f tok/s" % (FAMILIES[fam], len(anc),
              sum(x["hits"][0] for x in anc) / len(anc), e, 1000 * (1 + e) / (466 + e * 45)))


if __name__ == "__main__":
    main()
