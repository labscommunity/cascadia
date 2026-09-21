#!/usr/bin/env python3
"""Summaries of mtp_prodsim.py (prodsim.json) and lens_score.py (lens.json) restricted to prompts i < max_i.

    final_tables.py prodsim.json lens.json [max_i]
"""
import json, sys
FAM = ["explain", "code", "arithmetic", "story", "lists", "tables", "rewriting", "facts", "poem", "how-to", "translation", "true/false"]
max_i = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
L_MS, T_MS, D = 466.0, 45.0, 8
rate = lambda xs: sum(xs) / len(xs) if xs else float("nan")

rows = sorted([r for r in json.load(open(sys.argv[1])) if r["i"] < max_i], key=lambda r: r["i"])
def chain(rs):
    anc = [x for r in rs for x in r["anchors"]]; out = []; e = 0.0
    for k in range(D):
        cond = [x["hits"][k] for x in anc if x["avail"][k] and all(x["hits"][:k])]
        full = [all(x["hits"][:k + 1]) for x in anc if x["avail"][k]]
        e += rate(full); out.append((k + 1, len(cond), rate(cond), rate(full)))
    return anc, out, e
anc, st, e = chain(rows)
print("### Production simulation D (module 0 current at every token; modules 1..7 run ONLY at the anchor after each miss)\n")
print("%d sequences, %d chains (anchors), mean accepted run %.2f\n" % (len(rows), len(anc), rate([x["run"] for x in anc])))
print("| draft k | n (chain right so far) | a_k given drafts 1..k-1 right | P(drafts 1..k right) |"); print("|---|---|---|---|")
for k, n, c, p in st:
    print("| %d | %d | %.3f | %.3f |" % (k, n, c, p))
print("\nE[accepted drafts per chain] = %.2f -> (L + E*T)/(1+E) = %.0f ms per token = %.1f tok/s\n" % (e, (L_MS + e * T_MS) / (1 + e), 1000 * (1 + e) / (L_MS + e * T_MS)))
print("| family | seqs | chains | a1 at anchors | E[accepted] | implied tok/s |"); print("|---|---|---|---|---|---|")
for f in sorted({r["family"] for r in rows}):
    rs = [r for r in rows if r["family"] == f]; a, _, ef = chain(rs)
    print("| %s | %d | %d | %.3f | %.2f | %.1f |" % (FAM[f], len(rs), len(a), rate([x["hits"][0] for x in a]), ef, 1000 * (1 + ef) / (L_MS + ef * T_MS)))

lens = sorted([r for r in json.load(open(sys.argv[2])) if r["i"] < max_i], key=lambda r: r["i"])
LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
print("\n### E2 raw logit lens (%d sequences, %d generated positions)\n" % (len(lens), sum(r["n"] for r in lens)))
print("| stages waited k (state leaving rank k-1 = layer) | top-1 | top-5 | bar to beat today (spec section 1) |"); print("|---|---|---|---|")
bars = {1: "0.42", 2: "0.47", 3: "0.54", 4: "0.62", 5: "0.73", 6: "0.90"}
for L in LAYERS:
    k = (L + 1) // 6
    print("| %d (rank %d, layer %d) | %.3f | %.3f | %s |" % (k, k - 1, L, rate([h for r in lens for h in r["top1"][str(L)]]),
          rate([h for r in lens for h in r["top5"][str(L)]]), bars.get(k, "impossible" if k < 11 else "(final layer: self-check)")))
print("\ntop-1 by family:\n\n| family | " + " | ".join("L%d" % L for L in LAYERS[:-1]) + " |"); print("|" + "---|" * len(LAYERS))
for f in sorted({r["family"] for r in lens}):
    rr = [r for r in lens if r["family"] == f]
    print("| %s | " % FAM[f] + " | ".join("%.2f" % rate([h for r in rr for h in r["top1"][str(L)]]) for L in LAYERS[:-1]) + " |")
