#!/usr/bin/env python3
"""Aggregate mtp_score.py's results.json into the markdown tables of the E4 verdict.

    mtp_tables.py results.json [max_i]
"""
import json, sys

FAMILIES = ["explain", "code", "arithmetic", "story", "lists", "tables", "rewriting", "facts", "poem", "how-to",
            "translation", "true/false"]


def rate(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


def chain_stats(seqs, proto, depths):
    """Per depth k (1-based draft index): unconditional hit, hit given drafts 1..k-1 right, P(drafts 1..k right)."""
    out = []
    for k in range(depths):
        unc = []; cond = []; pref = []
        for s in seqs:
            h = s[proto]["hits"]
            n = len(h[k])                       # anchors that have a depth-k target
            for j in range(n):
                hit = h[k][j]
                before = all(h[d][j] for d in range(k))
                unc.append(hit); pref.append(hit and before)
                if before:
                    cond.append(hit)
        out.append(dict(k=k + 1, n=len(unc), a=rate(unc), cond=rate(cond), n_cond=len(cond), prefix=rate(pref)))
    return out


def main():
    path = sys.argv[1]
    depths = 8
    seqs = json.load(open(path))
    max_i = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9      # keep only prompts i < max_i (balanced families)
    seqs = sorted([s for s in seqs if s["i"] < max_i], key=lambda s: s["i"])
    depths = min(depths, len(seqs[0]["teacher"]["hits"]))
    n_pos = sum(len(s["teacher"]["hits"][0]) for s in seqs)
    print("sequences: %d, first-draft positions: %d, head self-check (python norm+unembed == dumped argmax): %.4f" % (
        len(seqs), n_pos, rate([s["head_agree"] for s in seqs])))
    fams = sorted({s["family"] for s in seqs})
    print("per family sequences:", {FAMILIES[f]: sum(1 for s in seqs if s["family"] == f) for f in fams})

    L_MS, T_MS = 466.0, 45.0
    for proto, label in (("teacher", "A: true-token context (draft context re-extended on verified tokens)"),
                         ("own", "B: own top-1 tokens as context (cheap chain, never corrected)"),
                         ("own_noctx", "C: own tokens, depths >= 2 with NO context (floor: deeper modules' state not maintained)")):
        if proto not in seqs[0]:
            continue
        st = chain_stats(seqs, proto, depths)
        print("\n### Protocol %s\n" % label)
        print("| draft k | n | a_k (unconditional) | a_k given drafts 1..k-1 right (n) | P(drafts 1..k all right) |")
        print("|---|---|---|---|---|")
        for r in st:
            print("| %d | %d | %.3f | %.3f (%d) | %.3f |" % (r["k"], r["n"], r["a"], r["cond"], r["n_cond"], r["prefix"]))
        e = sum(r["prefix"] for r in st)
        print("\nexpected accepted drafts per chain of %d: E = %.2f; renewal model (L + E*T)/(1 + E) with L = %.0f ms, T = %.0f ms: "
              "%.0f ms per token = %.1f tok/s (a constant a would need to be %.2f for the same E)" % (
                  depths, e, L_MS, T_MS, (L_MS + e * T_MS) / (1 + e), 1000.0 * (1 + e) / (L_MS + e * T_MS), e / (1 + e)))

    print("\n### By family (protocol A; a_k unconditional, in brackets: given the chain right so far)\n")
    hdr = "| family | seqs | n | a1 | a2 | a3 | a4 | P(1..2) | P(1..3) | P(1..4) | E[accepted of 8] | implied tok/s |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for f in fams + [-1]:
        ss = [s for s in seqs if s["family"] == f] if f >= 0 else seqs
        st = chain_stats(ss, "teacher", depths)
        name = FAMILIES[f] if f >= 0 else "**all**"
        cells = ["%.2f (%.2f)" % (st[k]["a"], st[k]["cond"]) if k else "%.3f" % st[k]["a"] for k in range(min(4, depths))]
        e = sum(r["prefix"] for r in st)
        print("| %s | %d | %d | %s | %.2f | %.2f | %.2f | %.2f | %.1f |" % (
            name, len(ss), st[0]["n"], " | ".join(cells), st[1]["prefix"], st[2]["prefix"], st[3]["prefix"],
            e, 1000.0 * (1 + e) / (466.0 + e * 45.0)))

    print("\n### By family, protocol B (own tokens as context)\n")
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for f in fams + [-1]:
        ss = [s for s in seqs if s["family"] == f] if f >= 0 else seqs
        st = chain_stats(ss, "own", depths)
        name = FAMILIES[f] if f >= 0 else "**all**"
        cells = ["%.2f (%.2f)" % (st[k]["a"], st[k]["cond"]) if k else "%.3f" % st[k]["a"] for k in range(min(4, depths))]
        e = sum(r["prefix"] for r in st)
        print("| %s | %d | %d | %s | %.2f | %.2f | %.2f | %.2f | %.1f |" % (
            name, len(ss), st[0]["n"], " | ".join(cells), st[1]["prefix"], st[2]["prefix"], st[3]["prefix"],
            e, 1000.0 * (1 + e) / (466.0 + e * 45.0)))

    print("\n### Teeth checks (first draft, a1; all positions)\n")
    ref = [h for s in seqs for h in s["teacher"]["hits"][0]]
    print("| variant | a1 |"); print("|---|---|")
    print("| reference (hidden first, both norms, post-final-norm input, no output norm, de-interleaved w13) | %.3f |" % rate(ref))
    for v in seqs[0]["variants"]:
        print("| %s | %.3f |" % (v, rate([h for s in seqs for h in s["variants"][v]])))
    prose = [s for s in seqs if s["family"] in (0, 3)]
    if prose:
        print("\nprose only (explain + story): reference %.3f; " % rate([h for s in prose for h in s["teacher"]["hits"][0]]) +
              "; ".join("%s %.3f" % (v, rate([h for s in prose for h in s["variants"][v]])) for v in seqs[0]["variants"]))

    print("\n### Vocabulary prefix (first draft)\n")
    tg = [t for s in seqs for t in s["target_ids"]]
    acc = [t for s in seqs for t, h in zip(s["target_ids"], s["teacher"]["hits"][0]) if h]
    print("| | all target tokens (n=%d) | accepted first drafts (n=%d) |" % (len(tg), len(acc))); print("|---|---|---|")
    for n in (32768, 65536, 131072):
        print("| id < %d | %.4f | %.4f |" % (n, rate([t < n for t in tg]), rate([t < n for t in acc])))
    print("\n| draft unembed restricted to ids < n | a1 |"); print("|---|---|")
    for n in ("32768", "65536", "131072"):
        print("| %s | %.3f |" % (n, rate([h for s in seqs for h in s["prefix"][n]])))
    print("| full (200058) | %.3f |" % rate(ref))


if __name__ == "__main__":
    main()
