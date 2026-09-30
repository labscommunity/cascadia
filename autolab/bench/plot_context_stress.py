#!/usr/bin/env python3
"""plot_context_stress.py EXP: figures and a per-size table from experiments/EXP/context_stress.json (autolab 047).

Writes context-prefill.{png,svg,pdf} (time to first token and prefill rate against context, log-log, with the fitted
c + aN + bN^2), context-decode.{png,svg,pdf} (decode tok/s against context, with the per-box probe of 034 when
present), context-boxes.{png,svg,pdf} (per box: CPU cores busy and GPU busy during decode at each context) and
context-summary.csv. Needs matplotlib (python3.11 on the operator's Mac).
"""
import csv, json, os, statistics as st, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit_quadratic_with_intercept(pts):
    """least squares T = c + a N + b N^2 (3x3 normal equations, no numpy)"""
    n = len(pts)
    s = lambda f: sum(f(x, y) for x, y in pts)
    A = [[n, s(lambda x, y: x), s(lambda x, y: x * x)],
         [s(lambda x, y: x), s(lambda x, y: x * x), s(lambda x, y: x ** 3)],
         [s(lambda x, y: x * x), s(lambda x, y: x ** 3), s(lambda x, y: x ** 4)]]
    B = [s(lambda x, y: y), s(lambda x, y: x * y), s(lambda x, y: x * x * y)]
    # gaussian elimination
    for i in range(3):
        p = A[i][i]
        for j in range(i + 1, 3):
            f = A[j][i] / p
            for k in range(3):
                A[j][k] -= f * A[i][k]
            B[j] -= f * B[i]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (B[i] - sum(A[i][k] * x[k] for k in range(i + 1, 3))) / A[i][i]
    return x


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "047_context_stress"
    d = os.path.join(LAB, "experiments", exp)
    data = json.load(open(os.path.join(d, "context_stress.json")))
    rows = [r for r in data["results"] if not r.get("skipped") and not r.get("in_flight") and not r.get("error")]
    by = {}
    for r in rows:
        by.setdefault((r["streams"], r["size"]), []).append(r)
    summary = []
    for (streams, size), rs in sorted(by.items()):
        n = st.mean(r["prompt_tokens"] for r in rs)
        cpu = [b["cpu_mean"] for r in rs for b in r.get("boxes", {}).values() if b.get("cpu_mean") is not None]
        gpu = [b["gpu_mean"] for r in rs for b in r.get("boxes", {}).values() if b.get("gpu_mean") is not None]
        pw = [b["pkg_w_mean"] for r in rs for b in r.get("boxes", {}).values() if b.get("pkg_w_mean") is not None]
        mem = [b["mem_avail_min_mb"] for r in rs for b in r.get("boxes", {}).values() if b.get("mem_avail_min_mb") is not None]
        summary.append(dict(streams=streams, size=size, prompt_tokens=round(n), repeats=len(rs),
                            ttft_s=round(st.mean(r["ttft_s"] for r in rs), 1), ttft_sd=round(st.pstdev(r["ttft_s"] for r in rs), 1),
                            prefill_tok_s=round(st.mean(r["prefill_tok_s"] for r in rs), 1),
                            decode_tok_s=round(st.mean(r["decode_tok_s"] for r in rs), 2), decode_sd=round(st.pstdev(r["decode_tok_s"] for r in rs), 2),
                            decode_min=min(r["decode_tok_s"] for r in rs), decode_max=max(r["decode_tok_s"] for r in rs),
                            aggregate_tok_s=round(st.mean(r.get("aggregate_decode_tok_s") or r["decode_tok_s"] for r in rs), 2),
                            needle=sum(1 for r in rs if r["needle_found"]), tokens=round(st.mean(r["tokens"] for r in rs)),
                            cores_busy_mean=round(st.mean(cpu) * 16, 2) if cpu else None, gpu_busy_pct=round(st.mean(gpu) * 100, 1) if gpu else None,
                            pkg_w_mean=round(st.mean(pw), 1) if pw else None, min_free_gb=round(min(mem) / 1024, 2) if mem else None))
    with open(os.path.join(d, "context-summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    one = [s for s in summary if s["streams"] == 1]
    pts = [(s["prompt_tokens"], s["ttft_s"]) for s in one]
    c, a, b = fit_quadratic_with_intercept(pts)
    print("fit: T_prefill(N) = %.1f s + %.3e s/token * N + %.3e s/token^2 * N^2  (linear part %.0f tok/s)" % (c, a, b, 1 / a))
    for n, t in pts:
        print("  N=%6d measured %7.1f s  fit %7.1f s" % (n, t, c + a * n + b * n * n))
    for n in (131072, 262144, 1048576):
        print("  predicted N=%7d: %.1f h" % (n, (c + a * n + b * n * n) / 3600))

    # figure 1: prefill
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [s["prompt_tokens"] for s in one]
    ax[0].errorbar(xs, [s["ttft_s"] for s in one], yerr=[s["ttft_sd"] for s in one], fmt="o-", label="measured (mean of repeats)")
    grid = [2 ** k for k in range(10, 19)]
    ax[0].plot(grid, [c + a * n + b * n * n for n in grid], "--", color="gray", label="fit c + aN + bN²")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("context (prompt tokens)"); ax[0].set_ylabel("time to first token (s)")
    ax[0].set_title("Prefill time against context (one stream)"); ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
    ax[1].plot(xs, [s["prefill_tok_s"] for s in one], "o-")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("context (prompt tokens)"); ax[1].set_ylabel("prefill throughput (tokens/s)")
    ax[1].set_title("Prefill rate against context"); ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(d, "context-prefill." + ext), dpi=150)
    # figure 2: decode, with the probe's per-box cost if present
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(xs, [s["decode_tok_s"] for s in one], yerr=[s["decode_sd"] for s in one], fmt="o-", label="one stream, measured")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("context (prompt tokens)"); ax[0].set_ylabel("decode (tokens/s)")
    ax[0].set_title("Decode speed against context (one stream)"); ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
    probe = os.path.join(LAB, "experiments", "034_context_scan", "probe_cx.json")
    if os.path.exists(probe):
        pr = [r for r in json.load(open(probe)) if int(r.get("fits", 0))]
        sizes = sorted({int(r["ctx"]) for r in pr})
        med = [st.median(int(r["decode_ms"]) for r in pr if int(r["ctx"]) == s) for s in sizes]
        att = [st.median(int(r["attn_ms"]) for r in pr if int(r["ctx"]) == s) for s in sizes]
        ax[1].plot(sizes, med, "o-", label="one token, one box (median of 11)")
        ax[1].plot(sizes, att, "s--", label="of which CPU attention")
        ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("context (positions)"); ax[1].set_ylabel("ms per token per box")
        ax[1].set_title("Per-box cost of one token (probe, 034)"); ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(d, "context-decode." + ext), dpi=150)
    # figure 3: what the boxes do
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, [s["cores_busy_mean"] for s in one], "o-", label="CPU cores busy (mean over boxes)")
    ax.plot(xs, [s["gpu_busy_pct"] / 10 for s in one], "s-", label="iGPU busy (% / 10)")
    ax.set_xscale("log", base=2); ax.set_xlabel("context (prompt tokens)"); ax.set_title("What a box does during a request")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(d, "context-boxes." + ext), dpi=150)
    print("| streams | context | n | first token (s) | prefill tok/s | decode tok/s (min-max) | needle | cores busy | iGPU busy | W | min free GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        print("| %d | %d | %d | %.0f ± %.0f | %.1f | %.2f (%.2f-%.2f) | %d/%d | %s | %s %% | %s | %s |" % (
            s["streams"], s["prompt_tokens"], s["repeats"], s["ttft_s"], s["ttft_sd"], s["prefill_tok_s"], s["decode_tok_s"], s["decode_min"], s["decode_max"],
            s["needle"], s["repeats"], s["cores_busy_mean"], s["gpu_busy_pct"], s["pkg_w_mean"], s["min_free_gb"]))


if __name__ == "__main__":
    main()
