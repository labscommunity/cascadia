#!/usr/bin/env python3
"""Figures for the B70 placement study (docs/perf/sycl-elastic/
placement-b70.md). All numbers come from placement-b70.json (frozen
medians from experiments/2026-10-05-placement-b70/). Same style + QC as
make_figs.py. Run with the campaign venv:

    experiments/2026-10-04-elastic-campaign-v3/venv/bin/python \
        make_figs_placement_b70.py
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
# reuse the theme + QC machinery from make_figs.py
from make_figs import (  # noqa: E402
    AMBER, BG, BLUE, ELASTIC, GRID, MUTED, QC_ERRORS, RED, REF_LABEL_BOX,
    STOCK, TEXT, footer, header, newfig, qc_figure, save, sf, style_ax,
)

DATA = json.loads((HERE / "placement-b70.json").read_text())

FOOTER_B70 = ("Arc Pro B70 (SYCL0, PCIe 4.0 x8) | stream-weights + "
              "host-buffer patches | 27B Q4_K_S, ctx 4096 | "
              "data: placement-b70.json")

B = DATA["B"]
NS = [62, 55, 34, 0]
XLAB = ["62", "55", "34 / 33", "0"]
STREAM_TPS = B["stream_tps"]
HOST_TPS = B["host_tps"]
LAW_TPS = B["law_13gbps_tps"]


def fig10():
    fig = newfig(h=6.4)
    header(fig, "27B on one B70: streaming vs pinned host memory "
                "(PCIe 4.0 x8)",
           "Decode t/s by number of weight layers kept on the card; "
           "streaming re-reads the rest from the file, host-in-place reads\n"
           "pinned host memory in place. Log scale. Temp 0, 64 tokens, "
           "median of 3. Streaming wins the extremes, host the middle.")
    ax = fig.add_axes([0.085, 0.13, 0.87, 0.58])
    style_ax(ax)
    ax.set_yscale("log")
    x = np.arange(len(NS))
    w = 0.38
    b1 = ax.bar(x - w / 2, STREAM_TPS, w, color=ELASTIC,
                label="weight streaming (patch 0001)")
    b2 = ax.bar(x + w / 2, HOST_TPS, w, color=BLUE,
                label="pinned host memory (patch 0003, -ot SYCL_Host)")
    ax.bar_label(b1, fmt="%.2f", fontsize=10.5, color=TEXT, padding=3)
    ax.bar_label(b2, fmt="%.2f", fontsize=10.5, color=TEXT, padding=3)
    # deficit-law markers sit between the two bars, clear of their labels
    ax.plot(x, LAW_TPS, linestyle="none", color=AMBER, marker="v",
            markersize=8, label="deficit law @ 13 GB/s")
    ax.axhline(B["resident_tps"], color=STOCK, linestyle="--", linewidth=1.2)
    ax.text(-0.42, B["resident_tps"] * 1.08,
            f"resident: {B['resident_tps']:.2f} t/s", fontsize=10,
            color=MUTED, ha="left", va="bottom", bbox=REF_LABEL_BOX)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=12)
    ax.set_xlabel("layers resident on device (of 65)", fontsize=12)
    ax.set_ylabel("decode t/s (log)", fontsize=12)
    ax.set_ylim(0.3, 42)
    ax.set_xlim(-0.55, 3.55)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.62), fontsize=10,
              frameon=True, facecolor="#161b22", edgecolor=GRID)
    footer(fig, FOOTER_B70)
    save(fig, "fig10_placement_b70.png")


def fig11():
    fig = newfig(h=6.0)
    header(fig, "How far the PCIe link actually gets",
           "Effective host-to-device read rate implied by each decode rate "
           "(streamed bytes per token x t/s). Host-in-place reads direct;\n"
           "streaming stages through a bounce buffer. Dashed line = PCIe 4.0 "
           "x8 theoretical peak; neither mechanism gets near it.")
    ax = fig.add_axes([0.085, 0.13, 0.87, 0.60])
    style_ax(ax)
    x = np.arange(len(NS))
    ax.plot(x, B["stream_gbps"], color=ELASTIC, marker="o", markersize=7,
            linewidth=2, label="weight streaming")
    ax.plot(x, B["host_gbps"], color=BLUE, marker="s", markersize=7,
            linewidth=2, label="pinned host memory")
    for xi, (s, h) in enumerate(zip(B["stream_gbps"], B["host_gbps"])):
        ax.annotate(f"{s:.1f}", (xi, s), textcoords="offset points",
                    xytext=(-12, -18), ha="right", fontsize=10, color=TEXT)
        ax.annotate(f"{h:.1f}", (xi, h), textcoords="offset points",
                    xytext=(12, -18), ha="left", fontsize=10, color=TEXT)
    ceil = DATA["link"]["theoretical_gbps"]
    ax.axhline(ceil, color=AMBER, linestyle="--", linewidth=1.4)
    ax.text(-0.4, ceil + 0.4,
            f"PCIe 4.0 x8 theoretical: {ceil} GB/s", fontsize=10,
            color=AMBER, ha="left", va="bottom", bbox=REF_LABEL_BOX)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=12)
    ax.set_xlabel("layers resident on device (of 65)", fontsize=12)
    ax.set_ylabel("effective link read rate, GB/s", fontsize=12)
    ax.set_ylim(0, 18.5)
    ax.set_xlim(-0.55, 3.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), fontsize=10,
              frameon=True, facecolor="#161b22", edgecolor=GRID, ncol=2)
    footer(fig, FOOTER_B70)
    save(fig, "fig11_link_b70.png")


def fig12():
    fig = newfig(h=6.4)
    header(fig, "Three 27B instances on one B70, all generating",
           "Stacked per-instance decode t/s, 64 tokens each fired "
           "concurrently. Streaming budgets give a bounded footprint but\n"
           "the xe driver's own oversubscription (instance pages evicted to "
           "host) wins on raw throughput.")
    ax = fig.add_axes([0.30, 0.14, 0.62, 0.58])
    style_ax(ax)
    ax.grid(axis="x", color=GRID, alpha=0.6)
    ax.grid(axis="y", visible=False)
    arms = DATA["C"]["arms"]
    insts = DATA["C"]["per_instance_tps"]
    cols = [STOCK, BLUE, ELASTIC]
    y = np.arange(len(arms))[::-1]
    for yi, (arm, vals) in zip(y, zip(arms, insts)):
        left = 0.0
        for k, v_ in enumerate(vals):
            ax.barh(yi, v_, left=left, height=0.6, color=cols[k],
                    edgecolor=BG, linewidth=1)
            if v_ >= 0.9:
                ax.text(left + v_ / 2, yi, f"{v_:.2f}", va="center",
                        ha="center", fontsize=9.5, fontweight="bold",
                        color=BG)
            elif v_ >= 0.28:
                ax.text(left + v_ / 2, yi + 0.36, f"{v_:.2f}", va="bottom",
                        ha="center", fontsize=8, color=MUTED)
            left += v_
        ax.text(12.6, yi, f"{left:.2f} combined", va="center",
                fontsize=11, fontweight="bold", color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(arms, fontsize=11, color=TEXT)
    ax.set_xlabel("decode t/s (stacked: instance 1 + 2 + 3)", fontsize=12)
    ax.set_xlim(0, 15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, ["instance 1", "instance 2", "instance 3"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=10,
              ncol=3, frameon=True, facecolor="#161b22", edgecolor=GRID)
    footer(fig, FOOTER_B70)
    save(fig, "fig12_cotenancy_b70.png")


def fig13():
    fig = newfig(h=6.0)
    header(fig, "Regression check on the review fixes (stack build, SYCL0)",
           "Median decode t/s, 3 runs x 3 prompts each. All targets met; "
           "streamed output is byte-identical to the unpatched upstream\n"
           "build on every prompt. Log scale.")
    axa = fig.add_axes([0.09, 0.14, 0.40, 0.52])
    axb = fig.add_axes([0.60, 0.14, 0.34, 0.52])
    for ax in (axa, axb):
        style_ax(ax)
        ax.set_yscale("log")
    arms = DATA["A"]["arms"]
    tps = DATA["A"]["tps"]
    cols = [STOCK, ELASTIC, ELASTIC, AMBER]
    bb = axa.bar(range(len(arms)), tps, 0.6, color=cols)
    axa.bar_label(bb, fmt="%.2f", fontsize=10.5, color=TEXT, padding=3)
    axa.set_xticks(range(len(arms)))
    axa.set_xticklabels(["resident", "vram 12\n(62/65)", "vram 0\n(0/65)",
                         "auto\n(fits)"], fontsize=9.5)
    axa.set_ylabel("decode t/s (log)", fontsize=11)
    axa.set_title("Qwen3.8-27B Q4_K_S", fontsize=11, color=MUTED)
    axa.set_ylim(0.4, 40)
    axa.set_yticks([1, 10])
    axa.minorticks_off()
    moe = DATA["A"]["moe"]
    bb2 = axb.bar(range(len(moe["arms"])), moe["tps"], 0.6,
                  color=[STOCK, ELASTIC, ELASTIC])
    axb.bar_label(bb2, fmt="%.2f", fontsize=10.5, color=TEXT, padding=3)
    axb.set_xticks(range(len(moe["arms"])))
    axb.set_xticklabels(["resident", "vram 0", "vram 10"], fontsize=9.5)
    axb.set_title("Qwen3.6-35B-A3B Q4_K_XL (MoE)", fontsize=11, color=MUTED)
    axb.set_ylim(0.8, 120)
    axb.set_yticks([1, 10, 100])
    axb.minorticks_off()
    footer(fig, FOOTER_B70)
    save(fig, "fig13_regression_b70.png")


if __name__ == "__main__":
    fig10(); fig11(); fig12(); fig13()
    if QC_ERRORS:
        print("\n=== QC FAILURES ===")
        for name, kind, detail in QC_ERRORS:
            print(f"{name} | {kind} | {detail}")
        sys.exit(1)
    print("QC: all figures clean")
