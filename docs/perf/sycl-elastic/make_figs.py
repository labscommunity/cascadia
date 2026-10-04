#!/usr/bin/env python3
"""Figures for the sycl-llama --elastic PR. All numbers come from
data.json (a verbatim copy of experiments/2026-10-04-elastic-campaign-v3/pr/
pr-summary.json). Run with the campaign venv (has matplotlib + Pillow):

    experiments/2026-10-04-elastic-campaign-v3/venv/bin/python make_figs.py

QC (runs on every figure before it is written): every rendered text must
lie fully inside the figure, no two texts may overlap, and no dashed
reference line may pass through a text label (labels on reference lines
carry a solid panel-colour backing box). Any violation exits nonzero.
"""
import itertools
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).parent
DATA = json.loads((HERE / "data.json").read_text())

# --- theme ---
BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GRID = "#21262d"
STOCK = "#8b949e"
ELASTIC = "#3fb950"
AMBER = "#d29922"
BLUE = "#58a6ff"
RED = "#f85149"

B70_HEAP_GIB = 31.9  # labelled constant: Arc Pro B70 device memory
PARITY_NOTE = ("64/64 greedy tokens byte-identical to stock (fusion off), "
               "Qwen3.8-27B")

FOOTER = ("Intel Arc Pro B70 32 GB | llama.cpp SYCL + stream-weights patch | "
          "Qwen GGUF Q4_K, ctx 4096, KV q8_0 | data: docs/perf/sycl-elastic/data.json")
FOOTER_LAPTOP = "Hunter laptop, Windows 11 | data: docs/perf/sycl-elastic/data.json"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "savefig.facecolor": BG,
})

QC_ERRORS = []
REF_LABEL_BOX = dict(boxstyle="round,pad=0.25", facecolor=PANEL,
                     edgecolor="none")


def newfig(h=6.75):
    return plt.figure(figsize=(12, h), dpi=160)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.text(0.045, 0.955, title, fontsize=20, fontweight="bold", color=TEXT,
             ha="left", va="top")
    fig.text(0.045, 0.885, subtitle, fontsize=12, color=MUTED, ha="left",
             va="top")


def footer(fig, text=FOOTER):
    fig.text(0.045, 0.018, text, fontsize=9, color=MUTED, ha="left",
             va="bottom")


def sf(v):
    """2 sig figs below 0.1, else 2 decimals."""
    return f"{v:.2f}" if v >= 0.1 else f"{v:.2g}"


def _ink(buf, b):
    H, W = buf.shape[0], buf.shape[1]
    x0 = max(int(b.x0) - 1, 0)
    x1 = min(int(b.x1) + 1, W)
    y0 = max(int(buf.shape[0] - b.y1) - 1, 0)
    y1 = min(int(buf.shape[0] - b.y0) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(buf[y0:y1, x0:x1, :3].max() > 60)


def qc_figure(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    buf = np.asarray(fig.canvas.buffer_rgba())
    W, H = fig.canvas.get_width_height()
    legend_text_ids = set()
    for lg in fig.findobj(matplotlib.legend.Legend):
        legend_text_ids.update(id(t) for t in lg.get_texts())
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in legend_text_ids:
            continue
        if not t.get_visible():
            continue
        s = t.get_text().strip()
        if not s:
            continue
        try:
            # for annotations, the window extent includes the leader arrow;
            # the visible label box is its FancyBboxPatch
            if t.get_bbox_patch() is not None:
                bb = t.get_bbox_patch().get_window_extent(renderer=r)
            else:
                bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 1 or bb.height <= 1:
            continue
        if not _ink(buf, bb):
            continue
        items.append((s, bb, t))
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
            QC_ERRORS.append((name, "text out of figure bounds",
                              f"{s[:50]!r} bbox={bb}"))
    for lg in fig.findobj(matplotlib.legend.Legend):
        if not lg.get_visible():
            continue
        try:
            bb = lg.get_window_extent(renderer=r)
        except Exception:
            continue
        items.append(("<legend>", bb, lg))
    for (s1, b1, _), (s2, b2, _) in itertools.combinations(items, 2):
        if b1.overlaps(b2):
            inter = b1.intersection(b1, b2)
            if inter is None:
                continue
            area = max(inter.width, 0) * max(inter.height, 0)
            mn = min(b1.width * b1.height, b2.width * b2.height)
            if mn and area / mn > 0.08:
                QC_ERRORS.append((name, "text overlap",
                                  f"{s1[:32]!r} <-> {s2[:32]!r} "
                                  f"({area / mn:.2f})"))
    # text/legend extents must not overlap bar Rectangles
    own_boxes = {id(t.get_bbox_patch()) for _, _, t in items
                 if hasattr(t, "get_bbox_patch")
                 and t.get_bbox_patch() is not None}
    bar_rects = []
    for ax in fig.axes:
        for p in ax.patches:
            if id(p) in own_boxes or p is ax.patch:
                continue
            if type(p) is not matplotlib.patches.Rectangle:
                continue
            try:
                bb = p.get_window_extent(renderer=r)
                # log-scale bars transform their y=0 base to -inf; clip to
                # the axes bounding box so extents stay meaningful
                bb = bb.intersection(bb, ax.get_window_extent(renderer=r))
            except Exception:
                continue
            if bb is not None and bb.width > 0 and bb.height > 0:
                bar_rects.append(bb)
    for s, bb, _ in items:
        for rb in bar_rects:
            if not bb.overlaps(rb):
                continue
            inter = bb.intersection(bb, rb)
            if inter is None:
                continue
            area = max(inter.width, 0) * max(inter.height, 0)
            own = bb.width * bb.height
            if not own:
                continue
            frac = area / own
            # >=95% inside a bar = deliberate in-segment label; only
            # flag partial straddles
            if 0.10 < frac < 0.95:
                QC_ERRORS.append((name, "text overlaps bar",
                                  f"{s[:40]!r} ({frac:.2f})"))
                break
    # data markers must not sit under a text label
    marker_rects = []
    dpi = fig.dpi
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_marker() or ln.get_marker() == "None":
                continue
            ms = ln.get_markersize()
            if ms <= 0:
                continue
            r = ms * dpi / 72.0 / 2.0 + 3
            try:
                xy = np.asarray(ln.get_xydata(), dtype=float)
                disp = ax.transData.transform(xy)
            except Exception:
                continue
            for px, py in disp:
                marker_rects.append(
                    matplotlib.transforms.Bbox.from_extents(
                        px - r, py - r, px + r, py + r))
    for s, bb, _ in items:
        for mb in marker_rects:
            if not bb.overlaps(mb):
                continue
            inter = bb.intersection(bb, mb)
            if inter is None:
                continue
            frac = (max(inter.width, 0) * max(inter.height, 0)
                    / (mb.width * mb.height))
            if frac > 0.2:
                QC_ERRORS.append((name, "text covers marker",
                                  f"{s[:40]!r} ({frac:.2f})"))
                break
    # dashed reference lines must not cross an unboxed text label
    for ax in fig.axes:
        for ln in ax.lines:
            if ln.get_linestyle() not in ("--", "-.", ":", "dashed",
                                        "dashdot", "dotted"):
                continue
            try:
                xy = np.asarray(ln.get_xydata(), dtype=float)
                disp = ln.get_transform().transform(xy)
            except Exception:
                continue
            t_lin = np.linspace(0.0, 1.0, 80)[:, None]
            pts = disp[0] + t_lin * (disp[-1] - disp[0])
            for s, bb, txt in items:
                if not hasattr(txt, "get_bbox_patch"):
                    continue
                if txt.get_bbox_patch() is not None:
                    continue  # label carries a solid backing box
                hit = ((pts[:, 0] >= bb.x0 - 2) & (pts[:, 0] <= bb.x1 + 2) &
                       (pts[:, 1] >= bb.y0 - 2) & (pts[:, 1] <= bb.y1 + 2))
                if hit.any():
                    QC_ERRORS.append((name, "reference line crosses text",
                                      f"{s[:50]!r}"))


def save(fig, name):
    qc_figure(fig, name)
    out = HERE / name
    fig.savefig(out)
    plt.close(fig)
    try:
        from PIL import Image
        im = Image.open(out).convert("RGB").quantize(colors=256)
        im.save(out)
    except Exception as e:
        print(f"quantize skipped for {name}: {e}")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


S = DATA["single"]


def v(key, field):
    return S[key][field][0]  # mean


MODELS = [
    ("Qwen2.5-1.5B", "1_5B_stock", "1_5B_elastic"),
    ("Qwen3.8-27B", "27B_stock", "27B_elastic"),
    ("Qwen3.6-35B-A3B MoE", "35B_a3b_stock", "35B_a3b_elastic"),
]


# ---------------- fig0 ----------------
def fig0():
    fig = newfig(4.9)
    fig.text(0.045, 0.955, "Run big models on whatever VRAM you have",
             fontsize=20, fontweight="bold", color=TEXT, va="top")
    fig.text(0.045, 0.88,
             "cascadia --engine sycl-llama --elastic keeps what fits on the "
             "GPU and streams the rest from the GGUF",
             fontsize=12, color=MUTED, va="top")

    conc = {f["fleet"]: f for f in DATA["concurrent_v3"]}
    n5 = conc["conc_5x27B"]
    p27s, p27e = v("27B_stock", "peak_vram_gib"), v("27B_elastic", "peak_vram_gib")
    p27 = {a["arm"]: a for a in DATA["partial"]["models"]["27B"]}
    v12 = p27["v12"]
    streamed_12 = v12["n_layers"] - v12["resident_layers"]
    cot2 = DATA["cotenant_auto"]["cotenant-auto-2"]
    par = DATA["partial"]["parity"]

    cards = [
        (f"{p27e:.1f} GiB", "27B fully streamed",
         f"was {p27s:.1f} GiB resident\n"
         f"(-{round(100 * (1 - p27e / p27s))}%)"),
        (f"{v12['decode_tps'][0]:.1f} t/s",
         f"27B in {v12['peak_vram_gib'][0]:.1f} GiB",
         f"{streamed_12} of {v12['n_layers']} layers streamed;\n"
         f"resident {v('27B_stock', 'decode_tps'):.1f} t/s"),
        ("5 x 27B", "on one 32 GB card",
         f"all 5 generating at once,\n{n5['peak_vram_gib']:.1f} GiB peak\n"
         "stock: 3 copies overflow"),
        ("2 x 27B", "auto-placed on one card",
         f"{cot2['aggregate_concurrent_tps']:.1f} t/s together,\n"
         f"{cot2['peak_vram_gib']:.1f} GiB peak"),
    ]
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, gap, x0, y0, h = 0.215, 0.025, 0.045, 0.40, 0.34
    for i, (big, mid, sub) in enumerate(cards):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.008",
                                    facecolor=PANEL, edgecolor=GRID,
                                    linewidth=1.2))
        ax.text(x + w / 2, y0 + h * 0.76, big, fontsize=28, fontweight="bold",
                color=ELASTIC, ha="center", va="center")
        ax.text(x + w / 2, y0 + h * 0.50, mid, fontsize=12.5, color=TEXT,
                ha="center", va="center")
        ax.text(x + w / 2, y0 + h * 0.20, sub, fontsize=9.5, color=MUTED,
                ha="center", va="center")
    v8 = p27["v8"]
    v0 = p27["v0"]
    fig.text(0.045, 0.30,
             f"{par['identical']}/{par['total']} runs token-identical to "
             "stock (fusion off when streaming)",
             fontsize=12, color=ELASTIC, fontweight="bold", va="top")
    fig.text(0.045, 0.23,
             f"Speed follows the streamed share: 27B "
             f"{v('27B_stock', 'decode_tps'):.1f} t/s resident -> "
             f"{v12['decode_tps'][0]:.1f} "
             f"({v12['streamed_mib'] / 1024:.1f} GiB/token streamed) -> "
             f"{v8['decode_tps'][0]:.1f} "
             f"({v8['streamed_mib'] / 1024:.1f} GiB) -> "
             f"{v0['decode_tps'][0]:.2f} t/s fully streamed.",
             fontsize=12, color=MUTED, va="top")
    footer(fig)
    save(fig, "fig0_hero.png")


# ---------------- fig1 ----------------
def fig1():
    fig = newfig()
    header(fig, "Peak VRAM: stock vs --elastic",
           "Peak device memory during load + generation, mean of 3 "
           "cold-cache runs (std < 0.01 GiB)")
    ax = fig.add_axes([0.20, 0.16, 0.68, 0.60])
    style_ax(ax)
    ax.grid(axis="x", color=GRID, alpha=0.6)
    ax.grid(axis="y", visible=False)

    y, labels = [], []
    for i, (name, ks, ke) in enumerate(MODELS):
        g = i * 1.0
        ax.barh(g + 0.19, v(ks, "peak_vram_gib"), height=0.36, color=STOCK,
                label="stock" if i == 0 else None)
        ax.barh(g - 0.19, v(ke, "peak_vram_gib"), height=0.36, color=ELASTIC,
                label="fully streamed (--elastic-vram 0)" if i == 0 else None)
        sv, ev = v(ks, "peak_vram_gib"), v(ke, "peak_vram_gib")
        ax.text(sv + 0.25, g + 0.19, f"{sv:.2f} GiB", va="center", fontsize=11,
                fontweight="bold", color=TEXT)
        ax.text(ev + 0.25, g - 0.19, f"{ev:.2f} GiB", va="center", fontsize=11,
                fontweight="bold", color=ELASTIC)
        ax.text(ev + 4.2, g - 0.19, f"-{round(100 * (1 - ev / sv))}%",
                va="center", fontsize=11, fontweight="bold", color=BG,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=ELASTIC,
                          edgecolor="none"))
        y.append(g)
        labels.append(name)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color=TEXT)
    ax.set_xlabel("GiB", fontsize=12)
    ax.set_xlim(0, 34)
    top = len(MODELS) - 0.3
    for x, lab in [(8, "8 GB card"), (16, "16 GB card"),
                   (B70_HEAP_GIB, "Arc Pro B70 device memory")]:
        ax.axvline(x, color=AMBER, linestyle="--", linewidth=1.2)
        ha, xx = ("right", x - 0.4) if x == B70_HEAP_GIB else ("center", x)
        ax.text(xx, top - 0.02, lab, fontsize=10, color=AMBER, ha=ha,
                va="top", bbox=REF_LABEL_BOX)
    ax.set_ylim(-0.55, top)
    ax.legend(loc="center right", bbox_to_anchor=(0.97, 0.42), fontsize=10,
              frameon=True, facecolor=PANEL, edgecolor=GRID)
    footer(fig)
    save(fig, "fig1_vram.png")


# ---------------- fig2 ----------------
def fig2():
    conc = {f["fleet"]: f for f in DATA["concurrent_v3"]}
    el = conc["conc_5x27B"]["vram_after_each_load_gib"]
    st = DATA["fleets_v2"]["fleet_3x27B_stock"]["vram_after_each_load_gib"]
    per = sum(el[i] - el[i - 1] for i in range(1, len(el))) / (len(el) - 1)

    fig = newfig()
    header(fig, "How many 27B models fit on one GPU",
           f"VRAM after each load. Each additional --elastic 27B costs "
           f"~{per:.1f} GiB; each stock 27B ~15 GiB")
    ax = fig.add_axes([0.09, 0.14, 0.86, 0.62])
    style_ax(ax)

    xs = list(range(1, len(el) + 1))
    ax.fill_between(xs, el, color=ELASTIC, alpha=0.15)
    ax.plot(xs, el, color=ELASTIC, marker="o", linewidth=2, label="fully streamed (--elastic-vram 0)")
    for x, yy in zip(xs, el):
        off = (0, -17) if x == xs[-1] else (0, 9)
        ax.annotate(f"{yy:.1f}", (x, yy), textcoords="offset points",
                    xytext=off, fontsize=11, fontweight="bold",
                    color=ELASTIC, ha="center")
    ax.text(5.05, 14.4, f"all 5 generating at once,\npeak "
            f"{conc['conc_5x27B']['peak_vram_gib']:.1f} GiB",
            fontsize=11, color=ELASTIC, ha="right", va="bottom")

    xs_s = list(range(1, len(st) + 1))
    ax.plot(xs_s, st, color=STOCK, marker="o", linewidth=2, label="stock")
    for x, yy in zip(xs_s, st):
        off = (10, -6) if x == xs_s[-1] else (0, -18)
        ax.annotate(f"{yy:.1f}", (x, yy), textcoords="offset points",
                    xytext=off, fontsize=11, fontweight="bold",
                    color=STOCK, ha="left" if x == xs_s[-1] else "center")
    ax.text(3.55, 27.6, "card full: +15 GiB spilled to host RAM,\n"
            "concurrent load crashed the host (OOM)",
            fontsize=10.5, color=RED, ha="left", va="top")
    ax.plot(xs_s[-1], st[-1], marker="x", color=RED, markersize=12,
            markeredgewidth=3)

    ax.axhline(B70_HEAP_GIB, color=AMBER, linestyle="--", linewidth=1.2)
    ax.text(4.6, B70_HEAP_GIB + 0.5, "Arc Pro B70 device memory (31.9 GiB)",
            fontsize=10, color=AMBER, ha="center", bbox=REF_LABEL_BOX)
    ax.set_xlabel("27B instances loaded on one card", fontsize=12)
    ax.set_ylabel("device VRAM (GiB)", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 36)
    ax.set_xlim(0.7, 5.4)
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor=PANEL,
              edgecolor=GRID)
    footer(fig)
    save(fig, "fig2_packing.png")


# ---------------- fig3 ----------------
def fig3():
    fig = newfig()
    header(fig, "Starts faster, because weights aren't copied to the GPU "
                "up front",
           "Cold page cache, process start to ready, mean of 3")
    ax = fig.add_axes([0.10, 0.14, 0.85, 0.60])
    style_ax(ax)
    for i, (name, ks, ke) in enumerate(MODELS):
        ax.bar(i - 0.19, v(ks, "load_s"), width=0.36, color=STOCK,
               label="stock" if i == 0 else None)
        ax.bar(i + 0.19, v(ke, "load_s"), width=0.36, color=ELASTIC,
               label="fully streamed (--elastic-vram 0)" if i == 0 else None)
        ax.text(i - 0.19, v(ks, "load_s") + 0.5, f"{v(ks, 'load_s'):.1f} s",
                ha="center", fontsize=11, fontweight="bold", color=TEXT)
        ax.text(i + 0.19, v(ke, "load_s") + 0.5, f"{v(ke, 'load_s'):.1f} s",
                ha="center", fontsize=11, fontweight="bold", color=ELASTIC)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[0] for m in MODELS], fontsize=12, color=TEXT)
    ax.set_ylabel("seconds", fontsize=12)
    ax.set_ylim(0, 34)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    footer(fig)
    save(fig, "fig3_load.png")


# ---------------- fig4 ----------------
def fig4():
    fig = newfig()
    header(fig, "The cost of streaming everything (--elastic-vram 0)",
           "Every token re-reads all layer weights at ~7 GB/s. With "
           "--elastic-vram auto, only what does not fit is\nstreamed "
           "(see fig8).")
    axl = fig.add_axes([0.07, 0.16, 0.40, 0.55])
    style_ax(axl)
    for i, (name, ks, ke) in enumerate(MODELS):
        sv, ev = v(ks, "decode_tps"), v(ke, "decode_tps")
        axl.bar(i - 0.19, sv, width=0.36, color=STOCK,
                label="stock" if i == 0 else None)
        axl.bar(i + 0.19, ev, width=0.36, color=ELASTIC,
                label="fully streamed (--elastic-vram 0)" if i == 0 else None)
        axl.text(i - 0.19, sv * 1.15, f"{sv:.1f}", ha="center", fontsize=11,
                 fontweight="bold", color=TEXT)
        axl.text(i + 0.19, ev * 1.15, f"{ev:.2f}" if ev < 1 else f"{ev:.1f}",
                 ha="center", fontsize=11, fontweight="bold", color=ELASTIC)
    axl.set_yscale("log")
    axl.set_xticks(range(len(MODELS)))
    axl.set_xticklabels([f"{m[0]}\n{round(v(m[1], 'decode_tps') / v(m[2], 'decode_tps'))}x slower" for m in MODELS], fontsize=11, color=TEXT)
    axl.set_ylabel("decode, tokens/s (log)", fontsize=12)
    axl.set_ylim(0.15, 900)
    axl.set_yticks([0.2, 1, 5, 20, 100, 500])
    axl.set_yticklabels(["0.2", "1", "5", "20", "100", "500"])
    axl.legend(loc="upper right", fontsize=11, frameon=False)

    axr = fig.add_axes([0.57, 0.16, 0.38, 0.55])
    style_ax(axr)
    lat = DATA["latency"]

    def m(key, field):
        vals = lat[key][field]
        return sum(vals) / len(vals)

    groups = [
        ("27B\nfirst token", m("lat_27B_stock", "warm_ttft_s"),
         m("lat_27B_elastic", "warm_ttft_s")),
        ("27B\nper token", m("lat_27B_stock", "warm_itl_median_s"),
         m("lat_27B_elastic", "warm_itl_median_s")),
        ("35B\nfirst token", m("lat_35B_a3b_stock", "warm_ttft_s"),
         m("lat_35B_a3b_elastic", "warm_ttft_s")),
        ("35B\nper token", m("lat_35B_a3b_stock", "warm_itl_median_s"),
         m("lat_35B_a3b_elastic", "warm_itl_median_s")),
    ]
    for i, (name, sv, ev) in enumerate(groups):
        axr.bar(i - 0.19, sv, width=0.36, color=STOCK,
                label="stock" if i == 0 else None)
        axr.bar(i + 0.19, ev, width=0.36, color=ELASTIC,
                label="fully streamed (--elastic-vram 0)" if i == 0 else None)
        axr.text(i - 0.19, sv * 1.15, sf(sv), ha="center", fontsize=10,
                 fontweight="bold", color=TEXT)
        axr.text(i + 0.19, ev * 1.15, sf(ev), ha="center", fontsize=10,
                 fontweight="bold", color=ELASTIC)
    axr.set_yscale("log")
    axr.set_xticks(range(len(groups)))
    axr.set_xticklabels([g[0] for g in groups], fontsize=10, color=TEXT)
    axr.set_ylabel("warm latency, seconds (log)", fontsize=12)
    axr.set_ylim(0.0095, 40)
    axr.set_yticks([0.01, 0.1, 1, 10])
    axr.set_yticklabels(["0.01", "0.1", "1", "10"])
    axr.legend(loc="upper right", fontsize=11, frameon=False)
    axr.set_title("latency: campaign v3", fontsize=10, color=MUTED, pad=8)
    footer(fig)
    save(fig, "fig4_tradeoff.png")


# ---------------- fig5 ----------------
def fig5():
    fig = newfig()
    header(fig, "A mixed lineup on one GPU, all generating at the same time",
           "2x Qwen3.8-27B + 3x Qwen2.5-1.5B, each its own cascadia "
           "--elastic instance")
    mix = next(f for f in DATA["concurrent_v3"] if f["fleet"] == "conc_mix")
    names = ["27B #1", "27B #2", "1.5B #1", "1.5B #2", "1.5B #3"]

    axl = fig.add_axes([0.06, 0.16, 0.42, 0.60])
    style_ax(axl)
    cols = [BLUE, BLUE, ELASTIC, ELASTIC, ELASTIC]
    axl.bar(range(5), mix["per_instance_tps"], color=cols, width=0.6)
    for i, tp in enumerate(mix["per_instance_tps"]):
        axl.text(i, tp + 0.05, f"{tp:.2f}", ha="center", fontsize=11,
                 fontweight="bold", color=TEXT)
    axl.set_xticks(range(5))
    axl.set_xticklabels(names, fontsize=11, color=TEXT)
    axl.set_ylabel("decode, tokens/s", fontsize=12)
    axl.set_ylim(0, 2.7)
    axl.text(2, 2.45, f"aggregate {mix['aggregate_tps']:.2f} t/s",
             ha="center", fontsize=13, fontweight="bold", color=TEXT)

    axr = fig.add_axes([0.60, 0.16, 0.34, 0.60])
    style_ax(axr)
    series = mix["vram_after_each_load_gib"]
    prev = 0.0
    for i, (n, upto) in enumerate(zip(names, series)):
        h = upto - prev
        axr.bar(0, h, bottom=prev, width=0.5, color=cols[i],
                edgecolor=BG, linewidth=1)
        if h > 1.5:
            axr.text(0.45, prev + h / 2, f"{n}: {h:.2f} GiB", fontsize=10.5,
                     color=TEXT, va="center")
        prev = upto
    axr.text(0.45, (series[1] + series[-1]) / 2,
             f"3x 1.5B: {series[-1] - series[1]:.2f} GiB", fontsize=10.5,
             color=TEXT, va="center")
    axr.axhline(B70_HEAP_GIB, color=AMBER, linestyle="--", linewidth=1.2)
    axr.text(0.4, B70_HEAP_GIB + 1.0, "B70 device memory", fontsize=10,
             color=AMBER, ha="center", bbox=REF_LABEL_BOX)
    axr.text(0, prev + 1.6,
             f"{prev:.1f} GiB loaded /\n{mix['peak_vram_gib']:.1f} GiB peak",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=TEXT)
    axr.set_xticks([])
    axr.set_ylabel("device VRAM (GiB)", fontsize=12)
    axr.set_ylim(0, 34)
    axr.set_xlim(-0.9, 1.7)
    footer(fig)
    save(fig, "fig5_fleet.png")


# ---------------- fig6 ----------------
def fig6():
    fig = newfig()
    header(fig, "On a laptop: pick the right device",
           "OpenVINO GenAI 2026.4 (the runtime under cascadia's ov-genai "
           "engine),\n48 tokens after a warmup pass. sycl-llama --elastic "
           "was not measured on this laptop.")
    bench = DATA["hunter"]["bench"]
    dev_color = {"CPU": AMBER, "GPU.0": BLUE, "GPU.1": "#a371f7",
                 "NPU": "#db61a2"}
    labels = {
        "qwen1.5b_cpu": "Qwen2.5-1.5B int8 - CPU",
        "qwen1.5b_igpu": "Qwen2.5-1.5B int8 - Arc 140T iGPU",
        "qwen1.5b_rtx": "Qwen2.5-1.5B int8 - RTX 5060 (via OpenVINO)",
        "qwen1.5b_npu": "Qwen2.5-1.5B int8 - NPU (AI Boost)",
        "gemma_e2b_igpu": "Gemma-4-E2B int4 - Arc 140T iGPU",
        "gemma_e2b_npu": "Gemma-4-E2B int4 - NPU",
    }
    order = ["qwen1.5b_cpu", "qwen1.5b_igpu", "qwen1.5b_rtx", "qwen1.5b_npu",
             "gemma_e2b_igpu", "gemma_e2b_npu"]
    rows = {b["tag"]: b for b in bench}

    axl = fig.add_axes([0.30, 0.16, 0.36, 0.58])
    style_ax(axl)
    axl.grid(axis="x", color=GRID, alpha=0.6)
    axl.grid(axis="y", visible=False)
    for i, tag in enumerate(order):
        b = rows[tag]
        y = len(order) - 1 - i
        if b["tok_s"] is None:
            axl.plot(1.2, y, marker="x", color=RED, markersize=12,
                     markeredgewidth=3)
            axl.text(2.4, y, "compile failed (NPU driver)", fontsize=10.5,
                     color=RED, va="center")
        else:
            axl.barh(y, b["tok_s"], height=0.55, color=dev_color[b["device"]])
            axl.text(b["tok_s"] + 0.6, y, f"{b['tok_s']:.1f} t/s",
                     fontsize=10.5, fontweight="bold", color=TEXT,
                     va="center")
    axl.set_yticks(range(len(order)))
    axl.set_yticklabels([labels[t] for t in reversed(order)], fontsize=10.5,
                        color=TEXT)
    axl.set_xlabel("decode, tokens/s", fontsize=12)
    axl.set_xlim(0, 46)

    axr = fig.add_axes([0.76, 0.16, 0.20, 0.58])
    style_ax(axr)
    cot = DATA["hunter"]["cpu_cotenancy"]
    cpu_shades = ["#d29922", "#e0aa35", "#b9831d"]
    axr.bar(0, cot["single"]["aggregate_tps"], width=0.55,
            color=cpu_shades[0])
    axr.text(0, cot["single"]["aggregate_tps"] + 1,
             f"{cot['single']['aggregate_tps']:.1f}", ha="center",
             fontsize=11, fontweight="bold", color=TEXT)
    prev = 0.0
    for tp, c in zip(cot["triple"]["per_instance_tps"], cpu_shades):
        axr.bar(1, tp, bottom=prev, width=0.55, color=c, edgecolor=BG,
                linewidth=1)
        axr.text(1, prev + tp / 2, f"{tp:.1f}", ha="center", fontsize=10,
                 color=BG, fontweight="bold")
        prev += tp
    axr.text(1, prev + 1, f"{cot['triple']['aggregate_tps']:.1f} aggregate",
             ha="center", fontsize=11, fontweight="bold", color=TEXT)
    axr.set_xticks([0, 1])
    axr.set_xticklabels(["1 process", "3 processes"], fontsize=11, color=TEXT)
    axr.set_ylabel("CPU decode, tokens/s", fontsize=12)
    axr.set_ylim(0, 48)
    footer(fig, FOOTER_LAPTOP)
    save(fig, "fig6_laptop.png")


# ---------------- fig7 ----------------
def fig7():
    fig = newfig()
    header(fig, "Repeatable and leak-free",
           "15 load/unload cycles of a 1.5B model, VRAM settled back to "
           "baseline every time.\nBefore this PR (10x 1.5B + 5x 27B, fixed "
           "120 s timeout): 11/15 cycles ready - after (15x 1.5B, auto "
           "timeout + retry): 15/15")

    lr = DATA["lifecycle_retry"]
    axl = fig.add_axes([0.06, 0.17, 0.38, 0.50])
    style_ax(axl)
    n = len(lr["load_s"])
    xs = list(range(1, n + 1))
    retried = [i + 1 for i, t in enumerate(lr["load_s"]) if t > 20]
    cols = [AMBER if x in retried else ELASTIC for x in xs]
    axl.bar(xs, lr["load_s"], width=0.6, color=cols)
    for x in retried:
        axl.text(x + 0.4, lr["load_s"][x - 1] - 6,
                 f"retried after xe reset - {lr['load_s'][x - 1]:.0f} s",
                 fontsize=10.5, color=AMBER, fontweight="bold",
                 ha="left", va="top")
    axl.set_xlabel("load/unload cycle (1.5B stock, SYCL1)", fontsize=12)
    axl.set_ylabel("ready time (s)", fontsize=12)
    axl.set_xticks(xs)
    axl.set_ylim(0, 85)
    settled = lr["vram_settled_gib"][0]
    fig.text(0.29, 0.062,
             f"VRAM after unload = baseline ({settled:.3f} GiB) in "
             f"{lr['gen_ok']}/{lr['cycles']} cycles - 0 leaks - child "
             "never outlives cascadia",
             fontsize=9.5, color=TEXT, ha="center")

    # parity: card1 = SYCL1 (single.* runs), card2 = SYCL0 (parity runs)
    p = {t["tag"]: t for t in DATA["parity"]}
    cats_dec = [("card 1\nstock", v("27B_stock", "decode_tps")),
                ("card 2\nstock", p["par_27B_stock_SYCL0"]["decode_tps"]),
                ("card 1\nelastic", v("27B_elastic", "decode_tps")),
                ("card 2\nelastic", p["par_27B_elastic_SYCL0"]["decode_tps"])]
    cats_vram = [("card 1\nstock", v("27B_stock", "peak_vram_gib")),
                 ("card 2\nstock", p["par_27B_stock_SYCL0"]["peak_vram_gib"]),
                 ("card 1\nelastic", v("27B_elastic", "peak_vram_gib")),
                 ("card 2\nelastic", p["par_27B_elastic_SYCL0"]["peak_vram_gib"])]
    cols = [STOCK, STOCK, ELASTIC, ELASTIC]
    hatches = [None, "//", None, "//"]

    axp = fig.add_axes([0.49, 0.17, 0.22, 0.50])
    style_ax(axp)
    for i, ((name, val), c) in enumerate(zip(cats_dec, cols)):
        axp.bar(i, val, width=0.55, color=c, hatch=hatches[i],
                edgecolor=BG if hatches[i] else "none")
        axp.text(i, val * 1.15, f"{val:.2f}" if val < 10 else f"{val:.1f}",
                 ha="center", fontsize=10, fontweight="bold", color=TEXT)
    axp.set_yscale("log")
    axp.set_xticks(range(4))
    axp.set_xticklabels([c[0] for c in cats_dec], fontsize=9.5, color=TEXT)
    axp.set_ylabel("27B decode, t/s (log)", fontsize=11)
    axp.set_ylim(0.4, 45)
    axp.set_yticks([0.5, 1, 5, 20])
    axp.set_yticklabels(["0.5", "1", "5", "20"])
    axp.set_title("decode parity\ncard1=SYCL1  card2=SYCL0", fontsize=11,
                  color=MUTED)

    axp2 = fig.add_axes([0.755, 0.17, 0.22, 0.50])
    style_ax(axp2)
    for i, ((name, val), c) in enumerate(zip(cats_vram, cols)):
        axp2.bar(i, val, width=0.55, color=c, hatch=hatches[i],
                 edgecolor=BG if hatches[i] else "none")
        axp2.text(i, val + 0.4, f"{val:.1f}", ha="center", fontsize=10,
                  fontweight="bold", color=TEXT)
    axp2.set_xticks(range(4))
    axp2.set_xticklabels([c[0] for c in cats_vram], fontsize=9.5, color=TEXT)
    axp2.set_ylabel("27B peak VRAM, GiB", fontsize=11)
    axp2.set_ylim(0, 19)
    axp2.set_title("VRAM parity\ncard1=SYCL1  card2=SYCL0", fontsize=11,
                   color=MUTED)
    footer(fig)
    save(fig, "fig7_reliability.png")


# ---------------- fig8 ----------------
def fig8():
    fig = newfig()
    header(fig, "--elastic-vram: speed follows how much of the model fits",
           "Keep the first N layers on the GPU, stream the rest from the "
           "GGUF each token.\nauto picks N from free VRAM; if the whole "
           "model fits, streaming turns off.")
    box = dict(boxstyle="round,pad=0.25", facecolor=PANEL,
               edgecolor=GRID, linewidth=0.8)
    lead = dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                shrinkA=4, shrinkB=6)
    # per-arm label anchor (x, y data coords) for the leadered label
    lab_pos = {
        "27B": {"v0": (2.4, 0.16), "v2": (9.5, 0.16), "v4": (3.0, 3.0),
                "v8": (7.5, 18.0), "v10": (11.5, 50.0), "v12": (19.0, 5.0)},
        "MoE": {"v0": (3.0, 0.14), "v4": (5.0, 2.2), "v10": (10.5, 2.6),
                "v16": (16.5, 8.0)},
    }
    clus_pos = {"27B": (18.5, 150.0), "MoE": (16.0, 148.0)}
    for ax, key, name, kstock in [
            (fig.add_axes([0.075, 0.15, 0.40, 0.60]), "27B",
             "Qwen3.8-27B", "27B_stock"),
            (fig.add_axes([0.565, 0.15, 0.40, 0.60]), "MoE",
             "Qwen3.6-35B-A3B MoE", "35B_a3b_stock")]:
        style_ax(ax)
        arms = DATA["partial"]["models"][key]
        pts = sorted(arms, key=lambda a: a["peak_vram_gib"][0])
        streamed = [a for a in pts if a["mode"] == "streamed"]
        fits = [a for a in pts if a["mode"] != "streamed"]
        xs = [a["peak_vram_gib"][0] for a in streamed]
        ys = [a["decode_tps"][0] for a in streamed]
        ax.plot(xs, ys, "-o", color=ELASTIC, markersize=7, linewidth=1.6)
        sv = v(kstock, "decode_tps")
        sp = v(kstock, "peak_vram_gib")
        ax.plot([sp], [sv], "s", color=STOCK, markersize=9)
        # fits arms land on the stock point: one hollow green ring over it
        if fits:
            ax.plot([sp], [sv], "o", color=ELASTIC, markersize=13,
                    markerfacecolor="none", markeredgewidth=1.8)
        for a in streamed:
            x, y = a["peak_vram_gib"][0], a["decode_tps"][0]
            lx, ly = lab_pos[key][a["arm"]]
            ax.annotate(f"--elastic-vram {a['budget_gib']}\n{y:.2f} t/s",
                        xy=(x, y), xytext=(lx, ly),
                        ha="center", va="center", fontsize=9.5, color=TEXT,
                        bbox=box, arrowprops=lead)
        fit_names = " / ".join(
            "auto" if a["arm"] == "auto" else f"v{a['budget_gib']}"
            for a in fits)
        cx, cy = clus_pos[key]
        ax.annotate(f"stock = {fit_names} (fits)\n{sv:.1f} t/s",
                    xy=(sp, sv), xytext=(cx, cy),
                    ha="center", va="center", fontsize=9.5, color=ELASTIC,
                    bbox=box, arrowprops=lead)
        ax.set_yscale("log")
        ax.set_xlim(0, 23)
        ax.set_ylim(0.1, 400)
        ax.set_yticks([0.2, 1, 5, 20, 100])
        ax.set_yticklabels(["0.2", "1", "5", "20", "100"])
        ax.set_xlabel("peak VRAM, GiB", fontsize=12)
        ax.set_ylabel("decode, tokens/s (log)", fontsize=12)
        ax.set_title(name, fontsize=13, color=TEXT)
        if key == "MoE":
            ax.text(22.6, 0.13,
                    "MoE streams every expert today\n(router-aware "
                    "streaming is future work)",
                    fontsize=9.5, color=MUTED, ha="right", va="bottom")
    par = DATA["partial"]["parity"]
    footer(fig, "Intel Arc Pro B70 32 GB | sycl-llama --elastic-vram sweep | "
                f"{par['identical']}/{par['total']} streamed/auto runs "
                "token-identical to reference")
    save(fig, "fig8_partial.png")


# ---------------- fig9 ----------------
WIN = {
    # Windows 11, Arc B390 iGPU (Core Ultra X7 358H), oneAPI 2026.0 + MSVC 19.44,
    # llama.cpp SYCL + stream-weights patch. Device memory = per-process GPU
    # perf counters (llama-cli), peak over load + generation, ~4 Hz sampling.
    # Both arms ran with -c 4096 (see data.json windows_delta notes).
    "1_5B": {"stock_gib": 1.463, "streamed_gib": 0.778,
             "stock_tps": 80.8, "streamed_tps": 5.7},
    "27B":  {"stock_gib": 15.376, "streamed_gib": 2.606,
             "stock_tps": 6.3, "streamed_tps": 0.4},
}
B390_POOL_GIB = 16.4


def fig9():
    fig = newfig()
    header(fig, "The same streaming, verified on Windows",
           "llama.cpp SYCL + stream-weights patch, built on Windows 11 "
           "(oneAPI 2026.0, MSVC 19.44)\nfor an Arc B390 iGPU (16.4 GiB "
           "device pool); device memory = per-process GPU counters "
           "(llama-cli),\npeak over load + generation; greedy tokens "
           "identical to stock.")
    rows = [
        ("Qwen2.5-1.5B\nLinux (B70)", v("1_5B_stock", "peak_vram_gib"),
         v("1_5B_elastic", "peak_vram_gib"), v("1_5B_stock", "decode_tps"),
         v("1_5B_elastic", "decode_tps")),
        ("Qwen2.5-1.5B\nWindows (B390)", WIN["1_5B"]["stock_gib"],
         WIN["1_5B"]["streamed_gib"], WIN["1_5B"]["stock_tps"],
         WIN["1_5B"]["streamed_tps"]),
        ("Qwen3.8-27B\nLinux (B70)", v("27B_stock", "peak_vram_gib"),
         v("27B_elastic", "peak_vram_gib"), v("27B_stock", "decode_tps"),
         v("27B_elastic", "decode_tps")),
        ("Qwen3.8-27B\nWindows (B390)", WIN["27B"]["stock_gib"],
         WIN["27B"]["streamed_gib"], WIN["27B"]["stock_tps"],
         WIN["27B"]["streamed_tps"]),
    ]

    axa = fig.add_axes([0.185, 0.16, 0.40, 0.58])
    style_ax(axa)
    axa.grid(axis="x", color=GRID, alpha=0.6)
    axa.grid(axis="y", visible=False)
    for i, (name, sv, ev, st, et) in enumerate(rows):
        g = i * 1.0
        axa.barh(g + 0.19, sv, height=0.36, color=STOCK,
                 label="stock" if i == 0 else None)
        axa.barh(g - 0.19, ev, height=0.36, color=ELASTIC,
                 label="streamed" if i == 0 else None)
        if sv > 5:
            axa.text(sv - 0.5, g + 0.19, f"{sv:.2f}", va="center", ha="right",
                     fontsize=11, fontweight="bold", color=BG)
        else:
            axa.text(sv + 0.35, g + 0.19, f"{sv:.2f}", va="center", ha="left",
                     fontsize=11, fontweight="bold", color=TEXT)
        axa.text(ev + 0.35, g - 0.19, f"{ev:.2f}", va="center", fontsize=11,
                 fontweight="bold", color=ELASTIC)
        axa.text(max(sv, ev) + 4.6, g, f"-{round(100 * (1 - ev / sv))}%",
                 va="center", fontsize=11, fontweight="bold", color=BG,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=ELASTIC,
                           edgecolor="none"))
    axa.set_yticks([0, 1, 2, 3])
    axa.set_yticklabels([r[0] for r in rows], fontsize=11, color=TEXT)
    axa.set_xlabel("peak device memory, GiB", fontsize=12)
    axa.set_xlim(0, 34)
    top = 3.75
    for x, lab in [(B390_POOL_GIB, "Arc B390 pool, 16.4"),
                   (B70_HEAP_GIB, "Arc Pro B70 pool, 31.9")]:
        axa.axvline(x, color=AMBER, linestyle="--", linewidth=1.2)
        axa.text(x - 0.4, top - 0.02, lab, fontsize=10, color=AMBER,
                 ha="right", va="top", bbox=REF_LABEL_BOX)
    axa.set_ylim(-0.55, top + 0.13)
    axa.legend(loc="center right", bbox_to_anchor=(0.98, 0.35), fontsize=10,
               frameon=True, facecolor=PANEL, edgecolor=GRID)

    axb = fig.add_axes([0.655, 0.16, 0.315, 0.58])
    style_ax(axb)
    axb.grid(axis="x", color=GRID, alpha=0.6)
    axb.grid(axis="y", visible=False)
    for i, (name, sv, ev, st, et) in enumerate(rows):
        ret = 100.0 * et / st
        col = BLUE if "Linux" in name else ELASTIC
        axb.barh(i, ret, height=0.5, color=col)
        axb.text(ret + 0.35, i, f"{ret:.1f}%   {et:.1f} vs {st:.1f} t/s",
                 va="center", fontsize=10.5, color=TEXT)
    axb.set_yticks([0, 1, 2, 3])
    axb.set_yticklabels(["" for _ in rows])
    axb.set_xlim(0, 15)
    axb.set_xlabel("decode retained under streaming, % of stock", fontsize=12)
    axb.set_title("capacity tier, not a speed path", fontsize=11, color=MUTED)
    footer(fig, "Arc Pro B70 32 GB (Linux) + Arc B390 iGPU (Windows) | "
                "llama.cpp SYCL + stream-weights patch | Qwen GGUF Q4_K | "
                "data: docs/perf/sycl-elastic/data.json")
    save(fig, "fig9_windows.png")


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9()
    if QC_ERRORS:
        print("\n=== QC FAILURES ===")
        for name, kind, detail in QC_ERRORS:
            print(f"{name} | {kind} | {detail}")
        sys.exit(1)
    print("QC: all figures clean")
