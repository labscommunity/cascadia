#!/usr/bin/env python3
"""Fold the 2026-10-04 router-aware MoE campaign into docs/perf/sycl-elastic/data.json.

Reads the partial.py/moe_cache.py run JSONs (moe2/, moec/, dense2/, b27/)
and rewrites the MoE sections. The pre-0002 (every-expert-streamed) MoE
arms are kept as MoE_all_experts so fig8 can draw the before/after.
"""
import json, glob, sys
from statistics import mean, pstdev

ROOT = "/home/sergio/projects/cascadia-upstream/experiments/2026-10-04-elastic-campaign-v3/pr/partial"
DATA = "/home/sergio/projects/cascadia-upstream/docs/perf/sycl-elastic/data.json"
GIB = 1024 ** 3


def runs(sub, arm, ready_only=True):
    subs = sub if isinstance(sub, list) else [sub]
    out = []
    for f in sorted(sum([glob.glob(f"{ROOT}/{s_}/MoE-{arm}-r*.json") for s_ in subs], [])):
        r = json.load(open(f))
        if not ready_only or r.get("ready"):
            out.append(r)
    return out


def stat(rs, key):
    vals = [r[key] for r in rs]
    err = pstdev(vals) if len(vals) > 1 else 0.0
    return [mean(vals), err]


def peak_gib(rs):
    vals = [(r["peak_vram_bytes"] - r["baseline_vram_bytes"]) / GIB for r in rs]
    return [mean(vals), pstdev(vals) if len(vals) > 1 else 0.0]


d = json.load(open(DATA))

old_moe = d["partial"]["models"]["MoE"]

new_arms = []
for arm, budget, mode, sub in [
    ("v0", 0, "streamed", ["moe2", "moefix"]), ("v4", 4, "streamed", "moe2"),
    ("v10", 10, "streamed", "moe2"), ("v16", 16, "streamed", "moe2"),
    ("auto", "auto", "fits", "moe2"),
]:
    rs = runs(sub, arm)
    if not rs:
        sys.exit(f"no runs for {arm}")
    e = {
        "arm": arm, "budget_gib": budget, "mode": mode, "n": len(rs),
        "peak_vram_gib": [round(x, 2) for x in peak_gib(rs)],
        "decode_tps": [round(x, 3) for x in stat(rs, "decode_tps")],
        "prefill_tps": [round(x, 2) for x in stat(rs, "prefill_tps")],
        "load_s": [round(mean([r["load_s"] for r in rs]), 1), 0.0],
        "expert_stream": "router-aware (0002)",
    }
    if arm == "v0":
        n_layers = 40
    new_arms.append(e)

if "MoE_all_experts" not in d["partial"]["models"]:
    d["partial"]["models"]["MoE_all_experts"] = old_moe  # pre-0002, kept for fig8
d["partial"]["models"]["MoE"] = new_arms

# parity: old 23/23 + every ready new streamed/auto arm verified token-identical
ref = json.load(open(f"{ROOT}/moe2/MoE-stock-r1.json"))["tokens"]
unf = {  # dense references are the unfused arms (streaming disables fusion)
    "1.5B": json.load(open(f"{ROOT}/sweep/1.5B-unfused-r1.json"))["tokens"],
    "27B": json.load(open(f"{ROOT}/sweep/27B-unfused-r1.json"))["tokens"],
}
checked = identical = 0
for sub, models in [("moe2", ["MoE"]), ("moec", ["MoE"]), ("moefix", ["MoE"]), ("dense2", ["1.5B"]), ("b27", ["27B"])]:
    for f in sorted(glob.glob(f"{ROOT}/{sub}/*-v*-r*.json")) + sorted(glob.glob(f"{ROOT}/{sub}/*-c*-r*.json")) + sorted(glob.glob(f"{ROOT}/{sub}/*-k*-r*.json")) + sorted(glob.glob(f"{ROOT}/{sub}/*-auto-r*.json")):
        if "/broken-" in f:
            continue  # pre-pin-fix run, known-bad tokens
        r = json.load(open(f))
        if not r.get("ready") or not r.get("tokens"):
            continue
        checked += 1
        want = unf.get(r["model"], ref)
        if r["tokens"] == want:
            identical += 1
        else:
            print("PARITY FAIL:", f, file=sys.stderr)
old_total = d["partial"]["parity"]["total"]
d["partial"]["parity"] = {
    "identical": 23 + identical, "total": 23 + checked,
    "note": "23 runs from the 0001 campaign + %d runs with router-aware expert "
            "streaming (0002); dense references are unfused stock" % checked,
}

# single-campaign MoE elastic rows -> new v0 (n=3)
v0 = runs(["moe2", "moefix"], "v0")
for sec in ("single_v3_campaign", "single"):
    if sec in d and "35B_a3b_elastic" in d.get(sec, {}):
        e = d[sec]["35B_a3b_elastic"]
        e["n"] = len(v0)
        e["load_s"] = [round(mean([r["load_s"] for r in v0]), 1), 0.0]
        e["peak_vram_gib"] = [round(x, 2) for x in peak_gib(v0)]
        e["vram_after_load_gib"] = [round(x, 2) for x in peak_gib(v0)]
        e["decode_tps"] = [round(x, 3) for x in stat(v0, "decode_tps")]
        e["prefill_tps"] = [round(x, 2) for x in stat(v0, "prefill_tps")]

# expert cache arms (experimental): c0/c4 (2 GiB cache) + k64/k512 controls
cache = {}
for arm in ["c0", "c4", "k64", "k512"]:
    rs = runs(["moec", "moefix"], arm)
    if not rs:
        continue
    cache[arm] = {
        "n": len(rs),
        "peak_vram_gib": [round(x, 2) for x in peak_gib(rs)],
        "decode_tps": [round(x, 3) for x in stat(rs, "decode_tps")],
        "prefill_tps": [round(x, 2) for x in stat(rs, "prefill_tps")],
    }
d["expert_cache_arms"] = {
    "note": "experimental GGML_STREAM_EXPERT_CACHE_MB hot-expert cache; "
            "2 GiB per-tensor LRU was measured SLOWER than selective-only "
            "despite 59-67% hit rates (see RESULTS doc); default is off",
    "arms": cache,
}

d.setdefault("metric_note", "")
d["metric_note"] += (" | 2026-10-04 router-aware campaign: moe2/ (partial.py, "
                     "n=3 stock+v0) and moec/ (moe_cache.py c/k arms) under "
                     "experiments/2026-10-04-elastic-campaign-v3/pr/partial/.")

json.dump(d, open(DATA, "w"), indent=1)
print(f"parity now {d['partial']['parity']['identical']}/{d['partial']['parity']['total']}")
print("MoE arms:", [(a['arm'], a['decode_tps'], a['peak_vram_gib']) for a in new_arms])
print("cache arms:", {k: v['decode_tps'] for k, v in cache.items()})
