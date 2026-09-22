#!/usr/bin/env python3
"""analyze.py <name>.jsonl <name>.phases  ->  per-phase, per-rank tables of where the pipeline's time goes."""
import sys, json, collections, datetime, statistics as st

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
phases = collections.OrderedDict()
for l in open(sys.argv[2]):
    f = l.split()
    if len(f) < 3: continue
    p = phases.setdefault(f[0], {})
    p[f[1]] = float(f[2]); p.setdefault("info", " ".join(f[3:]))
# Boxes are offline, their clocks disagree: place every record on rank 0's clock by arrival order.
last0, diffs = None, collections.defaultdict(list)
for r in recs:
    if r["rank"] == 0: last0 = r["t"]
    elif last0 is not None: diffs[r["rank"]].append(r["t"] - last0)
OFF = {k: st.median(v) for k, v in diffs.items()}; OFF[0] = 0.0
for r in recs: r["t"] -= OFF.get(r["rank"], 0.0)
print("clock offsets vs rank 0 (s):", {k: round(v, 1) for k, v in sorted(OFF.items())})
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"   # all | decode (windows without admissions) | admit (windows with)
def ts_raw(at):  # 2026-09-20T05:02:51.617172Z
    return datetime.datetime.strptime(at[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
ts = lambda at, rank=0: ts_raw(at) - OFF.get(rank, 0.0)
mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
BUSY = ("recv_ms", "compute_ms", "prefill_ms", "head_ms", "send_ms", "relay_ms", "emit_ms")
out = {}
for name, ph in phases.items():
    if "end" not in ph: continue
    s, e = ph["start"], ph["end"]
    print("\n" + "=" * 150); print("PHASE %s  (%s)  %.0f s" % (name, ph["info"], e - s))
    rows = []
    for rank in range(11):
        rr = [r for r in recs if r["rank"] == rank]
        sysr = [r["sys"] for r in rr if s + 2 <= r["t"] <= e]
        profs = [r["prof"] for r in rr if "prof" in r]
        inw = [p for p in profs if ts(p["at"], rank) - p["window_ms"] / 1e3 >= s - 0.5 and ts(p["at"], rank) <= e + 0.5]
        if MODE == "decode": inw = [p for p in inw if p.get("opens", 0) == 0]
        if MODE == "admit": inw = [p for p in inw if p.get("opens", 0) > 0]
        tot = collections.Counter()
        for p in inw:
            for k, v in p.items():
                if isinstance(v, int) and k not in ("rank", "total", "cache_mib", "cache_cap_mib", "max_compute_ms", "max_round_trip_ms"): tot[k] += v
        W = tot["window_ms"] or 1
        fr, rw = tot["frames"] or 1, tot["rows"] or 1
        busy = sum(tot[k] for k in BUSY)
        look = tot["cache_hits"] + tot["cache_misses"]
        d = dict(rank=rank, windows=len(inw), frames=tot["frames"], rows_per_frame=tot["rows"] / fr, util=busy / W, wait=tot["wait_ms"] / W,
                 other=max(0, 1 - busy / W - tot["wait_ms"] / W),
                 compute_per_frame=tot["compute_ms"] / fr, compute_per_row=tot["compute_ms"] / rw, max_compute=max([p["max_compute_ms"] for p in inw] or [0]),
                 attn_per_frame=tot["attn_ms"] / fr, mlp_per_frame=tot["mlp_ms"] / fr, ov_attn_per_frame=tot["ov_attn_ms"] / fr,
                 recv_per_frame=tot["recv_ms"] / fr, send_per_frame=tot["send_ms"] / fr, head_per_frame=tot["head_ms"] / fr,
                 relay_ms=tot["relay_ms"], relays=tot["relays"], prefill_ms=tot["prefill_ms"], opens=tot["opens"], open_rows=tot["open_rows"],
                 prefill_attn=tot["prefill_attn_ms"], prefill_mlp=tot["prefill_mlp_ms"], emit_ms=tot["emit_ms"],
                 rtt=tot["round_trip_ms"] / (tot["replies"] or 1), max_rtt=max([p["max_round_trip_ms"] for p in inw] or [0]), replies=tot["replies"],
                 miss=tot["cache_misses"] / look if look else 0.0, misses=tot["cache_misses"], cache_mib=(inw[-1]["cache_mib"] if inw else 0),
                 **{k: mean([x[k] for x in sysr if k in x]) for k in ("cpu", "iowait", "p_cores", "pkg_w", "mhz", "temp", "gpu", "rd_mb_s", "majflt_s", "swapin_s", "swapout_s", "rx_mb_s", "tx_mb_s", "retrans_s", "throttle")},
                 avail_min=min([x.get("mem_avail", 0) for x in sysr] or [0]), anon=max([x.get("p_rssanon", 0) for x in sysr] or [0]), swap=max([x.get("swap", 0) for x in sysr] or [0]),
                 pkg_w_max=max([x.get("pkg_w", 0) for x in sysr] or [0]), mhz_loaded=mean([x["mhz"] for x in sysr if x.get("cpu", 0) > 0.5 and "mhz" in x]))
        rows.append(d)
    out[name] = rows
    print("rank win frames rows/f  util%% wait%% oth%% | ms/frame  ms/row  maxms |  attn   mlp ovattn  recv  send  head | relays relay_ms | miss%%  misses cacheMiB")
    for d in rows:
        print("%4d %3d %6d %6.2f  %5.1f %5.1f %4.1f | %8.1f %7.1f %6d | %5.1f %5.1f %6.1f %5.1f %5.1f %5.1f | %6d %8d | %5.1f %7d %8d" % (
            d["rank"], d["windows"], d["frames"], d["rows_per_frame"], 100 * d["util"], 100 * d["wait"], 100 * d["other"], d["compute_per_frame"], d["compute_per_row"], d["max_compute"],
            d["attn_per_frame"], d["mlp_per_frame"], d["ov_attn_per_frame"], d["recv_per_frame"], d["send_per_frame"], d["head_per_frame"], d["relays"], d["relay_ms"], 100 * d["miss"], d["misses"], d["cache_mib"]))
    r0 = rows[0]
    act = [d for d in rows if d["windows"]]
    if act:
        worst = max(act, key=lambda d: d["compute_per_row"])
        rate = mean([d["frames"] * d["rows_per_frame"] / (d["windows"] * 10.2) for d in act])
        print("FLEET [%s windows]: mean util %.0f%%, rows/s through a rank %.2f; slowest stage rank %d at %.1f ms/row -> a full pipeline at this batch size would do %.1f tok/s" % (
            MODE, 100 * mean([d["util"] for d in act]), rate, worst["rank"], worst["compute_per_row"], 1000.0 / worst["compute_per_row"]))
    print("rank 0: frame round trip avg %.0f ms (max %d) over %d replies; sum of ranks' compute+recv+head per frame = %.0f ms; emit %d ms; admissions %d (%d rows) %d ms" % (
        r0["rtt"], r0["max_rtt"], r0["replies"], sum(d["compute_per_frame"] + d["recv_per_frame"] + d["head_per_frame"] + d["send_per_frame"] for d in rows), r0["emit_ms"], r0["opens"], r0["open_rows"], r0["prefill_ms"]))
    if any(d["opens"] for d in rows):
        print("prefill per rank (ms per prompt row | attn / mlp share):  " + "  ".join("r%d %.1f|%.0f/%.0f" % (d["rank"], d["prefill_ms"] / max(d["open_rows"], 1), d["prefill_attn"] / max(d["open_rows"], 1), d["prefill_mlp"] / max(d["open_rows"], 1)) for d in rows))
    print("rank  cpu%%  cores   pkgW  maxW   MHz MHz@load temp  gpu%%  rdMB/s majf/s swpin/s swpout/s  rxMB/s txMB/s retr/s thr | availMin  anon  swap")
    for d in rows:
        print("%4d %5.1f %6.2f %6.1f %5.1f %5.0f %8.0f %4.0f %5.1f %7.1f %6.0f %7.0f %8.0f %7.2f %6.2f %6.1f %3.0f | %8d %5d %5d" % (
            d["rank"], 100 * d["cpu"], d["p_cores"], d["pkg_w"], d["pkg_w_max"], d["mhz"], d["mhz_loaded"], d["temp"], 100 * d["gpu"], d["rd_mb_s"], d["majflt_s"], d["swapin_s"], d["swapout_s"],
            d["rx_mb_s"], d["tx_mb_s"], d["retrans_s"], d["throttle"], d["avail_min"], d["anon"], d["swap"]))
json.dump(out, open(sys.argv[1].replace(".jsonl", ".summary.json"), "w"))
