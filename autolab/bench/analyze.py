#!/usr/bin/env python3
"""analyze.py EXP [--mode all|decode|admit]: per phase and rank, where the pipeline's time went.

Reads experiments/EXP/phases.json (phase windows on this Mac's clock) and telemetry.jsonl (what lab.py
polled from /api/fleet/telemetry: "lt" = local poll time, "profs" = stage profiles with rank 0's receive time
"rt"). Rank 0's clock and this Mac's differ, so profile times are placed by the poll that first saw them.
"""
import sys, json, os, collections

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp = sys.argv[1]; mode = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--mode" else "all"
d = os.path.join(LAB, "experiments", exp)
phases = json.load(open(os.path.join(d, "phases.json")))
recs = [json.loads(l) for l in open(os.path.join(d, "telemetry.jsonl")) if l.strip()]
mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
BUSY = ("recv_ms", "compute_ms", "prefill_ms", "head_ms", "send_ms", "relay_ms", "emit_ms")
summary = {}
for ph in phases:
    s, e = ph["start"], ph["end"]
    print("\n" + "=" * 170)
    print("PHASE %s: %d streams, %d tok, %.0f s: aggregate %.2f tok/s, steady (sum of streams) %.2f, ttft mean %.1f s" % (
        ph["phase"], ph["streams"], ph["tokens"], ph["wall_s"], ph["aggregate_tok_s"], ph["sum_stream_tok_s"], ph["ttft_mean_s"]))
    rows = []
    for rank in range(11):
        rr = [r for r in recs if r.get("rank") == rank and s - 1 <= r["lt"] <= e + 14]
        profs = [p for r in rr for p in r.get("profs", []) if r["lt"] - p.get("window_ms", 0) / 1e3 >= s - 3]
        if mode == "decode": profs = [p for p in profs if p.get("opens", 0) == 0]
        if mode == "admit": profs = [p for p in profs if p.get("opens", 0) > 0]
        sysr = [r["sys"] for r in rr if r.get("sys") and s + 2 <= r["lt"] <= e]
        tot = collections.Counter()
        for p in profs:
            for k, v in p.items():
                if isinstance(v, (int, float)) and k not in ("rank", "total", "rt", "cache_mib", "cache_cap_mib", "max_compute_ms", "max_round_trip_ms"):
                    tot[k] += v
        W = tot["window_ms"] or 1; fr = tot["frames"] or 1; rw = tot["rows"] or 1
        busy = sum(tot[k] for k in BUSY); look = tot["cache_hits"] + tot["cache_misses"]
        g = lambda k: mean([x[k] for x in sysr if k in x and x[k] is not None])
        rows.append(dict(rank=rank, n=len(profs), frames=tot["frames"], rpf=tot["rows"] / fr, util=busy / W, wait=tot["wait_ms"] / W,
                         cpf=tot["compute_ms"] / fr, cpr=tot["compute_ms"] / rw, maxc=max([p.get("max_compute_ms", 0) for p in profs] or [0]),
                         attn=tot["attn_ms"] / fr, mlp=tot["mlp_ms"] / fr, ovattn=tot["ov_attn_ms"] / fr, ovmoe=tot["ov_moe_ms"] / fr,
                         moefb=tot["ov_moe_fallbacks"], moenf=tot["ov_moe_nonfinite"], recv=tot["recv_ms"] / fr, send=tot["send_ms"] / fr,
                         head=tot["head_ms"] / fr, relays=tot["relays"], relay_ms=tot["relay_ms"], opens=tot["opens"], open_rows=tot["open_rows"],
                         prefill_ms=tot["prefill_ms"], rtt=tot["round_trip_ms"] / (tot["replies"] or 1), maxrtt=max([p.get("max_round_trip_ms", 0) for p in profs] or [0]),
                         replies=tot["replies"], emit=tot["emit_ms"], miss=100.0 * tot["cache_misses"] / look if look else 0.0,
                         spec=(tot["spec_sent"], tot["spec_hits"], tot["spec_misses"]), cache=(profs[-1].get("cache_mib", 0) if profs else 0),
                         cpu=g("cpu"), cores=g("p_cores"), w=g("pkg_w"), mhz=g("mhz"), temp=g("temp"), gpu=g("gpu"), rd=g("rd_mb_s"), swi=g("swapin_s"),
                         rx=g("rx_mb_s"), tx=g("tx_mb_s"), avail=min([x.get("mem_avail", 0) for x in sysr] or [0]), anon=max([x.get("p_rssanon", 0) for x in sysr] or [0])))
    print("rank win frames rows/f  util% wait% | ms/frame ms/row maxms |  attn   mlp ovattn ovmoe(fb/nf) | recv send head | relays(ms) | miss% cacheMiB |  cpu% cores  pkgW  MHz temp gpu% rdMB/s swi/s rxMB/s | availMiB anonMiB")
    for r in rows:
        print("%4d %3d %6d %6.2f  %5.1f %5.1f | %8.1f %6.1f %5d | %5.1f %5.1f %6.1f %5.1f(%d/%d) | %4.1f %4.1f %4.1f | %5d(%4d) | %5.1f %8d | %5.1f %5.2f %5.1f %4.0f %4.0f %4.0f %6.1f %5.0f %6.2f | %8d %7d" % (
            r["rank"], r["n"], r["frames"], r["rpf"], 100 * r["util"], 100 * r["wait"], r["cpf"], r["cpr"], r["maxc"], r["attn"], r["mlp"], r["ovattn"],
            r["ovmoe"], r["moefb"], r["moenf"], r["recv"], r["send"], r["head"], r["relays"], r["relay_ms"], r["miss"], r["cache"],
            100 * r["cpu"], r["cores"], r["w"], r["mhz"], r["temp"], 100 * r["gpu"], r["rd"], r["swi"], r["rx"], r["avail"], r["anon"]))
    r0 = rows[0]; act = [r for r in rows if r["n"]]
    if act:
        worst = max(act, key=lambda r: r["cpr"])
        per_frame = sum(r["cpf"] + r["recv"] + r["head"] + r["send"] for r in rows)
        print("rank 0: round trip avg %.0f ms (max %d) over %d replies; all ranks' work per frame %.0f ms; admissions %d (%d rows, %d ms); emit %d ms; speculation sent/right/wrong %s" % (
            r0["rtt"], r0["maxrtt"], r0["replies"], per_frame, r0["opens"], r0["open_rows"], r0["prefill_ms"], r0["emit"], r0["spec"]))
        print("FLEET: mean util %.0f %%; slowest stage rank %d at %.1f ms/row -> %.1f tok/s if every rank were always busy at this batch size" % (
            100 * mean([r["util"] for r in act]), worst["rank"], worst["cpr"], 1000.0 / max(worst["cpr"], 1e-9)))
    summary[ph["phase"]] = rows
json.dump(summary, open(os.path.join(d, "analysis.json"), "w"))
