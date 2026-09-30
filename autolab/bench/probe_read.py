#!/usr/bin/env python3
"""Print the probe lines the ranks smuggled out as "stage profile" records (see experiments/014).

    probe_read.py [PREFIX]      PREFIX = tag prefix of the probe lines, default P14
"""
import json, os, sys, urllib.request

API = os.environ.get("INKLING_API", "http://localhost:18000")
prefix = sys.argv[1] if len(sys.argv) > 1 else "P14"
doc = json.loads(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(API + "/api/fleet/telemetry", timeout=15).read())
rows = []
for r, e in sorted(doc.get("ranks", {}).items(), key=lambda kv: int(kv[0])):
    for p in e.get("profs", []):
        if str(p.get("at", "")).startswith(prefix):
            rows.append(dict(p, rank=int(r)))
keys = ["rank", "at"] + sorted({k for p in rows for k in p} - {"rank", "at", "rt"})
print(" ".join(k.rjust(9) for k in keys))
for p in sorted(rows, key=lambda p: (p["at"], p["rank"])):
    print(" ".join(str(p.get(k, "")).rjust(9) for k in keys))
if len(sys.argv) > 2:
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
