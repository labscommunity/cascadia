#!/usr/bin/env python3
"""collect_pages.py PREFIX OUT [minutes]: poll the signed status for rank 0's log lines 'PREFIX<n>| text' until PREFIXEND."""
import json, os, re, subprocess, sys, time
prefix, out, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 30
seen, order, t0 = {}, [], time.time()
pat = re.compile(re.escape(prefix) + r"(\d+|END)\| ?(.*)$")
while time.time() - t0 < minutes * 60:
    try:
        r = subprocess.run(["/usr/bin/python3", os.path.expanduser("~/inkling-release/bin/release.py"), "status", "--json"], capture_output=True, text=True, timeout=60)
        doc = json.loads(r.stdout)
        text = json.dumps(doc)
        lines = []
        def walk(o):
            if isinstance(o, str): lines.extend(o.split("\n"))
            elif isinstance(o, list): [walk(v) for v in o]
            elif isinstance(o, dict): [walk(v) for v in o.values()]
        walk(doc)
        end = False
        for l in lines:
            m = pat.search(l)
            if not m: continue
            key = (m.group(1), m.group(2))
            if key not in seen:
                seen[key] = 1; order.append(key)
            if m.group(1) == "END": end = True
        with open(out, "w") as f:
            for k, v in order: f.write("%s| %s\n" % (k, v))
        if end: break
    except Exception as e:
        pass
    time.sleep(20)
print("collected", len(order), "lines ->", out)
