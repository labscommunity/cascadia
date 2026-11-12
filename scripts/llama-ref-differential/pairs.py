#!/usr/bin/env python3
"""Pairs for check (d): each schema that needs hoisting as the response,
with the next such schema as a tool. Reads the corpus, the engine's output
for it (response prefix and tool prefix `t0`) and the variants (two
prefixes, so the two never share a copy's name); writes three files of one
pair per line: `{"response":..., "tool":...}` inlined, the same as
variants, and the pair's corpus indices."""

import json
import sys

corpus, inl, inl_tool, var_a, var_b, out_inl, out_var, out_meta = sys.argv[1:9]
rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
inl, inl_tool, var_a, var_b = rows(inl), rows(inl_tool), rows(var_a), rows(var_b)
hoisted = [i for i, r in enumerate(inl) if r.get("hoisted")]
with open(out_inl, "w") as fi, open(out_var, "w") as fv, open(out_meta, "w") as fm:
    for a, b in zip(hoisted, hoisted[1:] + hoisted[:1]):
        fi.write(json.dumps({"response": inl[a]["inlined"], "tool": inl_tool[b]["inlined"]}) + "\n")
        fv.write(json.dumps({"response": var_a[a], "tool": var_b[b]}) + "\n")
        fm.write(f"{a}+{b}\n")
