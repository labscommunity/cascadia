#!/usr/bin/env python3
"""For each corpus schema (stdin, one per line) prints a variant (stdout,
same order) that llama.cpp's converter must turn into the same grammar up to
rule names: every `#/...` ref is pointed at a copy of its target under a
fresh, collision-free name. llama-server names a ref's rule after its
pointer and reuses a rule that already has the name without visiting the
target, so where the two grammars differ, a ref's rule held something other
than its own target."""

import json
import sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "zqref"


def pointer_target(doc, ref):
    """The target of `#` / `#/...` `ref` as the JSON Schema spec reads it
    (RFC 6901: `~1` is `/`, `~0` is `~`, an array index in decimal), which
    is what the engine's inlining follows."""
    if ref == "#":
        return doc
    if not ref.startswith("#/"):
        return None
    cur = doc
    for tok in ref[2:].split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        elif isinstance(cur, list) and tok.isdigit() and (tok == "0" or not tok.startswith("0")) and int(tok) < len(cur):
            cur = cur[int(tok)]
        else:
            return None
    return cur


def refs_in(v, out, in_literal=False):
    """Every `#/...` ref string outside `const`/`enum` literals."""
    if isinstance(v, list):
        for x in v:
            refs_in(x, out, in_literal)
    elif isinstance(v, dict):
        r = v.get("$ref")
        if isinstance(r, str) and r.startswith("#") and not in_literal:
            out.append(r)
        for k, c in v.items():
            refs_in(c, out, in_literal or k in ("const", "enum"))


def rename(v, names, in_literal=False):
    if isinstance(v, list):
        return [rename(x, names, in_literal) for x in v]
    if isinstance(v, dict):
        out = {}
        for k, c in v.items():
            if k == "$ref" and not in_literal and isinstance(c, str) and c in names:
                out[k] = names[c]
            else:
                out[k] = rename(c, names, in_literal or k in ("const", "enum"))
        return out
    return v


def variant(schema):
    if not isinstance(schema, dict):
        return schema
    refs = []
    refs_in(schema, refs)
    names, copies = {}, {}
    for r in dict.fromkeys(refs):
        target = pointer_target(schema, r)
        if target is None:
            continue
        key = f"{PREFIX}{len(names)}{PREFIX}"
        names[r] = f"#/$defs/{key}"
        copies[key] = target
    if not names:
        return schema
    out = rename(schema, names)
    if "$defs" not in out and out:
        out["$defs"] = {}
    defs = out.get("$defs")
    if not isinstance(defs, dict):
        return schema
    for key, target in copies.items():
        defs[key] = rename(target, names)
    return out


for line in sys.stdin:
    if line.strip():
        print(json.dumps(variant(json.loads(line)), separators=(",", ":")))
