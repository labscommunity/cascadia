#!/usr/bin/env python3
"""Joins the corpus, llama.cpp's verdicts (harness.cpp) and ours
(`sycl_ref_verdicts`), tags each schema, and checks:

(a) every schema llama.cpp degrades (a `$ref` it turned into "any") is
    refused by our sycl-llama check: no silent 200;
(b) every schema we refuse that llama.cpp converts faithfully falls in a
    documented conservative class.

Writes the tagged fixture (one JSON object per line: schema, llama, tag)
to the path given, and exits 1 on any mismatch.

Tags: `degraded`, `faithful`, `error` (llama.cpp refuses it: a 4xx, never
silent, either verdict of ours is fine) or `conservative:<class>`:
- `ref-beside-ref`: a `$ref` under a `$ref` object's siblings;
- `https-ref-unreached`: a no-fragment `https://` `$ref` (in data, or
  anywhere else the converter never reaches)."""

import hashlib
import json
import re
import sys

TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]|[A-Za-z0-9-]+|\{[0-9,]*\}|[()|*+?]|\S')


def parse_grammar(text):
    """`name ::= body` rules from the harness's `--grammar` line."""
    rules = {}
    for part in text.split("\t"):
        if " ::= " in part:
            name, body = part.split(" ::= ", 1)
            rules[name.strip()] = TOKEN.findall(body)
    return rules


def same_language_shape(g1, g2):
    """Whether the roots of two grammars are the same up to rule names:
    partition refinement over both grammars together (a rule's class is
    its previous class and its body with every rule name replaced by that
    rule's class), run until no class splits, then the roots compared."""
    rules = {("a", k): v for k, v in g1.items()}
    rules.update({("b", k): v for k, v in g2.items()})
    if ("a", "root") not in rules or ("b", "root") not in rules:
        return False
    color = {k: "" for k in rules}
    while True:
        new = {}
        for (side, name), body in rules.items():
            toks = ["@" + color[(side, t)] if (side, t) in rules else t for t in body]
            text = color[(side, name)] + "|" + " ".join(toks)
            new[(side, name)] = hashlib.sha1(text.encode()).hexdigest()[:16]
        done = len(set(new.values())) == len(set(color.values()))
        color = new
        if done:
            break
    return color[("a", "root")] == color[("b", "root")]


def refs_beside_ref(v, beside=False):
    """Whether a `$ref` sits under a `$ref` object's siblings."""
    if isinstance(v, list):
        return any(refs_beside_ref(x, beside) for x in v)
    if isinstance(v, dict):
        own = v.get("$ref")
        if isinstance(own, str) and beside:
            return True
        b = beside or "$ref" in v
        return any(refs_beside_ref(c, b) for k, c in v.items() if k != "$ref")
    return False


def no_fragment_https_refs(v):
    if isinstance(v, list):
        return any(no_fragment_https_refs(x) for x in v)
    if isinstance(v, dict):
        r = v.get("$ref")
        if isinstance(r, str) and r.startswith("https://") and r.split("#", 1)[1:] in ([], [""]):
            return True
        return any(no_fragment_https_refs(c) for c in v.values())
    return False


def conservative_class(schema):
    if refs_beside_ref(schema):
        return "ref-beside-ref"
    if no_fragment_https_refs(schema):
        return "https-ref-unreached"
    return None


def main():
    corpus, llama, ours, grammars, variant_grammars, fixture = sys.argv[1:7]
    schemas = [json.loads(l) for l in open(corpus) if l.strip()]
    llama_v = [l.rstrip("\n") for l in open(llama)]
    ours_v = [l.strip() for l in open(ours)]
    g_orig = [l.rstrip("\n") for l in open(grammars)]
    g_var = [l.rstrip("\n") for l in open(variant_grammars)]
    assert len(schemas) == len(llama_v) == len(ours_v) == len(g_orig) == len(g_var), "line counts differ"
    # A faithful verdict still hides a ref whose rule was reused: compare
    # the grammar with the one for the variant, where every ref has a rule
    # name of its own.
    for i, (lv, a, b) in enumerate(zip(llama_v, g_orig, g_var)):
        if lv == "faithful" and a != "error" and b != "error":
            if not same_language_shape(parse_grammar(a), parse_grammar(b)):
                llama_v[i] = "degraded (a ref's rule name held another rule)"
    counts = {}
    mismatches = []
    entries = []
    for schema, lv, ov in zip(schemas, llama_v, ours_v):
        kind = lv.split(" ", 1)[0]
        refused = ov == "refused"
        if kind == "degraded":
            tag = "degraded"
            if not refused:
                mismatches.append(("silent 200", schema, lv))
        elif kind == "error":
            tag = "error"
        elif refused:
            cls = conservative_class(schema)
            if cls is None:
                mismatches.append(("refused a faithful schema", schema, lv))
                tag = "faithful"
            else:
                tag = f"conservative:{cls}"
        else:
            tag = "faithful"
        counts[tag] = counts.get(tag, 0) + 1
        entries.append({"schema": schema, "llama": lv, "tag": tag})
    with open(fixture, "w") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    print(f"corpus {len(schemas)}; " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"mismatches {len(mismatches)}")
    for why, schema, lv in mismatches:
        print(f"MISMATCH {why}: {json.dumps(schema)} | llama.cpp: {lv}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
