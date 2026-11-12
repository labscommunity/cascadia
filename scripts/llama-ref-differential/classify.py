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


def same_language_shape(g1, g2, root):
    """Whether the roots of two grammars are the same up to rule names:
    partition refinement over both grammars together (a rule's class is
    its previous class and its body with every rule name replaced by that
    rule's class), run until no class splits, then the roots compared."""
    rules = {("a", k): v for k, v in g1.items()}
    rules.update({("b", k): v for k, v in g2.items()})
    if ("a", root) not in rules or ("b", root) not in rules:
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
    return color[("a", root)] == color[("b", root)]


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


def verdicts(llama, grammars, variant_grammars, root):
    """One endpoint's verdicts: the harness's, with a faithful one turned
    degraded where the grammar differs from the variant's (where every
    ref has a rule name of its own), so a reused rule shows."""
    out = [l.rstrip("\n") for l in open(llama)]
    g_orig = [l.rstrip("\n") for l in open(grammars)]
    g_var = [l.rstrip("\n") for l in open(variant_grammars)]
    assert len(out) == len(g_orig) == len(g_var), "line counts differ"
    for i, (a, b) in enumerate(zip(g_orig, g_var)):
        if out[i] == "faithful" and a != "error" and b != "error":
            if not same_language_shape(parse_grammar(a), parse_grammar(b), root):
                out[i] = "degraded (a ref's rule name held another rule)"
    return out


def tag_of(schema, lv, refused):
    """(tag, mismatch reason or None)."""
    kind = lv.split(" ", 1)[0]
    if kind == "degraded":
        return "degraded", None if refused else "silent 200"
    if kind == "error":
        return "error", None
    if refused:
        cls = conservative_class(schema)
        if cls is None:
            return "faithful", "refused a faithful schema"
        return f"conservative:{cls}", None
    return "faithful", None


def main():
    (corpus, ours, chat, chat_g, chat_vg, raw, raw_g, raw_vg, fixture) = sys.argv[1:10]
    schemas = [json.loads(l) for l in open(corpus) if l.strip()]
    ours_v = [l.strip() == "refused" for l in open(ours)]
    # The chat endpoint (what sycl-llama uses) decides; the plain
    # converter (`/completion`) is kept beside it for comparison.
    chat_v = verdicts(chat, chat_g, chat_vg, "response-format-schema")
    raw_v = verdicts(raw, raw_g, raw_vg, "root")
    assert len(schemas) == len(ours_v) == len(chat_v) == len(raw_v), "line counts differ"
    counts, raw_counts = {}, {}
    mismatches, flips, entries = [], [], []
    for i, (schema, refused, cv, rv) in enumerate(zip(schemas, ours_v, chat_v, raw_v)):
        tag, why = tag_of(schema, cv, refused)
        raw_tag, _ = tag_of(schema, rv, refused)
        counts[tag] = counts.get(tag, 0) + 1
        raw_counts[raw_tag] = raw_counts.get(raw_tag, 0) + 1
        if why:
            mismatches.append((why, schema, cv))
        if rv.startswith("degraded") and not cv.startswith("degraded"):
            flips.append((i, schema, rv, cv))
        entries.append({"schema": schema, "llama": cv, "raw": rv, "tag": tag})
    with open(fixture, "w") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    fmt = lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
    print(f"corpus {len(schemas)}")
    print(f"chat endpoint: {fmt(counts)}")
    print(f"plain converter: {fmt(raw_counts)}")
    print(f"degraded by the plain converter, not by the chat endpoint: {len(flips)}")
    for i, schema, rv, cv in flips:
        print(f"FLIP #{i}: {json.dumps(schema)[:160]} | plain: {rv} | chat: {cv}")
    print(f"mismatches {len(mismatches)}")
    for why, schema, lv in mismatches:
        print(f"MISMATCH {why}: {json.dumps(schema)} | llama.cpp chat: {lv}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
