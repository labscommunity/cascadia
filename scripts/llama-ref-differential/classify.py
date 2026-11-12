#!/usr/bin/env python3
"""Checks the sycl-llama engine's ref inlining against llama.cpp's own
converter, and writes the fixture the Rust test pins our verdicts with.

For every corpus schema the engine accepts (inlines) for llama-server's
chat endpoint:
(a) no `$ref` the client wrote reaches the converter: every `$ref` left in
    a schema position names a hoisted copy (`#/$defs/__cascadia_...`);
(b) the converter's grammar for it is the grammar of the ref-renamed
    variant (every ref pointing at a copy under a name of its own, which the
    converter resolves faithfully), up to rule names;
(c) without `--jinja` (the root visited as ``): every schema that needed
    hoisting is refused, and every other one gives the same grammar;
(d) a response schema and a tool's, both hoisted, through one converter
    and one table of refs, each give their own variant's grammar.

Exits 1 on any failure."""

import hashlib
import json
import re
import sys

TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]|[A-Za-z0-9-]+|\{[0-9,]*\}|[()|*+?]|\S')
CHAT = "response-format-schema"
TOOL = "tool-f-schema"
DATA = ("const", "enum", "default", "examples")
NAME_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")


def parse_grammar(text):
    rules = {}
    for part in text.split("\t"):
        if " ::= " in part:
            name, body = part.split(" ::= ", 1)
            rules[name.strip()] = TOKEN.findall(body)
    return rules


def without_aliases(rules):
    """`rules` with every rule whose body is one other rule's name replaced
    by that rule where it is used: the converter writes `name ::= other`
    for a ref (or a reused schema) and inlines a built-in primitive, so the
    same grammar can come with or without such a step."""
    alias = {k: v[0] for k, v in rules.items() if len(v) == 1 and v[0] in rules and v[0] != k}

    def final(name, seen=()):
        while name in alias and name not in seen:
            seen = seen + (name,)
            name = alias[name]
        return name

    out = {}
    for k, body in rules.items():
        out[k] = [final(t) if t in rules else t for t in body]
    return out


def final_of(rules, name):
    seen = set()
    while name in rules and len(rules[name]) == 1 and rules[name][0] in rules and name not in seen:
        seen.add(name)
        name = rules[name][0]
    return name


def same_shape(g1, root1, g2, root2):
    """Whether rule `root1` of `g1` and `root2` of `g2` are the same up to
    rule names: partition refinement over both grammars together."""
    if root1 not in g1 or root2 not in g2:
        return False
    root1, root2 = final_of(g1, root1), final_of(g2, root2)
    g1, g2 = without_aliases(g1), without_aliases(g2)
    rules = {("a", k): v for k, v in g1.items()}
    rules.update({("b", k): v for k, v in g2.items()})
    color = {k: "" for k in rules}
    while True:
        new = {}
        for (side, name), body in rules.items():
            toks = ["@" + color[(side, t)] if (side, t) in rules else t for t in body]
            new[(side, name)] = hashlib.sha1((color[(side, name)] + "|" + " ".join(toks)).encode()).hexdigest()[:16]
        done = len(set(new.values())) == len(set(color.values()))
        color = new
        if done:
            break
    return color[("a", root1)] == color[("b", root2)]


def client_refs(v, out):
    """`$ref`s in schema positions that don't name a hoisted copy."""
    if isinstance(v, list):
        for x in v:
            client_refs(x, out)
    elif isinstance(v, dict):
        r = v.get("$ref")
        if isinstance(r, str) and not r.startswith("#/$defs/__cascadia_"):
            out.append(r)
        for k, c in v.items():
            if k in DATA:
                continue
            if k in NAME_MAPS and isinstance(c, dict):
                for s in c.values():
                    client_refs(s, out)
            else:
                client_refs(c, out)


def lines(path):
    return [l.rstrip("\n") for l in open(path)]


def main():
    (corpus, inl, inl_nj, g_inl, g_var, g_inl_nj, pairs, g_pair_inl, g_pair_var, fixture) = sys.argv[1:11]
    schemas = [json.loads(l) for l in lines(corpus) if l]
    inl = [json.loads(l) for l in lines(inl)]
    inl_nj = [json.loads(l) for l in lines(inl_nj)]
    g_inl, g_var, g_inl_nj = lines(g_inl), lines(g_var), lines(g_inl_nj)
    assert len(schemas) == len(inl) == len(inl_nj) == len(g_inl) == len(g_var) == len(g_inl_nj)
    counts = {"accepted": 0, "hoisted": 0, "refused": 0, "(a) pass": 0, "(b) pass": 0,
              "(b) original errors at llama.cpp": 0, "(c) refused (hoisting)": 0, "(c) same grammar": 0,
              "(d) pairs pass": 0}
    reasons = {}
    failures = []
    entries = []
    for i, schema in enumerate(schemas):
        r, rn = inl[i], inl_nj[i]
        entry = {"schema": schema, "verdict": "inlined" if "inlined" in r else "refused",
                 "hoisted": r.get("hoisted", 0), "no_jinja": "inlined" if "inlined" in rn else "refused"}
        entries.append(entry)
        if "refused" in r:
            counts["refused"] += 1
            reason = r["refused"].split(" is not supported")[0].split(" makes")[0]
            msg = r["refused"]
            kind = ("too big" if "bytes" in msg else "llama-server cost" if "llama-server" in msg
                    else "recursive" if "recursive" in msg else "not inlinable")
            reasons[kind] = reasons.get(kind, 0) + 1
            continue
        counts["accepted"] += 1
        if r["hoisted"]:
            counts["hoisted"] += 1
        left = []
        client_refs(r["inlined"], left)
        if left:
            failures.append(("(a) a client ref reaches the converter", i, left[:3]))
        else:
            counts["(a) pass"] += 1
        if g_var[i] == "error":
            counts["(b) original errors at llama.cpp"] += 1
        elif g_inl[i] == "error" or not same_shape(parse_grammar(g_inl[i]), CHAT, parse_grammar(g_var[i]), CHAT):
            failures.append(("(b) grammar differs from the faithful one", i, g_inl[i][:200]))
        else:
            counts["(b) pass"] += 1
        if r["hoisted"]:
            if "refused" in rn:
                counts["(c) refused (hoisting)"] += 1
            else:
                failures.append(("(c) hoisted schema accepted without --jinja", i, ""))
        elif "inlined" in rn and g_var[i] != "error":
            if same_shape(parse_grammar(g_inl_nj[i]), "root", parse_grammar(g_var[i]), CHAT):
                counts["(c) same grammar"] += 1
            else:
                failures.append(("(c) grammar without --jinja differs", i, g_inl_nj[i][:200]))
    for meta, a, b in zip(lines(pairs), lines(g_pair_inl), lines(g_pair_var)):
        if b == "error":
            continue
        ga, gb = parse_grammar(a), parse_grammar(b)
        if a != "error" and same_shape(ga, CHAT, gb, CHAT) and same_shape(ga, TOOL, gb, TOOL):
            counts["(d) pairs pass"] += 1
        else:
            failures.append(("(d) response + tool differ", meta, a[:200]))
    with open(fixture, "w") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    print(f"corpus {len(schemas)}")
    print(", ".join(f"{k} {v}" for k, v in counts.items()))
    print("refused: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    print(f"failures {len(failures)}")
    for why, i, detail in failures:
        print(f"FAIL {why}: #{i}: {json.dumps(schemas[i])[:200] if isinstance(i, int) else i} | {detail}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
