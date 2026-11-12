#!/usr/bin/env python3
"""Prints the differential corpus, one JSON schema per line: the seed
schemas (every `$ref` probe of the structured-output reviews and the crash
sweep) and generated nestings of `$ref` forms in the places llama.cpp's
converter treats differently: beside another `$ref` at depths 1 to 3, in
tuple `items`, `allOf`/`anyOf` members, data keywords, and pointers into
those places. Deterministic: the same corpus on every run."""

import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Every `$ref` form: the root, a pointer that resolves, a pointer to
# nothing, remote documents with and without a fragment, a URN, a bare name.
REFS = [
    "#",
    "#/$defs/C",
    "#/$defs/missing",
    "https://example.com/a.json",
    "https://example.com/a.json#",
    "https://example.com/a.json#/x",
    "urn:z",
    "T",
]

DEFS = {"B": {"type": "string"}, "C": {"type": "integer"}}


def doc(defs, props, required=("p",), extra=None):
    d = {"$defs": {**DEFS, **defs}, "type": "object", "properties": props}
    if required:
        d["required"] = list(required)
    if extra:
        d.update(extra)
    return d


def nested(ref, depth):
    """`{"$ref": ref}` wrapped `depth - 1` levels deep under `y` keys,
    with the pointer suffix that reaches it."""
    inner = {"$ref": ref}
    path = ""
    for _ in range(depth - 1):
        inner = {"y": inner}
        path += "/y"
    return inner, path


def generated():
    for ref, depth, pointed, also_elsewhere in itertools.product(
        REFS, (1, 2, 3), (True, False), (True, False)
    ):
        inner, path = nested(ref, depth)
        a = {"$ref": "#/$defs/B", "x": inner}
        props = {"q": {"type": "string"}}
        if pointed:
            props["p"] = {"$ref": "#/$defs/A/x" + path}
        if also_elsewhere:
            props["r"] = {"$ref": ref}
        yield doc({"A": a}, props, required=("p",) if pointed else ())
    for ref in REFS:
        r = {"$ref": ref}
        # Containers llama-server's ref scan enters and its converter visits.
        yield {"type": "array", "items": [r], "minItems": 1}
        yield {"$id": "urn:x", "type": "array", "items": [r], "minItems": 1}
        yield doc({}, {"p": {"allOf": [r, {"description": "d"}]}})
        yield doc({}, {"p": {"allOf": [{"type": "object"}, {"properties": {"a": r}}]}})
        yield doc({}, {"p": {"anyOf": [r, {"type": "null"}]}})
        yield doc({}, {"p": {"type": "array", "items": r}})
        yield doc({}, {"p": r}, extra={"patternProperties": {"^x": {"type": "string"}}})
        # Data keywords, alone and reached by a pointer.
        for kw in ("const", "default", "examples", "enum"):
            data = [r] if kw in ("examples", "enum") else r
            yield doc({}, {"k": {kw: data, "type": "object"}}, required=())
            target = "/0" if kw in ("examples", "enum") else ""
            yield doc(
                {},
                {"k": {kw: data}, "p": {"$ref": f"#/properties/k/{kw}{target}"}},
            )
        # Beside a `$ref` with plain annotations, and an unused `$defs` entry.
        yield doc({}, {"p": {"$ref": "#/$defs/C", "description": "d", "default": r}})
        yield doc({"U": r}, {"p": {"type": "string"}})
        yield doc({"U": {"$ref": "#/$defs/B", "z": r}}, {"p": {"$ref": "#/$defs/U"}})


def main():
    seen = set()
    lines = []
    with open(os.path.join(HERE, "seed.jsonl")) as f:
        seeds = [json.loads(l) for l in f if l.strip()]
    for schema in itertools.chain(seeds, generated()):
        text = json.dumps(schema, separators=(",", ":"))
        if text not in seen:
            seen.add(text)
            lines.append(text)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
