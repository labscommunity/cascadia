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


def nested(ref, depth, in_array=False):
    """`{"$ref": ref}` wrapped `depth - 1` levels deep under `y` keys (and
    then in a one-item array), with the pointer suffix that reaches it."""
    inner = {"$ref": ref}
    path = ""
    if in_array:
        inner = [inner]
        path = "/0"
    for _ in range(depth - 1):
        inner = {"y": inner}
        path = "/y" + path
    return inner, path


# Names llama-server's rule names can't tell apart: `ref` + the pointer,
# every run of characters outside [a-zA-Z0-9-] made `-`.
TWINS = [("a_b", "a-b"), ("a.b", "a b"), ("a__b", "a-b"), ("aéb", "a-b"), ("a/b", "a~1b")]
OBJ_N = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
OBJ_S = {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}


def collisions():
    for one, two in TWINS:
        # Two `$defs` entries whose refs name the same rule, used both ways.
        for order in ((one, two), (two, one)):
            yield {
                "$defs": {order[0]: OBJ_N, order[1]: OBJ_S},
                "type": "object",
                "properties": {"x": {"$ref": f"#/$defs/{order[0]}"}, "y": {"$ref": f"#/$defs/{order[1]}"}},
                "required": ["x", "y"],
            }
        # Only one of them referenced: no collision.
        yield {
            "$defs": {one: OBJ_N, two: OBJ_S},
            "type": "object",
            "properties": {"x": {"$ref": f"#/$defs/{one}"}},
            "required": ["x"],
        }
    # A property whose rule name is a ref's rule name, before and after it,
    # object (a rule of its own) or primitive (inlined).
    for value in (OBJ_N, {"type": "integer"}):
        for first in ("prop", "ref"):
            props = {"ref-defs-B": value, "b": {"$ref": "#/$defs/B"}}
            if first == "ref":
                props = {"b": {"$ref": "#/$defs/B"}, "ref-defs-B": value}
            yield {"$defs": {"B": OBJ_S}, "type": "object", "properties": props, "required": list(props)}
        yield {"$defs": {"B": OBJ_S}, "type": "object",
               "properties": {"ref_defs_B": value, "b": {"$ref": "#/$defs/B"}}, "required": ["ref_defs_B", "b"]}
    # The review's cases as written: an enum property named like the ref's
    # rule, and the object one probed live.
    yield {
        "$defs": {"B": {"enum": ["u", "v"]}},
        "type": "object",
        "properties": {"ref-defs-B": {"enum": [1, 2]}, "y": {"$ref": "#/$defs/B"}},
        "required": ["ref-defs-B", "y"],
    }
    yield {
        "$defs": {"B": OBJ_N},
        "type": "object",
        "properties": {"ref-defs-B": OBJ_S, "z": {"$ref": "#/$defs/B"}},
        "required": ["ref-defs-B", "z"],
    }
    # A ref's rule name equal to a rule inside another ref's target:
    # `#/$defs/A-x` against A's property `x`, and `#/$defs/A/x` against it.
    for ref in ("#/$defs/A-x", "#/$defs/A/x", "#/$defs/A/properties/x"):
        yield {
            "$defs": {"A": {"type": "object", "properties": {"x": OBJ_N}, "required": ["x"], "x": OBJ_S}, "A-x": OBJ_S},
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": ref}},
            "required": ["a", "b"],
        }
    # A ref name one suffix away from another rule (`_add_rule` numbers a
    # second rule of the same name: `ref-defs-B0`).
    yield {
        "$defs": {"B": OBJ_S, "B0": OBJ_N},
        "type": "object",
        "properties": {"ref-defs-B": OBJ_N, "b": {"$ref": "#/$defs/B"}, "c": {"$ref": "#/$defs/B0"}},
        "required": ["ref-defs-B", "b", "c"],
    }
    # Items and alternatives named like a ref's rule.
    yield {
        "$defs": {"B": OBJ_S}, "type": "object",
        "properties": {"ref-defs-B": {"type": "array", "items": OBJ_N}, "b": {"$ref": "#/$defs/B"}},
        "required": ["ref-defs-B", "b"],
    }


def generated():
    yield from collisions()
    for ref, depth, in_array, pointed, also_elsewhere in itertools.product(
        REFS, (1, 2, 3), (False, True), (True, False), (True, False)
    ):
        inner, path = nested(ref, depth, in_array)
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
