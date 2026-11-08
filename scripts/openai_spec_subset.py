#!/usr/bin/env python3
"""Extract the OpenAI response schemas cascadia-api's tests validate against.

Reads a downloaded upstream ``openapi.json`` (openai/openai-openapi) and writes
the transitive ``$ref`` closure of the response schemas below, under
``components.schemas``, plus a top-level ``x-source`` recording where it came
from. Output is deterministic (sorted keys, fixed indent), so re-running on the
same pinned spec reproduces the committed fixture byte for byte.

Usage (stdlib only):

    curl -sSL -o openapi.json \\
      https://raw.githubusercontent.com/openai/openai-openapi/926a7b8133226630bac025350e84b0ca5c31d45a/openapi.json
    python3 scripts/openai_spec_subset.py openapi.json \\
      crates/cascadia-api/tests/fixtures/openai-spec-subset.json
"""

import json
import sys

UPSTREAM_COMMIT = "926a7b8133226630bac025350e84b0ca5c31d45a"
SPEC_URL = (
    "https://raw.githubusercontent.com/openai/openai-openapi/"
    + UPSTREAM_COMMIT
    + "/openapi.json"
)
ROOTS = [
    "CreateChatCompletionResponse",
    "CreateChatCompletionStreamResponse",
    "CreateCompletionResponse",
    "ListModelsResponse",
    "Response",
    "ResponseStreamEvent",
]
REF_PREFIX = "#/components/schemas/"


def refs_in(node):
    """Yield every schema name referenced by a ``$ref`` anywhere in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                if not value.startswith(REF_PREFIX):
                    raise SystemExit("unsupported $ref (not a schema): " + value)
                yield value[len(REF_PREFIX):]
            else:
                yield from refs_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from refs_in(item)


def closure(schemas, roots):
    seen = set()
    todo = list(roots)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        if name not in schemas:
            raise SystemExit("schema not found in spec: " + name)
        seen.add(name)
        todo.extend(refs_in(schemas[name]))
    return {name: schemas[name] for name in sorted(seen)}


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: openai_spec_subset.py <openapi.json> <output.json>")
    src, out = argv[1], argv[2]
    with open(src, encoding="utf-8") as f:
        spec = json.load(f)
    subset = {
        "x-source": {
            "repo": "openai/openai-openapi",
            "commit": UPSTREAM_COMMIT,
            "url": SPEC_URL,
            "roots": ROOTS,
            "command": "python3 scripts/openai_spec_subset.py openapi.json "
            "crates/cascadia-api/tests/fixtures/openai-spec-subset.json",
        },
        "components": {"schemas": closure(spec["components"]["schemas"], ROOTS)},
    }
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(subset, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print("wrote %d schemas to %s" % (len(subset["components"]["schemas"]), out))


if __name__ == "__main__":
    main(sys.argv)
