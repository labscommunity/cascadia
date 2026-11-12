#!/usr/bin/env bash
# Differential check of the API's sycl-llama `$ref` check against llama.cpp's
# own JSON-schema-to-grammar converter, at the llama.cpp commit sycl-llama
# builds (LLAMA_BASE in scripts/build-llama-stream.sh).
#
#   scripts/llama-ref-differential/run.sh
#
# Needs git, python3, a C++17 compiler and cargo. Env:
#   LLAMA_DIR  a llama.cpp checkout at that commit. Unset: that commit alone
#              is fetched from GitHub (a shallow fetch, about 30 MB) into
#              the work directory.
#   WORK       work directory, kept (default: a new temp directory, removed
#              at exit)
#   CARGO_CMD  cargo command (default: cargo)
#   CXX        C++ compiler (default: c++)
#
# The sycl-llama engine never sends llama-server a `$ref` the client wrote:
# it inlines every ref, and hoists a recursive schema's copies under
# reserved `$defs` names (see `cascadia_grammar::inline_refs`). This builds
# harness.cpp against the checkout and runs every schema corpus.py prints
# through the engine's inlining (the `llama_inline` example) and then
# through llama.cpp's converter as the chat endpoint runs it, and checks
# (see classify.py): no client ref reaches the converter; the grammar is
# the ref-renamed variant's (variants.py, which the converter resolves
# faithfully); without `--jinja` hoisted schemas are refused and the rest
# give the same grammar; and a hoisted response schema and a hoisted tool
# schema don't mix in llama-server's shared table of refs. Rewrites the
# fixture crates/cascadia-api/tests/fixtures/llama_ref_differential.jsonl,
# which a Rust test checks the inlining's verdicts against without a C++
# toolchain.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PIN="$(sed -n 's/^LLAMA_BASE="\${LLAMA_BASE:-\([0-9a-f]*\)}"/\1/p' "$ROOT/scripts/build-llama-stream.sh")"
CARGO_CMD="${CARGO_CMD:-cargo}"
CXX="${CXX:-c++}"
if [ -z "${WORK:-}" ]; then
    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' EXIT
fi

if [ -z "${LLAMA_DIR:-}" ]; then
    LLAMA_DIR="$WORK/llama.cpp"
    if [ ! -d "$LLAMA_DIR/.git" ]; then
        git init -q "$LLAMA_DIR"
        git -C "$LLAMA_DIR" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$PIN"
        git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
    fi
fi
HEAD="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
if [ "$HEAD" != "$PIN" ]; then
    echo "llama.cpp at $HEAD, expected $PIN" >&2
    exit 1
fi

"$CXX" -std=c++17 -O1 \
    -I "$LLAMA_DIR/common" -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" -I "$LLAMA_DIR/vendor" \
    "$HERE/harness.cpp" -o "$WORK/harness"
(cd "$ROOT" && $CARGO_CMD build -q -p cascadia-grammar --example llama_inline)
INLINE="${CARGO_TARGET_DIR:-$ROOT/target}/debug/examples/llama_inline"
H="$WORK/harness"
python3 "$HERE/corpus.py" > "$WORK/corpus.jsonl"
"$INLINE" < "$WORK/corpus.jsonl" > "$WORK/inlined.jsonl"
"$INLINE" --no-jinja < "$WORK/corpus.jsonl" > "$WORK/inlined-no-jinja.jsonl"
"$INLINE" --prefix t0 < "$WORK/corpus.jsonl" > "$WORK/inlined-tool.jsonl"
# Variants only of schemas the engine accepts (`{}` for the others): the
# converter can recurse without end on some it refuses (an `allOf` cycle).
accepted_only() { python3 -c 'import json,sys
for v, r in zip(open(sys.argv[1]), open(sys.argv[2])):
    print(v.rstrip("\n") if "inlined" in json.loads(r) else "{}")' "$1" "$WORK/inlined.jsonl"; }
python3 "$HERE/variants.py" zqref < "$WORK/corpus.jsonl" > "$WORK/variants-all.jsonl"
python3 "$HERE/variants.py" zqtool < "$WORK/corpus.jsonl" > "$WORK/variants-tool-all.jsonl"
accepted_only "$WORK/variants-all.jsonl" > "$WORK/variants.jsonl"
accepted_only "$WORK/variants-tool-all.jsonl" > "$WORK/variants-tool.jsonl"
# The inlined schema (or `{}` when refused), as the chat endpoint and as a
# `--no-jinja` server convert it, and the variant as the chat endpoint does.
schema_of() { python3 -c 'import json,sys
for l in sys.stdin: print(json.dumps(json.loads(l).get("inlined", {})))'; }
schema_of < "$WORK/inlined.jsonl" | "$H" --grammar --root response-format-schema > "$WORK/g-inlined.txt"
schema_of < "$WORK/inlined-no-jinja.jsonl" | "$H" --grammar > "$WORK/g-inlined-no-jinja.txt"
"$H" --grammar --root response-format-schema < "$WORK/variants.jsonl" > "$WORK/g-variants.txt"
python3 "$HERE/pairs.py" "$WORK/corpus.jsonl" "$WORK/inlined.jsonl" "$WORK/inlined-tool.jsonl" \
    "$WORK/variants.jsonl" "$WORK/variants-tool.jsonl" \
    "$WORK/pairs-inlined.jsonl" "$WORK/pairs-variants.jsonl" "$WORK/pairs.txt"
"$H" --pair < "$WORK/pairs-inlined.jsonl" > "$WORK/g-pairs-inlined.txt"
"$H" --pair < "$WORK/pairs-variants.jsonl" > "$WORK/g-pairs-variants.txt"
python3 "$HERE/classify.py" "$WORK/corpus.jsonl" "$WORK/inlined.jsonl" "$WORK/inlined-no-jinja.jsonl" \
    "$WORK/g-inlined.txt" "$WORK/g-variants.txt" "$WORK/g-inlined-no-jinja.txt" \
    "$WORK/pairs.txt" "$WORK/g-pairs-inlined.txt" "$WORK/g-pairs-variants.txt" \
    "$ROOT/crates/cascadia-api/tests/fixtures/llama_ref_differential.jsonl"
