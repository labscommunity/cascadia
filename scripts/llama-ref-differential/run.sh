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
# Builds harness.cpp against the checkout and runs every schema corpus.py
# prints through llama.cpp's converter and through our check
# (`sycl_ref_verdicts`). llama.cpp degrades a schema when a `$ref` it uses
# has no recorded target, or when its rule is one another rule already held
# (seen by comparing the grammar with variants.py's, where every ref has a
# rule name of its own). Fails when it degrades a schema our check passes,
# or we refuse a schema it converts faithfully outside the documented
# conservative classes (see classify.py). Rewrites the fixture
# crates/cascadia-api/tests/fixtures/llama_ref_differential.jsonl, which a
# Rust test checks our verdicts against without a C++ toolchain.
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
python3 "$HERE/corpus.py" > "$WORK/corpus.jsonl"
python3 "$HERE/variants.py" < "$WORK/corpus.jsonl" > "$WORK/variants.jsonl"
"$WORK/harness" < "$WORK/corpus.jsonl" > "$WORK/llama.txt"
"$WORK/harness" --grammar < "$WORK/corpus.jsonl" > "$WORK/grammars.txt"
"$WORK/harness" --grammar < "$WORK/variants.jsonl" > "$WORK/variant-grammars.txt"
(cd "$ROOT" && $CARGO_CMD build -q -p cascadia-api --example sycl_ref_verdicts)
"${CARGO_TARGET_DIR:-$ROOT/target}/debug/examples/sycl_ref_verdicts" < "$WORK/corpus.jsonl" > "$WORK/ours.txt"
python3 "$HERE/classify.py" "$WORK/corpus.jsonl" "$WORK/llama.txt" "$WORK/ours.txt" \
    "$WORK/grammars.txt" "$WORK/variant-grammars.txt" \
    "$ROOT/crates/cascadia-api/tests/fixtures/llama_ref_differential.jsonl"
