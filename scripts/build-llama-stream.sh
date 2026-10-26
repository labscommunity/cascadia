#!/usr/bin/env bash
# Build a llama-server (SYCL) with the weight-streaming patch applied.
#
# Usage: scripts/build-llama-stream.sh [DEST]
#   DEST        clone + build directory (default: ./llama-stream)
#
# Env:
#   LLAMA_REPO  upstream remote (default: https://github.com/ggml-org/llama.cpp)
#   LLAMA_BASE  base commit the patch applies to (see patches/llama.cpp/)
#   ONEAPI_ROOT oneAPI install root (default: /opt/intel/oneapi)
#   JOBS        cmake --build parallelism (default: nproc)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-./llama-stream}"
case "$DEST" in /*) ;; *) DEST="$PWD/$DEST" ;; esac   # absolutize: we cd below
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_BASE="${LLAMA_BASE:-1692f9e50bb20fd96b963af38a282daf78feea64}"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
JOBS="${JOBS:-$(nproc)}"
# patch, marker file (relative to DEST), marker string proving it applied
PATCHES=(
    "$REPO_ROOT/patches/llama.cpp/0001-sycl-stream-weights.patch:ggml/src/ggml-backend.cpp:GGML_STREAM_WEIGHTS"
    "$REPO_ROOT/patches/llama.cpp/0002-sycl-router-aware-moe.patch:ggml/src/ggml-sycl/ggml-sycl.cpp:GGML_STREAM_EXPERT_CACHE_MB"
    "$REPO_ROOT/patches/llama.cpp/0003-sycl-host-buffer-compute.patch:ggml/src/ggml-sycl/ggml-sycl.cpp:CASCADIA_HOST_BUFT"
)

if [ ! -d "$DEST/.git" ]; then
    git clone "$LLAMA_REPO" "$DEST"
fi
cd "$DEST"
git checkout "$LLAMA_BASE"
# a tree whose markers survive from an OLDER patch revision would fool the
# per-patch skip check below; pin the exact revision set in a manifest and
# refuse to build on top of a stale one
MANIFEST=".cascadia-patches.manifest"
{
    echo "base=$LLAMA_BASE"
    for SPEC in "${PATCHES[@]}"; do
        P="${SPEC%%:*}"
        echo "$(basename "$P") sha256:$(sha256sum "$P" | cut -c1-16)"
    done
} > "$MANIFEST.new"
if [ -f "$MANIFEST" ] && ! cmp -s "$MANIFEST" "$MANIFEST.new"; then
    echo "ERROR: $DEST has a different patch/base revision applied:" >&2
    diff "$MANIFEST" "$MANIFEST.new" >&2 || true
    echo "reset to a clean base first: git -C $DEST checkout $LLAMA_BASE -- . && rm $DEST/$MANIFEST" >&2
    exit 1
fi
# apply the chain in order (idempotent reruns): skip a patch only when its
# own marker already proves it applied - a missing 0002 cannot hide behind
# 0001's marker - otherwise verify it applies, apply it, and abort on any
# failure rather than building a silently unpatched tree
for SPEC in "${PATCHES[@]}"; do
    PATCH="${SPEC%%:*}"
    REST="${SPEC#*:}"
    MARKER_FILE="${REST%%:*}"
    MARKER="${REST#*:}"
    if grep -q "$MARKER" "$MARKER_FILE" 2>/dev/null; then
        echo "patch $(basename "$PATCH") already applied; continuing"
        continue
    fi
    if ! git apply --check "$PATCH"; then
        echo "ERROR: $(basename "$PATCH") does not apply to $LLAMA_BASE and its marker is absent" >&2
        exit 1
    fi
    if ! git apply "$PATCH"; then
        echo "ERROR: $(basename "$PATCH") failed to apply to $LLAMA_BASE" >&2
        exit 1
    fi
done
mv "$MANIFEST.new" "$MANIFEST"

# icx/icpx + SYCL headers come from the oneAPI environment. setvars.sh is
# not nounset-clean, so relax -u while sourcing it.
# shellcheck disable=SC1091
set +u
source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null
set -u

cmake -B build \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j "$JOBS"

BIN="$DEST/build/bin/llama-server"
if ! grep -l GGML_STREAM_WEIGHTS "$BIN" "$DEST"/build/bin/libggml-base* >/dev/null 2>&1; then
    echo "ERROR: GGML_STREAM_WEIGHTS marker not found in the build output — patch 0001 missing?" >&2
    exit 1
fi
if ! grep -l GGML_STREAM_EXPERT_CACHE_MB "$BIN" "$DEST"/build/bin/libggml-sycl* >/dev/null 2>&1; then
    echo "ERROR: GGML_STREAM_EXPERT_CACHE_MB marker not found in the build output — patch 0002 missing?" >&2
    exit 1
fi
echo
echo "built: $BIN (markers verified: GGML_STREAM_WEIGHTS, GGML_STREAM_EXPERT_CACHE_MB)"
echo "export CASCADIA_LLAMA_BIN=\"$BIN\""
