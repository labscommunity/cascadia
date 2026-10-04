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
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_BASE="${LLAMA_BASE:-1692f9e50bb20fd96b963af38a282daf78feea64}"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
JOBS="${JOBS:-$(nproc)}"
PATCH="$REPO_ROOT/patches/llama.cpp/0001-sycl-stream-weights.patch"

if [ ! -d "$DEST/.git" ]; then
    git clone "$LLAMA_REPO" "$DEST"
fi
cd "$DEST"
git checkout "$LLAMA_BASE"
git apply --check "$PATCH" 2>/dev/null && git apply "$PATCH" || \
    echo "patch already applied or not applicable; continuing"

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
if ! grep -l GGML_STREAM_WEIGHTS "$DEST"/build/bin/libggml-base* >/dev/null 2>&1; then
    echo "ERROR: GGML_STREAM_WEIGHTS marker not found in libggml-base — patch missing?" >&2
    exit 1
fi
echo
echo "built: $BIN (weight-streaming marker verified in libggml-base)"
echo "export CASCADIA_LLAMA_BIN=\"$BIN\""
