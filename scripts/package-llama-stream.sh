#!/usr/bin/env bash
# Package a patched SYCL llama-server build into a release tarball.
#
# Usage: scripts/package-llama-stream.sh BUILD_DIR OUT_TARBALL
#   BUILD_DIR    the DEST passed to build-llama-stream.sh (contains build/bin)
#   OUT_TARBALL  e.g. cascadia-llama-sycl-linux-x86_64-v0.3.0.tar.gz
#
# The bundle is self-contained: every oneAPI library the binary and its
# bundled libraries need is copied next to the binary, and RUNPATH=$ORIGIN
# picks siblings up at runtime. The host still needs the Intel GPU driver
# (Level-Zero) and libstdc++/glibc from its own OS. The script fails if the
# finished bundle does not load in a clean environment (no oneAPI vars).
#
# Env: ONEAPI_ROOT  oneAPI install root (default: /opt/intel/oneapi)
set -euo pipefail

BUILD_DIR="${1:?usage: package-llama-stream.sh BUILD_DIR OUT_TARBALL}"
OUT_TARBALL="${2:?usage: package-llama-stream.sh BUILD_DIR OUT_TARBALL}"
LLAMA_BIN_DIR="$BUILD_DIR/build/bin"
LLAMA_SERVER="$LLAMA_BIN_DIR/llama-server"

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: $LLAMA_SERVER missing — run build-llama-stream.sh first" >&2; exit 1; }
command -v patchelf >/dev/null 2>&1 || { echo "ERROR: patchelf is required (apt install patchelf)" >&2; exit 1; }

# the oneAPI runtime must be resolvable to be copied: without it ldd prints
# "not found" for libsycl/libsvml/... and they would silently be left out.
# setvars.sh is not nounset-clean, so relax -u while sourcing it.
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
if ldd "$LLAMA_SERVER" | grep -q "not found"; then
    # shellcheck disable=SC1091
    set +u
    source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null
    set -u
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/cascadia-llama-sycl"
mkdir -p "$PKG"

cp "$LLAMA_SERVER" "$PKG/"
# llama.cpp shared libs produced by the build (versioned + symlinks)
cp -a "$LLAMA_BIN_DIR"/libggml*.so* "$LLAMA_BIN_DIR"/libllama*.so* "$LLAMA_BIN_DIR"/libmtmd*.so* "$PKG/" 2>/dev/null || true

# SYCL loads its Level-Zero/UR adapters by name, not through DT_NEEDED
for d in "$ONEAPI_ROOT"/*/lib "$ONEAPI_ROOT"/compiler/*/lib; do
    for l in "$d"/libpi_level_zero.so* "$d"/libpi_unified_runtime.so* "$d"/libur_*.so*; do
        [ -e "$l" ] && cp -L "$l" "$PKG/" 2>/dev/null || true
    done
done

# copy the oneAPI dependency closure of everything in the bundle (the
# adapters above and the bundled ggml libs have deps of their own); repeat
# until a pass adds nothing
while :; do
    added=0
    while read -r lib; do
        base="$(basename "$lib")"
        [ -e "$PKG/$base" ] && continue
        cp -L "$lib" "$PKG/$base"
        added=1
    done < <(find "$PKG" -type f \( -name "*.so*" -o -name "llama-server" \) -exec ldd {} \; 2>/dev/null |
             awk '/=> \// {print $3}' | grep -E "^$ONEAPI_ROOT/|/oneapi/" | sort -u)
    [ "$added" -eq 1 ] || break
done

# RUNPATH is baked as the absolute build dir; repoint it at the bundle dir so
# the tarball stands alone on a host without the source tree
find "$PKG" -type f \( -name "*.so*" -o -name "llama-server" \) -print0 | \
    xargs -0 -n1 patchelf --set-rpath '$ORIGIN'

# bundle sanity, in a clean environment (no oneAPI vars, no LD_LIBRARY_PATH):
# every library must resolve from the bundle or the OS, and the server must
# start far enough to print its version
unresolved="$(find "$PKG" -type f \( -name "*.so*" -o -name "llama-server" \) -exec env -i PATH=/usr/bin:/bin ldd {} \; 2>/dev/null |
              awk '/not found/ {print $1}' | sort -u)"
if [ -n "$unresolved" ]; then
    echo "ERROR: bundle does not resolve without oneAPI:" >&2
    echo "$unresolved" >&2
    exit 1
fi
if ! env -i PATH=/usr/bin:/bin "$PKG/llama-server" --version > "$PKG/VERSION.txt" 2>&1; then
    echo "ERROR: bundled llama-server --version failed in a clean environment:" >&2
    tail -5 "$PKG/VERSION.txt" >&2
    exit 1
fi
tar -czf "$OUT_TARBALL" -C "$STAGE" cascadia-llama-sycl
echo "packaged: $OUT_TARBALL ($(du -h "$OUT_TARBALL" | cut -f1), $(ls "$PKG" | wc -l) files)"
