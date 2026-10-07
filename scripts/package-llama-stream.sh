#!/usr/bin/env bash
# Package a patched SYCL llama-server build into a release tarball.
#
# Usage: scripts/package-llama-stream.sh BUILD_DIR OUT_TARBALL
#   BUILD_DIR    the DEST passed to build-llama-stream.sh (contains build/bin)
#   OUT_TARBALL  e.g. cascadia-llama-sycl-linux-x86_64-v0.3.0.tar.gz
#
# The bundle is self-contained: every library ldd resolves under the oneAPI
# environment is copied next to the binary, and the binary's RUNPATH=$ORIGIN
# picks siblings up at runtime. The host still needs the Intel GPU driver
# (Level-Zero) and libstdc++/glibc from its own OS.
set -euo pipefail

BUILD_DIR="${1:?usage: package-llama-stream.sh BUILD_DIR OUT_TARBALL}"
OUT_TARBALL="${2:?usage: package-llama-stream.sh BUILD_DIR OUT_TARBALL}"
BIN_DIR="$BUILD_DIR/build/bin"
SERVER="$BIN_DIR/llama-server"

[ -x "$SERVER" ] || { echo "ERROR: $SERVER missing — run build-llama-stream.sh first" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/cascadia-llama-sycl"
mkdir -p "$PKG"

cp "$SERVER" "$PKG/"
# llama.cpp shared libs produced by the build (versioned + symlinks)
cp -a "$BIN_DIR"/libggml*.so* "$BIN_DIR"/libllama*.so* "$BIN_DIR"/libmtmd*.so* "$PKG/" 2>/dev/null || true

# oneAPI/SYCL runtime deps resolved through the current environment
ldd "$SERVER" | awk '/=> \// {print $3}' | while read -r lib; do
    case "$lib" in
        "$BIN_DIR"/*) ;;                      # already bundled above
        /opt/intel/oneapi/*|*/oneapi/*) cp -L "$lib" "$PKG/" ;;
    esac
done
# SYCL loads its Level-Zero plugin by name, not through DT_NEEDED
for d in /opt/intel/oneapi/*/lib /opt/intel/oneapi/compiler/*/lib; do
    for l in "$d"/libpi_level_zero.so* "$d"/libpi_unified_runtime.so* "$d"/libur_*.so*; do
        [ -e "$l" ] && cp -L "$l" "$PKG/" 2>/dev/null || true
    done
done

# RUNPATH is baked as the absolute build dir; repoint it at the bundle dir so
# the tarball stands alone on a host without the source tree
if command -v patchelf >/dev/null 2>&1; then
    find "$PKG" -type f \( -name "*.so*" -o -name "llama-server" \) -print0 | \
        xargs -0 -n1 patchelf --set-rpath '$ORIGIN' 2>/dev/null || true
else
    echo "WARN: patchelf not found; RUNPATH still points at $BIN_DIR" >&2
fi

# bundle sanity: every DT_NEEDED that is not a system lib resolves inside PKG
missing=0
while read -r need; do
    case "$need" in
        linux-vdso*|ld-linux*|libpthread*|libdl*|librt*|libm.so*|libc.so*|libgcc*|libstdc++*|libze_loader*|libOpenCL*|libz.so*|libnuma*|libssl*|libcrypto*) ;;
        *) [ -e "$PKG/$need" ] || [ -e "$PKG/${need%%.so*}.so" ] || { echo "WARN: unresolved in bundle: $need" >&2; missing=1; } ;;
    esac
done < <(readelf -d "$SERVER" | awk '/NEEDED/ {gsub(/[\[\]]/,"",$5); print $5}')
[ "$missing" -eq 0 ] || echo "WARN: bundle has unresolved libs (see above)"

"$SERVER" --version > "$PKG/VERSION.txt" 2>&1 || true
tar -czf "$OUT_TARBALL" -C "$STAGE" cascadia-llama-sycl
echo "packaged: $OUT_TARBALL ($(du -h "$OUT_TARBALL" | cut -f1), $(ls "$PKG" | wc -l) files)"
