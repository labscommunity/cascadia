#!/usr/bin/env bash
# Package a patched SYCL llama-server build into a release tarball.
#
# Usage: scripts/package-llama-stream.sh BUILD_DIR OUT_TARBALL
#   BUILD_DIR    the DEST passed to build-llama-stream.sh (contains build/bin)
#   OUT_TARBALL  e.g. cascadia-llama-sycl-linux-x86_64-v0.3.0.tar.gz
#
# Run in the oneAPI environment used to build the server. Requires patchelf,
# readelf and ldd. The selected SYCL runtime's dlopened UR/legacy PI plugins
# and every non-host transitive dependency are bundled with RUNPATH=$ORIGIN.
# Host dependencies: glibc/ELF loader, libgcc/libstdc++, zlib, NUMA, OpenSSL,
# and the OS GPU stack (Level Zero/OpenCL loaders and vendor drivers). These
# must be installed on the destination; no oneAPI installation is required.
set -euo pipefail
shopt -s nullglob

BUILD_DIR="${1:?usage: package-llama-stream.sh BUILD_DIR OUT_TARBALL}"
OUT_TARBALL="${2:?usage: package-llama-stream.sh BUILD_DIR OUT_TARBALL}"
BIN_DIR="$(realpath "$BUILD_DIR/build/bin")"
SERVER="$BIN_DIR/llama-server"

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -x "$SERVER" ] || fail "$SERVER missing — run build-llama-stream.sh first"
for tool in patchelf readelf ldd; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required for a relocatable bundle"
done

is_host_library() {
    case "$1" in
        linux-vdso.so.*|ld-linux*.so.*|ld64.so.*|libc.so.*|libm.so.*|libmvec.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*|libgcc_s.so.*|libstdc++.so.*|libz.so.*|libnuma.so.*|libssl.so.*|libcrypto.so.*|libze_loader.so.*|libze_intel_gpu.so.*|libOpenCL.so.*|libigdrcl.so.*) return 0 ;;
        *) return 1 ;;
    esac
}

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/cascadia-llama-sycl"
mkdir -p "$PKG"

# Inspect original files before changing their RPATH. Keep one canonical copy
# plus relative aliases for DT_NEEDED SONAMEs and dlopen's unversioned names.
declare -a sources=()
declare -A copied=() plugin_dirs=()
add_alias() {
    local name="$1" target="$2"
    [ "$name" = "$target" ] && return 0
    if [ -e "$PKG/$name" ] || [ -L "$PKG/$name" ]; then
        cmp -s "$PKG/$name" "$PKG/$target" || fail "conflicting library name: $name"
    else
        ln -s "$target" "$PKG/$name"
    fi
}
add_library() {
    local source="$1" real name soname dynamic
    real="$(realpath -e "$source")" || fail "missing library: $source"
    name="${real##*/}"
    if [ -z "${copied[$real]+present}" ]; then
        if [ -e "$PKG/$name" ]; then
            cmp -s "$real" "$PKG/$name" || fail "conflicting library name: $name"
        else
            cp "$real" "$PKG/$name"
        fi
        copied[$real]=1
        sources+=("$real")
        dynamic="$(readelf -d "$real")" || fail "cannot inspect $real"
        soname="$(awk '/\(SONAME\)/ {gsub(/[\[\]]/, "", $5); print $5}' <<< "$dynamic")"
        [ -z "$soname" ] || add_alias "$soname" "$name"
    fi
    add_alias "${source##*/}" "$name"
}

# ldd output is checked rather than piped into a subshell: a missing plugin
# dependency must abort packaging, not just produce an incomplete archive.
# Only this object's own DT_NEEDED entries are followed. Dependencies of host
# libraries (for example OpenSSL's libzstd) stay host-provided; every bundled
# object is inspected in turn, so the bundled closure is still complete.
inspect_dependencies() {
    local file="$1" mode="$2" listing line need resolved real dynamic
    local -A direct=()
    dynamic="$(readelf -d "$file")" || fail "cannot inspect $file"
    while read -r need; do
        [ -n "$need" ] && direct[$need]=1
    done < <(awk '/\(NEEDED\)/ {gsub(/[\[\]]/, "", $5); print $5}' <<< "$dynamic")
    if [ "$mode" = source ]; then
        listing="$(ldd "$file" 2>&1)" || fail "cannot resolve $file: $listing"
    else
        listing="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD ldd "$file" 2>&1)" || fail "relocation dependency check failed for $file: $listing"
    fi
    while IFS= read -r line; do
        if [[ "$line" == *"=> not found"* ]]; then
            fail "unresolved dependency in $file: $line"
        fi
        if [[ "$line" =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+\=\>[[:space:]]+(.*)[[:space:]]+\(0x[[:xdigit:]]+\) ]]; then
            need="${BASH_REMATCH[1]}"
            resolved="${BASH_REMATCH[2]}"
        elif [[ "$line" =~ ^[[:space:]]*(/.*)[[:space:]]+\(0x[[:xdigit:]]+\) ]]; then
            resolved="${BASH_REMATCH[1]}"
            need="${resolved##*/}"
        else
            continue
        fi
        [ -n "${direct[$need]+present}" ] || continue
        is_host_library "$need" && continue
        if [ "$mode" = source ]; then
            add_library "$resolved"
            add_alias "$need" "$(basename "$(realpath -e "$resolved")")"
        else
            real="$(realpath -e "$resolved")" || fail "unresolved dependency in $file: $need"
            case "$real" in
                "$PKG"/*) ;;
                *) fail "relocation resolves $need outside bundle: $real (from $file)" ;;
            esac
        fi
    done <<< "$listing"
}

add_library "$SERVER"
for library in "$BIN_DIR"/libggml*.so* "$BIN_DIR"/libllama*.so* "$BIN_DIR"/libmtmd*.so*; do
    add_library "$library"
done
for ((index=0; index<${#sources[@]}; index++)); do
    source="${sources[$index]}"
    inspect_dependencies "$source" source
    case "${source##*/}" in
        libsycl.so*|libur_loader.so*)
            # Use the runtime actually resolved by this build, never every
            # installed compiler version (or an unrelated `latest` symlink).
            runtime_dir="${source%/*}"
            if [ -z "${plugin_dirs[$runtime_dir]+present}" ]; then
                plugin_dirs[$runtime_dir]=1
                for plugin in "$runtime_dir"/libur_loader.so* \
                    "$runtime_dir"/libur_adapter_*.so* \
                    "$runtime_dir"/libpi_level_zero.so* \
                    "$runtime_dir"/libpi_unified_runtime.so* \
                    "$runtime_dir"/libpi_opencl.so*; do
                    add_library "$plugin"
                done
            fi
            ;;
    esac
done

# Each ELF object needs its own origin: DT_RUNPATH is not inherited by
# transitive dependencies or dlopened adapters. Never mask a patch failure.
for source in "${sources[@]}"; do
    file="$PKG/${source##*/}"
    patchelf --set-rpath '$ORIGIN' "$file" || fail "cannot relocate $file"
done
for source in "${sources[@]}"; do
    inspect_dependencies "$PKG/${source##*/}" relocated
done

# Exercise the staged executable, not the working build. The clean loader
# environment must also work for plugin loading performed by --version.
if ! env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PKG/llama-server" --version > "$PKG/VERSION.txt" 2>&1; then
    cat "$PKG/VERSION.txt" >&2
    fail "relocated llama-server --version failed"
fi
tar -czf "$OUT_TARBALL" -C "$STAGE" cascadia-llama-sycl
echo "packaged: $OUT_TARBALL ($(du -h "$OUT_TARBALL" | cut -f1), ${#sources[@]} ELF files)"
