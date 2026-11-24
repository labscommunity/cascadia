#!/usr/bin/env bash
# Full-disk fallback check for the Linux interposer (no model, no GPU).
#
# Big allocations are MAP_SHARED tmpfiles in ELASTIC_DIR. A sparse file is
# accepted even when the disk cannot hold it and then SIGBUSes on the first
# write fault, so the shim reserves the blocks up front on a disk-backed dir.
# This keeps 128 MB live against a 64 MB ext4 loop mount: the run must exit 0
# with some allocations file-backed and the rest fallen back to anonymous
# memory. Exit 0 = pass, 1 = fail, 77 = skipped (prerequisite missing).
set -euo pipefail

skip() { echo "SKIP: $*"; exit 77; }

[[ "$(uname -s)" == Linux ]] || skip "Linux-only (O_TMPFILE interposer)"
for tool in cc mkfs.ext4 truncate sudo; do
    command -v "$tool" >/dev/null 2>&1 || skip "$tool not found"
done
sudo -n true 2>/dev/null || skip "needs passwordless sudo to loop-mount a small ext4"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
mnt="$work/mnt"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
    if mountpoint -q "$mnt" 2>/dev/null; then sudo -n umount "$mnt" || true; fi
    rm -rf "$work"
}
trap cleanup EXIT

cc -O2 -shared -fPIC -o "$work/elastic.so" \
    "$repo/crates/cascadia-elastic/src/elastic_unix.c" -ldl -lpthread

cat > "$work/fill.c" <<'EOF'
/* N mallocs of M MB, every byte written, all kept live, one byte read back. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: fill N M\n"); return 2; }
    int n = atoi(argv[1]);
    size_t sz = (size_t)atol(argv[2]) << 20;
    if (n <= 0 || n > 1024 || sz == 0) { fprintf(stderr, "bad args\n"); return 2; }
    unsigned char *p[1024];
    for (int i = 0; i < n; i++) {
        p[i] = malloc(sz);
        if (!p[i]) { fprintf(stderr, "malloc %d failed\n", i); return 3; }
        memset(p[i], (int)(i + 1), sz);
    }
    for (int i = 0; i < n; i++)
        if (p[i][sz - 1] != (unsigned char)(i + 1)) {
            fprintf(stderr, "allocation %d lost its contents\n", i);
            return 4;
        }
    puts("done");
    return 0;
}
EOF
cc -O2 -o "$work/fill" "$work/fill.c"

truncate -s 64M "$work/disk.img"
mkfs.ext4 -q -F "$work/disk.img"
mkdir "$mnt"
sudo -n mount -o loop "$work/disk.img" "$mnt"
sudo -n chmod 1777 "$mnt"

rc=0
LD_PRELOAD="$work/elastic.so" ELASTIC_DIR="$mnt" ELASTIC_POOL_MB=0 ELASTIC_LOG=1 \
    "$work/fill" 8 16 >"$work/out" 2>"$work/err" || rc=$?

line="$(grep -m1 '^\[elastic\] big_allocs=' "$work/err" || true)"
big="$(sed -nE 's/.*big_allocs=([0-9]+).*/\1/p' <<<"$line")"
fb="$(sed -nE 's/.* fallbacks=([0-9]+).*/\1/p' <<<"$line")"
big="${big:-0}"
fb="${fb:-0}"

if [[ "$rc" -eq 0 && "$big" -ge 1 && "$fb" -ge 1 ]]; then
    echo "PASS: exit=0 big_allocs=$big fallbacks=$fb (128 MB live on a 64 MB ext4)"
    exit 0
fi
echo "FAIL: exit=$rc (135 = SIGBUS) big_allocs=$big fallbacks=$fb"
echo "--- stderr ---"
cat "$work/err"
exit 1
