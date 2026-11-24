#!/usr/bin/env bash
# Full-disk fallback check for the Linux interposer (no model, no GPU).
#
# Big allocations are MAP_SHARED tmpfiles in the backing dir. A sparse file is
# accepted even when the disk cannot hold it and then SIGBUSes on the first
# write fault, so when the shim escapes a tmpfs ELASTIC_DIR to an ext4/xfs dir
# it reserves the blocks up front and keeps 1 GiB free there.
# Inside a private mount namespace (nothing is mounted on the host) this puts
# ELASTIC_DIR on tmpfs and a 2 GiB ext4 over /var/tmp, then keeps 2 GiB live
# with HOME unset, so /var/tmp is the only escape target. The run must escape,
# exit 0, and serve some allocations from files while the rest fall back to
# anonymous memory. Exit 0 = pass, 1 = fail, 77 = skipped (prerequisite missing).
set -euo pipefail

skip() { echo "SKIP: $*"; exit 77; }

[[ "$(uname -s)" == Linux ]] || skip "Linux-only (O_TMPFILE interposer)"
for tool in cc mkfs.ext4 truncate unshare setpriv sudo; do
    command -v "$tool" >/dev/null 2>&1 || skip "$tool not found"
done
sudo -n true 2>/dev/null || skip "needs passwordless sudo for a private mount namespace"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { rm -rf "$work"; }
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
    if (n < 0 || n > 1024 || (n > 0 && sz == 0)) {
        fprintf(stderr, "bad args\n");
        return 2;
    }
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

rc=0
"$work/fill" 0 0 >/dev/null 2>&1 || rc=$?
if [[ "$rc" -eq 126 || ! -x "$work/fill" ]]; then
    skip "$work is mounted noexec; set TMPDIR to an exec-able dir"
elif [[ "$rc" -ne 0 ]]; then
    echo "FAIL: the test program does not run (exit=$rc)"
    exit 1
fi

truncate -s 2G "$work/disk.img"
mkfs.ext4 -q -F "$work/disk.img"
mkdir "$work/shm"
touch "$work/err" "$work/rc"    # user-owned, so the trap can remove them

# Runs as root inside the namespace; the workload drops back to the caller.
cat > "$work/inner.sh" <<'EOF'
set -eu
work=$1 uid=$2 gid=$3
mount -o loop "$work/disk.img" /var/tmp
chmod 1777 /var/tmp
mount -t tmpfs -o size=3G tmpfs "$work/shm"
chmod 1777 "$work/shm"
rc=0
setpriv --reuid="$uid" --regid="$gid" --init-groups \
    env -u HOME LD_PRELOAD="$work/elastic.so" ELASTIC_DIR="$work/shm" \
    ELASTIC_POOL_MB=0 ELASTIC_LOG=1 "$work/fill" 8 256 \
    >/dev/null 2>"$work/err" || rc=$?
echo "$rc" >"$work/rc"
EOF
sudo -n unshare -m --propagation private -- \
    bash "$work/inner.sh" "$work" "$(id -u)" "$(id -g)" || {
    echo "FAIL: could not set up the mount namespace"
    exit 1
}

rc="$(cat "$work/rc")"
rc="${rc:-?}"
escaped=no
grep -q 'using /var/tmp instead' "$work/err" && escaped=yes
line="$(grep -m1 '^\[elastic\] big_allocs=' "$work/err" || true)"
big="$(sed -nE 's/.*big_allocs=([0-9]+).*/\1/p' <<<"$line")"
fb="$(sed -nE 's/.* fallbacks=([0-9]+).*/\1/p' <<<"$line")"
big="${big:-0}"
fb="${fb:-0}"

if [[ "$escaped" == yes && "$rc" == 0 && "$big" -ge 1 && "$fb" -ge 1 ]]; then
    echo "PASS: exit=0 escaped to /var/tmp, big_allocs=$big fallbacks=$fb" \
        "(2 GiB live on a 2 GiB ext4, 1 GiB kept free)"
    exit 0
fi
echo "FAIL: exit=$rc (135 = SIGBUS) escaped=$escaped big_allocs=$big fallbacks=$fb"
echo "--- stderr ---"
cat "$work/err"
exit 1
