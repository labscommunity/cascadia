#!/bin/bash
# Chunked resilient transfer v3: keep-alives kill dead streams in ~60s so
# retries actually fire; unique per-invocation remote parts dir.
set -e
F="$1"; IP="$2"; DEST="$3"
SIZE=$(stat -c%s "$F")
SSHOPTS="-i $HOME/.ssh/fleet_ed25519 -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
TMP=/tmp/xfer_$(basename "$F" | tr -c 'a-zA-Z0-9' _)$$
rm -rf $TMP; mkdir -p $TMP
split -b 512M "$F" $TMP/part_
DESTW=$(echo "$DEST" | sed 's|/|\\|g')
RD="C:\\cascadia\\xp_$(basename "$F" | tr -c 'a-zA-Z0-9' _)$$"
RDF=$(echo "$RD" | sed 's|\\\\|/|g' | sed 's|\\|/|g')
ssh $SSHOPTS "devcloud@$IP" "powershell -Command \"New-Item -ItemType Directory '$RD' -Force | Out-Null\"" >/dev/null
send_part() {
  local p=$1 pn=$(basename $1)
  for try in 1 2 3 4 5; do
    scp $SSHOPTS "$p" "devcloud@$IP:$RDF/$pn" 2>/dev/null && return 0
    sleep 8
  done
  echo "PART_FAILED $pn"; return 1
}
FAIL=0
i=0
for p in $TMP/part_*; do
  send_part $p &
  i=$((i+1))
  [ $((i % 4)) = 0 ] && { wait || FAIL=1; }
done
wait || FAIL=1
[ $FAIL = 0 ] || { echo "XFER_PARTS_FAILED $(basename $F)"; rm -rf $TMP; exit 1; }
PLIST=$(for p in $TMP/part_*; do echo "$RD\\$(basename $p)"; done | paste -sd+ -)
ssh $SSHOPTS "devcloud@$IP" "cmd /c copy /b $PLIST \"$DESTW\"" >/dev/null
ssh $SSHOPTS "devcloud@$IP" "cmd /c rmdir /s /q $RD" >/dev/null 2>&1 || true
rm -rf $TMP
RSIZE=$(ssh $SSHOPTS "devcloud@$IP" "powershell -Command \"(Get-Item '$DEST').Length\"" | tr -d '\r')
if [ "$RSIZE" = "$SIZE" ]; then echo "XFER_VERIFIED $(basename $F) $SIZE"; else echo "XFER_SIZE_MISMATCH $(basename $F) local=$SIZE remote=$RSIZE"; exit 1; fi
