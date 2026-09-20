#!/bin/bash
# One-screen health for this rank.
PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PREFIX/rank.env" 2>/dev/null
echo "rank $RANK of $TOTAL  layers [$LAYER_START,$LAYER_END)  next=${NEXT:-none}  fused=${CASCADIA_INKLING_OV_MOE_LAYERS:-none}  igpu=${CASCADIA_INKLING_OV_ATTN:-0}"
systemctl --no-pager --lines=0 status cascadia-inkling.service 2>/dev/null | sed -n 3p
echo "last log lines:"
journalctl -u cascadia-inkling --no-pager -n 6 -o cat 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160
if [ "$RANK" = 0 ]; then
  echo "api: $(curl -s -m 3 http://127.0.0.1:8000/v1/models | head -c 120)"
fi
free -g | awk 'NR==2 {r="ram: used " $3 " GB, available " $7 " GB"} NR==3 {print r ", swap used " $3 " GB" ($3 > 1 ? "  <-- swapping: this rank will be very slow; lower FUSED_LAYERS_LINUX in fleet.env and re-run the installer" : "")}'
if [ -f "$PREFIX/beacon.py" ]; then
  echo "fleet (as heard on the LAN; beacon: $(systemctl is-active cascadia-inkling-beacon.service 2>/dev/null)):"
  python3 "$PREFIX/beacon.py" --show --wait 2 --fleet "${FLEET:-inkling}" --port "${BEACON_PORT:-9099}" || true
fi
python3 - <<'PY' 2>/dev/null || true
import json, time
try:
    u = json.load(open("/run/cascadia-inkling/update.json"))
    print("fleet files: version %s, %s" % (time.strftime("%m-%d %H:%M:%S", time.localtime(u["version"])) if u["version"] else "none yet", "in sync with rank 0" if u["ok"] else "last check failed: " + u["error"]))
except OSError:
    print("fleet files: this box is not enrolled in the updater")
PY
