#!/bin/bash
# Apply this update to the Inkling deployment SSD. Run on a Linux box with the SSD mounted (it is ext4):
#     sudo ./apply-update.sh /media/$USER/<ssd> [number of boxes, default 11]
# <ssd> is the folder that holds inkling-deploy/ and inkling/. Adds and replaces files under inkling-deploy/ and
# rewrites fleet.env for that many boxes with address discovery on (the old file is saved next to it). The model
# export and everything else on the SSD are left alone.
#
# Fixed addresses instead of discovery: add them, one per box in rank order (rank r sends to rank r+1):
#     sudo ./apply-update.sh /media/$USER/<ssd> 11 10.0.0.21 10.0.0.22 ... (11 addresses)
set -euo pipefail
SSD="${1:?usage: $0 /path/to/ssd [number of boxes] [fixed addresses in rank order]}"; TOTAL="${2:-11}"; shift; [ $# -gt 0 ] && shift; IPS=("$@")
HERE="$(cd "$(dirname "$0")" && pwd)"; D="$SSD/inkling-deploy"
[ -f "$D/install.sh" ] || { echo "no inkling-deploy/install.sh under $SSD - is this the SSD's top folder?"; exit 1; }
[ -f "$SSD/inkling/out/manifest.json" ] || { echo "no inkling/out/manifest.json under $SSD - is this the SSD's top folder?"; exit 1; }
[ -w "$D" ] || { echo "$D is not writable - run this with sudo"; exit 1; }
case "$TOTAL" in ''|*[!0-9]*) echo "number of boxes must be a number"; exit 1;; esac
LAYERS=$(python3 -c "import json;print(json.load(open('$SSD/inkling/out/manifest.json'))['num_layers'])")
[ "$TOTAL" -ge 1 ] && [ "$TOTAL" -le "$LAYERS" ] || { echo "number of boxes must be 1..$LAYERS"; exit 1; }
if [ ${#IPS[@]} -gt 0 ]; then
  [ ${#IPS[@]} -eq "$TOTAL" ] || { echo "got ${#IPS[@]} addresses for $TOTAL boxes: give exactly one per box, in rank order"; exit 1; }
  for a in "${IPS[@]}"; do [[ "$a" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo "not an IPv4 address: $a"; exit 1; }; done
  [ "$(printf '%s\n' "${IPS[@]}" | sort -u | wc -l)" -eq "$TOTAL" ] || { echo "the same address appears twice"; exit 1; }
fi

# 1. files
cd "$HERE/inkling-deploy"
find . -type f ! -name fleet.env.template | sort | while read -r f; do mkdir -p "$D/$(dirname "$f")"; cp -p "$f" "$D/$f"; done
chmod +x "$D/install.sh" "$D/serve.py" "$D/bin/linux/cascadia" "$D/fleet/"*.sh "$D/fleet/beacon.py"

# 2. fleet.env: the current template, with this fleet's size and the settings the old file carried
F="$D/fleet.env"; cp -p "$F" "$F.before-update"
python3 - "$F" "$HERE/inkling-deploy/fleet.env.template" "$TOTAL" "${IPS[@]}" <<'PY'
import re, sys
path, template, total, ips = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
old = dict(re.findall(r"^([A-Za-z_0-9]+)=(.*)$", open(path).read(), flags=re.M))
text = open(template).read()
keep = ["RELAY_PORT", "STREAMS", "MAX_SEQ", "FLEET", "BEACON_PORT", "FUSED_LAYERS_WINDOWS"]
values = {k: old[k].strip() for k in keep if k in old}
values["TOTAL"] = str(total)
values["DISCOVER"] = "0" if ips else "1"
for k, v in values.items():
    text = re.sub(r"^%s=.*$" % k, "%s=%s" % (k, v), text, flags=re.M)
addr = [ips[n] if ips else old.get("IP_%d" % n, "192.168.50.%d" % (10 + n)).strip() for n in range(total)]
block = "".join("IP_%d=%s\n" % (n, a) for n, a in enumerate(addr))
text = re.sub(r"(?:^IP_\d+=.*\n)+", block, text, count=1, flags=re.M)
open(path, "w").write(text)
PY
sync

# 3. verify
bad=0
while read -r f; do cmp -s "$f" "$D/$f" || { echo "MISMATCH: $f"; bad=1; }; done < <(find . -type f ! -name fleet.env.template | sort)
[ $bad = 0 ] || { echo "the copy did not verify - run this again"; exit 1; }
echo "update applied and verified: $(find . -type f ! -name fleet.env.template | wc -l | tr -d ' ') files"
LO_L=$((LAYERS / TOTAL)); HI_L=$(( (LAYERS + TOTAL - 1) / TOTAL ))
echo "fleet: $TOTAL boxes, $( [ $LO_L = $HI_L ] && echo $LO_L || echo "$LO_L or $HI_L" ) of $LAYERS layers each; ranks 0..$((TOTAL-1))"
if [ ${#IPS[@]} -gt 0 ]; then
  echo "addresses: fixed ($(grep -E '^IP_' "$F" | tr '\n' ' '))"
  echo "install each box with:  sudo <ssd>/inkling-deploy/install.sh auto     (takes the rank from the box's address)"
else
  echo "addresses: discovered on the LAN (the boxes may use DHCP and their addresses may change; all on one switch)"
  echo "install each box with:  sudo <ssd>/inkling-deploy/install.sh <rank>   (a different number 0..$((TOTAL-1)) on every box; rank 0 is the one clients talk to)"
  echo "who is where, any time: python3 <ssd>/inkling-deploy/fleet/beacon.py --show"
fi
