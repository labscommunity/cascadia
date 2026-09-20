#!/bin/bash
# Apply this update to the Inkling deployment SSD. Run on a Linux box with the SSD mounted (it is ext4):
#     ./apply-update.sh /media/$USER/<ssd> [number of boxes, default 11]
# <ssd> is the folder that holds inkling-deploy/ and inkling/. Adds and replaces files under inkling-deploy/,
# and sets the fleet size in fleet.env (your address edits are kept; the old file is saved next to it).
# The model export and everything else on the SSD are left alone.
set -euo pipefail
SSD="${1:?usage: $0 /path/to/ssd [number of boxes]}"; TOTAL="${2:-11}"
HERE="$(cd "$(dirname "$0")" && pwd)"; D="$SSD/inkling-deploy"
[ -f "$D/install.sh" ] || { echo "no inkling-deploy/install.sh under $SSD - is this the SSD's top folder?"; exit 1; }
[ -f "$SSD/inkling/out/manifest.json" ] || { echo "no inkling/out/manifest.json under $SSD - is this the SSD's top folder?"; exit 1; }
[ -w "$D" ] || { echo "$D is not writable - run this with sudo"; exit 1; }
case "$TOTAL" in ''|*[!0-9]*) echo "number of boxes must be a number"; exit 1;; esac
LAYERS=$(python3 -c "import json;print(json.load(open('$SSD/inkling/out/manifest.json'))['num_layers'])")
[ "$TOTAL" -ge 1 ] && [ "$TOTAL" -le "$LAYERS" ] || { echo "number of boxes must be 1..$LAYERS"; exit 1; }

# 1. files
cd "$HERE/inkling-deploy"
find . -type f | sort | while read -r f; do mkdir -p "$D/$(dirname "$f")"; cp -p "$f" "$D/$f"; done
chmod +x "$D/install.sh" "$D/serve.py" "$D/bin/linux/cascadia" "$D/fleet/"*.sh

# 2. fleet size (addresses already in the file are kept)
F="$D/fleet.env"; cp -p "$F" "$F.before-update"
sed -i -E "s/^TOTAL=.*/TOTAL=$TOTAL/; s/^FUSED_LAYERS_LINUX=.*/FUSED_LAYERS_LINUX=3/" "$F"
grep -q '^TOTAL=' "$F" || echo "TOTAL=$TOTAL" >> "$F"
for n in $(seq "$TOTAL" 64); do sed -i "/^IP_${n}=/d" "$F"; done
# the old note above FUSED_LAYERS_LINUX suggested raising it; replace it with what is now known
python3 - "$F" <<'PY'
import sys,re
p=sys.argv[1]; s=open(p).read()
note=("# Fused MoE layers the iGPU takes per box, at most (each is 8.3 GB of unified\n"
      "# memory; the rest of the rank's layers stay on the CPU at 7.7 GB each). The\n"
      "# iGPU can use half of RAM, so 3 on a 64 GB box, and the Ubuntu installer caps\n"
      "# this by the box's RAM anyway. If status.sh shows a box swapping, lower this\n"
      "# and re-run the installer there.\n")
s2=re.sub(r"(?:^#[^\n]*\n)+(?=FUSED_LAYERS_LINUX=)", note, s, count=1, flags=re.M) if re.search(r"^# Fused MoE layers", s, flags=re.M) else s
open(p,'w').write(s2)
PY
for n in $(seq 0 $((TOTAL-1))); do grep -q "^IP_${n}=" "$F" || echo "IP_${n}=192.168.50.$((10+n))" >> "$F"; done
sync

# 3. verify
bad=0
while read -r f; do cmp -s "$f" "$D/$f" || { echo "MISMATCH: $f"; bad=1; }; done < <(find . -type f | sort)
[ $bad = 0 ] || { echo "the copy did not verify - run this again"; exit 1; }
"$D/bin/linux/cascadia" --version > /dev/null 2>&1 && echo "binary runs on this machine: $("$D/bin/linux/cascadia" --version)" || echo "(the binary was not test-run here: it needs the OpenVINO runtime that each box's installer sets up; the installer tests it)"
echo "update applied and verified: $(find . -type f | wc -l | tr -d ' ') files"
LO_L=$((LAYERS / TOTAL)); HI_L=$(( (LAYERS + TOTAL - 1) / TOTAL ))
echo "fleet: $TOTAL boxes, $( [ $LO_L = $HI_L ] && echo $LO_L || echo "$LO_L or $HI_L" ) of $LAYERS layers each; ranks 0..$((TOTAL-1))"
grep -E '^(TOTAL|IP_|FUSED_LAYERS_LINUX)' "$F" | tr '\n' ' '; echo
echo "install each box with:  sudo $D/install.sh <rank>"
