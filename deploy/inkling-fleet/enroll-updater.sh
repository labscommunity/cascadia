#!/bin/bash
# One-time enrollment of a box into the fleet's updater. Run on the box (it must already be installed as a rank):
#     python3 -c "import urllib.request as u;print(u.urlopen('http://inkling-rank-0:8088/enroll-updater.sh').read().decode())" | sudo bash
# (or, where curl is installed:  curl -fsS http://inkling-rank-0:8088/enroll-updater.sh | sudo bash)
# Installs updater.py and its service and stores the fleet key. After that the box takes every later change
# (scripts, binary, settings) from rank 0 by itself. The key is fetched from rank 0 while enrollment is open
# (fleet.key is in the served folder); give it as an argument instead to enroll when it is closed.
set -euo pipefail
PREFIX="${PREFIX:-/opt/cascadia-inkling}"; SERVER="${SERVER:-http://inkling-rank-0:8088}"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 2; }
[ -f "$PREFIX/rank.env" ] || { echo "no rank installed under $PREFIX: run the installer first"; exit 1; }
fetch() { python3 -c "import sys, urllib.request; sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1], timeout=30).read())" "$1"; }
KEY="${1:-}"
[ -n "$KEY" ] || KEY=$(fetch "$SERVER/fleet.key" 2>/dev/null | tr -d '[:space:]' || true)
[ "${#KEY}" -ge 32 ] || { echo "no fleet key: enrollment is closed on rank 0 (ask for it to be opened, or pass the key as an argument)"; exit 1; }
umask 077; printf '%s\n' "$KEY" > "$PREFIX/fleet.key"; chmod 600 "$PREFIX/fleet.key"; umask 022
fetch "$SERVER/updater.py" > "$PREFIX/updater.py.new" && chmod 755 "$PREFIX/updater.py.new" && mv "$PREFIX/updater.py.new" "$PREFIX/updater.py"
set -a; . "$PREFIX/rank.env"; set +a
cat > /etc/systemd/system/cascadia-inkling-updater.service <<UNIT
[Unit]
Description=Cascadia Inkling fleet updater (pulls scripts and binary from rank 0)
After=network.target cascadia-inkling-beacon.service

[Service]
Type=simple
ExecStart=/usr/bin/env python3 $PREFIX/updater.py --prefix $PREFIX --fleet ${FLEET:-inkling}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable cascadia-inkling-updater.service > /dev/null 2>&1
systemctl restart cascadia-inkling-updater.service
sleep 4
echo "enrolled: rank ${RANK:-?}. First round:"
journalctl -u cascadia-inkling-updater --no-pager -n 6 -o cat 2>/dev/null || true
