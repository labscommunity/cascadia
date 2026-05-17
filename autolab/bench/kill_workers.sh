#!/usr/bin/env bash
# Stop tahoma workers on both matias boxes cleanly.
# Usage: autolab/bench/kill_workers.sh

set -euo pipefail

for host in cascadia-matias-02 cascadia-matias-03; do
  echo "[+] stopping tahoma on $host"
  ssh "$host" 'powershell -NoProfile -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2; Get-Process tahoma -ErrorAction SilentlyContinue | Select-Object Id,ProcessName"'
done
echo "[+] done"
