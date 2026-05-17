#!/usr/bin/env bash
# Bring up the 2-box matias K2.6 pipeline cleanly.
# Run from this Mac. SSH config must have cascadia-matias-{02,03} aliases.
# Assumes start_rank{0,1}.ps1 already deployed to each box's $USERPROFILE.
#
# Usage:
#   autolab/bench/start_workers.sh
#
# After return, both workers are launched detached and model load
# proceeds. Use poll_ready.sh (or your bench script's built-in poll) to
# wait for API readiness. Cold start ~40 min historically.

set -euo pipefail

RANK0=cascadia-matias-02
RANK1=cascadia-matias-03

echo "[+] killing existing tahoma on $RANK0 and $RANK1"
for host in "$RANK0" "$RANK1"; do
  ssh "$host" 'powershell -NoProfile -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 3"'
done

echo "[+] starting rank 1 on $RANK1 (listens, then blocks until rank 0 connects)"
ssh "$RANK1" 'powershell -NoProfile -Command "Start-Process powershell -ArgumentList @(\"-NoProfile\",\"-ExecutionPolicy\",\"Bypass\",\"-File\",\"$env:USERPROFILE\start_rank1.ps1\") -WindowStyle Hidden -PassThru | Select-Object Id,StartTime"'

# Give rank 1 a moment to bind 9100 before rank 0 tries to connect.
sleep 8

echo "[+] starting rank 0 on $RANK0 (API on :8000, connects outbound to rank 1)"
ssh "$RANK0" 'powershell -NoProfile -Command "Start-Process powershell -ArgumentList @(\"-NoProfile\",\"-ExecutionPolicy\",\"Bypass\",\"-File\",\"$env:USERPROFILE\start_rank0.ps1\") -WindowStyle Hidden -PassThru | Select-Object Id,StartTime"'

echo "[+] both ranks launched. Poll http://100.77.178.45:8000/health or run bench script with built-in poll."
