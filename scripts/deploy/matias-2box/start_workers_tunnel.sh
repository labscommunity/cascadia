#!/usr/bin/env bash
# Bring up the 2-box matias K2.6 pipeline when Tailscale is logged out.
# Run from this Mac. SSH config must have cascadia-matias-{02,03} aliases.
# Assumes start_rank{0_tunnel,1}.ps1 already deployed to each box's $USERPROFILE.
#
# Mechanism: chains the inter-rank wire through the Mac via SSH tunnels:
#   matias-02:9100  --(ssh -R)-->  Mac:19100  --(ssh -L)-->  matias-03:9100
# Plus an API tunnel Mac:18000 -> matias-02:8000 for the bench harness.
#
# Latency cost: ~117 ms RTT median through the chain vs ~22 ms direct
# DERP relay (still <2% of K2.6's per-token decode budget).
#
# Usage:
#   scripts/deploy/matias-2box/start_workers_tunnel.sh
#
# After return, both workers are launched detached and the tunnels are
# live. Cold start ~40 min historically; poll http://127.0.0.1:18000/health.

set -euo pipefail

RANK0=cascadia-matias-02
RANK1=cascadia-matias-03

echo "[+] killing existing tahoma + tunnel ssh on Mac and both boxes"
for host in "$RANK0" "$RANK1"; do
  ssh "$host" 'powershell -NoProfile -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' || true
done
# Best-effort kill of any stale tunnels on Mac side
pkill -f 'ssh.*-L 19100' 2>/dev/null || true
pkill -f 'ssh.*-R 9100'  2>/dev/null || true
pkill -f 'ssh.*-L 18000' 2>/dev/null || true
sleep 2

echo "[+] ensuring spawn helpers are present on both boxes"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
scp -q "$SCRIPT_DIR/start_rank0_tunnel.ps1" "$RANK0:start_rank0_tunnel.ps1" || true
scp -q "$SCRIPT_DIR/spawn_rank0_wmi.ps1"    "$RANK0:spawn_rank0_wmi.ps1"    || true
scp -q "$SCRIPT_DIR/spawn_rank1_wmi.ps1"    "$RANK1:spawn_rank1_wmi.ps1"    || true

echo "[+] opening SSH tunnels (Mac <-> matias-02 <-> matias-03 via bastions)"
# Mac:19100 -> matias-03:9100 (forward leg)
ssh -f -N -L 19100:127.0.0.1:9100 "$RANK1" -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes
# matias-02:9100 -> Mac:19100 (reverse leg, completes the chain)
ssh -f -N -R 9100:127.0.0.1:19100 "$RANK0" -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes
# Mac:18000 -> matias-02:8000 (so bench can hit the API)
ssh -f -N -L 18000:127.0.0.1:8000 "$RANK0" -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes
echo "    tunnels up"

echo "[+] starting rank 1 on $RANK1 (WMI-detached, listens on :9100, blocks until rank 0 connects)"
ssh "$RANK1" 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\devcloud\spawn_rank1_wmi.ps1'

# Give rank 1 a moment before rank 0 tries to connect through the chain.
sleep 8

echo "[+] starting rank 0 on $RANK0 (WMI-detached, API on :8000, --next 127.0.0.1:9100 via tunnels)"
ssh "$RANK0" 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\devcloud\spawn_rank0_wmi.ps1'

echo "[+] both ranks launched. API at http://127.0.0.1:18000/ (via tunnel) once shells finish loading."
