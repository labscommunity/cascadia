# matias 2-box K2.6 deployment — SSH-tunnel chain via controller Mac

Scripts for running the K2.6 sparse-MoE pipeline split across two Windows
matias hosts (`matias-02`, `matias-03`) when the boxes cannot reach each
other directly. The chain forwards the inter-rank wire through SSH
tunnels on a controller Mac that has SSH access to both hosts.

Use this pattern when the boxes have no shared L2 / L3 path (e.g., when
Tailscale is down or unavailable on the workers, when one host sits
behind a NAT, or when you want to drive a 2-box cluster from a laptop
without configuring a VPN on the workers themselves).

## Tunnel layout

The two-box pipeline needs three connections:

```
matias-02:9100  --(ssh -R)-->  Mac:19100  --(ssh -L)-->  matias-03:9100
```

Plus an API tunnel `Mac:18000 -> matias-02:8000` so the bench harness
on the Mac can hit rank-0's OpenAI-compatible API.

Latency cost: about 117 ms RTT median through the chain vs about
22 ms direct over a shared LAN / overlay. Still well under K2.6's
per-token decode budget, so transport is not the bottleneck on either
path.

## Critical: `127.0.0.1` not `localhost`

Both bind and target addresses in the SSH forwards must use
`127.0.0.1` literally. With `localhost`, OpenSSH's `direct-tcpip`
channel resolves to `::1` on the remote end and the forward gets
`channel_free`d immediately — the listener never sees data.

## WMI-detached spawn (Windows / OpenSSH)

Both ranks are launched via `Invoke-WmiMethod -Class Win32_Process
-Name Create` (see `spawn_rank{0,1}_wmi.ps1`). On Windows OpenSSH,
`Start-Process -WindowStyle Hidden -PassThru` inherits the SSH job
object and gets killed the moment the SSH session closes — silently
losing the worker. WMI `Win32_Process` is the only reliable Windows-
OpenSSH detachment path.

## Mechanism step by step

1. From the Mac:

   ```bash
   ssh -f -N -L 19100:127.0.0.1:9100 cascadia-matias-03  # forward leg
   ssh -f -N -R 9100:127.0.0.1:19100 cascadia-matias-02  # reverse leg
   ssh -f -N -L 18000:127.0.0.1:8000 cascadia-matias-02  # API tunnel
   ```

2. Launch rank-1 on matias-03 with `--listen :9100`.
3. Launch rank-0 on matias-02 with `--next 127.0.0.1:9100`. From
   rank-0's POV it just connects to `localhost:9100`; the kernel
   routes via the `ssh -R` reverse forward to Mac:19100, where
   `ssh -L` forwards on to matias-03:9100.

## Files

- `start_workers_tunnel.sh` — top-level orchestrator. Run from the Mac.
  Kills stale workers and tunnels, opens the three SSH forwards, then
  triggers detached spawn on each box.
- `start_rank0_tunnel.ps1` — rank-0 launcher (deployed to matias-02).
- `spawn_rank0_wmi.ps1` / `spawn_rank1_wmi.ps1` — WMI-detached spawn
  wrappers that survive the SSH session ending.
- `k26_bench_matias_10.sh` — 10-prompt eval client. Hits the API
  through the local tunnel at `http://127.0.0.1:18000`.

## Quick start

```bash
# From Mac, with cascadia-matias-{02,03} SSH config aliases set up:
scripts/deploy/matias-2box/start_workers_tunnel.sh
# Wait for shells to load (cold: tens of minutes; warm: ~1 min).
# Poll: curl http://127.0.0.1:18000/health
scripts/deploy/matias-2box/k26_bench_matias_10.sh http://127.0.0.1:18000 /tmp/k26-bench.jsonl 32
```

## Prerequisites

- SSH config aliases `cascadia-matias-02` and `cascadia-matias-03`
  resolving to the two Windows hosts.
- Each box has tahoma built at `$USERPROFILE\tahoma\target\release\tahoma.exe`
  and the K2.6 model on disk at `$USERPROFILE\kimi-k26-model`.
- OpenVINO GenAI runtime present at
  `$USERPROFILE\openvino\openvino_genai_windows_2026.1.0.0_x86_64\`
  (paths inlined in the spawn scripts; edit if you move it).
- `jq` available on the Mac for the bench client.
