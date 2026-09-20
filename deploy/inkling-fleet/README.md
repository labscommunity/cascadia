# Inkling on 11 boxes — from this SSD

Everything the installation needs is on this drive:

```
inkling-deploy/          this folder: installers, binaries, runtimes, tools, fleet layout
inkling/out/             the Inkling int4 export (549 GB) + int8 attention/head IRs
```

Eleven Intel Panther Lake boxes running Ubuntu, one wired switch, no
internet. Each box becomes one rank of an 11-rank pipeline (6 of the model's
66 layers per box) that serves the OpenAI-compatible API (and the dashboard)
from rank 0. The number of boxes is `TOTAL` in `fleet.env`; the installers
split the layers evenly over whatever it says, so set it before the first
install and keep it the same for every box.

## Per box (about 10 minutes, mostly copying)

1. Plug the SSD in. Decide the box's rank (0–10): rank 0 is the box clients
   talk to, and rank r sends to rank r+1, so give each box a different rank.
2. **Ubuntu** (24.04 or newer; the installer checks that the binary starts
   before it does anything slow):
   ```
   sudo /media/$USER/<ssd>/inkling-deploy/install.sh <rank>
   ```
3. Unplug the SSD. The rank starts by itself, now and at every boot, and
   restarts if it stops. `status.sh` / `status.ps1` in the install folder show
   one screen of health.

Order does not matter: a rank keeps retrying its downstream neighbour until
it is up. Rank 0 answers on `http://<IP_0>:8000` once every rank is up.

## If a Windows 11 box joins the fleet (over the LAN)

The SSD is ext4, which Windows cannot read (it offers to format the drive:
say no), so a Windows box installs over the LAN:

1. Plug the SSD into any Ubuntu box on the switch (one that is already
   installed is fine) and run
   ```
   python3 /media/$USER/<ssd>/inkling-deploy/serve.py
   ```
   It serves the kit and the model read-only on port 8080 and prints two
   lines. (If that box has a firewall on: `sudo ufw allow 8080/tcp`.)
2. On each Windows box, open PowerShell as administrator and type those two
   lines with the box's rank:
   ```
   Set-ExecutionPolicy -Scope Process Bypass -Force
   curl.exe -s -o $env:TEMP\bootstrap.ps1 http://<that box>:8080/inkling-deploy/bootstrap.ps1; & $env:TEMP\bootstrap.ps1 -Rank <rank> -Server http://<that box>:8080
   ```
   It pulls the Windows half of the kit (about 400 MB) into `C:\inkling-kit`, then
   runs the normal installer, which pulls that rank's slice of the model (about
   44 GB, a few minutes on 2.5 GbE) and sets the rank up exactly as on Ubuntu.
   Both Windows boxes can pull at the same time. Running it again resumes.
3. Ctrl-C `serve.py` when both are done and carry on with the SSD.

The Windows box has to reach the Ubuntu box before its own installer has
given it a fleet address. If the switch has no DHCP server, give the wired
port its `fleet.env` address first (PowerShell as administrator; the port is
usually called `Ethernet`, see `Get-NetAdapter`):
```
netsh interface ipv4 set interface "Ethernet" dhcpstaticipcoexistence=enabled
netsh interface ipv4 add address "Ethernet" 192.168.50.<10 + rank> 255.255.255.0
```
The installer later finds the address already there and keeps it.

## Addresses

`fleet.env` assigns `192.168.50.10 + rank` to each box's wired port (the one
with link). Where that port holds a DHCP lease the address is added next to
DHCP; on a switch with no DHCP server an Ubuntu box's port becomes static
only, because a "DHCP + static" profile does not stay up there (to undo:
remove `/etc/netplan/60-cascadia-inkling.yaml`, then `sudo netplan apply`).
Windows keeps DHCP on either way. Change the addresses there before
installing if the venue's network is different, or install with `--no-net` /
`-NoNet` and set addresses yourself; the `NEXT` line in each box's `rank.env`
must point at the next rank.

## What runs where

- Experts stay on the CPU with the tuned read profile (the whole slice is
  resident in the expert cache after the first request), except the layers
  the iGPU takes as fused MoE (3 of a 64 GB box's 6, generated on the box at
  install time in about a minute each; `fleet.env` sets the count and the
  installer caps it at what half of the box's RAM can hold).
- Attention projections and the output head run on the iGPU from the int8
  IRs on the SSD.
- Without a usable iGPU (no `/dev/dri` render node, no Intel GPU, or
  `--cpu-only`) the rank runs the same CPU profile for everything.

## Showing throughput

From any box on the network (Python 3):

```
python3 inkling-deploy/bench.py http://192.168.50.10:8000 --streams 16 --tokens 64
```

prints per-stream tokens/s, time to first token and the aggregate
tokens/s. `CASCADIA_STREAMS` in `fleet.env` (16) is the most simultaneous
requests the pipeline serves; raise it on every box together.

## If something is off

- `journalctl -u cascadia-inkling -f` (Ubuntu) / `C:\cascadia-inkling\logs\worker.log` (Windows).
- A rank that says `stream slot ... beyond this rank's capacity` has a
  different `CASCADIA_STREAMS` than rank 0.
- `no /dev/dri render node`: the kernel does not expose the Panther Lake
  iGPU; the rank runs on the CPU. Ubuntu 24.04 needs its HWE kernel (6.14+).
- A rank that is far slower than the others is swapping (`status.sh` says
  so): lower `FUSED_LAYERS_LINUX` in `fleet.env` by one and run the installer
  on that box again.
- Re-running the installer is safe; it skips what is already done. It
  restarts that box's rank; the ranks behind rank 0 then restart once under
  their supervisors (about five seconds plus load time) and rank 0 reconnects
  by itself, so the next request just works. Same after a box reboots.
- Binaries built before 2026-09-19 (`bin/BUILD.txt` names a commit other than
  `ea7a54ed` or later) lack that reconnect: if requests keep failing after a
  box restarted, restart rank 0 once (`sudo systemctl restart cascadia-inkling`,
  or on Windows end the `cascadia-ov` process; the task relaunches it).
