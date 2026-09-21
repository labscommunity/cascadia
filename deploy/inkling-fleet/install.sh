#!/bin/bash
# Inkling rank installer for Ubuntu boxes — everything comes from this SSD.
#
#   sudo ./install.sh <rank>               # rank 0..TOTAL-1 (TOTAL from fleet.env)
#   sudo ./install.sh auto                 # the rank whose fleet.env address (IP_<rank>) this box already has
#   sudo ./install.sh <rank> --cpu-only    # skip the iGPU (no driver / no fused IRs)
#   sudo ./install.sh <rank> --no-net      # never touch the network configuration
#   sudo ./install.sh <rank> --add-address # add the fleet.env address even though the port already has a fixed one
#   sudo ./install.sh <rank> --prefix DIR  # install elsewhere than /opt/cascadia-inkling
#
# What it does, in order, all idempotent: copies the binary + OpenVINO runtime,
# installs the Intel GPU packages offline, copies this rank's model slice from
# the SSD to local disk, unpacks a private Python (system python3 + bundled
# wheels) and generates the fused MoE IRs for the rank's iGPU layers, writes
# the rank's environment, installs and starts the cascadia-inkling systemd
# service. Nothing system-wide is replaced: OpenVINO lives under the prefix,
# the GPU packages are the vendor .debs (they do add /dev/dri access), and the
# service runs as its own unit.
set -Eeuo pipefail
# Never stop without saying where: any command that fails unexpectedly names itself.
trap 'rc=$?; echo "[install] ERROR: stopped at line $LINENO (exit $rc) while running: $BASH_COMMAND" >&2; echo "[install] nothing was removed; fix the cause and run the same command again (finished steps are skipped)" >&2' ERR
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK="${1:?usage: install.sh <rank>|auto [--cpu-only] [--no-net] [--add-address] [--prefix DIR] [--model-source DIR]}"; shift
PREFIX=/opt/cascadia-inkling; CPU_ONLY=0; NO_NET=0; ADD_ADDRESS=0; MODEL_SRC="$HERE/../inkling/out"
while [ $# -gt 0 ]; do case "$1" in
  --cpu-only) CPU_ONLY=1;; --no-net) NO_NET=1;; --add-address) ADD_ADDRESS=1;; --prefix) PREFIX="$2"; shift;; --model-source) MODEL_SRC="$2"; shift;;
  *) echo "unknown option $1"; exit 2;; esac; shift; done
# shellcheck source=fleet.env
source "$HERE/fleet.env"
TOTAL="${TOTAL:-12}"
# DISCOVER=1 (default): the ranks find each other through the beacon (fleet/beacon.py) and the network is never
# touched. DISCOVER=0: fixed addresses from fleet.env's IP_<rank> lines.
DISCOVER="${DISCOVER:-1}"; FLEET="${FLEET:-inkling}"; BEACON_PORT="${BEACON_PORT:-9099}"
if [ "$RANK" = auto ] && [ "$DISCOVER" = 1 ]; then
  echo "auto takes the rank from an address list, and this fleet discovers addresses instead (DISCOVER=1 in fleet.env):"
  echo "give the rank number, a different one on every box (0..$((TOTAL-1)))."; exit 2
fi
if [ "$RANK" = auto ]; then
  RANK=""
  for r in $(seq 0 $((TOTAL-1))); do v="IP_$r"; a="${!v:-}"; if [ -n "$a" ] && ip -4 -o addr show | grep -q " $a/"; then RANK=$r; break; fi; done
  [ -n "$RANK" ] || { echo "auto: none of this box's addresses ($(ip -4 -o addr show scope global | awk '{print $4}' | tr '\n' ' ')) is listed in $HERE/fleet.env as IP_0..IP_$((TOTAL-1))"; exit 2; }
  echo "auto: this box holds IP_$RANK, so it is rank $RANK"
fi
[ "$RANK" -ge 0 ] && [ "$RANK" -lt "$TOTAL" ] || { echo "rank must be 0..$((TOTAL-1))"; exit 2; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 2; }
log() { echo "[install $(date +%T)] $*"; }
STATE="$PREFIX/install.state"; mkdir -p "$PREFIX" "$PREFIX/logs"; touch "$STATE"
done_step() { grep -qx "$1" "$STATE"; }
mark() { echo "$1" >> "$STATE"; }

# ---------- 0. facts about this box ----------
. /etc/os-release
UB="${VERSION_ID%%.*}"
RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 ))
PY=$(command -v python3 || true)
PYV=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)
# the wired port: one with link if there is one (boxes with two ports), else the first
NICS=$(ip -o link show | awk -F': ' '{n=$2; sub(/@.*/,"",n)} n !~ /^(lo|docker|veth|br-|virbr|tailscale|wl|ww|tun|wg)/ {print n, ($0 ~ /LOWER_UP/ ? "up" : "down")}')
NIC=$(echo "$NICS" | awk '$2=="up"{print $1; exit}'); [ -n "$NIC" ] || NIC=$(echo "$NICS" | awk 'NR==1{print $1}')
log "rank $RANK of $TOTAL on Ubuntu $VERSION_ID, ${RAM_GB} GB RAM, python $PYV, nic ${NIC:-?}, prefix $PREFIX"

# Addresses. The neighbouring ranks dial this box at fleet.env's IP_$RANK. If the box already holds that address
# the network is left alone. If its wired port has some other fixed address, stop now rather than rewrite it:
# netplan would add ours next to an address configured in netplan, but an address set in Settings/nmcli lives in
# its own NetworkManager profile, only one profile runs per port, and ours could take its place.
v="IP_$RANK"; MYIP="${!v:-}"
HAVE_MYIP=0; if [ -n "$MYIP" ] && ip -4 -o addr show | grep -q " $MYIP/"; then HAVE_MYIP=1; fi
if [ "$DISCOVER" = 0 ] && [ "$NO_NET" = 0 ] && [ "$ADD_ADDRESS" = 0 ] && [ "$HAVE_MYIP" = 0 ] && [ -n "$MYIP" ] && [ -n "$NIC" ]; then
  FIXED_NOW=$(ip -4 -o addr show dev "$NIC" scope global | grep -v dynamic | awk '{print $4}' | tr '\n' ' ' || true)
  if [ -n "$FIXED_NOW" ]; then
    echo
    echo "This box's wired port ($NIC) already has a fixed address: $FIXED_NOW"
    echo "but fleet.env says rank $RANK is $MYIP, and that is the address the neighbouring rank will dial."
    echo "  * The boxes already have their addresses: put them in $HERE/fleet.env (IP_0..IP_$((TOTAL-1)), in rank"
    echo "    order) and run this again. The installer then leaves the network alone, and 'install.sh auto' takes"
    echo "    the rank from the address."
    echo "  * To add $MYIP to $NIC anyway: run this again with --add-address. On Ubuntu Desktop that can replace an"
    echo "    address that was set in Settings, because NetworkManager runs one profile per port."
    exit 2
  fi
fi

# ---------- 0. stop a rank that is already running here (re-install) ----------
systemctl stop cascadia-inkling.service 2>/dev/null || true

# ---------- 1. binary + OpenVINO runtime (side by side) ----------
install -m 0755 "$HERE/bin/linux/cascadia" "$PREFIX/cascadia"
ARCH="$HERE/runtime/openvino_genai_ubuntu24_2026.3.1.0_x86_64.tar.gz"
[ "$UB" -le 22 ] && ARCH="$HERE/runtime/openvino_genai_ubuntu22_2026.3.1.0_x86_64.tar.gz"
if ! done_step runtime; then
  log "OpenVINO runtime -> $PREFIX/ov"; rm -rf "$PREFIX/ov"; mkdir -p "$PREFIX/ov"
  tar -xzf "$ARCH" -C "$PREFIX/ov"; mark runtime
fi
OVDIR=$(ls -d "$PREFIX"/ov/openvino_genai_* 2>/dev/null | head -1 || true)
[ -n "$OVDIR" ] || { log "ERROR: the OpenVINO runtime did not unpack under $PREFIX/ov (is $ARCH on the SSD and readable?)"; exit 1; }
# OpenCL ICD loader: OpenVINO GenAI's library links libOpenCL.so.1, so without it the
# binary does not start (CPU-only ranks included) and Intel's OpenCL package will not
# configure. Ubuntu only has it where something else pulled it in (ffmpeg, another ICD).
if ! ldconfig -p | grep -q 'libOpenCL\.so\.1'; then
  log "OpenCL ICD loader (ocl-icd-libopencl1)"
  if ls "$HERE"/gpu-debs/ocl-icd-libopencl1_*.deb > /dev/null 2>&1; then
    dpkg -i "$HERE"/gpu-debs/ocl-icd-libopencl1_*.deb >> "$PREFIX/logs/gpu-debs.log" 2>&1 || log "ocl-icd-libopencl1 did not install (see logs/gpu-debs.log)"
  else
    log "gpu-debs/ocl-icd-libopencl1_*.deb is not on this SSD"
  fi
fi
# Self-test before anything slow: the binary must start against the side-by-side
# runtime on this OS, or the service would only crash-loop later.
if ! ( set +u; . "$OVDIR/setupvars.sh" > /dev/null 2>&1; "$PREFIX/cascadia" --version > "$PREFIX/logs/selftest.out" 2>&1 ); then
  log "ERROR: $PREFIX/cascadia does not start on this system:"; head -5 "$PREFIX/logs/selftest.out" | sed 's/^/    /'
  if grep -q 'libOpenCL' "$PREFIX/logs/selftest.out"; then
    log "libOpenCL.so.1 is missing: copy ocl-icd-libopencl1_*.deb into gpu-debs/ on the SSD (or, with internet, apt install ocl-icd-libopencl1) and run this again."
  elif grep -q 'GLIBC' "$PREFIX/logs/selftest.out"; then
    log "the Linux binary on this SSD is built on Ubuntu 24.04 (glibc 2.39) and needs 24.04 or newer; this box runs Ubuntu $VERSION_ID."
  fi
  exit 1
fi
log "binary self-test: $(head -1 "$PREFIX/logs/selftest.out")"

# ---------- 2. Intel GPU packages (offline .debs) ----------
GPU_OK=0
if [ "$CPU_ONLY" = 0 ]; then
  if ! done_step gpudebs; then
    log "Intel GPU compute packages"
    # Level Zero loader built for this Ubuntu release, else the 24.04 build (it installs fine on newer releases).
    # "|| true": with set -e and pipefail a glob that matches nothing would otherwise end the install on the spot.
    ZE=$(ls "$HERE"/gpu-debs/libze1_*u${UB}.04_*.deb 2>/dev/null | head -1 || true)
    [ -n "$ZE" ] || ZE=$(ls "$HERE"/gpu-debs/libze1_*u24.04_*.deb 2>/dev/null | head -1 || true)
    [ -n "$ZE" ] || log "no libze1 package on the SSD for Ubuntu $VERSION_ID"
    dpkg -i "$HERE"/gpu-debs/libigdgmm12_*.deb "$HERE"/gpu-debs/intel-igc-core-2_*.deb "$HERE"/gpu-debs/intel-igc-opencl-2_*.deb \
            ${ZE:+"$ZE"} "$HERE"/gpu-debs/libze-intel-gpu1_*.deb "$HERE"/gpu-debs/intel-opencl-icd_*.deb \
            "$HERE"/gpu-debs/intel-ocloc_*.deb >> "$PREFIX/logs/gpu-debs.log" 2>&1 || log "some GPU packages did not install (see logs/gpu-debs.log)"
    getent group render > /dev/null && usermod -aG render,video root || true
    mark gpudebs
  fi
  if ls /dev/dri/renderD* > /dev/null 2>&1; then GPU_OK=1; else log "no /dev/dri render node: the kernel does not expose the iGPU; continuing CPU-only"; fi
fi

# ---------- 3. this rank's model slice ----------
$PY - "$MODEL_SRC" "$PREFIX/model" "$RANK" "$TOTAL" "$GPU_OK" <<'PY' | tee -a "$PREFIX/logs/slice.log"
import json, os, shutil, sys, time
src, dst, rank, total, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
m = json.load(open(os.path.join(src, "manifest.json"))); n = int(m["num_layers"])
base, rem = divmod(n, total); lo = rank * base + min(rank, rem); hi = lo + base + (1 if rank < rem else 0)
files = ["manifest.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "source_config.json"]
for l in range(lo, hi):
    files.append(f"shells/layer_{l:02d}.safetensors")
    files += [f"experts/layer_{l:02d}/{x}" for x in sorted(os.listdir(os.path.join(src, "experts", f"layer_{l:02d}")))]
    adir = os.path.join(src, "attn_ov", f"layer_{l:02d}")
    if os.path.isdir(adir):
        for root, _, names in os.walk(adir):
            files += [os.path.relpath(os.path.join(root, x), src) for x in names]
if rank == 0: files.append("embed.safetensors")
if rank == total - 1:
    files.append("head.safetensors")
    if os.path.isdir(os.path.join(src, "head_ov")): files += [f"head_ov/{x}" for x in os.listdir(os.path.join(src, "head_ov"))]
files = [f for f in files if os.path.exists(os.path.join(src, f))]
total_bytes = sum(os.path.getsize(os.path.join(src, f)) for f in files)
print(f"slice: layers [{lo},{hi}) {len(files)} files {total_bytes/2**30:.1f} GiB -> {dst}", flush=True)
t0 = time.time(); done = 0
for f in files:
    s, d = os.path.join(src, f), os.path.join(dst, f)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s): done += os.path.getsize(s); continue
    shutil.copyfile(s, d); done += os.path.getsize(s)
    if int(done / total_bytes * 10) != int((done - os.path.getsize(s)) / total_bytes * 10): print(f"  {done/2**30:.0f}/{total_bytes/2**30:.0f} GiB {time.time()-t0:.0f}s", flush=True)
open(os.path.join(dst, "rank.json"), "w").write(json.dumps({"rank": rank, "total": total, "layer_start": lo, "layer_end": hi, "num_layers": n}))
print(f"slice done in {time.time()-t0:.0f}s")
PY
LO=$($PY -c "import json;print(json.load(open('$PREFIX/model/rank.json'))['layer_start'])")
HI=$($PY -c "import json;print(json.load(open('$PREFIX/model/rank.json'))['layer_end'])")

# ---------- 4. private Python (for IR generation) ----------
PYLIB="$PREFIX/pylib"
if [ "$GPU_OK" = 1 ] && ! done_step pylib; then
  TAG="cp${PYV/./}"
  mkdir -p "$PYLIB"
  for w in "$HERE"/wheels/numpy-*-"$TAG"-*manylinux*.whl "$HERE"/wheels/openvino-*-"$TAG"-*manylinux*.whl; do
    [ -e "$w" ] && $PY -m zipfile -e "$w" "$PYLIB"
  done
  if PYTHONPATH="$PYLIB" $PY -c "import numpy, openvino" 2>/dev/null; then mark pylib; else log "no OpenVINO wheel for python $PYV: fused MoE IRs cannot be generated here (attention/head IRs still run on the iGPU)"; fi
fi

# ---------- 5. fused MoE IRs for the iGPU layers ----------
FUSED=""
if [ "$GPU_OK" = 1 ]; then
  # device (=unified) memory per fused layer is 8.3 GB; leave RAM for the CPU
  # layers (7.7 GB each) and ~14 GB for the OS, the slots and compile transients.
  MAX_FUSED="${FUSED_LAYERS_LINUX:-3}"
  # The iGPU can take at most half of RAM (the kernel's default limit for the xe driver, the same
  # share Windows gives it), so cap the request by this box's RAM: 3 layers on 64 GB, 1 on 32 GB.
  RAM_CAP=$(( RAM_GB * 10 / 2 / 83 ))
  if [ "$MAX_FUSED" -gt "$RAM_CAP" ]; then log "fused layers capped at $RAM_CAP for ${RAM_GB} GB RAM (fleet.env asks for $MAX_FUSED)"; MAX_FUSED=$RAM_CAP; fi
  cands=""; for l in $(seq "$LO" $((HI-1))); do [ "$l" -ge 2 ] && cands="$cands $l"; done
  n=0; for l in $cands; do [ $n -lt "$MAX_FUSED" ] && FUSED="${FUSED:+$FUSED,}$l" && n=$((n+1)); done
  if [ -n "$FUSED" ] && done_step pylib && ! done_step "fused:$FUSED"; then
    log "generating fused MoE IRs for layers $FUSED (about 30 s each)"
    PYTHONPATH="$PYLIB" $PY "$HERE/tools/inkling_moe_layer_ov.py" --src "$PREFIX/model" --layers "$FUSED" --layout u4zp --pad-experts 4 --skip-existing >> "$PREFIX/logs/fused-ir.log" 2>&1 \
      && mark "fused:$FUSED" || { log "fused IR generation failed (logs/fused-ir.log); the iGPU will serve attention only"; FUSED=""; }
  elif [ -n "$FUSED" ] && ! done_step pylib; then FUSED=""; fi
fi

# ---------- 6. rank environment ----------
IP_NEXT=""
if [ "$RANK" -lt $((TOTAL-1)) ]; then
  if [ "$DISCOVER" = 1 ]; then IP_NEXT="$FLEET-rank-$((RANK+1))"   # a name the beacon keeps current in /etc/hosts
  else
    v="IP_$((RANK+1))"; IP_NEXT="${!v:-}"
    [ -n "$IP_NEXT" ] || { log "ERROR: fleet.env has no IP_$((RANK+1)): rank $RANK would not know where to send"; exit 1; }
  fi
fi
{
  echo "# generated by install.sh for rank $RANK of $TOTAL — layers [$LO,$HI)"
  echo "RANK=$RANK"; echo "TOTAL=$TOTAL"; echo "LAYER_START=$LO"; echo "LAYER_END=$HI"; echo "NEXT=${IP_NEXT:+$IP_NEXT:$((RELAY_PORT+RANK+1))}"
  echo "DISCOVER=$DISCOVER"; echo "FLEET=$FLEET"; echo "BEACON_PORT=$BEACON_PORT"
  echo "OVDIR=$OVDIR"
  echo "RAYON_NUM_THREADS=$(nproc)"
  echo "CASCADIA_INKLING_MAX_SEQ=${MAX_SEQ:-1024}"; echo "CASCADIA_STREAMS=${STREAMS:-16}"; echo "CASCADIA_API_MAX_CONCURRENT=$(( ${STREAMS:-16} * 4 ))"
  # no frame-start idle ceiling between requests (default 900 s would drop an idle rank's link; TCP keepalive still catches a dead peer)
  echo "CASCADIA_FRAME_IDLE_CEILING_SECS=0"
  # the tuned CPU read profile with the expert cache holding the whole slice
  cat <<'ENV'
CASCADIA_INKLING_SERIAL_EXPERTS=0
CASCADIA_INKLING_SEQ_READS=0
CASCADIA_INKLING_PIN_EXPERTS=0
CASCADIA_BF16_GEMV_ROWS=2
CASCADIA_INT4_GEMV_ROWS=4
CASCADIA_INKLING_MMAP_EMBED=1
CASCADIA_INKLING_OWN_SHARED=1
CASCADIA_INKLING_REUSE_READ_BUFFERS=1
CASCADIA_INKLING_SKIP_BULK_PREFETCH=1
CASCADIA_INKLING_UNCACHED_READS=1
CASCADIA_INKLING_PIPELINE_READS=1
CASCADIA_INKLING_EXPERT_CACHE_MIB=8000
CASCADIA_INKLING_PREFILL_READS=1
CASCADIA_INKLING_CACHE_RESET_HISTORY=0
CASCADIA_INKLING_CACHE_DECAY_REQUESTS=4096
CASCADIA_INKLING_CACHE_RECENT_TIES=1
CASCADIA_INKLING_PREDICT_READS=1
CASCADIA_INKLING_EARLY_PREDICT_READS=0
CASCADIA_INKLING_SECOND_PREDICT_READS=1
CASCADIA_INKLING_SECOND_PREDICT_RANK=2
CASCADIA_INKLING_THIRD_PREDICT_READS=0
RUST_LOG=info
ENV
  if [ "$GPU_OK" = 1 ]; then
    echo "CASCADIA_INKLING_OV_ATTN=1"; echo "CASCADIA_INKLING_OV_ATTN_DEVICE=GPU"; echo "CASCADIA_INKLING_OV_ATTN_DIR=attn_ov"; echo "CASCADIA_INKLING_OV_ATTN_DROP_RUST=1"
    echo "CASCADIA_INKLING_OV_HEAD=1"; echo "CASCADIA_INKLING_OV_HEAD_DEVICE=GPU"; echo "OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=0"
    # f32: at the plugin's f16 the fused layers of the deeper ranks overflow and every logit is NaN (output "!!!!")
    if [ -n "$FUSED" ]; then echo "CASCADIA_INKLING_OV_MOE=1"; echo "CASCADIA_INKLING_OV_MOE_DEVICE=GPU"; echo "CASCADIA_INKLING_OV_MOE_LAYERS=$FUSED"; echo "CASCADIA_INKLING_OV_MOE_PRECISION=f32"; fi
  fi
} > "$PREFIX/rank.env"
install -m 0755 "$HERE/fleet/run.sh" "$PREFIX/run.sh"
install -m 0755 "$HERE/fleet/status.sh" "$PREFIX/status.sh"
log "rank.env written: layers [$LO,$HI), iGPU=$GPU_OK, fused=[${FUSED:-none}], next=${IP_NEXT:-none}"

# ---------- 7. static IP on the wired NIC ----------
if [ "$DISCOVER" = 1 ]; then
  log "addresses are discovered (beacon): network configuration left alone"
elif [ "$HAVE_MYIP" = 1 ]; then
  log "this box already has $MYIP: network configuration left alone"
elif [ "$NO_NET" = 1 ]; then
  if [ -n "$MYIP" ] && [ "$RANK" -gt 0 ]; then log "WARNING: --no-net and $MYIP (rank $RANK in fleet.env) is not on this box: rank $((RANK-1)) cannot reach it until it is"; fi
elif [ -n "$MYIP" ] && [ -n "$NIC" ]; then
  # With a DHCP lease on the port right now, keep DHCP and add the fleet address. Without one (an offline
  # switch, no DHCP server) "DHCP + static" is not safe: NetworkManager fails the whole connection when
  # DHCP times out and takes the static address down with it, so the port becomes static only.
  if ip -4 -o addr show dev "$NIC" | grep -q dynamic; then DHCP4=true; else DHCP4=false; fi
  cat > /etc/netplan/60-cascadia-inkling.yaml <<YAML
network:
  version: 2
  ethernets:
    $NIC:
      dhcp4: $DHCP4
      addresses: [$MYIP/24]
YAML
  chmod 600 /etc/netplan/60-cascadia-inkling.yaml
  netplan apply 2>>"$PREFIX/logs/netplan.log" || log "netplan apply failed (see logs/netplan.log)"
  sleep 2
  if ip -4 -o addr show dev "$NIC" | grep -q " $MYIP/"; then
    if [ "$DHCP4" = true ]; then log "static address $MYIP/24 on $NIC (DHCP kept: the port holds a lease)"
    else log "static address $MYIP/24 on $NIC (static only: no DHCP lease on this port; to undo, remove /etc/netplan/60-cascadia-inkling.yaml and run netplan apply)"; fi
  else
    log "WARNING: $MYIP/24 is not on $NIC after netplan apply (see logs/netplan.log); set it by hand or the neighbouring ranks cannot reach this box"
  fi
fi

# ---------- 8. systemd services ----------
BEACON_UNIT=/etc/systemd/system/cascadia-inkling-beacon.service
if [ "$DISCOVER" = 1 ]; then
  install -m 0755 "$HERE/fleet/beacon.py" "$PREFIX/beacon.py"
  cat > "$BEACON_UNIT" <<UNIT
[Unit]
Description=Cascadia Inkling fleet beacon (rank $RANK of $TOTAL): who is at which address
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/env python3 $PREFIX/beacon.py --rank $RANK --total $TOTAL --fleet $FLEET --port $BEACON_PORT --on-change "systemctl restart cascadia-inkling.service"
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT
fi
cat > /etc/systemd/system/cascadia-inkling.service <<UNIT
[Unit]
Description=Cascadia Inkling rank $RANK of $TOTAL
After=network-online.target cascadia-inkling-beacon.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=$PREFIX/run.sh
Restart=always
RestartSec=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity
WorkingDirectory=$PREFIX

[Install]
WantedBy=multi-user.target
UNIT
# a firewall that is on would drop the beacon and the ranks' traffic (Ubuntu ships ufw switched off)
if command -v ufw > /dev/null 2>&1 && ufw status 2> /dev/null | grep -q "Status: active"; then
  log "ufw is active: allowing the fleet's ports (udp $BEACON_PORT, tcp 8000, tcp 9100-9120)"
  ufw allow "$BEACON_PORT/udp" > /dev/null; ufw allow 8000/tcp > /dev/null; ufw allow 9100:9120/tcp > /dev/null
fi
systemctl daemon-reload
if [ "$DISCOVER" = 1 ]; then
  systemctl enable cascadia-inkling-beacon.service > /dev/null 2>&1
  systemctl restart cascadia-inkling-beacon.service
elif [ -f "$BEACON_UNIT" ]; then
  systemctl disable --now cascadia-inkling-beacon.service > /dev/null 2>&1 || true; rm -f "$BEACON_UNIT"; systemctl daemon-reload
fi
systemctl enable cascadia-inkling.service > /dev/null 2>&1
systemctl restart cascadia-inkling.service
sleep 3
systemctl --no-pager --lines=0 status cascadia-inkling.service 2>/dev/null | head -3 || true
if [ "$DISCOVER" = 1 ]; then
  log "the fleet as this box hears it (ranks not installed yet are 'not heard'):"
  $PY "$PREFIX/beacon.py" --show --wait 4 --fleet "$FLEET" --port "$BEACON_PORT" && rc=0 || rc=$?
  [ "$rc" != 2 ] || log "WARNING: a rank is claimed by more than one box (above). Every box needs its own rank: re-run this installer with the right number on the box that has the wrong one."
elif [ -n "$IP_NEXT" ] && command -v ping > /dev/null 2>&1; then
  if ping -c1 -W1 "$IP_NEXT" > /dev/null 2>&1; then log "next rank's box ($IP_NEXT) answers"
  else log "note: next rank's box ($IP_NEXT) does not answer ping (fine if it is not plugged in or set up yet)"; fi
fi
# Fleet updater: if rank 0 is already serving the fleet's files, enroll now, so later changes (scripts, binary,
# settings) reach this box by themselves. Not fatal: it can be done any time later.
if [ "$DISCOVER" = 1 ]; then
  ENROLL="curl --noproxy '*' -fsS http://$FLEET-rank-0:8088/enroll-updater.sh | sudo bash"
  for i in 1 2 3 4 5; do getent hosts "$FLEET-rank-0" > /dev/null 2>&1 && break; sleep 2; done
  if $PY -c "import sys, urllib.request as u; sys.stdout.buffer.write(u.build_opener(u.ProxyHandler({})).open('http://$FLEET-rank-0:8088/enroll-updater.sh', timeout=5).read())" > "$PREFIX/enroll-updater.sh" 2>/dev/null; then
    PREFIX="$PREFIX" bash "$PREFIX/enroll-updater.sh" 2>&1 | tail -3 || log "updater enrollment did not complete; later: $ENROLL"
  else
    rm -f "$PREFIX/enroll-updater.sh"; log "rank 0 is not serving fleet files right now; to get updates by themselves later, run: $ENROLL"
  fi
fi
log "installed. Follow with: $PREFIX/status.sh   (logs: journalctl -u cascadia-inkling -f)"
