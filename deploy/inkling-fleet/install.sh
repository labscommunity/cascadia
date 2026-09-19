#!/bin/bash
# Inkling rank installer for Ubuntu boxes — everything comes from this SSD.
#
#   sudo ./install.sh <rank>               # rank 0..TOTAL-1 (TOTAL from fleet.env, default 12)
#   sudo ./install.sh <rank> --cpu-only    # skip the iGPU (no driver / no fused IRs)
#   sudo ./install.sh <rank> --no-net      # do not set the static IP from fleet.env
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
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK="${1:?usage: install.sh <rank> [--cpu-only] [--no-net] [--prefix DIR] [--model-source DIR]}"; shift
PREFIX=/opt/cascadia-inkling; CPU_ONLY=0; NO_NET=0; MODEL_SRC="$HERE/../inkling/out"
while [ $# -gt 0 ]; do case "$1" in
  --cpu-only) CPU_ONLY=1;; --no-net) NO_NET=1;; --prefix) PREFIX="$2"; shift;; --model-source) MODEL_SRC="$2"; shift;;
  *) echo "unknown option $1"; exit 2;; esac; shift; done
# shellcheck source=fleet.env
source "$HERE/fleet.env"
TOTAL="${TOTAL:-12}"
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
NIC=$(ip -o link show | awk -F': ' '$2 !~ /lo|docker|veth|tailscale|wl/ {print $2; exit}')
log "rank $RANK of $TOTAL on Ubuntu $VERSION_ID, ${RAM_GB} GB RAM, python $PYV, nic ${NIC:-?}, prefix $PREFIX"

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
OVDIR=$(ls -d "$PREFIX"/ov/openvino_genai_* | head -1)
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
    ZE=$(ls "$HERE"/gpu-debs/libze1_*u${UB}.04_*.deb 2>/dev/null | head -1)
    [ -n "$ZE" ] || ZE=$(ls "$HERE"/gpu-debs/libze1_*u24.04_*.deb | head -1)   # 26.04: the 24.04 build installs fine
    dpkg -i "$HERE"/gpu-debs/libigdgmm12_*.deb "$HERE"/gpu-debs/intel-igc-core-2_*.deb "$HERE"/gpu-debs/intel-igc-opencl-2_*.deb \
            "$ZE" "$HERE"/gpu-debs/libze-intel-gpu1_*.deb "$HERE"/gpu-debs/intel-opencl-icd_*.deb \
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
  MAX_FUSED="${FUSED_LAYERS_LINUX:-4}"
  cands=""; for l in $(seq "$LO" $((HI-1))); do [ "$l" -ge 2 ] && cands="$cands $l"; done
  n=0; for l in $cands; do [ $n -lt "$MAX_FUSED" ] && FUSED="${FUSED:+$FUSED,}$l" && n=$((n+1)); done
  if [ -n "$FUSED" ] && done_step pylib && ! done_step "fused:$FUSED"; then
    log "generating fused MoE IRs for layers $FUSED (about 30 s each)"
    PYTHONPATH="$PYLIB" $PY "$HERE/tools/inkling_moe_layer_ov.py" --src "$PREFIX/model" --layers "$FUSED" --layout u4zp --pad-experts 4 --skip-existing >> "$PREFIX/logs/fused-ir.log" 2>&1 \
      && mark "fused:$FUSED" || { log "fused IR generation failed (logs/fused-ir.log); the iGPU will serve attention only"; FUSED=""; }
  elif [ -n "$FUSED" ] && ! done_step pylib; then FUSED=""; fi
fi

# ---------- 6. rank environment ----------
IP_NEXT=""; MYIP=""
if [ "$RANK" -lt $((TOTAL-1)) ]; then v="IP_$((RANK+1))"; IP_NEXT="${!v}"; fi
v="IP_$RANK"; MYIP="${!v}"
{
  echo "# generated by install.sh for rank $RANK of $TOTAL — layers [$LO,$HI)"
  echo "RANK=$RANK"; echo "TOTAL=$TOTAL"; echo "LAYER_START=$LO"; echo "LAYER_END=$HI"; echo "NEXT=${IP_NEXT:+$IP_NEXT:$((RELAY_PORT+RANK+1))}"
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
    if [ -n "$FUSED" ]; then echo "CASCADIA_INKLING_OV_MOE=1"; echo "CASCADIA_INKLING_OV_MOE_DEVICE=GPU"; echo "CASCADIA_INKLING_OV_MOE_LAYERS=$FUSED"; fi
  fi
} > "$PREFIX/rank.env"
install -m 0755 "$HERE/fleet/run.sh" "$PREFIX/run.sh"
install -m 0755 "$HERE/fleet/status.sh" "$PREFIX/status.sh"
log "rank.env written: layers [$LO,$HI), iGPU=$GPU_OK, fused=[${FUSED:-none}], next=${IP_NEXT:-none}"

# ---------- 7. static IP on the wired NIC ----------
if [ "$NO_NET" = 0 ] && [ -n "$MYIP" ] && [ -n "$NIC" ]; then
  cat > /etc/netplan/60-cascadia-inkling.yaml <<YAML
network:
  version: 2
  ethernets:
    $NIC:
      dhcp4: true
      addresses: [$MYIP/24]
YAML
  chmod 600 /etc/netplan/60-cascadia-inkling.yaml
  netplan apply 2>>"$PREFIX/logs/netplan.log" || log "netplan apply failed (see logs/netplan.log)"
  log "static address $MYIP/24 added on $NIC (DHCP kept)"
fi

# ---------- 8. systemd service ----------
cat > /etc/systemd/system/cascadia-inkling.service <<UNIT
[Unit]
Description=Cascadia Inkling rank $RANK of $TOTAL
After=network-online.target
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
systemctl daemon-reload
systemctl enable cascadia-inkling.service > /dev/null 2>&1
systemctl restart cascadia-inkling.service
sleep 3
systemctl --no-pager --lines=0 status cascadia-inkling.service 2>/dev/null | head -3 || true
log "installed. Follow with: $PREFIX/status.sh   (logs: journalctl -u cascadia-inkling -f)"
