#!/usr/bin/env bash
# setup-openvino.sh — install the Linux GPU runtime stack OpenVINO needs.
#
# OpenVINO GPU inference is NOT a single install: beyond the SDK, the GPU
# plugin needs the Intel Compute Runtime, OpenCL ICD, and Level-Zero
# loader, and your user must be in the `render` group. A correct SDK +
# driver can still leave OpenVINO seeing only the CPU until these are in
# place — this script installs them so `cascadia doctor` reports the iGPU.
#
# It does NOT download the OpenVINO GenAI SDK itself (that's a separate,
# sometimes click-through download from intel.com) — see INSTALL.md.
#
# Safe to re-run. Ubuntu 22.04 / 24.04. Other distros: see INSTALL.md.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script targets Linux. On Windows, run scripts/setup-openvino.ps1."
  echo "On macOS there is no Intel GPU runtime to install (dev/stub use only)."
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script uses apt-get (Ubuntu/Debian). For other distros, install the"
  echo "equivalents of: intel-opencl-icd intel-level-zero-gpu level-zero ocl-icd-libopencl1"
  echo "and add your user to the 'render' group. See INSTALL.md."
  exit 1
fi

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

echo "==> Installing Intel GPU runtime packages (OpenCL + Level-Zero + Compute Runtime)"
$SUDO apt-get update
$SUDO apt-get install -y \
  ocl-icd-libopencl1 \
  intel-opencl-icd \
  intel-level-zero-gpu \
  level-zero

echo "==> Adding $USER to the 'render' group (needed for GPU device access)"
if id -nG "$USER" | tr ' ' '\n' | grep -qx render; then
  echo "    already in 'render' group."
else
  $SUDO usermod -a -G render "$USER"
  echo "    added. LOG OUT AND BACK IN (or reboot) — group changes don't apply"
  echo "    to the current shell session."
fi

echo
echo "Done. Next:"
echo "  1. Download the OpenVINO GenAI 2026.2+ SDK and set INTEL_OPENVINO_DIR (see INSTALL.md)."
echo "  2. Build:   INTEL_OPENVINO_DIR=... cargo build --release -p cascadia --features openvino"
echo "  3. Verify:  cascadia doctor   (should list a GPU device, not just CPU)"
