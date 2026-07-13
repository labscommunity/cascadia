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
  echo "equivalents of: intel-opencl-icd, ocl-icd-libopencl1, and the Level-Zero"
  echo "driver + loader (Intel's Ubuntu repo names them libze-intel-gpu1 + libze1)."
  echo "Then add your user to the 'render' group. See INSTALL.md."
  exit 1
fi

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

# shellcheck disable=SC1091
. /etc/os-release
# Intel publishes its graphics repo for Ubuntu only; the suite is the *Ubuntu*
# codename, which on derivatives (Mint, Pop!_OS) is UBUNTU_CODENAME, not
# VERSION_CODENAME.
UBUNTU_SUITE="${UBUNTU_CODENAME:-}"
if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *ubuntu* ]]; then UBUNTU_SUITE=""; fi

INTEL_LIST=/etc/apt/sources.list.d/intel-gpu.list
# Intel's graphics signing keys (production, and the 2024 predecessor). Pinning
# the fingerprint means a hijacked key/URL cannot become an apt trust anchor.
INTEL_KEY_FPR="E0258B57D9C442D5DB1855C271740E4DE392BFE3"
INTEL_KEY_FPR_OLD="4E9EFCDEF82800256C1E7C64B02DB9BD8C321DCB"

echo "==> Installing Intel GPU runtime packages (OpenCL + Level-Zero + Compute Runtime)"
$SUDO apt-get update

if [[ -n "$UBUNTU_SUITE" ]]; then
  # Prefer Intel's repo, not the distro's: Ubuntu 24.04 ships Compute Runtime
  # 23.43, which predates Lunar Lake and Arc B — the hardware cascadia targets.
  # `unified` carries 25.18; `client` is stuck on 24.39, which is older than
  # Battlemage support (24.48).
  echo "==> Adding Intel's graphics repo (${UBUNTU_SUITE})"
  $SUDO apt-get install -y --no-install-recommends ca-certificates gnupg wget

  # Dearmor to a temp file and check the fingerprint before installing it: a
  # failed fetch must not truncate an existing keyring (gpg -o opens for write
  # before it reads), which would leave apt unable to verify the repo.
  KEY_TMP="$(mktemp)"
  trap 'rm -f "$KEY_TMP"' EXIT
  if ! wget -qO- https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor > "$KEY_TMP"; then
    echo "ERROR: could not fetch Intel's graphics key. Nothing was changed."
    exit 1
  fi
  if ! gpg --show-keys --with-colons "$KEY_TMP" | awk -F: '/^fpr:/{print $10}' \
       | grep -qx -e "$INTEL_KEY_FPR" -e "$INTEL_KEY_FPR_OLD"; then
    echo "ERROR: Intel's graphics key does not match a known fingerprint. Refusing to trust it."
    exit 1
  fi
  $SUDO install -m 0644 "$KEY_TMP" /usr/share/keyrings/intel-graphics.gpg

  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${UBUNTU_SUITE} unified" \
    | $SUDO tee "$INTEL_LIST" >/dev/null
  # If the suite doesn't exist, remove the source again rather than leaving a
  # broken entry that fails every future `apt-get update` on this machine.
  if ! $SUDO apt-get update; then
    $SUDO rm -f "$INTEL_LIST"
    echo "ERROR: Intel's repo has no '${UBUNTU_SUITE}' suite — removed $INTEL_LIST."
    echo "Install the GPU runtime by hand; see INSTALL.md."
    exit 1
  fi
  LEVEL_ZERO_PKGS=(libze-intel-gpu1 libze1)
elif apt-cache show libze-intel-gpu1 >/dev/null 2>&1; then
  # Non-Ubuntu with the packages in its own repos: use them, but say what you get.
  echo "==> Not Ubuntu; using distro packages (older Compute Runtime — new Intel"
  echo "    GPUs may not be detected). See INSTALL.md."
  LEVEL_ZERO_PKGS=(libze-intel-gpu1 libze1)
else
  echo "ERROR: no Level-Zero GPU driver available for ${PRETTY_NAME:-this distro}."
  echo "Intel publishes its graphics repo for Ubuntu only. See INSTALL.md."
  exit 1
fi

$SUDO apt-get install -y \
  ocl-icd-libopencl1 \
  intel-opencl-icd \
  "${LEVEL_ZERO_PKGS[@]}"

# Under `sudo`, $USER is root — adding *root* to the render group leaves the
# actual human without GPU access, which is the failure this script exists to
# prevent. Prefer the invoking user.
TARGET_USER="${SUDO_USER:-${USER:-$(id -un)}}"
if ! getent group render >/dev/null 2>&1; then
  # No render group means no /dev/dri — a container, or a host with no Intel GPU.
  # The packages are installed; don't fail the run over a group that can't exist.
  echo "==> No 'render' group on this host (no /dev/dri?); skipping that step."
elif [[ "$TARGET_USER" == "root" ]]; then
  echo "==> Running as root with no SUDO_USER; skipping the 'render' group step."
  echo "    Add your own account by hand:  sudo usermod -a -G render <you>"
else
  echo "==> Adding $TARGET_USER to the 'render' group (needed for GPU device access)"
  if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx render; then
    echo "    already in 'render' group."
  else
    $SUDO usermod -a -G render "$TARGET_USER"
    echo "    added. LOG OUT AND BACK IN (or reboot) — group changes don't apply"
    echo "    to the current shell session."
  fi
fi

echo
echo "Done. Next:"
echo "  1. Download the OpenVINO GenAI 2026.2+ SDK and set INTEL_OPENVINO_DIR (see INSTALL.md)."
echo "  2. Build:   INTEL_OPENVINO_DIR=... cargo build --release -p cascadia --features openvino"
echo "  3. Verify:  cascadia doctor   (should list a GPU device, not just CPU)"
