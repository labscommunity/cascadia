# setup-openvino.ps1 — Windows OpenVINO GPU readiness check.
#
# On Windows the OpenCL runtime ships INSIDE the Intel graphics driver,
# so there's no separate runtime stack to apt-install (unlike Linux).
# GPU readiness comes down to: a current Intel graphics driver + the
# OpenVINO GenAI SDK + the C++ build toolchain. This script checks for
# those and points you at the downloads; it does not silently install
# system drivers.
#
# Run from a "Developer Command Prompt for VS 2022" (PowerShell) so the
# MSVC toolchain is on PATH for `--features openvino` builds.

Write-Host "==> Windows OpenVINO readiness check`n"

# 1. Graphics driver / GPU presence (informational).
Write-Host "Intel GPU(s) detected by Windows:"
try {
    Get-CimInstance Win32_VideoController |
        Where-Object { $_.Name -match 'Intel' } |
        ForEach-Object { Write-Host "  - $($_.Name)  (driver $($_.DriverVersion))" }
} catch {
    Write-Host "  (could not query Win32_VideoController)"
}
Write-Host "  If your iGPU/Arc isn't listed or the driver is old, update it:"
Write-Host "  https://www.intel.com/content/www/us/en/download-center/home.html`n"

# 2. MSVC toolchain.
if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    Write-Host "MSVC compiler (cl.exe): found on PATH"
} else {
    Write-Host "MSVC compiler (cl.exe): NOT on PATH"
    Write-Host "  Install 'Visual Studio 2022 Build Tools' and run this from a"
    Write-Host "  'Developer Command Prompt for VS 2022'."
}

# 3. OpenVINO SDK env.
if ($env:INTEL_OPENVINO_DIR -and (Test-Path (Join-Path $env:INTEL_OPENVINO_DIR 'runtime\include'))) {
    Write-Host "INTEL_OPENVINO_DIR: $env:INTEL_OPENVINO_DIR (looks valid)"
} else {
    Write-Host "INTEL_OPENVINO_DIR: not set / not valid"
    Write-Host "  Download the OpenVINO GenAI 2026.2+ SDK (see INSTALL.md) and set it, e.g.:"
    Write-Host '    $env:INTEL_OPENVINO_DIR = "C:\openvino_genai_2026.2.0.0"'
}

Write-Host "`nNext:"
Write-Host "  cargo build --release -p cascadia --features openvino"
Write-Host "  cascadia doctor   # should list a GPU device, not just CPU"
