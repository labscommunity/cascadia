# Package a patched SYCL llama-server build (Windows) into a release zip.
#
# Usage: scripts\package-llama-stream-windows.ps1 BUILD_DIR OUT_ZIP [-NoRunCheck]
#   BUILD_DIR    the DEST passed to build-llama-stream-windows.bat (contains build\bin)
#   OUT_ZIP      e.g. cascadia-llama-sycl-windows-x86_64-v0.3.0.zip
#   -NoRunCheck  skip running the staged server. Use it only on a builder with
#                no Intel GPU (hosted CI): the SYCL runtime aborts at startup
#                when it finds no device, even for --version. The import walk
#                below still fails on any DLL it cannot resolve.
#
# Run in the environment used to build the server: vcvars64.bat (for
# dumpbin.exe) and oneAPI setvars.bat (so PATH resolves the SYCL runtime the
# build linked against). Every non-host DLL that the server, its ggml/llama
# DLLs and the dynamically loaded Unified Runtime adapters import is copied flat
# beside llama-server.exe; Windows searches the executable's own directory
# first. Host dependencies, which the destination must provide: Windows system
# DLLs, the MSVC runtime (VC++ redistributable) and the Intel GPU driver
# (Level Zero / OpenCL loaders and ICDs). No oneAPI installation is required.
# License texts for llama.cpp, the oneAPI compiler runtime and oneMKL go in
# licenses\.
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$OutZip,
    [switch]$NoRunCheck
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Fail([string]$msg) { [Console]::Error.WriteLine("ERROR: $msg"); exit 1 }

$binDir = (Resolve-Path (Join-Path $BuildDir "build\bin")).Path
$server = Join-Path $binDir "llama-server.exe"
if (-not (Test-Path $server)) { Fail "$server missing - run build-llama-stream-windows.bat first" }
$dumpbin = (Get-Command dumpbin.exe -ErrorAction SilentlyContinue).Source
if (-not $dumpbin) { Fail "dumpbin.exe not on PATH - call vcvars64.bat first" }

# setvars.bat normally puts the runtime on PATH, but the offline installer does
# not have to (observed on a 2026.2 windows-2022 runner, where ur_loader.dll
# sat under the oneAPI root and no PATH entry had it). Look on PATH first, then
# under the oneAPI roots, preferring a "latest" component tree.
function Find-OneApiFile([string]$name) {
    foreach ($dir in ($env:PATH -split ';' | Where-Object { $_ })) {
        $candidate = Join-Path $dir $name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    $roots = @(
        $env:ONEAPI_ROOT,
        $(if ($env:ProgramFiles) { Join-Path $env:ProgramFiles "Intel\oneAPI" }),
        $(if (${env:ProgramFiles(x86)}) { Join-Path ${env:ProgramFiles(x86)} "Intel\oneAPI" })
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }
    foreach ($root in $roots) {
        $hit = Get-ChildItem -LiteralPath $root -Filter $name -File -Recurse -ErrorAction SilentlyContinue |
            Sort-Object -Property @{ Expression = { $_.FullName -notmatch '[\\/]latest[\\/]' } }, FullName |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

$urLoader = Find-OneApiFile "ur_loader.dll"
if (-not $urLoader) {
    # Say what the install actually contains instead of failing blind: a
    # toolkit without the DPC++ runtime DLLs looks identical to a PATH problem.
    $bin = Join-Path $env:ONEAPI_ROOT "compiler\latest\bin"
    if (Test-Path -LiteralPath $bin) {
        Write-Output ("compiler\latest\bin DLLs: " + ((Get-ChildItem -LiteralPath $bin -Filter *.dll -ErrorAction SilentlyContinue | Select-Object -First 25 -ExpandProperty Name) -join ", "))
    } else {
        Write-Output "compiler\latest\bin does not exist under $env:ONEAPI_ROOT"
    }
    Fail "ur_loader.dll not on PATH or under the oneAPI roots - install the DPC++ runtime (intel.oneapi.win.cpp-dpcpp-common) or call setvars.bat first"
}
$runtimeDir = Split-Path -Parent $urLoader
Write-Output "oneAPI runtime dir: $runtimeDir"
$mklCore = Find-OneApiFile "mkl_core.3.dll"
if (-not $mklCore) { Fail "mkl_core.3.dll not on PATH or under the oneAPI roots - install oneMKL (intel.oneapi.win.mkl.devel) or call setvars.bat first" }
$mklDir = Split-Path -Parent $mklCore
Write-Output "oneMKL dir: $mklDir"
# Resolve imports against the trees we just found, PATH or not.
$env:PATH = "$runtimeDir;$mklDir;$env:PATH"

# The GPU driver owns these; a bundled copy would shadow the driver's own.
$hostNames = '^(ze_loader|ze_intel_gpu.*|ze_tracing_layer|ze_validation_layer|OpenCL|intelocl.*|igdrcl.*|igc.*|msvcp\d+.*|vcruntime\d+.*|concrt\d+|vcomp\d+|ucrtbase.*)\.dll$'

# Native tools write diagnostics to stderr; under "Stop" Windows PowerShell
# turns those into terminating errors. Merge the streams in cmd.exe instead.
function Invoke-Native([string]$exe, [string]$arguments) {
    $out = cmd.exe /d /c "`"$exe`" $arguments 2>&1" | Out-String
    return @{ Code = $LASTEXITCODE; Text = $out }
}

function Get-Imports([string]$file) {
    $r = Invoke-Native $dumpbin "/nologo /dependents `"$file`""
    if ($r.Code -ne 0) { Fail "dumpbin failed on ${file}: $($r.Text)" }
    # Both "Image has the following dependencies" and the delay-load section.
    $r.Text -split "`r?`n" | Select-String '^\s+(\S+\.dll)\s*$' | ForEach-Object { $_.Matches[0].Groups[1].Value }
}

function Resolve-Dll([string]$name, [string]$stage) {
    $staged = Join-Path $stage $name
    if (Test-Path $staged) { return $staged }
    foreach ($dir in ($env:PATH -split ';' | Where-Object { $_ })) {
        $candidate = Join-Path $dir $name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    return $null
}

$stageRoot = Join-Path ([IO.Path]::GetTempPath()) ("cascadia-llama-sycl-" + [guid]::NewGuid().ToString("N"))
$pkg = Join-Path $stageRoot "cascadia-llama-sycl"
New-Item -ItemType Directory -Force $pkg | Out-Null
try {
    # The build outputs: server, its wrapper DLLs and the ggml backends.
    Copy-Item $server $pkg
    Get-ChildItem $binDir -Filter *.dll | Copy-Item -Destination $pkg
    # Unified Runtime loads its adapters by name at run time, so imports
    # alone never reach them. Take them from the runtime the build resolved.
    foreach ($pattern in "ur_loader.dll", "ur_win_proxy_loader.dll", "ur_adapter_level_zero*.dll", "ur_adapter_opencl.dll") {
        $found = @(Get-ChildItem $runtimeDir -Filter $pattern)
        if ($found.Count -eq 0 -and $pattern -ne "ur_adapter_opencl.dll") { Fail "$pattern not found in $runtimeDir" }
        $found | Copy-Item -Destination $pkg
    }

    # Walk imports of every bundled PE to a fixed point.
    $queue = [Collections.Generic.Queue[string]]::new()
    $seen = @{}
    Get-ChildItem $pkg -Include *.exe, *.dll -Recurse | ForEach-Object { $queue.Enqueue($_.FullName) }
    $hostDeps = [Collections.Generic.SortedSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    while ($queue.Count -gt 0) {
        $file = $queue.Dequeue()
        if ($seen[$file]) { continue }
        $seen[$file] = $true
        foreach ($dep in Get-Imports $file) {
            if ($dep -match '^(api|ext)-ms-') { continue }          # API sets: resolved by the OS loader
            if ($dep -match $hostNames) { [void]$hostDeps.Add($dep); continue }
            # OS DLLs (dbghelp, wintrust, ...) also ship inside VS/oneAPI dirs
            # that sit earlier on PATH; System32 wins, like the real loader.
            if (Test-Path -LiteralPath (Join-Path "$env:SystemRoot\System32" $dep) -PathType Leaf) { [void]$hostDeps.Add($dep); continue }
            $resolved = Resolve-Dll $dep $pkg
            if (-not $resolved) { Fail "unresolved dependency of ${file}: $dep" }
            $dest = Join-Path $pkg $dep
            if (-not (Test-Path $dest)) {
                Copy-Item -LiteralPath $resolved $dest
                $queue.Enqueue($dest)
            }
        }
    }

    # Patch markers must have survived into the packaged binaries. Each one is
    # checked in the DLL that owns the patched code (0001: ggml-backend.cpp ->
    # ggml-base.dll; 0002: ggml-sycl.cpp -> ggml-sycl.dll), never "any file".
    # Exact byte search: Select-String decodes binaries and still matched a
    # DLL whose only marker bytes had been overwritten.
    foreach ($m in @(
            @{ Dll = "ggml-base.dll"; Marker = "GGML_STREAM_WEIGHTS"; Patch = "0001" },
            @{ Dll = "ggml-sycl.dll"; Marker = "GGML_STREAM_EXPERT_CACHE_MB"; Patch = "0002" })) {
        $file = Join-Path $pkg $m.Dll
        if (-not (Test-Path $file)) { Fail "$($m.Dll) missing from the build" }
        $latin1 = [Text.Encoding]::GetEncoding(28591)
        if ($latin1.GetString([IO.File]::ReadAllBytes($file)).IndexOf($m.Marker, [StringComparison]::Ordinal) -lt 0) {
            Fail "$($m.Marker) marker missing from $($m.Dll) - patch $($m.Patch) not in this build"
        }
    }

    # License texts: llama.cpp (source tree) and the Intel runtimes we
    # redistribute, found relative to the DLLs actually bundled.
    $licenses = Join-Path $pkg "licenses"
    New-Item -ItemType Directory -Force $licenses | Out-Null
    $llamaLicense = Join-Path $BuildDir "LICENSE"
    if (-not (Test-Path $llamaLicense)) { Fail "llama.cpp LICENSE not found at $llamaLicense" }
    Copy-Item $llamaLicense (Join-Path $licenses "llama.cpp-LICENSE")
    Get-ChildItem (Join-Path $BuildDir "licenses") -File -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName (Join-Path $licenses "llama.cpp-$($_.Name)") }
    foreach ($src in @(
            @{ Dir = Join-Path $runtimeDir "..\share\doc\compiler\licensing\c"; Prefix = "oneapi-compiler" },
            @{ Dir = Join-Path $mklDir "..\share\doc\mkl\licensing"; Prefix = "onemkl" })) {
        $files = @(Get-ChildItem $src.Dir -File -ErrorAction SilentlyContinue | Where-Object Name -match '^(license|third-party-programs)')
        if ($files.Count -eq 0) { Fail "no license texts in $($src.Dir)" }
        $files | ForEach-Object { Copy-Item $_.FullName (Join-Path $licenses "$($src.Prefix)-$($_.Name)") }
    }

    # Exercise the staged server as a user would: no oneAPI, no build tree on
    # PATH. --version loads every import; --list-devices loads the UR adapters.
    if ($NoRunCheck) {
        Write-Output "run check skipped (-NoRunCheck): import closure verified statically only"
    } else {
        $savedPath = $env:PATH
        $savedOneApi = $env:ONEAPI_ROOT
        try {
            $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
            Remove-Item Env:ONEAPI_ROOT -ErrorAction SilentlyContinue
            $staged = Join-Path $pkg "llama-server.exe"
            $version = Invoke-Native $staged "--version"
            if ($version.Code -ne 0) { Fail "relocated llama-server --version failed (exit $($version.Code); no Intel GPU on this builder? see -NoRunCheck): $($version.Text)" }
            $devices = Invoke-Native $staged "--list-devices"
            if ($devices.Code -ne 0) { Fail "relocated llama-server --list-devices failed: $($devices.Text)" }
        } finally {
            $env:PATH = $savedPath
            if ($savedOneApi) { $env:ONEAPI_ROOT = $savedOneApi }
        }
        Set-Content -Path (Join-Path $pkg "VERSION.txt") -Value $version.Text.Trim()
        Set-Content -Path (Join-Path $pkg "DEVICES.txt") -Value $devices.Text.Trim()
    }
    Set-Content -Path (Join-Path $pkg "HOST-DEPENDENCIES.txt") -Value ($hostDeps -join "`r`n")

    $outFull = [IO.Path]::GetFullPath($OutZip)
    if (Test-Path $outFull) { Remove-Item $outFull }
    Compress-Archive -Path $pkg -DestinationPath $outFull
    $count = @(Get-ChildItem $pkg -Include *.exe, *.dll -Recurse).Count
    $mb = [math]::Round((Get-Item $outFull).Length / 1MB)
    Write-Output "packaged: $outFull ($mb MB, $count PE files)"
    Write-Output "host DLLs (destination provides): $($hostDeps -join ', ')"
} finally {
    Remove-Item -Recurse -Force $stageRoot -ErrorAction SilentlyContinue
}
