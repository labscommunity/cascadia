# Inkling rank installer for Windows 11 boxes - everything comes from this SSD.
#
#   Right-click PowerShell "Run as administrator", then:
#     Set-ExecutionPolicy -Scope Process Bypass -Force
#     E:\inkling-deploy\install.ps1 -Rank 3            # rank 0..TOTAL-1
#     E:\inkling-deploy\install.ps1 -Rank 3 -CpuOnly   # no iGPU
#     E:\inkling-deploy\install.ps1 -Rank 3 -NoNet     # keep the box's current IP
#
# Copies the binary and the OpenVINO 2026.3.1 runtime side by side under
# C:\cascadia-inkling (nothing system-wide is replaced; other Cascadia or
# OpenVINO installs on the box are untouched), copies this rank's model slice
# to local disk, unpacks a private Python and generates the fused MoE IRs for
# the iGPU layers, writes the rank's environment, opens the firewall for the
# relay/API ports, and registers a scheduled task that starts the rank at
# boot and restarts it if it stops.
param(
  [Parameter(Mandatory=$true)][int]$Rank,
  [string]$Prefix = 'C:\cascadia-inkling',
  [string]$ModelSource = '',
  [switch]$CpuOnly,
  [switch]$NoNet
)
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
function Log($m) { Write-Host "[install $(Get-Date -Format T)] $m" }
function Or($v, $d) { if ($null -eq $v -or "$v" -eq '') { $d } else { $v } }
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw "run as administrator" }

# fleet.env (KEY=VALUE lines)
$fleet = @{}
Get-Content "$Here\fleet.env" | Where-Object { $_ -match '^\s*([A-Za-z_0-9]+)\s*=\s*(.*)$' -and $_ -notmatch '^\s*#' } | ForEach-Object { $fleet[$matches[1]] = $matches[2].Trim() }
$Total = [int](Or $fleet['TOTAL'] 12); $RelayPort = [int](Or $fleet['RELAY_PORT'] 9100)
if ($Rank -lt 0 -or $Rank -ge $Total) { throw "rank must be 0..$($Total-1)" }
if ($ModelSource -eq '') { $ModelSource = Join-Path (Split-Path -Parent $Here) 'inkling\out' }
New-Item -ItemType Directory -Force -Path $Prefix, "$Prefix\logs" | Out-Null
$state = "$Prefix\install.state"; if (-not (Test-Path $state)) { New-Item $state -ItemType File | Out-Null }
function Done($s) { (Get-Content $state) -contains $s }
function Mark($s) { Add-Content $state $s }
$ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Log "rank $Rank of $Total, $ramGb GB RAM, prefix $Prefix, model source $ModelSource"

# ---------- 0. stop a rank that is already running here (re-install) ----------
# Order matters: the task's run.ps1 loop relaunches its worker within 5 s, so end every
# copy of the loop before the workers. Only processes under this prefix are touched;
# other cascadia deployments on the box keep running.
Stop-ScheduledTask -TaskName 'CascadiaInkling' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'CascadiaInkling' -Confirm:$false -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run\.ps1' -and $_.CommandLine -match [regex]::Escape($Prefix) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process cascadia, cascadia-ov -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Prefix\*" } | Stop-Process -Force -ErrorAction SilentlyContinue
foreach ($i in 1..10) { if (-not (Get-Process cascadia, cascadia-ov -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Prefix\*" })) { break }; Start-Sleep -Seconds 1 }
Start-Sleep -Seconds 2

# ---------- 1. binaries + OpenVINO runtime ----------
Copy-Item "$Here\bin\windows\cascadia-ov.exe" "$Prefix\cascadia-ov.exe" -Force
Copy-Item "$Here\bin\windows\cascadia.exe" "$Prefix\cascadia.exe" -Force
$ovZip = "$Here\runtime\openvino_genai_windows_2026.3.1.0_x86_64.zip"
if (-not (Done 'runtime')) {
  if (Test-Path "$Prefix\ov\openvino_genai_windows_2026.3.1.0_x86_64\setupvars.bat") {
    Log "OpenVINO runtime already present under $Prefix\ov"
  } else {
    Log "OpenVINO runtime -> $Prefix\ov"
    if (Test-Path "$Prefix\ov") { Remove-Item -Recurse -Force "$Prefix\ov" }
    Expand-Archive -Path $ovZip -DestinationPath "$Prefix\ov" -Force
  }
  Mark 'runtime'
}
$OvDir = (Get-ChildItem "$Prefix\ov" -Directory | Select-Object -First 1).FullName

# ---------- 2. iGPU present? ----------
$GpuOk = $false
if (-not $CpuOnly) {
  $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' } | Select-Object -First 1
  if ($gpu) { $GpuOk = $true; Log "iGPU: $($gpu.Name) driver $($gpu.DriverVersion)" } else { Log "no Intel GPU found; CPU only" }
}

# ---------- 3. model slice ----------
$m = Get-Content "$ModelSource\manifest.json" | ConvertFrom-Json
$n = [int]$m.num_layers; $base = [math]::Floor($n / $Total); $rem = $n % $Total
$Lo = [int]($Rank * $base + [math]::Min($Rank, $rem)); $Hi = [int]($Lo + $base + $(if ($Rank -lt $rem) { 1 } else { 0 }))
$files = @('manifest.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'source_config.json')
for ($li = $Lo; $li -lt $Hi; $li++) {
  $tag = '{0:d2}' -f [int]$li
  $files += "shells\layer_$tag.safetensors"
  $files += Get-ChildItem "$ModelSource\experts\layer_$tag" -File | ForEach-Object { "experts\layer_$tag\$($_.Name)" }
  if (Test-Path "$ModelSource\attn_ov\layer_$tag") { $files += Get-ChildItem "$ModelSource\attn_ov\layer_$tag" -Recurse -File | ForEach-Object { $_.FullName.Substring($ModelSource.Length + 1) } }
}
if ($Rank -eq 0) { $files += 'embed.safetensors' }
if ($Rank -eq $Total - 1) { $files += 'head.safetensors'; if (Test-Path "$ModelSource\head_ov") { $files += Get-ChildItem "$ModelSource\head_ov" -File | ForEach-Object { "head_ov\$($_.Name)" } } }
$files = $files | Where-Object { Test-Path "$ModelSource\$_" }
$bytes = ($files | ForEach-Object { (Get-Item "$ModelSource\$_").Length } | Measure-Object -Sum).Sum
Log ("slice: layers [{0},{1}) {2} files {3:n1} GiB -> {4}\model" -f $Lo, $Hi, $files.Count, ($bytes / 1GB), $Prefix)
$t0 = Get-Date; $done = 0
foreach ($f in $files) {
  $src = "$ModelSource\$f"; $dst = "$Prefix\model\$f"
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  if (-not ((Test-Path $dst) -and ((Get-Item $dst).Length -eq (Get-Item $src).Length))) { Copy-Item $src $dst -Force }
  $done += (Get-Item $src).Length
}
Log ("slice done in {0:n0} s" -f ((Get-Date) - $t0).TotalSeconds)
@{ rank = $Rank; total = $Total; layer_start = $Lo; layer_end = $Hi; num_layers = $n } | ConvertTo-Json | Set-Content "$Prefix\model\rank.json"

# ---------- 4. private Python for IR generation ----------
$Py = "$Prefix\py\python.exe"
if ($GpuOk -and -not (Done 'pylib')) {
  Log "private Python (embeddable 3.12 + numpy + openvino wheels)"
  if (Test-Path "$Prefix\py") { Remove-Item -Recurse -Force "$Prefix\py" }
  Expand-Archive -Path "$Here\runtime\python-3.12.10-embed-amd64.zip" -DestinationPath "$Prefix\py" -Force
  $pth = Get-ChildItem "$Prefix\py" -Filter 'python*._pth' | Select-Object -First 1
  Add-Content $pth.FullName "Lib\site-packages"; Add-Content $pth.FullName "import site"
  New-Item -ItemType Directory -Force -Path "$Prefix\py\Lib\site-packages" | Out-Null
  foreach ($w in (Get-ChildItem "$Here\wheels" -Filter '*cp312*win_amd64.whl')) {
    Copy-Item $w.FullName "$Prefix\logs\$($w.BaseName).zip" -Force
    Expand-Archive -Path "$Prefix\logs\$($w.BaseName).zip" -DestinationPath "$Prefix\py\Lib\site-packages" -Force
    Remove-Item "$Prefix\logs\$($w.BaseName).zip"
  }
  & $Py -c "import numpy, openvino; print('python ok', openvino.__version__)" 2>&1 | Tee-Object -FilePath "$Prefix\logs\pylib.log"
  if ($LASTEXITCODE -eq 0) { Mark 'pylib' } else { Log "private Python failed (logs\pylib.log): the iGPU will serve attention only" }
}

# ---------- 5. fused MoE IRs ----------
$Fused = ''
if ($GpuOk) {
  $maxFused = [int](Or $fleet['FUSED_LAYERS_WINDOWS'] 3)
  $cands = @(); for ($li = $Lo; $li -lt $Hi; $li++) { if ($li -ge 2) { $cands += $li } }
  $Fused = ($cands | Select-Object -First $maxFused) -join ','
  if ($Fused -ne '' -and (Done 'pylib') -and -not (Done "fused:$Fused")) {
    Log "generating fused MoE IRs for layers $Fused (about a minute each)"
    & $Py "$Here\tools\inkling_moe_layer_ov.py" --src "$Prefix\model" --layers $Fused --layout u4zp --pad-experts 4 --skip-existing 2>&1 | Tee-Object -FilePath "$Prefix\logs\fused-ir.log"
    if ($LASTEXITCODE -eq 0) { Mark "fused:$Fused" } else { Log "fused IR generation failed (logs\fused-ir.log); attention only"; $Fused = '' }
  } elseif (-not (Done 'pylib')) { $Fused = '' }
}

# ---------- 6. rank environment ----------
$next = ''; if ($Rank -lt $Total - 1) { $next = "$($fleet["IP_$($Rank+1)"]):$($RelayPort + $Rank + 1)" }
$env_lines = @(
  "# generated by install.ps1 for rank $Rank of $Total - layers [$Lo,$Hi)",
  "RANK=$Rank", "TOTAL=$Total", "LAYER_START=$Lo", "LAYER_END=$Hi", "NEXT=$next", "OVDIR=$OvDir",
  "RAYON_NUM_THREADS=$([Environment]::ProcessorCount)", "CASCADIA_INKLING_MAX_SEQ=$(Or $fleet['MAX_SEQ'] 1024)", "CASCADIA_STREAMS=$(Or $fleet['STREAMS'] 16)", "CASCADIA_FRAME_IDLE_CEILING_SECS=0", "CASCADIA_API_MAX_CONCURRENT=$([int](Or $fleet['STREAMS'] 16) * 4)",
  "CASCADIA_INKLING_SERIAL_EXPERTS=0", "CASCADIA_INKLING_SEQ_READS=0", "CASCADIA_INKLING_PIN_EXPERTS=0", "CASCADIA_BF16_GEMV_ROWS=2", "CASCADIA_INT4_GEMV_ROWS=4",
  "CASCADIA_INKLING_MMAP_EMBED=1", "CASCADIA_INKLING_OWN_SHARED=1", "CASCADIA_INKLING_REUSE_READ_BUFFERS=1", "CASCADIA_INKLING_SKIP_BULK_PREFETCH=1",
  "CASCADIA_INKLING_UNCACHED_READS=1", "CASCADIA_INKLING_PIPELINE_READS=1", "CASCADIA_INKLING_EXPERT_CACHE_MIB=8000", "CASCADIA_INKLING_PREFILL_READS=1",
  "CASCADIA_INKLING_CACHE_RESET_HISTORY=0", "CASCADIA_INKLING_CACHE_DECAY_REQUESTS=4096", "CASCADIA_INKLING_CACHE_RECENT_TIES=1", "CASCADIA_INKLING_PREDICT_READS=1",
  "CASCADIA_INKLING_EARLY_PREDICT_READS=0", "CASCADIA_INKLING_SECOND_PREDICT_READS=1", "CASCADIA_INKLING_SECOND_PREDICT_RANK=2", "CASCADIA_INKLING_THIRD_PREDICT_READS=0", "RUST_LOG=info"
)
if ($GpuOk) {
  $env_lines += @("CASCADIA_INKLING_OV_ATTN=1", "CASCADIA_INKLING_OV_ATTN_DEVICE=GPU", "CASCADIA_INKLING_OV_ATTN_DIR=attn_ov", "CASCADIA_INKLING_OV_ATTN_DROP_RUST=1",
                  "CASCADIA_INKLING_OV_HEAD=1", "CASCADIA_INKLING_OV_HEAD_DEVICE=GPU", "OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=0")
  if ($Fused -ne '') { $env_lines += @("CASCADIA_INKLING_OV_MOE=1", "CASCADIA_INKLING_OV_MOE_DEVICE=GPU", "CASCADIA_INKLING_OV_MOE_LAYERS=$Fused") }
}
$env_lines | Set-Content "$Prefix\rank.env"
Copy-Item "$Here\fleet\run.ps1" "$Prefix\run.ps1" -Force
Copy-Item "$Here\fleet\status.ps1" "$Prefix\status.ps1" -Force
Log "rank.env written: layers [$Lo,$Hi), iGPU=$GpuOk, fused=[$Fused], next=$next"

# ---------- 7. firewall + static IP ----------
Get-NetFirewallRule -DisplayName 'cascadia inkling*' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName 'cascadia inkling program' -Direction Inbound -Program "$Prefix\cascadia-ov.exe" -Action Allow -Profile Any | Out-Null
New-NetFirewallRule -DisplayName 'cascadia inkling program cpu' -Direction Inbound -Program "$Prefix\cascadia.exe" -Action Allow -Profile Any | Out-Null
New-NetFirewallRule -DisplayName 'cascadia inkling ports' -Direction Inbound -Protocol TCP -LocalPort 8000,9100-9120 -Action Allow -Profile Any | Out-Null
$myIp = $fleet["IP_$Rank"]
if (-not $NoNet -and $myIp) {
  $nic = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' -and $_.Name -notmatch 'Tailscale|Wi-Fi|Bluetooth|vEthernet' } | Sort-Object -Property LinkSpeed -Descending | Select-Object -First 1
  if ($nic) {
    if (-not (Get-NetIPAddress -InterfaceIndex $nic.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -eq $myIp })) {
      New-NetIPAddress -InterfaceIndex $nic.ifIndex -IPAddress $myIp -PrefixLength 24 -ErrorAction SilentlyContinue | Out-Null
    }
    Log "static address $myIp/24 added on $($nic.Name) (existing addresses kept)"
  }
}

# ---------- 8. scheduled task (starts at boot, restarts on failure) ----------
$taskName = 'CascadiaInkling'
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Prefix\run.ps1`""
$trigger = New-ScheduledTaskTrigger -AtStartup
# (run.ps1 restarts the worker itself within 5 s; the task restart is the backstop and Windows wants >= 1 minute)
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -User 'SYSTEM' | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
Log "installed. Check with: $Prefix\status.ps1   (log: $Prefix\logs\worker.log)"
