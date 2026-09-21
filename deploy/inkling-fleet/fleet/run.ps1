# Runs this box's Inkling rank (started by the CascadiaInkling scheduled task); restarts the worker if it exits.
$Prefix = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-Content "$Prefix\rank.env" | Where-Object { $_ -match '^\s*([A-Za-z_0-9]+)\s*=\s*(.*)$' -and $_ -notmatch '^\s*#' } | ForEach-Object { Set-Item -Path "Env:$($matches[1])" -Value $matches[2].Trim() }
$exe = "$Prefix\cascadia.exe"
if ($env:CASCADIA_INKLING_OV_ATTN -eq '1' -and (Test-Path "$env:OVDIR\setupvars.bat")) {
  $vars = cmd /c "call `"$env:OVDIR\setupvars.bat`" > nul 2>&1 && set"
  foreach ($line in $vars) { if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] } }
  $exe = "$Prefix\cascadia-ov.exe"
}
$args = @('worker', '--rank', $env:RANK, '--total', $env:TOTAL, '--engine', 'sparse-moe', '--device', 'CPU', '--model', "$Prefix\model", '--layer-start', $env:LAYER_START, '--layer-end', $env:LAYER_END)
if ([int]$env:RANK -gt 0) { $args += @('--listen', ":$(9100 + [int]$env:RANK)") }
if ($env:NEXT) { $args += @('--next', $env:NEXT) }
if ([int]$env:RANK -eq 0) { $args += @('--api', ':8000') }
while ($true) {
  # this task is the only launcher for this prefix: a worker left behind by an earlier supervisor would hold the ports
  Get-Process cascadia, cascadia-ov -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Prefix\*" } | Stop-Process -Force -ErrorAction SilentlyContinue
  $log = "$Prefix\logs\worker.log"
  if ((Test-Path $log) -and ((Get-Item $log).Length -gt 200MB)) { Move-Item $log "$Prefix\logs\worker.prev.log" -Force }
  $p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  try { $p.PriorityClass = 'High' } catch {}
  $p.WaitForExit()
  Add-Content "$Prefix\logs\restarts.log" "$(Get-Date -Format s) worker exited with $($p.ExitCode); restarting in 5 s"
  Start-Sleep -Seconds 5
}
