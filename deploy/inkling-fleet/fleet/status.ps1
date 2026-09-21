# One-screen health for this rank.
$Prefix = Split-Path -Parent $MyInvocation.MyCommand.Path
$e = @{}; Get-Content "$Prefix\rank.env" | Where-Object { $_ -match '^\s*([A-Za-z_0-9]+)\s*=\s*(.*)$' -and $_ -notmatch '^\s*#' } | ForEach-Object { $e[$matches[1]] = $matches[2].Trim() }
"rank $($e.RANK) of $($e.TOTAL)  layers [$($e.LAYER_START),$($e.LAYER_END))  next=$($e.NEXT)  fused=$($e.CASCADIA_INKLING_OV_MOE_LAYERS)  igpu=$($e.CASCADIA_INKLING_OV_ATTN)"
$t = Get-ScheduledTask -TaskName CascadiaInkling -ErrorAction SilentlyContinue
"task: " + $(if ($t) { $t.State } else { 'not installed' }) + "  worker process: " + ((Get-Process cascadia, cascadia-ov -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Prefix\*" } | Measure-Object).Count)
"last log lines:"
Get-Content "$Prefix\logs\worker.log", "$Prefix\logs\worker.log.err" -ErrorAction SilentlyContinue | Select-Object -Last 6 | ForEach-Object { $t = ($_ -replace '\x1b\[[0-9;]*m', ''); $t.Substring(0, [Math]::Min(160, $t.Length)) }
if ($e.RANK -eq '0') { try { "api: " + (Invoke-WebRequest -Uri http://127.0.0.1:8000/v1/models -TimeoutSec 3 -UseBasicParsing).Content.Substring(0, 120) } catch { "api: not answering" } }
$os = Get-CimInstance Win32_OperatingSystem
"ram: free " + [math]::Round($os.FreePhysicalMemory/1MB, 1) + " GB of " + [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
