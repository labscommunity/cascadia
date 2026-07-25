param(
    [string]$Shards = "C:\cascadia\models\hybrid-test\l32-1b-1stage",
    [string]$Device = "CPU",
    [string]$PrefillDevice = "",
    [int]$MaxNew = 32,
    [int]$PromptRepeat = 0,
    [int]$Park = 0,
    [int]$GemvOffload = 0,
    [int]$GemvMax = -1,
    [string]$GemvSkip = "",
    [int]$GemvStats = 0,
    [int]$PerfDump = 0,
    [int]$Dnnl = 0,
    [int]$ParitySoft = 0,
    [int]$Tasks = 0,
    [string]$CacheDir = ""
)
$ErrorActionPreference = "Stop"
$sdk = "C:\cascadia\ovgenai\openvino_genai_windows_2026.1.0.0_x86_64"
$env:PATH = "$sdk\runtime\bin\intel64\Release;$sdk\runtime\3rdparty\tbb\bin;C:\cascadia\dnnl\bin;" + $env:PATH
$env:CASCADIA_STATIC_SHARDS = $Shards
$env:CASCADIA_STATIC_DEVICE = $Device
if ($PrefillDevice -ne "") { $env:CASCADIA_PREFILL_DEVICE = $PrefillDevice }
$env:CASCADIA_STATIC_MAX_NEW = "$MaxNew"
if ($PromptRepeat -gt 0) {
    $para = "The quick brown fox jumps over the lazy dog near the quiet river bank at dawn. "
    $env:CASCADIA_STATIC_PROMPT = ($para * $PromptRepeat) + "Now explain how rainbows form."
}
if ($CacheDir -ne "") { $env:CASCADIA_OV_CACHE = $CacheDir }
if ($Park -eq 1) { $env:CASCADIA_PARK = "1" } else { Remove-Item Env:CASCADIA_PARK -ErrorAction SilentlyContinue }
if ($GemvOffload -eq 1) { $env:CASCADIA_GEMV_OFFLOAD = "1" } else { Remove-Item Env:CASCADIA_GEMV_OFFLOAD -ErrorAction SilentlyContinue }
if ($GemvMax -ge 0) { $env:CASCADIA_GEMV_MAX = "$GemvMax" } else { Remove-Item Env:CASCADIA_GEMV_MAX -ErrorAction SilentlyContinue }
if ($GemvSkip -ne "") { $env:CASCADIA_GEMV_SKIP = $GemvSkip } else { Remove-Item Env:CASCADIA_GEMV_SKIP -ErrorAction SilentlyContinue }
if ($GemvStats -eq 1) { $env:CASCADIA_GEMV_STATS = "1" } else { Remove-Item Env:CASCADIA_GEMV_STATS -ErrorAction SilentlyContinue }
if ($PerfDump -eq 1) { $env:CASCADIA_PERF_DUMP = "1"; $env:CASCADIA_OV_PROPS = "PERF_COUNT=YES" } else { Remove-Item Env:CASCADIA_PERF_DUMP -ErrorAction SilentlyContinue }
if ($Dnnl -ge 1) { $env:CASCADIA_GEMV_DNNL = "$Dnnl" } else { Remove-Item Env:CASCADIA_GEMV_DNNL -ErrorAction SilentlyContinue }
if ($ParitySoft -eq 1) { $env:CASCADIA_PARITY_SOFT = "1" } else { Remove-Item Env:CASCADIA_PARITY_SOFT -ErrorAction SilentlyContinue }
if ($Tasks -ge 2) { $env:CASCADIA_STATIC_TASKS = "$Tasks" } else { Remove-Item Env:CASCADIA_STATIC_TASKS -ErrorAction SilentlyContinue }
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("free-before-GB=" + [math]::Round($os.FreePhysicalMemory/1MB,1))
$exe = Get-ChildItem C:\cascadia\oss-hybrid\target\release\deps\static_prefill_parity-*.exe |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
cmd /c "`"$($exe.FullName)`" --nocapture > C:\cascadia\hybrid_run.log 2>&1"
$code = $LASTEXITCODE
Get-Content C:\cascadia\hybrid_run.log
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("free-after-GB=" + [math]::Round($os.FreePhysicalMemory/1MB,1))
Write-Output ("exit=" + $code)
exit $code
