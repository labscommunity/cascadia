param(
    [string]$Model = "C:\cascadia\models\hybrid-test\q25-14b-1stage",
    [string]$Device = "CPU",
    [string]$PrefillDevice = "NPU",
    [int]$MaxTokens = 24
)
$ErrorActionPreference = "Stop"
$sdk = "C:\cascadia\ovgenai\openvino_genai_windows_2026.1.0.0_x86_64"
$env:PATH = "$sdk\runtime\bin\intel64\Release;$sdk\runtime\3rdparty\tbb\bin;C:\cascadia\dnnl\bin;" + $env:PATH
$bin = "C:\cascadia\oss-hybrid\target\release\cascadia.exe"
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1) }
Write-Output ("free-start-GB=" + (FreeGB))
$args0 = @("worker","--rank","0","--total","1","--engine","ov-runtime","--model",$Model,"--device",$Device,"--api",":8098","--log-level","info")
if ($PrefillDevice -ne "") { $args0 += @("--prefill-device",$PrefillDevice) }
$p0 = $null
try {
    $p0 = Start-Process -FilePath $bin -ArgumentList $args0 -RedirectStandardError C:\cascadia\one_r0.err -RedirectStandardOutput C:\cascadia\one_r0.out -PassThru -NoNewWindow
    $ok = $false
    for ($i = 0; $i -lt 240; $i++) {
        Start-Sleep -Seconds 5
        if ($p0.HasExited) { throw "worker exited early" }
        try { Invoke-RestMethod -Uri http://127.0.0.1:8098/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
    }
    if (-not $ok) { throw "health never came up" }
    Write-Output ("HEALTH-OK free-GB=" + (FreeGB))
    $para = "The quick brown fox jumps over the lazy dog near the quiet river bank at dawn. "
    $lp = ($para * 25) + "Now explain how rainbows form."
    $sb = '{"model":"m","messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}],"max_tokens":' + $MaxTokens + ',"temperature":0}'
    $lb = '{"model":"m","messages":[{"role":"user","content":"' + $lp + '"}],"max_tokens":' + $MaxTokens + ',"temperature":0}'
    foreach ($n in 1..2) {
        $t = Measure-Command { $script:r = Invoke-RestMethod -Uri http://127.0.0.1:8098/v1/chat/completions -Method Post -Body $sb -ContentType "application/json" -TimeoutSec 1800 }
        Write-Output ("SHORT-REQ${n}: " + [math]::Round($t.TotalSeconds,2) + "s -> " + ($script:r.choices[0].message.content -replace "`n"," "))
    }
    foreach ($n in 1..2) {
        $t = Measure-Command { $script:r = Invoke-RestMethod -Uri http://127.0.0.1:8098/v1/chat/completions -Method Post -Body $lb -ContentType "application/json" -TimeoutSec 1800 }
        Write-Output ("LONG-REQ${n}: " + [math]::Round($t.TotalSeconds,2) + "s -> " + ($script:r.choices[0].message.content -replace "`n"," "))
    }
    Write-Output ("free-loaded-GB=" + (FreeGB))
} finally {
    if ($p0 -and -not $p0.HasExited) { Stop-Process -Id $p0.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    Write-Output ("free-end-GB=" + (FreeGB))
}
Write-Output "---- worker log tail ----"
Select-String -Path C:\cascadia\one_r0.err -Pattern "prefill|task done|importing" -ErrorAction SilentlyContinue | Select-Object -Last 6 | ForEach-Object { $_.Line }
