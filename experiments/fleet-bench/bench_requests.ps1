param(
    [string]$Tag = "m",
    [int]$ApiPort = 8099,
    [int]$MaxTokens = 24,
    [int]$HealthTries = 480
)
$ErrorActionPreference = "Continue"
$ok = $false
for ($i = 0; $i -lt $HealthTries; $i++) {
    Start-Sleep -Seconds 5
    try { Invoke-RestMethod -Uri ("http://127.0.0.1:" + $ApiPort + "/health") -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
}
if (-not $ok) { Write-Output "HEALTH-TIMEOUT"; exit 1 }
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("HEALTH-OK free-GB=" + [math]::Round($os.FreePhysicalMemory/1MB,1))
$para = "The quick brown fox jumps over the lazy dog near the quiet river bank at dawn. "
$lp = ($para * 25) + "Now explain how rainbows form."
$sb = '{"model":"' + $Tag + '","messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}],"max_tokens":' + $MaxTokens + ',"temperature":0}'
$lb = '{"model":"' + $Tag + '","messages":[{"role":"user","content":"' + $lp + '"}],"max_tokens":' + $MaxTokens + ',"temperature":0}'
$uri = "http://127.0.0.1:" + $ApiPort + "/v1/chat/completions"
foreach ($n in 1..2) {
    $t = Measure-Command { $script:r = Invoke-RestMethod -Uri $uri -Method Post -Body $sb -ContentType "application/json" -TimeoutSec 3600 }
    Write-Output ("SHORT-REQ${n}: " + [math]::Round($t.TotalSeconds,2) + "s -> " + ($script:r.choices[0].message.content -replace "`n"," "))
}
foreach ($n in 1..2) {
    $t = Measure-Command { $script:r = Invoke-RestMethod -Uri $uri -Method Post -Body $lb -ContentType "application/json" -TimeoutSec 3600 }
    Write-Output ("LONG-REQ${n}: " + [math]::Round($t.TotalSeconds,2) + "s -> " + ($script:r.choices[0].message.content -replace "`n"," "))
}
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("free-loaded-GB=" + [math]::Round($os.FreePhysicalMemory/1MB,1))
