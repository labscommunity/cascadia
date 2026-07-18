param(
    [string]$Model = "C:\cascadia\models\hybrid-test\l31-8b-2stage",
    [string]$Device = "NPU",
    [string]$PrefillDevice = "",
    [string]$CacheDir = "C:\cascadia\ovcache8b2",
    [int]$MaxTokens = 24
)
$ErrorActionPreference = "Stop"
$sdk = "C:\cascadia\ovgenai\openvino_genai_windows_2026.1.0.0_x86_64"
$env:PATH = "$sdk\runtime\bin\intel64\Release;$sdk\runtime\3rdparty\tbb\bin;C:\cascadia\dnnl\bin;" + $env:PATH
$bin = "C:\cascadia\oss-hybrid\target\release\cascadia.exe"
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1) }
Write-Output ("free-start-GB=" + (FreeGB))

$commonArgs = @("--engine","ov-runtime","--model",$Model,"--device",$Device,"--ov-cache-dir",$CacheDir,"--log-level","info")
if ($PrefillDevice -ne "") { $commonArgs += @("--prefill-device",$PrefillDevice) }

$p0 = $null; $p1 = $null
try {
    # Both ranks start together (rank 1 blocks in transport accept until
    # rank 0 dials in — serialized startup deadlocks). NPU compile transients
    # never overlap because the blob cache is PRE-WARMED sequentially via the
    # compile_warm_probe test — both stages only import blobs here.
    Remove-Item C:\cascadia\smoke8b_r1.err,C:\cascadia\smoke8b_r0.err -ErrorAction SilentlyContinue
    $p1 = Start-Process -FilePath $bin -ArgumentList (@(
        "worker","--rank","1","--total","2","--listen",":9401") + $commonArgs
    ) -RedirectStandardError C:\cascadia\smoke8b_r1.err -RedirectStandardOutput C:\cascadia\smoke8b_r1.out -PassThru -NoNewWindow
    Start-Sleep -Seconds 3

    $p0 = Start-Process -FilePath $bin -ArgumentList (@(
        "worker","--rank","0","--total","2","--listen",":9400",
        "--next","127.0.0.1:9401","--api",":8099") + $commonArgs
    ) -RedirectStandardError C:\cascadia\smoke8b_r0.err -RedirectStandardOutput C:\cascadia\smoke8b_r0.out -PassThru -NoNewWindow
    $ok = $false
    for ($i = 0; $i -lt 480; $i++) {
        Start-Sleep -Seconds 5
        if ($p0.HasExited -or $p1.HasExited) { throw "a worker exited early" }
        try {
            Invoke-RestMethod -Uri http://127.0.0.1:8099/health -TimeoutSec 2 | Out-Null
            $ok = $true; break
        } catch {}
    }
    if (-not $ok) { throw "health endpoint never came up" }
    Write-Output ("HEALTH-OK free-GB=" + (FreeGB))

    $para = "The quick brown fox jumps over the lazy dog near the quiet river bank at dawn. "
    $longPrompt = ($para * 25) + "Now explain how rainbows form."
    $shortBody = '{"model":"l31","messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}],"max_tokens":' + $MaxTokens + ',"temperature":0}'
    $longBody = '{"model":"l31","messages":[{"role":"user","content":"' + $longPrompt + '"}],"max_tokens":' + $MaxTokens + ',"temperature":0}'

    foreach ($n in 1..2) {
        $t = Measure-Command {
            $script:resp = Invoke-RestMethod -Uri http://127.0.0.1:8099/v1/chat/completions -Method Post -Body $shortBody -ContentType "application/json" -TimeoutSec 900
        }
        Write-Output ("SHORT-REQ${n}: " + [math]::Round($t.TotalSeconds,2) + "s -> " + ($script:resp.choices[0].message.content -replace "`n"," "))
    }
    foreach ($n in 1..2) {
        $t = Measure-Command {
            $script:resp = Invoke-RestMethod -Uri http://127.0.0.1:8099/v1/chat/completions -Method Post -Body $longBody -ContentType "application/json" -TimeoutSec 900
        }
        Write-Output ("LONG-REQ${n}: " + [math]::Round($t.TotalSeconds,2) + "s -> " + ($script:resp.choices[0].message.content -replace "`n"," "))
    }
    Write-Output ("free-loaded-GB=" + (FreeGB))
} finally {
    foreach ($p in @($p0, $p1)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Seconds 2
    $left = @(Get-Process cascadia -ErrorAction SilentlyContinue)
    Write-Output ("cleanup: remaining cascadia processes = " + $left.Count)
    Write-Output ("free-end-GB=" + (FreeGB))
}
Write-Output "---- rank0 task lines ----"
Select-String -Path C:\cascadia\smoke8b_r0.err -Pattern "prefill_ms|task done|decode_tok_s" -ErrorAction SilentlyContinue | Select-Object -Last 8 | ForEach-Object { $_.Line }
