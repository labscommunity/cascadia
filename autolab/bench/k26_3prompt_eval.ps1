# K2.6 3-prompt baseline eval - runs on matias-02 (rank 0, API host).
# Polls API readiness, then issues Paris/Pacific/four prompts and
# emits a JSON result per prompt + an aggregate line.
#
# Usage on matias-02:
#   powershell -NoProfile -File k26_3prompt_eval.ps1
#   # writes to ~/k26-bench-baseline.jsonl by default
#
# Override: -ApiBase http://127.0.0.1:8000  -MaxTokens 8  -OutFile X

param(
    [string]$ApiBase = "http://127.0.0.1:8000",
    [int]$MaxTokens = 8,
    [int]$ReadyTimeoutMin = 90,
    [string]$OutFile = "$env:USERPROFILE\k26-bench-baseline.jsonl"
)

$prompts = @(
    @{ prompt = "The capital of France is"; substr = "paris" },
    @{ prompt = "The largest ocean on Earth is the"; substr = "pacific" },
    @{ prompt = "Two plus two equals"; substr = "four" }
)

function Try-Health {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri "$Url/health" -TimeoutSec 5 -ErrorAction Stop
        return $r.StatusCode -eq 200
    } catch { return $false }
}

function Try-TinyRequest {
    param([string]$Base)
    try {
        $body = @{
            model = "k26"
            messages = @(@{ role = "user"; content = "Hi" })
            max_tokens = 1
            temperature = 0
        } | ConvertTo-Json -Depth 5
        $r = Invoke-RestMethod -Uri "$Base/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600 -ErrorAction Stop
        return $true
    } catch { return $false }
}

Write-Host "Polling API readiness at $ApiBase ..."
$deadline = (Get-Date).AddMinutes($ReadyTimeoutMin)
$ready = $false
while ((Get-Date) -lt $deadline) {
    if (Try-Health -Url $ApiBase) {
        Write-Host "  /health OK at $(Get-Date -Format HH:mm:ss); trying tiny inference ..."
        if (Try-TinyRequest -Base $ApiBase) {
            $ready = $true
            break
        } else {
            Write-Host "  tiny request failed; not warmed yet"
        }
    }
    Start-Sleep -Seconds 30
}

if (-not $ready) {
    Write-Error "API not ready within $ReadyTimeoutMin minutes - aborting bench"
    exit 2
}

Write-Host "API warm. Running 3-prompt eval ..."
"" | Out-File $OutFile -Force

$totalTokens = 0
$totalWallMs = 0
$passCount = 0
$results = @()

foreach ($p in $prompts) {
    $body = @{
        model = "k26"
        messages = @(@{ role = "user"; content = $p.prompt })
        max_tokens = $MaxTokens
        temperature = 0
    } | ConvertTo-Json -Depth 5

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-RestMethod -Uri "$ApiBase/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 900 -ErrorAction Stop
        $sw.Stop()
        $content = $r.choices[0].message.content
        $completionTokens = if ($r.usage.completion_tokens) { $r.usage.completion_tokens } else { 0 }
        $pass = $content.ToLower().Contains($p.substr)
        if ($pass) { $passCount++ }
        $totalTokens += $completionTokens
        $totalWallMs += $sw.ElapsedMilliseconds
        $tokPerSec = if ($sw.ElapsedMilliseconds -gt 0) { [math]::Round(($completionTokens * 1000.0) / $sw.ElapsedMilliseconds, 4) } else { 0 }
        $row = @{
            prompt = $p.prompt
            substr = $p.substr
            content = $content
            completion_tokens = $completionTokens
            wall_ms = $sw.ElapsedMilliseconds
            tok_per_sec = $tokPerSec
            quality_pass = $pass
        } | ConvertTo-Json -Compress
        Write-Host $row
        $row | Out-File $OutFile -Append
        $results += $row
    } catch {
        $sw.Stop()
        Write-Host "  FAILED prompt='$($p.prompt)' err=$($_.Exception.Message) after $($sw.ElapsedMilliseconds)ms"
        @{ prompt = $p.prompt; error = $_.Exception.Message; wall_ms = $sw.ElapsedMilliseconds } | ConvertTo-Json -Compress | Out-File $OutFile -Append
    }
}

$aggTokPerSec = if ($totalWallMs -gt 0) { [math]::Round(($totalTokens * 1000.0) / $totalWallMs, 4) } else { 0 }
$summary = @{
    aggregate = $true
    total_tokens = $totalTokens
    total_wall_ms = $totalWallMs
    tok_per_sec = $aggTokPerSec
    quality_pass = "$passCount/$($prompts.Count)"
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json -Compress
Write-Host "AGG: $summary"
$summary | Out-File $OutFile -Append
