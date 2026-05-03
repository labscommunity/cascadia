#!/bin/bash
set -e
mkdir -p experiments/e5-profile/logs

# Charlie: ov-runtime worker on 16/16
ssh cascadia@charlie.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' >/dev/null 2>&1
ssh cascadia@charlie.local 'powershell -Command "$env:Path=\"C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;\" + $env:Path; cd C:\Users\cascadia\tahoma-rust; .\target\release\tahoma.exe worker --rank 1 --total 2 --engine ov-runtime --device GPU --model C:\cascadia\shards_2stage_v3 --listen 10.10.10.2:9100 *>C:\Users\cascadia\e5-worker.log"' >/dev/null 2>&1 &
SSH_PID=$!
sleep 30

# Alpha trial
ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 2 --engine ov-runtime --device GPU --model C:\cascadia\shards_2stage_v3 --next 10.10.10.2:9100 --max-tokens 256 < C:\Users\cascadia\e0-prompt.txt > C:\Users\cascadia\e5-trial.log 2>&1'; Select-String -Path C:\Users\cascadia\e5-trial.log -Pattern 'task active|task done' | ForEach-Object { Write-Output \$_.Line }\"" 2>&1
kill $SSH_PID 2>/dev/null || true
echo done
