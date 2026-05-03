#!/bin/bash
# Quick test: swap dist_spec's draft from FastDraft 150M to Llama 3.2 1B INT4.
# If accept rate jumps, the math changes. Even if it's slower per-token,
# higher accept may net win.
# 3 trials each on factual workload, K=1.
set -e

start_charlie() {
    ssh cascadia@charlie.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' >/dev/null 2>&1
    ssh cascadia@charlie.local 'powershell -Command "$env:Path=\"C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;\" + $env:Path; cd C:\Users\cascadia\tahoma-rust; .\target\release\tahoma.exe worker --rank 1 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --listen 10.10.10.2:9100 *>C:\Users\cascadia\q3-worker.log"' >/dev/null 2>&1 &
    SSH_PID=$!
    sleep 30
    echo $SSH_PID
}

echo "=== Llama 3.2 1B INT4 as draft, K=1 factual ==="
for i in 1 2 3; do
    SSH_PID=$(start_charlie)
    ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --draft-model C:\cascadia\models\srang992-llama-3.2-1b-int4-ov --next 10.10.10.2:9100 --spec-k 1 --max-tokens 256 < C:\Users\cascadia\e7-factual-prompt.txt > C:\Users\cascadia\q3-1b-trial-$i.log 2>&1'; (Select-String -Path C:\Users\cascadia\q3-1b-trial-$i.log -Pattern 'task active').Line; (Select-String -Path C:\Users\cascadia\q3-1b-trial-$i.log -Pattern 'ov-dist-spec done').Line; (Select-String -Path C:\Users\cascadia\q3-1b-trial-$i.log -Pattern 'spec_decode timing').Line\""
    kill $SSH_PID 2>/dev/null || true
done
echo done
