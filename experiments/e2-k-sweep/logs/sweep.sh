#!/bin/bash
# e2: K-sweep on ov-dist-spec + FastDraft, creative workload, alpha+charlie/TB4
# 3 trials per K. K=3 already measured in e1; included here for parity check.
set -e

start_charlie() {
    ssh cascadia@charlie.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' >/dev/null 2>&1
    ssh cascadia@charlie.local 'powershell -Command "$env:Path=\"C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;\" + $env:Path; cd C:\Users\cascadia\tahoma-rust; .\target\release\tahoma.exe worker --rank 1 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --listen 10.10.10.2:9100 *>C:\Users\cascadia\e2-worker.log"' >/dev/null 2>&1 &
    SSH_PID=$!
    sleep 30
    echo $SSH_PID
}

run_alpha_trial() {
    local K=$1
    local i=$2
    ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --draft-model C:\cascadia\models\fastdraft-150m-int8-ov --next 10.10.10.2:9100 --spec-k $K --max-tokens 256 < C:\Users\cascadia\e0-prompt.txt > C:\Users\cascadia\e2-K${K}-trial-$i.log 2>&1'; \$start = (Select-String -Path C:\Users\cascadia\e2-K${K}-trial-$i.log -Pattern 'task active' | Select -First 1).Line; \$end = (Select-String -Path C:\Users\cascadia\e2-K${K}-trial-$i.log -Pattern 'ov-dist-spec done' | Select -First 1).Line; Write-Output \\\"START: \$start\\\"; Write-Output \\\"END: \$end\\\"\"" 2>&1
}

for K in 1 2 4 5 6; do
    for i in 1 2 3; do
        echo "=== K=$K trial=$i ==="
        SSH_PID=$(start_charlie)
        run_alpha_trial $K $i
        kill $SSH_PID 2>/dev/null || true
    done
done
echo "done"
