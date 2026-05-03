#!/bin/bash
# e4: ov-runtime no-spec on 22/10 layer split (alpha-heavy)
# 5 trials, charlie restarted between each.
set -e

start_charlie() {
    ssh cascadia@charlie.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' >/dev/null 2>&1
    ssh cascadia@charlie.local 'powershell -Command "$env:Path=\"C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;\" + $env:Path; cd C:\Users\cascadia\tahoma-rust; .\target\release\tahoma.exe worker --rank 1 --total 2 --engine ov-runtime --device GPU --model C:\cascadia\shards_2stage_v3_22_10 --listen 10.10.10.2:9100 *>C:\Users\cascadia\e4-worker.log"' >/dev/null 2>&1 &
    SSH_PID=$!
    sleep 30
    echo $SSH_PID
}

run_alpha_trial() {
    local i=$1
    ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 2 --engine ov-runtime --device GPU --model C:\cascadia\shards_2stage_v3_22_10 --next 10.10.10.2:9100 --max-tokens 256 < C:\Users\cascadia\e0-prompt.txt > C:\Users\cascadia\e4-trial-$i.log 2>&1'; \$start = (Select-String -Path C:\Users\cascadia\e4-trial-$i.log -Pattern 'task active' | Select -First 1).Line; \$end = (Select-String -Path C:\Users\cascadia\e4-trial-$i.log -Pattern 'task done' | Select -First 1).Line; Write-Output \\\"START: \$start\\\"; Write-Output \\\"END: \$end\\\"\"" 2>&1
}

for i in 1 2 3 4 5; do
    echo "=== trial $i ==="
    SSH_PID=$(start_charlie)
    run_alpha_trial $i
    kill $SSH_PID 2>/dev/null || true
done
echo "done"
