#!/bin/bash
set -e

echo "=== Single-node ov-genai + FastDraft K=5 (factual workload) ==="
ssh cascadia@alpha.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force"' >/dev/null 2>&1
for i in 1 2 3 4 5; do
    ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 1 --engine ov-genai --device GPU --model C:\cascadia\models\llama-3.1-8b-int4 --draft-model C:\cascadia\models\fastdraft-150m-int8-ov --spec-k 5 --max-tokens 256 < C:\Users\cascadia\e7-factual-prompt.txt > C:\Users\cascadia\e7-factual-genai-$i.log 2>&1'; (Select-String -Path C:\Users\cascadia\e7-factual-genai-$i.log -Pattern 'task done').Line\""
done

echo ""
echo "=== Distributed ov-dist-spec K=3 + FastDraft (factual workload) ==="
start_charlie() {
    ssh cascadia@charlie.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' >/dev/null 2>&1
    ssh cascadia@charlie.local 'powershell -Command "$env:Path=\"C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;\" + $env:Path; cd C:\Users\cascadia\tahoma-rust; .\target\release\tahoma.exe worker --rank 1 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --listen 10.10.10.2:9100 *>C:\Users\cascadia\e7-worker.log"' >/dev/null 2>&1 &
    SSH_PID=$!
    sleep 30
    echo $SSH_PID
}
for i in 1 2 3; do
    SSH_PID=$(start_charlie)
    ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --draft-model C:\cascadia\models\fastdraft-150m-int8-ov --next 10.10.10.2:9100 --spec-k 3 --max-tokens 256 < C:\Users\cascadia\e7-factual-prompt.txt > C:\Users\cascadia\e7-factual-dist-$i.log 2>&1'; \$start = (Select-String -Path C:\Users\cascadia\e7-factual-dist-$i.log -Pattern 'task active').Line; \$end = (Select-String -Path C:\Users\cascadia\e7-factual-dist-$i.log -Pattern 'ov-dist-spec done').Line; Write-Output \\\"trial $i\\\"; Write-Output \\\"  \$start\\\"; Write-Output \\\"  \$end\\\"\""
    kill $SSH_PID 2>/dev/null || true
done
echo done
