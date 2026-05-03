#!/bin/bash
# e1 distributed bench: alpha (driver) <-> charlie (worker, listens on 10.10.10.2:9100)
# Charlie's worker only handles one connection per restart, so we kill+respawn between trials.
# Engine-internal tok_s is steady-state (excludes cold start).
set -e

start_charlie() {
    ssh cascadia@charlie.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 2"' >/dev/null 2>&1
    ssh cascadia@charlie.local 'powershell -Command "$env:Path=\"C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;\" + $env:Path; cd C:\Users\cascadia\tahoma-rust; .\target\release\tahoma.exe worker --rank 1 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --listen 10.10.10.2:9100 *>C:\Users\cascadia\e1-worker-trial.log"' >/dev/null 2>&1 &
    echo $!  # ssh pid; tahoma keeps running until killed via Stop-Process
    sleep 30  # wait for bind + warmup
}

run_alpha_trial() {
    local i=$1
    ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 2 --engine ov-dist-spec --device GPU --model C:\cascadia\shards_2stage_v5_beam --draft-model C:\cascadia\models\fastdraft-150m-int8-ov --next 10.10.10.2:9100 --spec-k 3 --max-tokens 256 < C:\Users\cascadia\e0-prompt.txt > C:\Users\cascadia\e1-trial-$i.log 2>&1'; Select-String -Path C:\Users\cascadia\e1-trial-$i.log -Pattern 'ov-dist-spec done' | Select-Object -Last 1\"" 2>&1
}

for i in 1 2 3 4 5; do
    echo "--- trial $i: starting charlie worker ---"
    SSH_PID=$(start_charlie)
    echo "--- trial $i: running alpha driver ---"
    run_alpha_trial $i
    kill $SSH_PID 2>/dev/null || true
done

echo "done"
