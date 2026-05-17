# Launches tahoma rank 0 (API server + first 30 K2.6 layers) on matias-02.
# Deployed to matias-02:%USERPROFILE%\start_rank0.ps1 — invoked by
# autolab/bench/start_workers.sh.

$ov = "$env:USERPROFILE\openvino\openvino_genai_windows_2026.1.0.0_x86_64"
$env:Path = "$ov\runtime\bin\intel64\Release;$ov\runtime\3rdparty\tbb\bin;$env:Path"
& "$env:USERPROFILE\tahoma\target\release\tahoma.exe" worker `
    --engine sparse-moe `
    --rank 0 --total 2 `
    --device CPU `
    --model "$env:USERPROFILE\kimi-k26-model" `
    --listen :9100 `
    --next 100.123.40.123:9100 `
    --api :8000 `
    --max-tokens 8 *>&1 | Out-File -FilePath "$env:USERPROFILE\tahoma-rank0.log" -Encoding utf8
