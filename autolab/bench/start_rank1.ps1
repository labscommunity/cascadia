# Launches tahoma rank 1 (last 30 K2.6 layers + head + sampler) on matias-03.
# Deployed to matias-03:%USERPROFILE%\start_rank1.ps1 — invoked by
# autolab/bench/start_workers.sh.

$ov = "$env:USERPROFILE\openvino\openvino_genai_windows_2026.1.0.0_x86_64"
$env:Path = "$ov\runtime\bin\intel64\Release;$ov\runtime\3rdparty\tbb\bin;$env:Path"
& "$env:USERPROFILE\tahoma\target\release\tahoma.exe" worker `
    --engine sparse-moe `
    --rank 1 --total 2 `
    --device CPU `
    --model "$env:USERPROFILE\kimi-k26-model" `
    --listen :9100 `
    --max-tokens 8 *>&1 | Out-File -FilePath "$env:USERPROFILE\tahoma-rank1.log" -Encoding utf8
