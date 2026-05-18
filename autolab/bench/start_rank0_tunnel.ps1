# Launches tahoma rank 0 (API server + first 30 K2.6 layers) on matias-02.
# Tunnel-aware variant for when Tailscale is logged out and the boxes can
# only reach each other via SSH tunnels chained through the controller Mac.
#
# --next points at the matias-02-local SSH reverse-forward port (9100),
# which the Mac forwards on to matias-03:9100. From rank-0's POV it just
# connects to 127.0.0.1:9100.
#
# Deployed to matias-02:%USERPROFILE%\start_rank0_tunnel.ps1 — invoked by
# autolab/bench/start_workers_tunnel.sh.

$ov = "$env:USERPROFILE\openvino\openvino_genai_windows_2026.1.0.0_x86_64"
$env:Path = "$ov\runtime\bin\intel64\Release;$ov\runtime\3rdparty\tbb\bin;$env:Path"
& "$env:USERPROFILE\tahoma\target\release\tahoma.exe" worker `
    --engine sparse-moe `
    --rank 0 --total 2 `
    --device CPU `
    --model "$env:USERPROFILE\kimi-k26-model" `
    --listen :9101 `
    --next 127.0.0.1:9100 `
    --api :8000 `
    --max-tokens 8 `
    --top-k-override $(if ($env:TAHOMA_TOPK) { $env:TAHOMA_TOPK } else { 8 }) *>&1 | Out-File -FilePath "$env:USERPROFILE\tahoma-rank0.log" -Encoding utf8
