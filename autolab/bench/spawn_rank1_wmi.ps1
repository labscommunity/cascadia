# Detached spawn for rank-1 worker via WMI (survives SSH disconnect).
$ov  = "$env:USERPROFILE\openvino\openvino_genai_windows_2026.1.0.0_x86_64"
$exe = "$env:USERPROFILE\tahoma\target\release\tahoma.exe"
$log = "$env:USERPROFILE\tahoma-rank1.log"

Set-Content -Path $log -Value "" -ErrorAction SilentlyContinue

$pathPrefix = "$ov\runtime\bin\intel64\Release;$ov\runtime\3rdparty\tbb\bin"
$args = "worker --engine sparse-moe --rank 1 --total 2 --device CPU --model `"$env:USERPROFILE\kimi-k26-model`" --listen :9100 --max-tokens 8 --top-k-override 8"
$cmd = "cmd /c set PATH=$pathPrefix;%PATH% && `"$exe`" $args > `"$log`" 2>&1"

$result = Invoke-WmiMethod -Class Win32_Process -Name Create -ArgumentList $cmd
if ($result.ReturnValue -eq 0) {
  Write-Host ("WMI_SPAWN_OK pid=" + $result.ProcessId)
} else {
  Write-Host ("WMI_SPAWN_FAILED ReturnValue=" + $result.ReturnValue)
}
