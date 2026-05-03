# e0 bench: single-node ov-genai + FastDraft K=5 on alpha
# 5 trials @ 256-tok creative prompt, capture tok/s from each.
$env:Path="C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;" + $env:Path
Set-Location C:\Users\cascadia\tahoma-rust

$prompt = "Write a long detailed creative story about a curious robot named Atlas who explores an abandoned space station orbiting Mars. Include vivid descriptions of the rusted corridors, mysterious signals from a forgotten control room, and the moment Atlas discovers an old log entry from the station's last human crew. Keep going."
$prompt | Out-File -Encoding ASCII C:\Users\cascadia\e0-prompt.txt

for ($i = 1; $i -le 5; $i++) {
    $log = "C:\Users\cascadia\e0-trial-$i.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    cmd /c ".\target\release\tahoma.exe worker --rank 0 --total 1 --engine ov-genai --device GPU --model C:\cascadia\models\llama-3.1-8b-int4 --draft-model C:\cascadia\models\fastdraft-150m-int8-ov --spec-k 5 --max-tokens 256 < C:\Users\cascadia\e0-prompt.txt > $log 2>&1"
    $sw.Stop()
    $line = Select-String -Path $log -Pattern "task done" | Select-Object -Last 1
    Write-Host "trial=$i wall_ms=$($sw.ElapsedMilliseconds) $line"
}
