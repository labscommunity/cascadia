#!/bin/bash
# M1 step 1: try Mixtral 8x7B INT4 single-node on alpha. Expected outcome: OOM
# (12 GB GPU mem, 12+ GB Mixtral weights + KV cache likely OOMs on B390).
# If it DOES fit, report tok/s as the new single-node baseline.
ssh cascadia@alpha.local 'powershell -Command "Get-Process tahoma -ErrorAction SilentlyContinue | Stop-Process -Force"' 2>&1 | tail -1
ssh cascadia@alpha.local "powershell -ExecutionPolicy Bypass -Command \"\$env:Path='C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\bin\intel64\Release;C:\tahoma\ov_genai_sdk\openvino_genai_windows_2026.1.0.0_x86_64\runtime\3rdparty\tbb\bin;' + \$env:Path; Set-Location C:\Users\cascadia\tahoma-rust; cmd /c '.\target\release\tahoma.exe worker --rank 0 --total 1 --engine ov-genai --device GPU --model C:\cascadia\models\mixtral-8x7b-int4-ov-fresh --max-tokens 32 < C:\Users\cascadia\e7-factual-prompt.txt > C:\Users\cascadia\m1-mixtral-single.log 2>&1'; (Select-String -Path C:\Users\cascadia\m1-mixtral-single.log -Pattern 'task done|Error|out of memory|OOM|allocate|memory').Line\""
