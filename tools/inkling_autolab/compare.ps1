param([ValidateSet('adaptive','direct','tiled','tiled2','bf16only')][string]$Arm='direct', [int]$Slot=-1, [int]$Repeat=1,
      [int]$Samples=7, [string]$Kind='layer', [switch]$UseFinal)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\devcloud\inkling-autolab'
if ($Slot -ge 0) { $Arm = @('direct','tiled','adaptive')[($Slot + $Repeat - 1) % 3] }
$variant = 'baseline'
$reads = 1
$bf16 = 1
$int4 = 1
switch ($Arm) {
    'adaptive' { $reads = 0 }
    'tiled' { $variant = 'int4-rows'; $bf16 = 2; $int4 = 4 }
    'tiled2' { $variant = 'int4-rows'; $int4 = 2 }
    'bf16only' { $variant = 'int4-rows'; $bf16 = 2 }
}
if ($UseFinal) { $variant = 'final' }
"arm=$Arm variant=$variant reads=$reads bf16_rows=$bf16 int4_rows=$int4"
Get-FileHash "$root\bin\$variant.exe" | ForEach-Object { "binary_sha256=$($_.Hash)" }
& "$root\run-bench.ps1" -Threads 16 -Mask 65535 -Serial 0 -Reads $reads -Bf16Rows $bf16 -Int4Rows $int4 -Samples $Samples -Variant $variant -Kind $Kind -ExpectedHash '4e89f0793afa1015'
exit $LASTEXITCODE
