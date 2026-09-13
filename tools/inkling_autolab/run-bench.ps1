param([int]$Threads=16, [int]$Mask=65535, [int]$Serial=0, [int]$Reads=0,
      [int]$Samples=5, [int]$Tokens=32, [string]$Kind='layer', [string]$Variant='baseline', [string]$ExpectedHash='', [string]$Priority='High', [int]$Bf16Rows=1)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\devcloud\inkling-autolab'
$exe = Join-Path $root "bin\$Variant.exe"
$env:RAYON_NUM_THREADS = "$Threads"
$env:CASCADIA_INKLING_SERIAL_EXPERTS = "$Serial"
$env:CASCADIA_INKLING_SEQ_READS = "$Reads"
$env:CASCADIA_INKLING_PIN_EXPERTS = '0'
$env:CASCADIA_BF16_GEMV_ROWS = "$Bf16Rows"
# Inherited by the child; applies affinity before Rayon starts any workers.
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]$Mask
$benchArgs = @('--expert-dir', "$root\synthetic-experts", '--samples', "$Samples", '--tokens', "$Tokens", '--kind', $Kind)
if ($ExpectedHash) { $benchArgs += @('--expect-hash', $ExpectedHash) }
$p = Start-Process -FilePath $exe -ArgumentList $benchArgs -PassThru -NoNewWindow
try {
    $p.PriorityClass = $Priority
    $p.ProcessorAffinity = [IntPtr]$Mask
    "process_priority=$($p.PriorityClass) affinity=$($p.ProcessorAffinity) bf16_rows=$Bf16Rows"
    $p.WaitForExit()
    exit $p.ExitCode
} finally {
    if (!$p.HasExited) { $p.Kill(); $p.WaitForExit() }
    $p.Dispose()
}
