param([Parameter(Mandatory=$true)][string]$Model,
      [Parameter(Mandatory=$true)][string]$Cases,
      [int]$Reads=0, [int]$Bf16Rows=1, [int]$Int4Rows=1,
      [int]$Tokens=64, [int]$Samples=3)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\devcloud\inkling-autolab'
if (!(Test-Path (Join-Path $Model 'manifest.json'))) {
    throw "Full Inkling checkpoint unavailable at $Model; cannot measure full-model throughput."
}
if (!(Test-Path $Cases)) { throw "Missing benchmark cases: $Cases" }
$env:RAYON_NUM_THREADS = '16'
$env:CASCADIA_INKLING_SERIAL_EXPERTS = '0'
$env:CASCADIA_INKLING_SEQ_READS = "$Reads"
$env:CASCADIA_INKLING_PIN_EXPERTS = '0'
$env:CASCADIA_BF16_GEMV_ROWS = "$Bf16Rows"
$env:CASCADIA_INT4_GEMV_ROWS = "$Int4Rows"
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]65535
$benchArgs = @('--export', "`"$Model`"", '--cases', "`"$Cases`"", '--tokens', "$Tokens", '--samples', "$Samples")
$p = Start-Process -FilePath "$root\bin\full-decode.exe" -ArgumentList $benchArgs -PassThru -NoNewWindow
try {
    $p.PriorityClass = 'High'
    $p.ProcessorAffinity = [IntPtr]65535
    "bf16_rows=$Bf16Rows int4_rows=$Int4Rows reads=$Reads"
    $p.WaitForExit()
    exit $p.ExitCode
} finally {
    if (!$p.HasExited) { $p.Kill(); $p.WaitForExit() }
    $p.Dispose()
}
