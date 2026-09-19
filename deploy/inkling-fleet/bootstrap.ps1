# Inkling fleet - Windows box installed over the LAN (the SSD is ext4; Windows cannot read it).
# serve.py on the Ubuntu box that holds the SSD prints the two lines to type here. This script
# pulls the Windows half of the kit into -Kit, then runs the normal installer, which pulls this
# rank's slice of the model (about 44 GB, a few minutes on 2.5 GbE) straight into the prefix.
param(
  [Parameter(Mandatory=$true)][int]$Rank,
  [Parameter(Mandatory=$true)][string]$Server,
  [string]$Kit = 'C:\inkling-kit',
  [string]$Prefix = 'C:\cascadia-inkling',
  [switch]$CpuOnly,
  [switch]$NoNet
)
$ErrorActionPreference = 'Stop'
function Log($m) { Write-Host "[bootstrap $(Get-Date -Format T)] $m" }
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw "run as administrator" }
$Server = $Server.TrimEnd('/')
$list = & curl.exe --fail -s "$Server/filelist.txt"
if ($LASTEXITCODE -ne 0 -or -not $list) { throw "cannot read $Server/filelist.txt (is serve.py running on the box with the SSD, and is this the address on the fleet's switch?)" }
# the Windows half of the kit: scripts, Windows binaries and runtimes, Windows and pure-Python wheels, tools
$want = '^inkling-deploy/([^/]+\.(ps1|env|py|md|txt)|fleet/[^/]+\.ps1|bin/BUILD\.txt|bin/windows/.+|runtime/[^/]*(windows|embed-amd64|vc_redist)[^/]*|wheels/[^/]*(win_amd64|none-any)[^/]*|tools/.+)$'
$files = @{}
foreach ($line in $list) { $p = $line -split "`t", 2; if ($p.Count -eq 2 -and $p[1] -match $want) { $files[$p[1]] = [int64]$p[0] } }
if (-not $files.ContainsKey('inkling-deploy/install.ps1')) { throw "the server's file list has no inkling-deploy/install.ps1" }
$todo = $files.Keys | Where-Object { $dst = Join-Path $Kit $_.Replace('/', '\'); -not ((Test-Path $dst) -and ((Get-Item $dst).Length -eq $files[$_])) } | Sort-Object
Log ("kit: {0} files, {1:n0} MB, {2} to fetch -> {3}" -f $files.Count, (($files.Values | Measure-Object -Sum).Sum / 1MB), @($todo).Count, $Kit)
if ($todo) {
  New-Item -ItemType Directory -Force -Path $Kit | Out-Null
  $cfg = Join-Path $Kit 'kit.curl'
  $todo | ForEach-Object { 'url = "{0}/{1}"' -f $Server, [uri]::EscapeUriString($_); 'output = "{0}"' -f (Join-Path $Kit $_).Replace('\', '/') } | Set-Content -Encoding ASCII $cfg
  & curl.exe --fail --silent --show-error --retry 3 --parallel --parallel-max 8 --create-dirs -K $cfg
  if ($LASTEXITCODE -ne 0) { throw "kit download failed (curl exit $LASTEXITCODE)" }
}
$bad = $files.Keys | Where-Object { $dst = Join-Path $Kit $_.Replace('/', '\'); -not ((Test-Path $dst) -and ((Get-Item $dst).Length -eq $files[$_])) }
if ($bad) { throw "kit incomplete: $(@($bad).Count) files missing or short, e.g. $(@($bad)[0])" }
$args2 = @{ Rank = $Rank; Prefix = $Prefix; ModelSource = "$Server/inkling/out" }
if ($CpuOnly) { $args2.CpuOnly = $true }
if ($NoNet) { $args2.NoNet = $true }
Log "running the installer from $Kit\inkling-deploy"
& "$Kit\inkling-deploy\install.ps1" @args2
