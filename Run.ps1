[CmdletBinding()]
param([string]$Python='', [string]$CacheDirectory='', [switch]$Inspect)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) { throw 'Use Windows PowerShell 5.1 (powershell.exe).' }
$g1PackageRoot=$PSScriptRoot
if (-not $Python) { $Python=Join-Path $g1PackageRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python is missing. Provide -Python pointing to the isolated environment. This script does not install software.' }
$g1SelectedPython=(Resolve-Path -LiteralPath $Python).Path
if (-not $CacheDirectory) { $CacheDirectory=Join-Path $g1PackageRoot '.cache' }
$g1Cache=[IO.Path]::GetFullPath($CacheDirectory)
$g1Temp=Join-Path $g1Cache 'temp'
New-Item -ItemType Directory -Path $g1Temp -Force | Out-Null
$env:G1_PACKAGE_PYTHON=$g1SelectedPython
$env:G1_PACKAGE_CACHE=$g1Cache
$env:TEMP=$g1Temp;$env:TMP=$g1Temp;$env:TMPDIR=$g1Temp
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8'
$env:XDG_CACHE_HOME=$g1Cache;$env:MPLCONFIGDIR=Join-Path $g1Cache 'matplotlib'
$g1Outputs=Join-Path $g1PackageRoot 'outputs'
New-Item -ItemType Directory -Path $g1Outputs -Force | Out-Null
if ($Inspect) {
    & $g1SelectedPython -B (Join-Path $g1PackageRoot 'tools\verify_runtime.py')
    if ($LASTEXITCODE -ne 0) { throw 'Static package inspection failed. No physics or viewer started.' }
    return
}
Write-Output '7 start; HOLD arrows for world X/Y, PageUp/PageDown for Z; 6 close/open; 8 camera; 9 save/end.'
Write-Output 'Manual offline simulation. No fixed key sequence. Task observations do not lock the gripper.'
& $g1SelectedPython -B (Join-Path $g1PackageRoot 'runtime\manual_pickplace.py')
$g1Code=$LASTEXITCODE
if ($g1Code -ne 0) { Write-Warning ('Run ended with exit='+$g1Code+'. Preserve the printed MANUAL RUN folder; no automatic retry.') }
exit $g1Code
