[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PackageRoot,
    [string]$Python='',
    [string]$CacheDirectory=''
)
# Dot-source this helper. It changes only the calling process environment.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) {
    throw 'Use Windows PowerShell 5.1 (powershell.exe).'
}
$g1PackageRoot=(Resolve-Path -LiteralPath $PackageRoot).ProviderPath
function Assert-G1NoReparse([string]$Path) {
    $g1CheckedPath=[IO.Path]::GetFullPath($Path)
    while ($g1CheckedPath) {
        if (Test-Path -LiteralPath $g1CheckedPath) {
            if (((Get-Item -LiteralPath $g1CheckedPath -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw ('Reparse paths are not supported: '+$g1CheckedPath)
            }
        }
        $g1CheckedPath=[IO.Path]::GetDirectoryName($g1CheckedPath)
    }
}
Assert-G1NoReparse $g1PackageRoot
if (-not $Python) { $Python=Join-Path $g1PackageRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python is missing. Provide -Python pointing to an existing isolated environment. No software is installed by this script.'
}
$g1SelectedPython=(Resolve-Path -LiteralPath $Python).ProviderPath
if (-not $CacheDirectory) { $CacheDirectory=Join-Path $g1PackageRoot '.cache' }
$g1Cache=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CacheDirectory)
Assert-G1NoReparse $g1Cache
$g1Temp=Join-Path $g1Cache 'temp'
Assert-G1NoReparse $g1Temp
New-Item -ItemType Directory -Path $g1Temp -Force | Out-Null
$g1ProbePath=Join-Path $g1Temp ('write_probe_'+[Guid]::NewGuid().ToString('N')+'.tmp')
$g1Probe=[IO.FileStream]::new($g1ProbePath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None,4096,[IO.FileOptions]::DeleteOnClose)
try { $g1Probe.WriteByte(1); $g1Probe.Flush() } finally { $g1Probe.Dispose() }
$env:G1_PACKAGE_PYTHON=$g1SelectedPython
$env:G1_PACKAGE_CACHE=$g1Cache
$env:TEMP=$g1Temp; $env:TMP=$g1Temp; $env:TMPDIR=$g1Temp
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
$env:XDG_CACHE_HOME=$g1Cache; $env:MPLCONFIGDIR=Join-Path $g1Cache 'matplotlib'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$OutputEncoding=[Console]::OutputEncoding
$g1Outputs=Join-Path $g1PackageRoot 'outputs'
Assert-G1NoReparse $g1Outputs
New-Item -ItemType Directory -Path $g1Outputs -Force | Out-Null
function Get-G1OutputInput([string]$Path) {
    $g1Input=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    Assert-G1NoReparse $g1Input
    $g1Prefix=$g1Outputs.TrimEnd('\')+'\'
    if (-not $g1Input.StartsWith($g1Prefix,[StringComparison]::OrdinalIgnoreCase)) {
        throw ('Run and episode inputs must be inside this package outputs directory: '+$g1Outputs)
    }
    if (-not (Test-Path -LiteralPath $g1Input -PathType Container)) { throw ('Directory is missing: '+$g1Input) }
    return (Resolve-Path -LiteralPath $g1Input).ProviderPath
}
function New-G1OutputPath([string]$Kind) {
    $g1Stamp=[DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('yyyyMMdd_HHmmss_fff')
    return Join-Path $g1Outputs ($Kind+'_'+$g1Stamp+'_'+[Guid]::NewGuid().ToString('N').Substring(0,6))
}
