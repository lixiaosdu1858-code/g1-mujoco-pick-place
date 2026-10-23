[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RunPath,
    [string]$SourceRun, [string]$OutputDirectory,
    [string]$Python='', [string]$CacheDirectory='',
    [switch]$NoShow, [switch]$ToolCheck
)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'tools\demonstration\Show-OrientationComparison.ps1') @PSBoundParameters
