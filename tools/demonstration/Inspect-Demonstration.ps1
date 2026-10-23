[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunPath, [string]$Python='', [string]$CacheDirectory='')
# Compatibility entry. The package-root command is the documented entry.
$ErrorActionPreference='Stop'
$v02PackageRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
& (Join-Path $v02PackageRoot 'Inspect-Demonstration.ps1') @PSBoundParameters