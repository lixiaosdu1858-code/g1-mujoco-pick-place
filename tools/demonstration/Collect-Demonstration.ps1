[CmdletBinding()]
param([string]$Python='', [string]$CacheDirectory='', [switch]$Inspect)
# Compatibility entry. The package-root command is the documented entry.
$ErrorActionPreference='Stop'
$v02PackageRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
& (Join-Path $v02PackageRoot 'Collect-Demonstration.ps1') @PSBoundParameters