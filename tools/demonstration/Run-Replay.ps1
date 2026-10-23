[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$EpisodeCheckPath, [string]$Python='', [string]$CacheDirectory='', [switch]$Inspect)
# Compatibility entry. The package-root command is the documented entry.
$ErrorActionPreference='Stop'
$v02PackageRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
& (Join-Path $v02PackageRoot 'Run-Replay.ps1') @PSBoundParameters