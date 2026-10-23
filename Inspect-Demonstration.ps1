[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunPath, [string]$Python='', [string]$CacheDirectory='')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'tools\Enable-PackageEnvironment.ps1') -PackageRoot $PSScriptRoot -Python $Python -CacheDirectory $CacheDirectory
$g1RunPath=Get-G1OutputInput $RunPath
$g1CheckOutput=New-G1OutputPath 'episode_check'
Write-Output ('PYTHON: '+$g1SelectedPython)
Write-Output 'Read saved complete episodes only. No simulation, installation or download.'
& $g1SelectedPython -B (Join-Path $g1PackageRoot 'tools\demonstration\check_episode.py') --run $g1RunPath --output $g1CheckOutput
$g1Exit=$LASTEXITCODE
if ($g1Exit -ne 0) { Write-Warning ('Episode check failed with exit='+$g1Exit+'. Preserve the output.'); exit $g1Exit }
Write-Output ('EPISODE CHECK: '+$g1CheckOutput)
