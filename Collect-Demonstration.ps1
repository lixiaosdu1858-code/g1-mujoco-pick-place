[CmdletBinding()]
param([string]$Python='', [string]$CacheDirectory='', [switch]$Inspect)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'tools\Enable-PackageEnvironment.ps1') -PackageRoot $PSScriptRoot -Python $Python -CacheDirectory $CacheDirectory
$g1Kind=if ($Inspect) { 'demonstration_inspect' } else { 'demonstration_user' }
$g1RunOutput=New-G1OutputPath $g1Kind
$g1Arguments=@('-B',(Join-Path $g1PackageRoot 'tools\demonstration\collect_episode.py'),'--output',$g1RunOutput)
if ($Inspect) { $g1Arguments+='--inspect' }
Write-Output ('PYTHON: '+$g1SelectedPython)
Write-Output ('CACHE: '+$g1Cache)
Write-Output ('DEMONSTRATION RUN: '+$g1RunOutput)
if ($Inspect) { Write-Output 'Static inspection only. No viewer or physics steps.' }
else { Write-Output '7 start; HOLD directions; 6 close/open; 8 camera; 9 save. Full episode logging is automatic.' }
& $g1SelectedPython @g1Arguments
$g1Exit=$LASTEXITCODE
if ($g1Exit -ne 0) { Write-Warning ('Collection failed with exit='+$g1Exit+'. Preserve the run; no automatic retry.'); exit $g1Exit }
$g1Summary=Join-Path $g1RunOutput 'summary.json'
if (Test-Path -LiteralPath $g1Summary -PathType Leaf) {
    Get-Content -LiteralPath $g1Summary -Raw -Encoding UTF8 | ConvertFrom-Json |
        Select-Object status,end_reason,steps,simulation_s,physics_loop_wall_s,manual_task_completed,placement_passed,tracking_thresholds_passed | Format-List
}
$g1Index=Join-Path $g1RunOutput 'episode\episode-index.json'
if (Test-Path -LiteralPath $g1Index -PathType Leaf) {
    Write-Output ('EPISODE CHECK: '+(Split-Path $g1Index -Parent))
    Get-Content -LiteralPath $g1Index -Raw -Encoding UTF8 | ConvertFrom-Json |
        Select-Object result,complete_v02_episode,state_count,action_count | Format-List
}
Write-Output ('DEMONSTRATION RUN: '+$g1RunOutput)
