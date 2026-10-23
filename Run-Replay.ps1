[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$EpisodeCheckPath,
    [string]$Python='', [string]$CacheDirectory='', [switch]$Inspect
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'tools\Enable-PackageEnvironment.ps1') -PackageRoot $PSScriptRoot -Python $Python -CacheDirectory $CacheDirectory
$g1EpisodePath=Get-G1OutputInput $EpisodeCheckPath
$g1EpisodeIndex=Join-Path $g1EpisodePath 'episode-index.json'
if (-not (Test-Path -LiteralPath $g1EpisodeIndex -PathType Leaf)) { throw 'Episode index missing. Pass the directory containing episode-index.json with -EpisodeCheckPath.' }
$g1Kind=if ($Inspect) { 'replay_inspect' } else { 'replay_user' }
$g1RunOutput=New-G1OutputPath $g1Kind
$g1Arguments=@('-B',(Join-Path $g1PackageRoot 'tools\demonstration\replay_episode.py'),'--episode-index',$g1EpisodeIndex,'--output',$g1RunOutput)
if ($Inspect) { $g1Arguments+='--inspect' }
Write-Output ('PYTHON: '+$g1SelectedPython)
Write-Output ('CACHE: '+$g1Cache)
Write-Output ('EPISODE INDEX: '+$g1EpisodeIndex)
Write-Output ('REPLAY RUN: '+$g1RunOutput)
if ($Inspect) { Write-Output 'Static inspection only. No viewer or physics steps.' }
& $g1SelectedPython @g1Arguments
$g1Exit=$LASTEXITCODE
if ($g1Exit -ne 0) { Write-Warning ('Replay failed with exit='+$g1Exit+'. Preserve the run; no automatic retry.'); exit $g1Exit }
$g1Summary=Join-Path $g1RunOutput 'summary.json'
if (Test-Path -LiteralPath $g1Summary -PathType Leaf) {
    Get-Content -LiteralPath $g1Summary -Raw -Encoding UTF8 | ConvertFrom-Json |
        Select-Object status,end_reason,steps,simulation_s,physics_loop_wall_s,sequence_completed,replay_completed,placement_passed,tracking_thresholds_passed | Format-List
}
if (-not $Inspect -and (Test-Path -LiteralPath (Join-Path $g1RunOutput 'manual.csv') -PathType Leaf)) {
    Write-Output ('Comparison command: & "'+(Join-Path $g1PackageRoot 'Show-ReplayComparison.ps1')+'" -RunPath "'+$g1RunOutput+'" -Python "'+$g1SelectedPython+'" -CacheDirectory "'+$g1Cache+'"')
}
