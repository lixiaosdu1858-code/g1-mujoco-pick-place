[CmdletBinding()]
param([string]$Python='', [string]$CacheDirectory='')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'tools\Enable-PackageEnvironment.ps1') -PackageRoot $PSScriptRoot -Python $Python -CacheDirectory $CacheDirectory
$g1ExampleOutput=New-G1OutputPath 'example'
Write-Output ('PYTHON: '+$g1SelectedPython)
Write-Output ('EXAMPLE OUTPUT: '+$g1ExampleOutput)
Write-Output 'Materialize and validate the saved example. No viewer, physics steps, installation or download.'
& $g1SelectedPython -B (Join-Path $g1PackageRoot 'tools\prepare_example.py') --output $g1ExampleOutput
$g1Exit=$LASTEXITCODE
if ($g1Exit -ne 0) { Write-Warning ('Example preparation failed with exit='+$g1Exit+'. Preserve the output.'); exit $g1Exit }
$g1ExampleIndex=Join-Path $g1ExampleOutput 'demonstration\episode\episode-index.json'
$g1SavedReplay=Join-Path $g1ExampleOutput 'saved-replay'
if (-not (Test-Path -LiteralPath $g1ExampleIndex -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $g1SavedReplay 'summary.json') -PathType Leaf)) {
    throw 'Example preparation returned success without the expected episode index and saved replay. Preserve its output.'
}
Write-Output ('EPISODE CHECK: '+(Split-Path $g1ExampleIndex -Parent))
Write-Output ('SAVED REPLAY: '+$g1SavedReplay)
