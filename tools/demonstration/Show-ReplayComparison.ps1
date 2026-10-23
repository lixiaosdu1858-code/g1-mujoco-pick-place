[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RunPath,
    [string]$SourceRun,
    [string]$OutputDirectory,
    [string]$Python='',
    [string]$CacheDirectory='',
    [switch]$NoShow,
    [switch]$ToolCheck
)
# Saved-file comparison/plotting only. No MuJoCo import or physics stepping.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) { throw 'Use Windows PowerShell 5.1 (powershell.exe).' }
$rcRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $rcRoot 'tools\Enable-PackageEnvironment.ps1') -PackageRoot $rcRoot -Python $Python -CacheDirectory $CacheDirectory
$rcPython=$g1SelectedPython
$rcArgs=@('-B',(Join-Path $PSScriptRoot 'compare_replay.py'),'--replay-run',$RunPath)
if ($SourceRun) { $rcArgs+=@('--source-run',$SourceRun) }
if ($OutputDirectory) { $rcArgs+=@('--output-dir',$OutputDirectory) }
if ($ToolCheck) { $rcArgs+='--tool-check' }
Write-Output ('PYTHON: '+$rcPython)
Write-Output 'Read saved records only. No simulation or installation. Outcome flags are not changed.'
$rcStdout=@(& $rcPython @rcArgs)
if ($LASTEXITCODE -ne 0) { throw ('Comparison validation failed with exit='+$LASTEXITCODE+'; preserve output.') }
if ($rcStdout.Count -ne 1) { throw 'Comparison did not return exactly one receipt; preserve output.' }
$rcReceipt=([string]$rcStdout[0]) | ConvertFrom-Json
if ($rcReceipt.kind -ne 'replay_comparison_receipt' -or $rcReceipt.protocol -ne 1) { throw 'Invalid comparison receipt.' }
$rcPayloadPath=[string]$rcReceipt.plotdata_json
if ((Get-FileHash -LiteralPath $rcPayloadPath -Algorithm SHA256).Hash -ne $rcReceipt.plotdata_sha256) { throw 'Plot payload digest mismatch.' }
$rcPayload=Get-Content -LiteralPath $rcPayloadPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($rcPayload.kind -ne 'replay_comparison_plot_payload') { throw 'Invalid comparison payload.' }
$rcRows=@($rcPayload.rows)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$rcChart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$rcForm=$null
$rcAreas=@{}
$rcBlue=[Drawing.Color]::RoyalBlue
$rcOrange=[Drawing.Color]::DarkOrange
$rcPurple=[Drawing.Color]::DarkViolet
$rcGreen=[Drawing.Color]::SeaGreen
function Add-RCTitle([string]$Text,[double]$Y,[bool]$Bold=$false) {
    $rcTitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $rcTitle.Text=$Text
    $rcTitle.Font=New-Object Drawing.Font('Segoe UI',10)
    if ($Bold) { $rcTitle.Font=New-Object Drawing.Font('Segoe UI',14,[Drawing.FontStyle]::Bold) }
    $rcTitle.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(0,$Y,100,2)
    [void]$rcChart.Titles.Add($rcTitle)
}
function Add-RCSeries([string]$Area,[string]$Name,[string]$Field,[Drawing.Color]$Color,[bool]$Dashed=$false) {
    $rcSeries=New-Object System.Windows.Forms.DataVisualization.Charting.Series ($Area+' '+$Name)
    $rcSeries.ChartArea=$Area;$rcSeries.Legend=$Area;$rcSeries.LegendText=$Name
    $rcSeries.ChartType='Line';$rcSeries.Color=$Color;$rcSeries.BorderWidth=2
    if ($Dashed) { $rcSeries.BorderDashStyle='Dash' }
    foreach ($rcRow in $rcRows) {
        if ($null -ne $rcRow.$Field) { [void]$rcSeries.Points.AddXY([double]$rcRow.sim_s,[double]$rcRow.$Field) }
    }
    if ($rcSeries.Points.Count -eq 1) { $rcSeries.ChartType='Point';$rcSeries.MarkerSize=6;$rcSeries.MarkerStyle='Circle' }
    [void]$rcChart.Series.Add($rcSeries)
}
function Add-RCPair([string]$Area,[string]$Field) {
    Add-RCSeries $Area 'Demonstration' ('source_'+$Field) $rcOrange $true
    Add-RCSeries $Area 'Replay' ('replay_'+$Field) $rcBlue
}
try {
    $rcChart.Size=New-Object Drawing.Size(1800,1500)
    $rcChart.BackColor=[Drawing.Color]::White
    $rcSpecs=@(
        @('ToolX','Tool X (mm)'),@('ToolY','Tool Y (mm)'),@('ToolZ','Tool Z (mm)'),
        @('CubeX','Cube X (mm)'),@('CubeY','Cube Y (mm)'),@('CubeZ','Cube Z (mm)'),
        @('Finger1','Finger 1 position (mm)'),@('Finger2','Finger 2 position (mm)'),@('Tracking','Tracking error (mm)'),
        @('Contact','Finger/cube force (N)'),@('Placement','Valid placement (s)'),@('Trajectory','Trajectory delta (mm)'))
    $rcT0=0.;$rcT1=[double]$rcPayload.source_duration_s
    if ($rcT1 -le $rcT0) { $rcT1=$rcT0+.01 }
    for ($rcI=0;$rcI -lt $rcSpecs.Count;$rcI++) {
        $rcName=[string]$rcSpecs[$rcI][0]
        $rcX=1+33*($rcI%3);$rcY=11+21*[Math]::Floor($rcI/3)
        $rcArea=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea $rcName
        $rcArea.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($rcX,$rcY,32,17)
        $rcArea.InnerPlotPosition=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(18,5,77,63)
        $rcArea.AxisX.Title='Simulation time (s)';$rcArea.AxisY.Title=[string]$rcSpecs[$rcI][1]
        $rcArea.AxisX.Minimum=$rcT0;$rcArea.AxisX.Maximum=$rcT1
        $rcArea.AxisY.IsStartedFromZero=$false
        foreach ($rcAxis in @($rcArea.AxisX,$rcArea.AxisY)) {
            $rcAxis.TitleFont=New-Object Drawing.Font('Segoe UI',10)
            $rcAxis.LabelStyle.Font=New-Object Drawing.Font('Segoe UI',9)
            $rcAxis.LabelStyle.Format='0.###';$rcAxis.MajorGrid.LineColor=[Drawing.Color]::Gainsboro
        }
        [void]$rcChart.ChartAreas.Add($rcArea);$rcAreas[$rcName]=$rcArea
        $rcLegend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend $rcName
        $rcLegend.Font=New-Object Drawing.Font('Segoe UI',9)
        $rcLegend.LegendStyle='Row';$rcLegend.IsTextAutoFit=$false
        $rcLegend.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($rcX,($rcY+18.5),32,2)
        [void]$rcChart.Legends.Add($rcLegend)
    }
    foreach ($rcAxisName in @('x','y','z')) {
        Add-RCPair ('Tool'+$rcAxisName.ToUpperInvariant()) ('actual_'+$rcAxisName+'_mm')
        Add-RCPair ('Cube'+$rcAxisName.ToUpperInvariant()) ('cube_'+$rcAxisName+'_mm')
    }
    Add-RCPair 'Finger1' 'finger1_q_mm'
    Add-RCPair 'Finger2' 'finger2_q_mm'
    Add-RCPair 'Tracking' 'position_error_mm'
    Add-RCPair 'Placement' 'placement_continuous_s'
    Add-RCSeries 'Contact' 'Demo F1' 'source_finger1_normal_n' $rcOrange $true
    Add-RCSeries 'Contact' 'Replay F1' 'replay_finger1_normal_n' $rcBlue
    Add-RCSeries 'Contact' 'Demo F2' 'source_finger2_normal_n' $rcPurple $true
    Add-RCSeries 'Contact' 'Replay F2' 'replay_finger2_normal_n' $rcGreen
    Add-RCSeries 'Trajectory' 'Tool' 'actual_trajectory_delta_mm' $rcBlue
    Add-RCSeries 'Trajectory' 'Cube' 'cube_trajectory_delta_mm' $rcPurple
    $rcHeading='G1 saved demonstration vs dynamic replay | comparison only'
    if ($rcPayload.tool_check) { $rcHeading='TOOL CHECK ONLY | NO NEW DYNAMIC REPLAY' }
    Add-RCTitle $rcHeading .4 $true
    Add-RCTitle ('Source: {0}; replay: {1} / {2}; source timeline covered: {3}.' -f $rcPayload.source_status,$rcPayload.replay_status,$rcPayload.replay_end_reason,$rcPayload.full_coverage) 2.8
    Add-RCTitle ('Aligned samples: {0}; plotted: {1}. Statistics use all aligned steps; no interpolation, smoothing, or time warping.' -f $rcPayload.overlap_samples,$rcPayload.plotted_samples) 4.8
    Add-RCTitle 'Orange dashed = demonstration; blue solid = replay. XYZ is world-frame displacement from the SAME demonstration initial position.' 6.8
    Add-RCTitle 'Tracking panel: actual vs current reference. Trajectory panel: replay actual vs demonstration actual. These are different errors.' 8.8
    Add-RCTitle 'Only overlapping saved states are drawn; the horizontal axis retains the full demonstration duration. Empty tail is not an inferred trajectory.' 96
    Add-RCTitle 'Task flags are quoted from saved summaries. This plot does not run physics, change task success, or establish new-initial-state generalization.' 98
    $rcStream=[IO.File]::Open($rcPayload.output_png,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $rcChart.SaveImage($rcStream,[System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png) } finally { $rcStream.Dispose() }
    $rcMeta=[ordered]@{
        kind='replay_comparison_plot_source';tool_check=[bool]$rcPayload.tool_check
        comparison_json=[string]$rcReceipt.comparison_json
        comparison_sha256=(Get-FileHash -LiteralPath $rcReceipt.comparison_json -Algorithm SHA256).Hash
        plotdata_json=$rcPayloadPath;plotdata_sha256=$rcReceipt.plotdata_sha256
        renderer=$PSCommandPath;renderer_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
        output_png=[string]$rcPayload.output_png
        output_png_sha256=(Get-FileHash -LiteralPath $rcPayload.output_png -Algorithm SHA256).Hash
        new_physics_steps=0
    }
    $rcBytes=(New-Object Text.UTF8Encoding($false)).GetBytes(($rcMeta|ConvertTo-Json -Depth 12))
    $rcStream=[IO.File]::Open($rcPayload.output_source_json,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $rcStream.Write($rcBytes,0,$rcBytes.Length) } finally { $rcStream.Dispose() }
    Write-Output ('COMPARISON: '+$rcReceipt.comparison_dir)
    Write-Output ('REPORT: '+$rcReceipt.report)
    Write-Output ('PNG: '+$rcPayload.output_png)
    Write-Output ('SOURCE: '+$rcPayload.output_source_json)
    if (-not $NoShow) {
        $rcForm=New-Object Windows.Forms.Form
        $rcForm.Text='G1 demonstration/replay comparison | saved states'
        $rcForm.StartPosition='CenterScreen';$rcForm.AutoScroll=$true
        $rcScreen=[Windows.Forms.Screen]::PrimaryScreen.WorkingArea
        $rcForm.ClientSize=New-Object Drawing.Size([Math]::Min(1800,$rcScreen.Width-80),[Math]::Min(1500,$rcScreen.Height-100))
        $rcForm.Controls.Add($rcChart);[void]$rcForm.ShowDialog()
    }
} finally {
    if ($null -ne $rcForm) { $rcForm.Dispose() }
    $rcChart.Dispose()
}
