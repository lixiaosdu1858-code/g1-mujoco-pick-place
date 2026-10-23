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
# Saved-state orientation comparison only. No MuJoCo import or physics steps.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) { throw 'Use Windows PowerShell 5.1 (powershell.exe).' }
$ocRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $ocRoot 'tools\Enable-PackageEnvironment.ps1') -PackageRoot $ocRoot -Python $Python -CacheDirectory $CacheDirectory
$ocPython=$g1SelectedPython
$ocArgs=@('-B',(Join-Path $PSScriptRoot 'compare_orientation.py'),'--replay-run',$RunPath)
if ($SourceRun) { $ocArgs+=@('--source-run',$SourceRun) }
if ($OutputDirectory) { $ocArgs+=@('--output-dir',$OutputDirectory) }
if ($ToolCheck) { $ocArgs+='--tool-check' }
Write-Output ('PYTHON: '+$ocPython)
Write-Output 'Read saved orientations only. No simulation or installation. Outcome flags are not changed.'
$ocStdout=@(& $ocPython @ocArgs)
if ($LASTEXITCODE -ne 0) { throw ('Orientation comparison validation failed with exit='+$LASTEXITCODE+'; preserve output.') }
if ($ocStdout.Count -ne 1) { throw 'Orientation comparison did not return exactly one receipt; preserve output.' }
$ocReceipt=([string]$ocStdout[0]) | ConvertFrom-Json
if ($ocReceipt.kind -ne 'orientation_comparison_receipt' -or $ocReceipt.protocol -ne 1) { throw 'Invalid orientation comparison receipt.' }
$ocPayloadPath=[string]$ocReceipt.plotdata_json
if ((Get-FileHash -LiteralPath $ocPayloadPath -Algorithm SHA256).Hash -ne $ocReceipt.plotdata_sha256) { throw 'Plot payload digest mismatch.' }
$ocPayload=Get-Content -LiteralPath $ocPayloadPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($ocPayload.kind -ne 'orientation_comparison_plot_payload') { throw 'Invalid orientation comparison payload.' }
$ocRows=@($ocPayload.rows)
if ($ocRows.Count -eq 0) { throw 'Orientation comparison contains no plotted samples.' }
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$ocChart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$ocForm=$null
$ocBlue=[Drawing.Color]::RoyalBlue
$ocOrange=[Drawing.Color]::DarkOrange
function Add-OCTitle([string]$Text,[double]$Y,[bool]$Bold=$false) {
    $ocTitle=New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $ocTitle.Text=$Text
    $ocTitle.Font=New-Object Drawing.Font('Segoe UI',10)
    if ($Bold) { $ocTitle.Font=New-Object Drawing.Font('Segoe UI',14,[Drawing.FontStyle]::Bold) }
    $ocTitle.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(0,$Y,100,2)
    [void]$ocChart.Titles.Add($ocTitle)
}
function New-OCSegment([string]$Area,[string]$Name,[Drawing.Color]$Color,[bool]$Dashed,[int]$Index) {
    $ocSegment=New-Object System.Windows.Forms.DataVisualization.Charting.Series ($Area+' '+$Name+' '+$Index)
    $ocSegment.ChartArea=$Area;$ocSegment.Legend=$Area;$ocSegment.LegendText=$Name
    $ocSegment.ChartType='Line';$ocSegment.Color=$Color;$ocSegment.BorderWidth=2
    $ocSegment.IsVisibleInLegend=($Index -eq 0)
    if ($Dashed) { $ocSegment.BorderDashStyle='Dash' }
    return $ocSegment
}
function Add-OCFinishedSegment($Segment) {
    if ($Segment.Points.Count -eq 1) { $Segment.ChartType='Point';$Segment.MarkerSize=5;$Segment.MarkerStyle='Circle' }
    if ($Segment.Points.Count -gt 0) { [void]$ocChart.Series.Add($Segment) }
    else { $Segment.Dispose() }
}
function Add-OCSeries([string]$Area,[string]$Name,[string]$Field,[Drawing.Color]$Color,[bool]$Dashed=$false,[string]$BreakField='') {
    $ocSegmentIndex=0
    $ocSeries=New-OCSegment $Area $Name $Color $Dashed $ocSegmentIndex
    foreach ($ocRow in $ocRows) {
        $ocBreak=($BreakField -and [bool]$ocRow.$BreakField)
        # Separate series guarantee no connection across a branch jump or missing value.
        # The first value after a break remains in the new segment.
        if (($ocBreak -or $null -eq $ocRow.$Field) -and $ocSeries.Points.Count -gt 0) {
            Add-OCFinishedSegment $ocSeries
            $ocSegmentIndex++
            $ocSeries=New-OCSegment $Area $Name $Color $Dashed $ocSegmentIndex
        }
        if ($null -ne $ocRow.$Field) { [void]$ocSeries.Points.AddXY([double]$ocRow.sim_s,[double]$ocRow.$Field) }
    }
    Add-OCFinishedSegment $ocSeries
}
function Add-OCPair([string]$Area,[string]$Field,[string]$BreakPrefix='') {
    $ocSourceBreak='';$ocReplayBreak=''
    if ($BreakPrefix) { $ocSourceBreak='source_'+$BreakPrefix+'_break';$ocReplayBreak='replay_'+$BreakPrefix+'_break' }
    Add-OCSeries $Area 'Demonstration' ('source_'+$Field) $ocOrange $true $ocSourceBreak
    Add-OCSeries $Area 'Replay' ('replay_'+$Field) $ocBlue $false $ocReplayBreak
}
try {
    $ocChart.Size=New-Object Drawing.Size(1800,1250)
    $ocChart.BackColor=[Drawing.Color]::White
    $ocSpecs=@(
        @('ToolX','Tool rotation X (deg)'),@('ToolY','Tool rotation Y (deg)'),@('ToolZ','Tool rotation Z (deg)'),
        @('CubeX','Cube rotation X (deg)'),@('CubeY','Cube rotation Y (deg)'),@('CubeZ','Cube rotation Z (deg)'),
        @('ToolDelta','Tool delta angle (deg)'),@('CubeDelta','Cube delta angle (deg)'),@('Tracking','Tool tracking angle (deg)'))
    $ocT0=0.;$ocT1=[double]$ocPayload.source_duration_s
    if ($ocT1 -le $ocT0) { $ocT1=$ocT0+.01 }
    for ($ocI=0;$ocI -lt $ocSpecs.Count;$ocI++) {
        $ocName=[string]$ocSpecs[$ocI][0]
        $ocX=1+33*($ocI%3);$ocY=12+28*[Math]::Floor($ocI/3)
        $ocArea=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea $ocName
        $ocArea.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($ocX,$ocY,32,22)
        $ocArea.InnerPlotPosition=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(18,5,77,65)
        $ocArea.AxisX.Title='Simulation time (s)';$ocArea.AxisY.Title=[string]$ocSpecs[$ocI][1]
        $ocArea.AxisX.Minimum=$ocT0;$ocArea.AxisX.Maximum=$ocT1
        $ocArea.AxisY.IsStartedFromZero=$false
        foreach ($ocAxis in @($ocArea.AxisX,$ocArea.AxisY)) {
            $ocAxis.TitleFont=New-Object Drawing.Font('Segoe UI',10)
            $ocAxis.LabelStyle.Font=New-Object Drawing.Font('Segoe UI',9)
            $ocAxis.LabelStyle.Format='0.###';$ocAxis.MajorGrid.LineColor=[Drawing.Color]::Gainsboro
        }
        [void]$ocChart.ChartAreas.Add($ocArea)
        $ocLegend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend $ocName
        $ocLegend.Font=New-Object Drawing.Font('Segoe UI',9)
        $ocLegend.LegendStyle='Row';$ocLegend.IsTextAutoFit=$false
        $ocLegend.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($ocX,($ocY+24),32,2)
        [void]$ocChart.Legends.Add($ocLegend)
    }
    foreach ($ocAxisName in @('x','y','z')) {
        Add-OCPair ('Tool'+$ocAxisName.ToUpperInvariant()) ('actual_rotvec_'+$ocAxisName+'_deg') 'actual'
        Add-OCPair ('Cube'+$ocAxisName.ToUpperInvariant()) ('cube_rotvec_'+$ocAxisName+'_deg') 'cube'
    }
    Add-OCSeries 'ToolDelta' 'Replay vs demonstration' 'actual_orientation_delta_deg' $ocBlue
    Add-OCSeries 'CubeDelta' 'Replay vs demonstration' 'cube_orientation_delta_deg' $ocBlue
    Add-OCPair 'Tracking' 'tracking_orientation_deg'
    $ocHeading='G1 saved demonstration vs dynamic replay | orientation comparison'
    if ($ocPayload.tool_check) { $ocHeading='ORIENTATION TOOL CHECK ONLY | NO NEW DYNAMIC REPLAY' }
    Add-OCTitle $ocHeading .4 $true
    Add-OCTitle ('Source: {0}; replay: {1} / {2}; source timeline covered: {3}.' -f $ocPayload.source_status,$ocPayload.replay_status,$ocPayload.replay_end_reason,$ocPayload.full_coverage) 2.8
    Add-OCTitle ('Aligned samples: {0}; plotted: {1}. Statistics use all aligned steps; no interpolation, smoothing, or time warping.' -f $ocPayload.overlap_samples,$ocPayload.plotted_samples) 4.8
    Add-OCTitle 'Orange dashed = demonstration; blue solid = replay. Each object uses its SAME demonstration initial orientation as the shared baseline.' 6.8
    Add-OCTitle 'XYZ panels: world-axis rotation-vector components in degrees, NOT Euler angles. Lines break at recorded rotation-vector branch jumps.' 8.8
    Add-OCTitle 'Delta angle: replay actual vs demonstration actual (shortest rotation). Tracking angle: actual vs current reference. These are different errors.' 96
    Add-OCTitle 'Only overlapping saved states are drawn. An empty tail is not extrapolated. No physics, new task verdict, or unseen-initial-state test is performed.' 98
    $ocStream=[IO.File]::Open($ocPayload.output_png,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $ocChart.SaveImage($ocStream,[System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png) } finally { $ocStream.Dispose() }
    $ocMeta=[ordered]@{
        kind='orientation_comparison_plot_source';tool_check=[bool]$ocPayload.tool_check
        comparison_json=[string]$ocReceipt.comparison_json
        comparison_sha256=(Get-FileHash -LiteralPath $ocReceipt.comparison_json -Algorithm SHA256).Hash
        plotdata_json=$ocPayloadPath;plotdata_sha256=$ocReceipt.plotdata_sha256
        renderer=$PSCommandPath;renderer_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
        output_png=[string]$ocPayload.output_png
        output_png_sha256=(Get-FileHash -LiteralPath $ocPayload.output_png -Algorithm SHA256).Hash
        new_physics_steps=0
    }
    $ocBytes=(New-Object Text.UTF8Encoding($false)).GetBytes(($ocMeta|ConvertTo-Json -Depth 12))
    $ocStream=[IO.File]::Open($ocPayload.output_source_json,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $ocStream.Write($ocBytes,0,$ocBytes.Length) } finally { $ocStream.Dispose() }
    Write-Output ('COMPARISON: '+$ocReceipt.comparison_dir)
    Write-Output ('REPORT: '+$ocReceipt.report)
    Write-Output ('PNG: '+$ocPayload.output_png)
    Write-Output ('SOURCE: '+$ocPayload.output_source_json)
    if (-not $NoShow) {
        $ocForm=New-Object Windows.Forms.Form
        $ocForm.Text='G1 orientation comparison | saved states'
        $ocForm.StartPosition='CenterScreen';$ocForm.AutoScroll=$true
        $ocScreen=[Windows.Forms.Screen]::PrimaryScreen.WorkingArea
        $ocForm.ClientSize=New-Object Drawing.Size([Math]::Min(1800,$ocScreen.Width-80),[Math]::Min(1250,$ocScreen.Height-100))
        $ocForm.Controls.Add($ocChart);[void]$ocForm.ShowDialog()
    }
} finally {
    if ($null -ne $ocForm) { $ocForm.Dispose() }
    $ocChart.Dispose()
}
