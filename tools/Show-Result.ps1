[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunPath,[string]$Python='', [string]$CacheDirectory='', [switch]$NoShow,[string]$OutputDirectory)
# Existing-record plotting only; this script does not load or step MuJoCo.
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
if ($PSVersionTable.PSEdition -ne 'Desktop' -or $PSVersionTable.PSVersion.Major -ne 5) { throw 'Use Windows PowerShell 5.1 (powershell.exe).' }
$mpRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Python) { $Python=Join-Path $mpRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python is missing. Provide -Python pointing to an isolated environment. This script does not install software.' }
$mpPython=(Resolve-Path -LiteralPath $Python).Path
if (-not $CacheDirectory) { $CacheDirectory=Join-Path $mpRoot '.cache' }
$mpCache=[IO.Path]::GetFullPath($CacheDirectory)
$mpTemp=Join-Path $mpCache 'temp'
New-Item -ItemType Directory -Path $mpTemp -Force | Out-Null
$env:G1_PACKAGE_PYTHON=$mpPython;$env:G1_PACKAGE_CACHE=$mpCache
$env:TEMP=$mpTemp;$env:TMP=$mpTemp;$env:TMPDIR=$mpTemp
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8'
$env:XDG_CACHE_HOME=$mpCache;$env:MPLCONFIGDIR=Join-Path $mpCache 'matplotlib'
$mpPlot=Join-Path $mpRoot 'runtime\plot_manual_task.py'
$mpArgs=@('-B',$mpPlot,'--run',$RunPath,'--prepare-only','--receipt-json')
if (-not [string]::IsNullOrWhiteSpace($OutputDirectory)) { $mpArgs+=@('--output-directory',$OutputDirectory) }
$mpPrepared=@(& $mpPython @mpArgs)
if ($LASTEXITCODE -ne 0 -or $mpPrepared.Count -ne 1) { throw 'Saved-record validation failed; no plot generated.' }
# Native stdout carries ASCII JSON only: do not rely on the console code page
# to decode a Chinese path emitted by a UTF-8 Python child in PowerShell 5.1.
$mpReceipt=([string]$mpPrepared[0]) | ConvertFrom-Json
if ($mpReceipt.kind -ne 'manual_plot_prepare_receipt' -or $mpReceipt.protocol -ne 1) { throw 'Unexpected plotting receipt.' }
$mpPayloadPath=[string]$mpReceipt.plotdata_json
if (-not (Test-Path -LiteralPath $mpPayloadPath -PathType Leaf)) { throw 'Prepared plot payload is missing.' }
if ((Get-FileHash -LiteralPath $mpPayloadPath -Algorithm SHA256).Hash -ne $mpReceipt.plotdata_sha256) { throw 'Prepared plot payload hash differs from receipt.' }
$mpPayload=Get-Content -LiteralPath $mpPayloadPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($mpPayload.kind -ne 'manual_pickplace_plot_payload') { throw 'Unexpected plot payload.' }
$mpRows=@($mpPayload.rows)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$mpChart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$mpForm=$null; $mpAreas=@{}
$mpBlue=[Drawing.Color]::RoyalBlue; $mpOrange=[Drawing.Color]::DarkOrange
$mpPurple=[Drawing.Color]::DarkViolet; $mpGreen=[Drawing.Color]::SeaGreen; $mpBlack=[Drawing.Color]::Black
function Add-MPTitle([string]$Text,[double]$Y,[bool]$Bold=$false,[double]$Height=1.7) {
    $mpT=New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $mpT.Text=$Text; $mpT.Font=New-Object Drawing.Font('Segoe UI',10)
    if ($Bold) { $mpT.Font=New-Object Drawing.Font('Segoe UI',14,[Drawing.FontStyle]::Bold) }
    $mpT.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(0,$Y,100,$Height)
    [void]$mpChart.Titles.Add($mpT)
}
function Set-MPRange($Axis,[string[]]$Fields,[double[]]$Include=@(),[double]$Padding=.1,[double]$MinimumPadding=.001) {
    $mpValues=New-Object 'System.Collections.Generic.List[double]'
    foreach ($mpValue in $Include) { $mpValues.Add($mpValue) }
    foreach ($mpRow in $mpRows) { foreach ($mpField in $Fields) {
        if ($null -ne $mpRow.$mpField) { $mpValues.Add([double]$mpRow.$mpField) }
    } }
    if ($mpValues.Count -eq 0) { $Axis.Minimum=0; $Axis.Maximum=1; return }
    $mpMin=($mpValues | Measure-Object -Minimum).Minimum; $mpMax=($mpValues | Measure-Object -Maximum).Maximum
    $mpPad=[Math]::Max($MinimumPadding,($mpMax-$mpMin)*$Padding)
    $Axis.Minimum=$mpMin-$mpPad; $Axis.Maximum=$mpMax+$mpPad
}
function Add-MPSeries([string]$Area,[string]$Name,[string]$Field,[Drawing.Color]$Color,[bool]$Secondary=$false,[bool]$Dashed=$false,[bool]$Step=$false) {
    # Separate finite runs: absent grip/release measurements are gaps, never zero.
    $mpS=$null; $mpPart=0
    foreach ($mpRow in $mpRows) {
        if ($null -eq $mpRow.$Field) { $mpS=$null; continue }
        if ($null -eq $mpS) {
            $mpS=New-Object System.Windows.Forms.DataVisualization.Charting.Series ($Area+' '+$Name+' '+$mpPart)
            $mpS.ChartArea=$Area; $mpS.Legend=$Area; $mpS.LegendText=$Name
            $mpS.IsVisibleInLegend=($mpPart -eq 0); $mpS.ChartType='Line'; $mpS.Color=$Color; $mpS.BorderWidth=2
            if ($Step) { $mpS.ChartType='StepLine' }; if ($Secondary) { $mpS.YAxisType='Secondary' }
            if ($Dashed) { $mpS.BorderDashStyle='Dash' }
            [void]$mpChart.Series.Add($mpS); $mpPart++
        }
        [void]$mpS.Points.AddXY([double]$mpRow.sim_s,[double]$mpRow.$Field)
    }
    foreach ($mpS in @($mpChart.Series | Where-Object { $_.ChartArea -eq $Area -and $_.LegendText -eq $Name })) {
        if ($mpS.Points.Count -eq 1) { $mpS.ChartType='Point'; $mpS.MarkerStyle='Circle'; $mpS.MarkerSize=5 }
    }
}
function Add-MPGate([string]$Area,[double]$Value,[Drawing.Color]$Color,[bool]$Secondary=$false) {
    $mpAxis=$mpAreas[$Area].AxisY; if ($Secondary) { $mpAxis=$mpAreas[$Area].AxisY2 }
    $mpLine=New-Object System.Windows.Forms.DataVisualization.Charting.StripLine
    $mpLine.IntervalOffset=$Value; $mpLine.StripWidth=0; $mpLine.BorderWidth=1
    $mpLine.BorderColor=$Color; $mpLine.BorderDashStyle='Dash'; [void]$mpAxis.StripLines.Add($mpLine)
}
try {
    $mpChart.Size=New-Object Drawing.Size(1500,1600); $mpChart.BackColor=[Drawing.Color]::White
    $mpSpecs=@(
        @('X',1,12,48,16,'World X displacement (mm)',''),@('Y',51,12,48,16,'World Y displacement (mm)',''),
        @('Z',1,33,48,16,'World Z displacement (mm)',''),@('Tracking',51,33,48,16,'Position error (mm)','Orientation error (rad)'),
        @('Finger',1,54,48,16,'Finger position (mm)','Contact force (N)'),@('Height',51,54,48,16,'Cube bottom / table (mm)',''),
        @('Slip',1,75,48,16,'Held translation (mm)','Held rotation (rad)'),@('Placement',51,75,48,16,'Valid placement (s)','Instant valid (0/1)'))
    $mpT0=0.;$mpT1=1.
    if ($mpRows.Count -gt 0) { $mpT0=[double]$mpRows[0].sim_s; $mpT1=[double]$mpRows[$mpRows.Count-1].sim_s }
    if ($mpT1 -le $mpT0) { $mpT0-=.005; $mpT1+=.005 }
    foreach ($mpSpec in $mpSpecs) {
        $mpName=[string]$mpSpec[0]; $mpArea=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea $mpName
        $mpArea.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($mpSpec[1],$mpSpec[2],$mpSpec[3],$mpSpec[4])
        $mpArea.InnerPlotPosition=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(17,4,71,70)
        $mpArea.AxisX.Title='Simulation time (s)'; $mpArea.AxisY.Title=$mpSpec[5]
        $mpArea.AxisX.Enabled='True'; $mpArea.AxisY.Enabled='True'
        $mpArea.AxisX.Minimum=$mpT0; $mpArea.AxisX.Maximum=$mpT1
        if ($mpSpec[6]) {
            $mpArea.InnerPlotPosition=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(17,4,64,70)
            $mpArea.AxisY2.Enabled='True'; $mpArea.AxisY2.Title=$mpSpec[6]; $mpArea.AxisY2.MajorGrid.Enabled=$false
        }
        foreach ($mpAxis in @($mpArea.AxisX,$mpArea.AxisY,$mpArea.AxisY2)) {
            $mpAxis.TitleFont=New-Object Drawing.Font('Segoe UI',10);$mpAxis.LabelStyle.Font=New-Object Drawing.Font('Segoe UI',9)
            $mpAxis.LabelStyle.Format='0.###';$mpAxis.MajorGrid.LineColor=[Drawing.Color]::Gainsboro
        }
        [void]$mpChart.ChartAreas.Add($mpArea);$mpAreas[$mpName]=$mpArea
        $mpLegend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend $mpName
        $mpLegend.Font=New-Object Drawing.Font('Segoe UI',9);$mpLegend.LegendStyle='Row';$mpLegend.IsTextAutoFit=$false
        $mpLegend.Position=New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($mpSpec[1],([double]$mpSpec[2]+[double]$mpSpec[4]+2),48,2)
        [void]$mpChart.Legends.Add($mpLegend)
    }
    $mpSpatialFields=@();foreach ($mpAxis in @('x','y','z')) { foreach ($mpPrefix in @('target','ref','actual')) { $mpSpatialFields+=($mpPrefix+'_'+$mpAxis+'_mm') } }
    foreach ($mpAxis in @('x','y','z')) {
        $mpArea=$mpAxis.ToUpperInvariant();Set-MPRange $mpAreas[$mpArea].AxisY $mpSpatialFields @(0) .08 .5
        Add-MPSeries $mpArea 'Accepted target' ('target_'+$mpAxis+'_mm') $mpBlack $false $true $true
        Add-MPSeries $mpArea 'FK reference' ('ref_'+$mpAxis+'_mm') $mpOrange
        Add-MPSeries $mpArea 'Physics actual' ('actual_'+$mpAxis+'_mm') $mpBlue
    }
    $mpG=$mpPayload.thresholds
    Add-MPSeries 'Tracking' 'Position (L)' 'position_error_mm' $mpBlue
    Add-MPSeries 'Tracking' 'Orientation (R)' 'orientation_error_rad' $mpPurple $true
    Set-MPRange $mpAreas.Tracking.AxisY @('position_error_mm') @(0,$mpG.position_mm) .1 .1
    Set-MPRange $mpAreas.Tracking.AxisY2 @('orientation_error_rad') @(0,$mpG.orientation_rad) .1 .001
    Add-MPGate 'Tracking' $mpG.position_mm $mpBlue;Add-MPGate 'Tracking' $mpG.orientation_rad $mpPurple $true
    Add-MPSeries 'Finger' 'Ref (L)' 'finger_ref_mm' $mpBlack $false $true
    Add-MPSeries 'Finger' 'Finger 1 (L)' 'finger1_q_mm' $mpBlue
    Add-MPSeries 'Finger' 'Finger 2 (L)' 'finger2_q_mm' $mpOrange
    Add-MPSeries 'Finger' 'Contact 1 (R)' 'finger1_normal_n' $mpPurple $true
    Add-MPSeries 'Finger' 'Contact 2 (R)' 'finger2_normal_n' $mpGreen $true
    Set-MPRange $mpAreas.Finger.AxisY @('finger_ref_mm','finger1_q_mm','finger2_q_mm') @(0) .1 1
    Set-MPRange $mpAreas.Finger.AxisY2 @('finger1_normal_n','finger2_normal_n') @(0) .1 .1
    Add-MPSeries 'Height' 'Cube lowest point' 'cube_bottom_clearance_mm' $mpBlue
    $mpHeights=@(0,[double]$mpG.lift_mm)
    Add-MPGate 'Height' 0 ([Drawing.Color]::Gray);Add-MPGate 'Height' $mpG.lift_mm $mpBlack
    if ('table_top_z_m' -in @($mpPayload.geometry.PSObject.Properties.Name)) {
        foreach ($mpH in @(@('tray_floor_z_m',$mpGreen),@('tray_rim_z_m',$mpOrange))) {
            if ($mpH[0] -in @($mpPayload.geometry.PSObject.Properties.Name)) {
                $mpHeight=1000*([double]$mpPayload.geometry.($mpH[0])-[double]$mpPayload.geometry.table_top_z_m)
                $mpHeights+=$mpHeight;Add-MPGate 'Height' $mpHeight $mpH[1]
            }
        }
    }
    Set-MPRange $mpAreas.Height.AxisY @('cube_bottom_clearance_mm') $mpHeights .1 1
    Add-MPSeries 'Slip' 'Translation (L)' 'held_slip_mm' $mpBlue
    Add-MPSeries 'Slip' 'Rotation (R)' 'held_rotation_rad' $mpPurple $true
    Set-MPRange $mpAreas.Slip.AxisY @('held_slip_mm') @(0,$mpG.slip_mm) .1 .1
    Set-MPRange $mpAreas.Slip.AxisY2 @('held_rotation_rad') @(0,$mpG.rotation_rad) .1 .001
    Add-MPGate 'Slip' $mpG.slip_mm $mpBlue;Add-MPGate 'Slip' $mpG.rotation_rad $mpPurple $true
    Add-MPSeries 'Placement' 'Valid duration (L)' 'postrelease_hold_s' $mpBlue
    Add-MPSeries 'Placement' 'Instant valid (R)' 'postrelease_valid' $mpGreen $true $false $true
    Set-MPRange $mpAreas.Placement.AxisY @('postrelease_hold_s') @(0,$mpG.placement_s) .1 .1
    $mpAreas.Placement.AxisY2.Minimum=-.1;$mpAreas.Placement.AxisY2.Maximum=1.1
    Add-MPGate 'Placement' $mpG.placement_s $mpBlack
    $mpHeading='Fixed-base G1 | manual XYZ positioning, grasp and tray placement'
    if ($mpPayload.diagnostic_only) { $mpHeading='DIAGNOSTIC RECORD | manual-task acceptance incomplete' }
    if ($mpPayload.synthetic_tool_check) { $mpHeading='SYNTHETIC TOOL CHECK | NOT a robot task result' }
    Add-MPTitle $mpHeading .3 $true 2.2
    $mpRecordedEnd='none';if ($mpRows.Count -gt 0) { $mpRecordedEnd=('{0:0.###} s' -f $mpRows[$mpRows.Count-1].sim_s) }
    Add-MPTitle ('Status: {0}; end: {1}; steps: {2}; summary sim: {3:0.###} s; saved through: {4}. Rows: {5}; plotted: {6}.' -f $mpPayload.run_status,$mpPayload.end_reason,$mpPayload.steps,$mpPayload.simulation_s,$mpRecordedEnd,$mpPayload.source_rows,$mpPayload.plotted_rows) 2.6
    if ($mpPayload.held_direction_mode) {
        Add-MPTitle ('Saved gates: manual={0}; grasp={1}; placement={2}; tracking={3}. Jog references: {4} accepted / {5} refused (not key presses).' -f $mpPayload.gates.manual_task_completed,$mpPayload.gates.grasp_hold_passed,$mpPayload.gates.placement_passed,$mpPayload.gates.tracking_thresholds_passed,$mpPayload.accepted_jog_reference_segments,$mpPayload.refused_jog_reference_proposals) 4.3
    } else {
        Add-MPTitle ('Saved gates: manual={0}; grasp={1}; placement={2}; tracking={3}. Command events: {4} accepted / {5} refused.' -f $mpPayload.gates.manual_task_completed,$mpPayload.gates.grasp_hold_passed,$mpPayload.gates.placement_passed,$mpPayload.gates.tracking_thresholds_passed,$mpPayload.accepted_events,$mpPayload.refused_events) 4.3
    }
    Add-MPTitle 'Target = accepted manual goal; FK = current interpolated joint reference; actual = physics state. XYZ are world axes; shared mm scale.' 6
    Add-MPTitle 'Grip slip is blank before baseline and after opening. Finger positions are joint coordinates, not jaw width. Curves join discrete samples.' 7.7
    if ($mpRows.Count -eq 0) { Add-MPTitle 'NO SAVED STATE SAMPLES: no motion, tracking or task outcome can be inferred from these empty panels.' 9.4 $true 2 }
    elseif ($mpPayload.diagnostic_only) { Add-MPTitle ('DIAGNOSTIC: '+[string]$mpPayload.recording_issues[0]) 9.4 $false 2 }
    elseif ($mpRows.Count -eq 1) { Add-MPTitle 'ONE SAVED STATE ONLY: dots are the initial/terminal state; no time response or completed movement can be inferred.' 9.4 $false 2 }
    else { Add-MPTitle ('XYZ origin is first FK reference: [{0:0.######}, {1:0.######}, {2:0.######}] m. All time axes are simulation seconds.' -f $mpPayload.origin_world_m[0],$mpPayload.origin_world_m[1],$mpPayload.origin_world_m[2]) 9.4 }
    Add-MPTitle ('Dashed limits: tracking {0:0.###} mm / {1:0.###} rad; held slip {2:0.###} mm / {3:0.###} rad; placement {4:0.###} s.' -f $mpG.position_mm,$mpG.orientation_rad,$mpG.slip_mm,$mpG.rotation_rad,$mpG.placement_s) 95
    Add-MPTitle ('Height guides: gray table; black lift {0:0.###} mm; green tray floor; orange tray rim. No continuous collision certificate or real-robot claim.' -f $mpG.lift_mm) 96.6
    Add-MPTitle 'Outcome labels quote the saved summary; plotting performs no new simulation and does not establish repeatability or v0.1 release acceptance.' 98.2
    $mpStream=[IO.File]::Open($mpPayload.output_png,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $mpChart.SaveImage($mpStream,[System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png) } finally { $mpStream.Dispose() }
    $mpMeta=[ordered]@{}
    foreach ($mpProperty in $mpPayload.PSObject.Properties) { if ($mpProperty.Name -ne 'rows') { $mpMeta[$mpProperty.Name]=$mpProperty.Value } }
    $mpMeta.kind='manual_pickplace_record_plot';$mpMeta['plotdata_json']=$mpPayloadPath
    $mpMeta['plotdata_sha256']=(Get-FileHash -LiteralPath $mpPayloadPath -Algorithm SHA256).Hash
    $mpMeta['renderer']=$PSCommandPath;$mpMeta['renderer_sha256']=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    $mpMeta['output_png_sha256']=(Get-FileHash -LiteralPath $mpPayload.output_png -Algorithm SHA256).Hash
    $mpBytes=(New-Object Text.UTF8Encoding($false)).GetBytes(($mpMeta|ConvertTo-Json -Depth 18))
    $mpStream=[IO.File]::Open($mpPayload.output_source_json,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try { $mpStream.Write($mpBytes,0,$mpBytes.Length) } finally { $mpStream.Dispose() }
    Write-Output ('PNG: '+$mpPayload.output_png);Write-Output ('SOURCE: '+$mpPayload.output_source_json)
    if (-not $NoShow) {
        $mpForm=New-Object Windows.Forms.Form;$mpForm.Text='G1 manual pick-place | saved evidence';$mpForm.StartPosition='CenterScreen'
        $mpScreen=[Windows.Forms.Screen]::PrimaryScreen.WorkingArea
        $mpForm.ClientSize=New-Object Drawing.Size([Math]::Min(1500,$mpScreen.Width-80),[Math]::Min(1600,$mpScreen.Height-100))
        $mpForm.AutoScroll=$true
        # Retain legibility with scroll bars rather than shrinking eight panels to tiny type.
        $mpForm.Controls.Add($mpChart);[void]$mpForm.ShowDialog()
    }
} finally { if ($null -ne $mpForm) { $mpForm.Dispose() };$mpChart.Dispose() }

