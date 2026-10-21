param(
  [string]$YoloPython="python",
  [string]$VdaPython=".\.venv-vda\Scripts\python.exe",
  [double]$Fps=5,
  [double]$MeasurementDistance=15
)
$ErrorActionPreference="Stop"
$Root=Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$Video="data\videos\dashcam.mp4"
if (!(Test-Path $Video)) { throw "Put your video at data\videos\dashcam.mp4" }
if (!(Test-Path "calibration\homography.json")) { throw "Calibration required first. Run: python src\03_calibrate_homography.py --image data\test_frames\0000.jpg" }

Write-Host "[1/3] Preparing sampled frames at $Fps FPS"
& $YoloPython src\02_prepare_sequence.py --video $Video --output data\sequence --fps $Fps

Write-Host "[2/3] Running YOLOP"
& $YoloPython src\01_yolop_frames.py --input data\sequence --output data\outputs\yolop

Write-Host "[3/3] Running Metric Video Depth Anything Small + geometry fusion"
$Py=(Resolve-Path (Join-Path $Root $VdaPython)).Path
& $Py src\07_vda_fusion.py --video $Video --yolop data\outputs\yolop --homography calibration\homography.json --output data\outputs\final --fps $Fps --distance $MeasurementDistance

Write-Host "DONE. Results:"
Write-Host "  data\outputs\final\lane_width.csv"
Write-Host "  data\outputs\final\summary.json"
