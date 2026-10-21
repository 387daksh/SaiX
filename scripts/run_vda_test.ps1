$ErrorActionPreference="Stop"
$Root=Resolve-Path (Join-Path $PSScriptRoot "..")
$Py=Join-Path $Root ".venv-vda\Scripts\python.exe"
$Repo=Join-Path $Root "third_party\Video-Depth-Anything"
$Video=Join-Path $Root "data\videos\dashcam.mp4"
$Out=Join-Path $Root "data\outputs\depth"
if (!(Test-Path $Py)) { throw "Run scripts\setup_vda.ps1 first." }
if (!(Test-Path $Video)) { throw "Put dashcam.mp4 in data\videos first." }
Push-Location $Repo
& $Py run.py --input_video $Video --output_dir $Out --encoder vits --metric --target_fps 5 --max_res 960 --save_npz
Pop-Location
