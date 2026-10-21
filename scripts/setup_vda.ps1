$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root ".venv-vda"
$Repo = Join-Path $Root "third_party\Video-Depth-Anything"

Write-Host "[1/5] Checking Python 3.10..."
py -3.10 --version

if (-not (Test-Path $Venv)) {
  py -3.10 -m venv $Venv
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip setuptools wheel

if (-not (Test-Path $Repo)) {
  git clone https://github.com/DepthAnything/Video-Depth-Anything.git $Repo
}

Write-Host "[2/5] Installing official VDA requirements..."
Push-Location $Repo
& $Py -m pip install -r requirements.txt
Pop-Location

Write-Host "[3/5] Downloading Metric VDA-S checkpoint..."
$CkptDir=Join-Path $Repo "checkpoints"
New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
$Ckpt=Join-Path $CkptDir "metric_video_depth_anything_vits.pth"
if (-not (Test-Path $Ckpt)) {
  Invoke-WebRequest -Uri "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth" -OutFile $Ckpt
}

Write-Host "[4/5] Verifying imports..."
& $Py -c "import torch; import video_depth_anything; print('torch',torch.__version__); print('cuda',torch.cuda.is_available()); print('VDA import OK')"

Write-Host "[5/5] Ready: $Py"
