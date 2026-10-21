$ErrorActionPreference="Stop"
$Root=Resolve-Path (Join-Path $PSScriptRoot "..")
& (Join-Path $PSScriptRoot "setup_vda.ps1")
Write-Host "VDA-S Metric weights are installed in third_party\Video-Depth-Anything\checkpoints"
