# Local verification for Grok Bot (Windows).
# C++ tests plus the Python parity slice of scripts/verify_all.sh against a Release tree.
param(
    [string]$BuildDir = $(if ($env:FLOWEDGE_BUILD_DIR) { $env:FLOWEDGE_BUILD_DIR } else { "build-win-clang" }),
    [string]$Model = "models/mamba_flow.safetensors"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Build = Join-Path $Root $BuildDir
if (-not (Test-Path $Build)) {
    throw "build dir not found: $Build (configure FLOWEDGE_TESTS=ON FLOWEDGE_PYTHON=ON and build Release first)"
}

$env:PYTHONPATH = "$Build;$(Join-Path $Root 'integrations\lerobot\src')"

Write-Host "ctest $Build"
ctest --test-dir $Build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Tests = Get-ChildItem -Path $Build -Filter "flowedge_tests.exe" | Select-Object -First 1
if ($Tests) {
    & $Tests.FullName --gtest_filter="Matmul*:Conv*:DenseConv1d*:DiffusionConvolution*:Mish*:GroupNorm*"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "note: python unavailable; C++ tests passed"
    Write-Host "FlowEdge local verification passed (C++ only)"
    exit 0
}

python -m unittest discover -s integrations/lerobot/tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ModelPath = Join-Path $Root $Model
if (-not (Test-Path $ModelPath)) {
    Write-Host "note: $Model missing; skipped ULP and python parity"
    Write-Host "FlowEdge local verification passed (C++ + adapter tests)"
    exit 0
}

python -m flowedge_dev verify head $BuildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m flowedge_dev verify diffusion $BuildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ErrorActionPreference = "Continue"
python -c "import torch, numpy, safetensors"
$HasTorch = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if ($HasTorch) {
    python -m flowedge_dev verify ulp $ModelPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "note: PyTorch ULP dependencies unavailable"
}

Write-Host "FlowEdge local verification passed"
