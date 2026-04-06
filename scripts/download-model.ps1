param(
    [string] $OutputDir = (Join-Path $PSScriptRoot "..\models")
)

$ErrorActionPreference = "Stop"

$ModelUrl = "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/bonsai-8b-q1_0_g128.gguf"
$FileName = "bonsai-8b-q1_0_g128.gguf"
$OutputPath = Join-Path $OutputDir $FileName

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

if (Test-Path $OutputPath) {
    Write-Host "Model already exists at: $OutputPath"
    Write-Host "Delete it first if you want to re-download."
    exit 0
}

Write-Host "Downloading Bonsai-8B Q1_0 (g128) from HuggingFace..."
Write-Host "URL: $ModelUrl"
Write-Host "Destination: $OutputPath"
Write-Host ""

try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $ModelUrl -OutFile $OutputPath -UseBasicParsing
    $ProgressPreference = 'Continue'

    $size = (Get-Item $OutputPath).Length
    $sizeMB = [math]::Round($size / 1MB, 1)
    Write-Host "Download complete: $sizeMB MB"
} catch {
    Write-Error "Download failed: $_"
    if (Test-Path $OutputPath) {
        Remove-Item $OutputPath -Force
    }
    exit 1
}
