$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Host "=== The House of Bonsai - Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check prerequisites
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

$missing = @()
if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) { $missing += "Rust (https://rustup.rs)" }
if (-not (Get-Command "cmake" -ErrorAction SilentlyContinue)) { $missing += "CMake (https://cmake.org)" }
if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) { $missing += "Git" }
if (-not (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) { $missing += "NVIDIA GPU drivers" }

# Check for CUDA toolkit
$cudaPath = $env:CUDA_PATH
if ([string]::IsNullOrWhiteSpace($cudaPath) -or -not (Test-Path $cudaPath)) {
    $missing += "CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)"
}

# Check for Visual Studio
$vsDevCmd = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ([string]::IsNullOrWhiteSpace($vsDevCmd)) {
    $missing += "Visual Studio 2022 with C++ workload"
}

if ($missing.Count -gt 0) {
    Write-Host "Missing prerequisites:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "  - $m" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Install the above and re-run this script."
    exit 1
}

Write-Host "  All prerequisites found." -ForegroundColor Green

# Step 2: Clone and patch llama.cpp
$llamaDir = Join-Path $ProjectRoot "llama-cpp"
Write-Host "[2/5] Setting up llama.cpp with Q1_0 CUDA support..." -ForegroundColor Yellow

if (-not (Test-Path $llamaDir)) {
    git clone https://github.com/ggml-org/llama.cpp.git $llamaDir
    Write-Host "  Applying Q1_0 CUDA patch..."
    Push-Location $llamaDir
    git apply (Join-Path $ProjectRoot "patches\q1_0-cuda-support.patch")
    Pop-Location
} else {
    Write-Host "  llama.cpp directory exists, skipping clone."
}

# Step 3: Build llama.cpp with available GPU backends
Write-Host "[3/5] Building llama.cpp..." -ForegroundColor Yellow

$buildDir = Join-Path $llamaDir "build"
$backendFlags = "-DGGML_NATIVE=ON -DGGML_CCACHE=OFF"

# Detect CUDA
$cudaPath = $env:CUDA_PATH
if ([string]::IsNullOrWhiteSpace($cudaPath) -or -not (Test-Path $cudaPath)) {
    # Try to find any CUDA installation
    $cudaSearch = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($cudaSearch) { $cudaPath = $cudaSearch.FullName; $env:CUDA_PATH = $cudaPath }
}
if (-not [string]::IsNullOrWhiteSpace($cudaPath) -and (Test-Path $cudaPath)) {
    Write-Host "  CUDA found: $cudaPath" -ForegroundColor Green
    $backendFlags += " -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON"
    $env:PATH = "$cudaPath\bin;$env:PATH"
} else {
    Write-Host "  CUDA not found — skipping CUDA backend" -ForegroundColor DarkYellow
}

# Detect Vulkan
$vulkanSdk = $env:VULKAN_SDK
if ([string]::IsNullOrWhiteSpace($vulkanSdk)) {
    $vulkanSearch = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($vulkanSearch) { $vulkanSdk = $vulkanSearch.FullName; $env:VULKAN_SDK = $vulkanSdk }
}
if (-not [string]::IsNullOrWhiteSpace($vulkanSdk) -and (Test-Path "$vulkanSdk\Include\vulkan\vulkan.h")) {
    Write-Host "  Vulkan SDK found: $vulkanSdk" -ForegroundColor Green
    $backendFlags += " -DGGML_VULKAN=ON"
} else {
    Write-Host "  Vulkan SDK not found — skipping Vulkan backend" -ForegroundColor DarkYellow
    Write-Host "  Install from https://vulkan.lunarg.com/sdk/home for AMD/Intel GPU support" -ForegroundColor DarkGray
}

Write-Host "  Backend flags: $backendFlags" -ForegroundColor Gray

$configureCmd = "cmake -S `"$llamaDir`" -B `"$buildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release $backendFlags"
$buildCmd = "cmake --build `"$buildDir`" --config Release --parallel 12 --target llama-server"

# Run in VS dev shell
$fullCmd = "call `"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && $configureCmd && $buildCmd"
& cmd.exe /c $fullCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. Check the output above for errors." -ForegroundColor Red
    exit 1
}

$serverExe = Join-Path $buildDir "bin\llama-server.exe"
if (-not (Test-Path $serverExe)) {
    Write-Host "Build completed but llama-server.exe not found at expected path." -ForegroundColor Red
    exit 1
}
Write-Host "  llama-server.exe built successfully." -ForegroundColor Green

# Step 4: Download model
Write-Host "[4/5] Downloading Bonsai-8B model..." -ForegroundColor Yellow
& powershell.exe -NoLogo -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts\download-model.ps1")

# Step 5: Build launcher
Write-Host "[5/5] Building launcher..." -ForegroundColor Yellow
$launcherDir = Join-Path $ProjectRoot "launcher"
Push-Location $launcherDir
cargo build --release
Pop-Location

$launcherExe = Join-Path $launcherDir "target\release\turboquant-launcher.exe"
if (-not (Test-Path $launcherExe)) {
    Write-Host "Launcher build failed." -ForegroundColor Red
    exit 1
}

Write-Host "" -ForegroundColor Green
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To start:" -ForegroundColor Cyan
Write-Host "  $launcherExe"
Write-Host ""
Write-Host "The launcher will auto-detect your GPU and optimize settings."
Write-Host "Select the Bonsai model, click Start, then Chat."
