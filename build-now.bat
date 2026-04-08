@echo off
setlocal enabledelayedexpansion

REM ==========================================================================
REM  The House of Bonsai — Multi-Backend Build Script
REM  Builds llama-server with all available GPU backends + Rust launcher
REM ==========================================================================

REM --- Visual Studio ---
set VS_ROOT=
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" (
    set VS_ROOT=C:\Program Files\Microsoft Visual Studio\18\Community
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
    set VS_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community
) else (
    echo ERROR: Visual Studio 2026 or 2022 not found
    exit /b 1
)
call "%VS_ROOT%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

REM --- Ensure VS-bundled CMake is on PATH (takes priority over Python cmake) ---
set "VS_CMAKE_DIR=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
if exist "%VS_CMAKE_DIR%\cmake.exe" (
    set "PATH=%VS_CMAKE_DIR%;%PATH%"
    echo [DETECT] Using VS-bundled CMake from "%VS_CMAKE_DIR%"
)

set PROJECT_ROOT=%~dp0
REM Strip trailing backslash
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%
set LLAMA_SRC=%PROJECT_ROOT%\llama-cpp
set BUILD_DIR=%LLAMA_SRC%\build
set BIN_DIR=%BUILD_DIR%\bin

REM --- Detect available backends ---
set HAS_CUDA=0
set HAS_VULKAN=0

REM Check CUDA
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
    set PATH=!CUDA_PATH!\bin;!PATH!
    set HAS_CUDA=1
    echo [DETECT] CUDA 13.2 found
) else (
    for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%D\bin\nvcc.exe" (
            set CUDA_PATH=%%D
            set PATH=!CUDA_PATH!\bin;!PATH!
            set HAS_CUDA=1
            echo [DETECT] CUDA found at %%D
        )
    )
)
if !HAS_CUDA!==0 echo [DETECT] No CUDA Toolkit found — skipping CUDA backend

REM Check Vulkan — env var first, then scan C:\VulkanSDK\*
if defined VULKAN_SDK (
    if exist "%VULKAN_SDK%\Include\vulkan\vulkan.h" (
        set HAS_VULKAN=1
        echo [DETECT] Vulkan SDK found at %VULKAN_SDK%
    )
)
if !HAS_VULKAN!==0 (
    for /d %%D in ("C:\VulkanSDK\*") do (
        if exist "%%D\Include\vulkan\vulkan.h" (
            set VULKAN_SDK=%%D
            set HAS_VULKAN=1
            echo [DETECT] Vulkan SDK found at %%D
        )
    )
)
if !HAS_VULKAN!==0 echo [DETECT] No Vulkan SDK found — skipping Vulkan backend

REM --- Build CMake flags ---
set CMAKE_BACKEND_FLAGS=-DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON

if !HAS_CUDA!==1 (
    set CMAKE_BACKEND_FLAGS=!CMAKE_BACKEND_FLAGS! -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
)

if !HAS_VULKAN!==1 (
    set CMAKE_BACKEND_FLAGS=!CMAKE_BACKEND_FLAGS! -DGGML_VULKAN=ON
)

if !HAS_CUDA!==0 if !HAS_VULKAN!==0 (
    echo.
    echo [WARNING] No GPU SDK detected — building CPU-only backend
    echo           Install CUDA Toolkit or Vulkan SDK for GPU acceleration
    echo.
)

echo.
echo === Configuring CMake ===
echo Backends: CUDA=!HAS_CUDA! Vulkan=!HAS_VULKAN!
echo Flags: !CMAKE_BACKEND_FLAGS!
echo.

cmake -S "%LLAMA_SRC%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release !CMAKE_BACKEND_FLAGS!
if %ERRORLEVEL% NEQ 0 (
    echo CMake configure FAILED
    exit /b 1
)

echo.
echo === Building llama-server ===
cmake --build "%BUILD_DIR%" --config Release --parallel 12 --target llama-server
if %ERRORLEVEL% NEQ 0 (
    echo Build FAILED
    exit /b 1
)

REM --- Verify built backends ---
echo.
echo === Built Backends ===
if exist "%BIN_DIR%\ggml-cuda.dll" (echo   [OK] CUDA backend: ggml-cuda.dll) else (echo   [--] CUDA backend: not built)
if exist "%BIN_DIR%\ggml-vulkan.dll" (echo   [OK] Vulkan backend: ggml-vulkan.dll) else (echo   [--] Vulkan backend: not built)
if exist "%BIN_DIR%\ggml-cpu.dll" (echo   [OK] CPU backend: ggml-cpu.dll) else (echo   [--] CPU backend: not built)

REM --- Build Rust launcher ---
echo.
echo === Building launcher ===
pushd "%PROJECT_ROOT%\launcher"
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    popd
    echo Launcher build FAILED
    exit /b 1
)
popd

echo.
echo ===================================================
echo   BUILD COMPLETE
echo ===================================================
echo   llama-server: %BIN_DIR%\llama-server.exe
echo   launcher:     %PROJECT_ROOT%\launcher\target\release\turboquant-launcher.exe
echo.
echo   Backends:
if exist "%BIN_DIR%\ggml-cuda.dll" echo     CUDA   — NVIDIA GPU acceleration
if exist "%BIN_DIR%\ggml-vulkan.dll" echo     Vulkan — AMD / Intel / NVIDIA GPU acceleration
echo     CPU    — universal fallback
echo.
echo   To add Vulkan support: install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
echo ===================================================

endlocal
