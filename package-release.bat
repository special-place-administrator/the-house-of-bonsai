@echo off
setlocal enabledelayedexpansion

REM ==========================================================================
REM  The House of Bonsai — Package Release
REM  Creates a self-contained distributable folder with all dependencies
REM ==========================================================================

set PROJECT_ROOT=%~dp0
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

set BUILD_BIN=%PROJECT_ROOT%\llama-cpp\build\bin
set LAUNCHER_EXE=%PROJECT_ROOT%\launcher\target\release\turboquant-launcher.exe
set RELEASE_DIR=%PROJECT_ROOT%\release\Bonsai
set SYSTEM_DIR=%RELEASE_DIR%\system
set MODELS_DIR=%RELEASE_DIR%\models

echo.
echo ===================================================
echo   The House of Bonsai — Package Release
echo ===================================================
echo.

REM --- Verify build artifacts exist ---
if not exist "%BUILD_BIN%\llama-server.exe" (
    echo ERROR: llama-server.exe not found. Run build-now.bat first.
    exit /b 1
)
if not exist "%LAUNCHER_EXE%" (
    echo ERROR: Launcher not found. Run build-now.bat first.
    exit /b 1
)

REM --- Clean and create release directories ---
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%SYSTEM_DIR%"
mkdir "%MODELS_DIR%"

echo [1/5] Copying launcher...
copy /y "%LAUNCHER_EXE%" "%RELEASE_DIR%\bonsai-launcher.exe" >nul
echo   OK

echo [2/5] Copying llama-server and backend DLLs...
for %%F in (
    llama-server.exe
    llama.dll
    ggml.dll
    ggml-base.dll
    ggml-cpu.dll
    mtmd.dll
) do (
    if exist "%BUILD_BIN%\%%F" (
        copy /y "%BUILD_BIN%\%%F" "%SYSTEM_DIR%\" >nul
        echo   %%F
    )
)

REM --- GPU backend DLLs (optional, included if built) ---
if exist "%BUILD_BIN%\ggml-cuda.dll" (
    copy /y "%BUILD_BIN%\ggml-cuda.dll" "%SYSTEM_DIR%\" >nul
    echo   ggml-cuda.dll
)
if exist "%BUILD_BIN%\ggml-vulkan.dll" (
    copy /y "%BUILD_BIN%\ggml-vulkan.dll" "%SYSTEM_DIR%\" >nul
    echo   ggml-vulkan.dll
)

echo [3/5] Bundling CUDA runtime DLLs...
REM --- Find CUDA toolkit and copy required runtime DLLs ---
set CUDA_FOUND=0

REM Try v13.2 first, then scan for any version
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64" (
    set CUDA_BIN_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64
    set CUDA_FOUND=1
) else (
    for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%D\bin\x64" (
            set CUDA_BIN_DIR=%%D\bin\x64
            set CUDA_FOUND=1
        ) else if exist "%%D\bin" (
            set CUDA_BIN_DIR=%%D\bin
            set CUDA_FOUND=1
        )
    )
)

if !CUDA_FOUND!==1 (
    REM Copy essential CUDA runtime DLLs
    for %%F in (
        cublas64_13.dll
        cublasLt64_13.dll
        cudart64_13.dll
    ) do (
        if exist "!CUDA_BIN_DIR!\%%F" (
            copy /y "!CUDA_BIN_DIR!\%%F" "%SYSTEM_DIR%\" >nul
            echo   %%F
        ) else (
            REM Try without version suffix (older CUDA)
            echo   [SKIP] %%F not found in !CUDA_BIN_DIR!
        )
    )
    REM Also try cublas64_12 for CUDA 12.x
    if not exist "%SYSTEM_DIR%\cublas64_13.dll" (
        for %%F in (cublas64_12.dll cublasLt64_12.dll cudart64_12.dll) do (
            if exist "!CUDA_BIN_DIR!\%%F" (
                copy /y "!CUDA_BIN_DIR!\%%F" "%SYSTEM_DIR%\" >nul
                echo   %%F
            )
        )
    )
) else (
    echo   [SKIP] No CUDA Toolkit found — CUDA DLLs not bundled
    echo          Users with NVIDIA GPUs will need CUDA Toolkit installed
)

echo [4/5] Bundling VC++ runtime...
REM --- Copy Visual C++ runtime DLLs from build output ---
for %%F in (
    msvcp140.dll
    msvcp140_1.dll
    vcruntime140.dll
    vcruntime140_1.dll
) do (
    if exist "%BUILD_BIN%\%%F" (
        copy /y "%BUILD_BIN%\%%F" "%SYSTEM_DIR%\" >nul
        echo   %%F
    )
)

echo [5/6] Bundling MCP server executables...
REM --- Copy pre-built MCP server exes ---
if exist "%PROJECT_ROOT%\prism-mcp\prism-mcp-server.exe" (
    copy /y "%PROJECT_ROOT%\prism-mcp\prism-mcp-server.exe" "%RELEASE_DIR%\" >nul
    echo   prism-mcp-server.exe
) else (
    echo   [SKIP] prism-mcp-server.exe not found — run: cd prism-mcp ^&^& bun build --compile --target=bun-windows-x64 src/server.ts --outfile=prism-mcp-server.exe
)
if exist "%PROJECT_ROOT%\bonsai-mcp\bonsai-mcp-server.exe" (
    copy /y "%PROJECT_ROOT%\bonsai-mcp\bonsai-mcp-server.exe" "%RELEASE_DIR%\" >nul
    echo   bonsai-mcp-server.exe
) else (
    echo   [SKIP] bonsai-mcp-server.exe not found — run: cd bonsai-mcp ^&^& bun build --compile --target=bun-windows-x64 src/server.ts --outfile=bonsai-mcp-server.exe
)

echo [6/6] Verifying package...
echo.
echo === Package Contents ===
echo   %RELEASE_DIR%\
echo     bonsai-launcher.exe
dir /b "%SYSTEM_DIR%" 2>nul | findstr /v "^$" > nul
if %ERRORLEVEL%==0 (
    echo     system\
    for %%F in ("%SYSTEM_DIR%\*") do (
        echo       %%~nxF
    )
)
echo     models\  (empty — user adds GGUF files)

REM --- Count files and total size ---
set FILE_COUNT=0
set /a TOTAL_SIZE=0
for %%F in ("%RELEASE_DIR%\*" "%SYSTEM_DIR%\*") do (
    set /a FILE_COUNT+=1
)

echo.
echo ===================================================
echo   PACKAGE READY: %RELEASE_DIR%
echo ===================================================
echo.
echo   To distribute:
echo     1. Copy the Bonsai folder to C:\TOOLS\Bonsai
echo     2. Add GGUF model files to the models\ subfolder
echo     3. Run bonsai-launcher.exe
echo.
echo   To create a ZIP:
echo     powershell Compress-Archive -Path "%RELEASE_DIR%" -DestinationPath "%PROJECT_ROOT%\release\Bonsai.zip"
echo ===================================================

endlocal
