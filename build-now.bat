@echo off
REM Try VS 2026 first, fall back to VS 2022
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
) else (
    echo ERROR: Visual Studio 2026 or 2022 not found
    exit /b 1
)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_PATH%\bin;%PATH%

echo === Configuring CMake ===
cmake -S "E:\project\the-house-of-bonsai\llama-cpp" -B "E:\project\the-house-of-bonsai\llama-cpp\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CCACHE=OFF
if %ERRORLEVEL% NEQ 0 (
    echo CMake configure FAILED
    exit /b 1
)

echo === Building llama-server ===
cmake --build "E:\project\the-house-of-bonsai\llama-cpp\build" --config Release --parallel 12 --target llama-server
if %ERRORLEVEL% NEQ 0 (
    echo Build FAILED
    exit /b 1
)

echo === Building launcher ===
cd "E:\project\the-house-of-bonsai\launcher"
cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo Launcher build FAILED
    exit /b 1
)

echo === BUILD COMPLETE ===
echo llama-server: E:\project\the-house-of-bonsai\llama-cpp\build\bin\llama-server.exe
echo launcher: E:\project\the-house-of-bonsai\launcher\target\release\turboquant-launcher.exe
