@echo off
REM Build llama-cli / llama-server (SYCL) with the weight-streaming patch on Windows.
REM
REM Requires (all on PATH or at the paths below):
REM   - Visual Studio 2022 Build Tools with the VCTools workload
REM   - Intel oneAPI (tested: Toolkit 2026.0) -> %ONEAPI_ROOT%\setvars.bat
REM   - git, cmake, ninja
REM
REM Usage: scripts\build-llama-stream-windows.bat [DEST]
REM   DEST  clone + build directory (default: .\llama-stream)
REM
REM Env:
REM   LLAMA_REPO  upstream remote (default: https://github.com/ggml-org/llama.cpp)
REM   LLAMA_BASE  base commit the patch applies to (see patches\llama.cpp\)
setlocal
set "REPO_ROOT=%~dp0.."
set "DEST=%~1"
if "%DEST%"=="" set "DEST=%CD%\llama-stream"
if "%LLAMA_REPO%"=="" set "LLAMA_REPO=https://github.com/ggml-org/llama.cpp"
if "%LLAMA_BASE%"=="" set "LLAMA_BASE=1692f9e50bb20fd96b963af38a282daf78feea64"
set "PATCH1=%REPO_ROOT%\patches\llama.cpp\0001-sycl-stream-weights.patch"
set "PATCH2=%REPO_ROOT%\patches\llama.cpp\0002-sycl-router-aware-moe.patch"

if not exist "%DEST%\.git" (
  git clone "%LLAMA_REPO%" "%DEST%" || exit /b 1
)
cd /d "%DEST%"
git checkout "%LLAMA_BASE%" || exit /b 1
git apply --check "%PATCH1%" 2>nul && git apply "%PATCH1%" || echo patch1 already applied or not applicable; continuing
git apply --check "%PATCH2%" 2>nul && git apply "%PATCH2%" || echo patch2 already applied or not applicable; continuing

REM VS + oneAPI environments. VS2022INSTALLDIR lets setvars find the Build Tools.
if "%VS2022INSTALLDIR%"=="" set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
if "%ONEAPI_ROOT%"=="" set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
call "%ONEAPI_ROOT%\setvars.bat" || exit /b 1

cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build --target llama-cli llama-server -j || exit /b 1

echo.
echo built: %DEST%\build\bin\llama-server.exe
echo set CASCADIA_LLAMA_BIN=%DEST%\build\bin\llama-server.exe
echo note: the child needs the oneAPI runtime on PATH - call "%ONEAPI_ROOT%\setvars.bat" first.
