@echo off
call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if errorlevel 1 exit /b %errorlevel%
cd /d C:\Users\devcloud\inkling-autolab\repo
set CARGO_TARGET_DIR=C:\Users\devcloud\inkling-autolab\target
rustc +stable-x86_64-pc-windows-msvc -vV
cargo +stable-x86_64-pc-windows-msvc -V
cargo +stable-x86_64-pc-windows-msvc build --release -p cascadia-engine-sparse-moe --example inkling_bench --example inkling_decode_bench
set BUILD_STATUS=%ERRORLEVEL%
echo BUILD_EXIT %BUILD_STATUS%
exit /b %BUILD_STATUS%
