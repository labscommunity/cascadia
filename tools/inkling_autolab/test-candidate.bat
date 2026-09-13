@echo off
call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if errorlevel 1 exit /b %errorlevel%
cd /d C:\Users\devcloud\inkling-autolab\repo
set CARGO_TARGET_DIR=C:\Users\devcloud\inkling-autolab\target
set RAYON_NUM_THREADS=16
set CASCADIA_BF16_GEMV_ROWS=4
set CASCADIA_INKLING_SEQ_READS=1
cargo +stable-x86_64-pc-windows-msvc test --release -p cascadia-engine-sparse-moe --lib bf16_tiled_tests
if errorlevel 1 exit /b %errorlevel%
cargo +stable-x86_64-pc-windows-msvc test --release -p cascadia-engine-sparse-moe --test inkling_attn --test inkling_conv --test inkling_ep --test inkling_gate --test inkling_loader --test inkling_model --test inkling_relpos --test inkling_wire
set TEST_STATUS=%ERRORLEVEL%
echo TEST_EXIT %TEST_STATUS%
exit /b %TEST_STATUS%
