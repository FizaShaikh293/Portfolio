@echo off
REM ============================================================
REM  Export a specific block range from your local Monero node
REM  FINAL PATH VERSION
REM ============================================================

REM --- Path to monero-blockchain-export.exe ---
SET MONERO_BIN="C:\Monero\monero-x86_64-w64-mingw32-v0.18.4.4"

REM --- Output file path ---
SET OUTPUT_FILE="C:\MoneroData\block_range.raw"

REM --- Block range to export ---
SET BLOCK_START=3000000
SET BLOCK_STOP=3030000

echo ============================================================
echo Exporting Monero block range: %BLOCK_START% to %BLOCK_STOP%
echo Source folder: %MONERO_BIN%
echo Output file:   %OUTPUT_FILE%
echo ============================================================
echo.

%MONERO_BIN%\monero-blockchain-export.exe ^
    --block-start %BLOCK_START% ^
    --block-stop %BLOCK_STOP% ^
    --blocks-file %OUTPUT_FILE%

echo.
echo Export completed.
pause
