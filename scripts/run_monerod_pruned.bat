@echo off
REM ============================================================
REM  Start Monero Daemon in PRUNED mode (FINAL PATH VERSION)
REM ============================================================

SET MONERO_BIN="C:\Monero\monero-x86_64-w64-mingw32-v0.18.4.4"
SET DATA_DIR="C:\MoneroData"

echo ============================================================
echo Starting monerod in PRUNED mode
echo Monero binary: %MONERO_BIN%
echo Data directory: %DATA_DIR%
echo ============================================================
echo.

%MONERO_BIN%\monerod.exe --data-dir %DATA_DIR% --prune-blockchain

echo.
echo monerod stopped.
pause
