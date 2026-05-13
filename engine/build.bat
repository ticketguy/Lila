@echo off
REM ═══════════════════════════════════════════════════════════════════════════════
REM  Lila Engine — Windows Build Script
REM
REM  Requirements: GCC (MinGW-w64) on PATH
REM    Install via: https://www.msys2.org/ then: pacman -S mingw-w64-x86_64-gcc
REM    Or: choco install mingw
REM    Or: winget install -e --id GnuWin32.Make (for make)
REM
REM  Usage: Just double-click this file, or run from PowerShell/cmd:
REM    .\build.bat
REM ═══════════════════════════════════════════════════════════════════════════════

echo.
echo   Building lila-asi.exe ...
echo.

REM Check if gcc is available
where gcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   ERROR: gcc not found on PATH.
    echo   Install MinGW-w64: https://www.msys2.org/
    echo   Then add C:\msys64\mingw64\bin to your PATH
    pause
    exit /b 1
)

set CC=gcc
set CFLAGS=-O2 -Wall -std=c11 -D_WIN32 -DWIN32_LEAN_AND_MEAN
set INCLUDES=-I runtime/ -I asi/

REM Compile runtime
echo   [1/7] runtime/model.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/model.o runtime/model.c

echo   [2/7] runtime/inference.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/inference.o runtime/inference.c

echo   [3/7] runtime/attention.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/attention.o runtime/attention.c

echo   [4/7] runtime/transformer.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/transformer.o runtime/transformer.c

echo   [5/7] runtime/tokenizer.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/tokenizer.o runtime/tokenizer.c

echo   [6/7] runtime/detect.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/detect.o runtime/detect.c

echo   [7/7] runtime/dispatch.c
%CC% %CFLAGS% %INCLUDES% -c -o runtime/dispatch.o runtime/dispatch.c

REM Compile ASI
echo   [+] asi/asi_runtime.c
%CC% %CFLAGS% %INCLUDES% -c -o asi/asi_runtime.o asi/asi_runtime.c

echo   [+] asi/lilavm.c
%CC% %CFLAGS% %INCLUDES% -c -o asi/lilavm.o asi/lilavm.c

echo   [+] asi/asi_cli.c
%CC% %CFLAGS% %INCLUDES% -c -o asi/asi_cli.o asi/asi_cli.c

REM Link
echo.
echo   Linking...
%CC% %CFLAGS% -o lila-asi.exe ^
    runtime/model.o runtime/inference.o runtime/attention.o ^
    runtime/transformer.o runtime/tokenizer.o runtime/detect.o ^
    runtime/dispatch.o asi/asi_runtime.o asi/lilavm.o asi/asi_cli.o ^
    -lm

if %ERRORLEVEL% neq 0 (
    echo.
    echo   BUILD FAILED
    pause
    exit /b 1
)

echo.
echo   ======================================
echo     Built: lila-asi.exe
echo     Run:   lila-asi.exe lila.asi
echo   ======================================
echo.
pause
