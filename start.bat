@echo off
setlocal enabledelayedexpansion
title Ultima

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "ULTIMA_EXE=%SCRIPT_DIR%build\windows-msvc-release\bin\Release\ultima.exe"

REM ---- Build if needed -------------------------------------------------------
if exist "%ULTIMA_EXE%" goto menu
call :build || exit /b 1

REM ---- Interactive menu ------------------------------------------------------
:menu
cls
echo.
echo ============================================================
echo   Ultima - independent local LLM runtime
echo   (temporary batch launcher, replaced by Go/Wails launcher)
echo ============================================================
echo.
"%ULTIMA_EXE%" --version
echo.
echo   [V]  Show version
echo   [H]  Show help
echo   [T]  Run tests
echo   [B]  Rebuild
echo   [S]  Open a shell in this project (cmd)
echo   [E]  Open Windows Explorer here
echo   [Q]  Quit
echo.
set /p "CHOICE=Choose: "
echo.

if not defined CHOICE goto menu
if "%CHOICE%"==""    goto menu

if /i "%CHOICE%"=="V" ( "%ULTIMA_EXE%" --version & pause & goto menu )
if /i "%CHOICE%"=="H" ( "%ULTIMA_EXE%" --help    & pause & goto menu )
if /i "%CHOICE%"=="T" ( call :test              & pause & goto menu )
if /i "%CHOICE%"=="B" ( call :build             & pause & goto menu )
if /i "%CHOICE%"=="S" ( start "" cmd /k "cd /d %SCRIPT_DIR%"     & goto menu )
if /i "%CHOICE%"=="E" ( start "" explorer "%SCRIPT_DIR%"          & goto menu )
if /i "%CHOICE%"=="Q" ( exit /b 0 )

echo Unknown choice: %CHOICE%
pause
goto menu

REM ---- :build ----------------------------------------------------------------
:build
echo Locating Visual Studio 2022...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [!] vswhere.exe not found - Visual Studio 2022 does not appear to be installed.
    echo     Install VS 2022 ^(Community or Build Tools^) with "Desktop development with C++":
    echo     https://visualstudio.microsoft.com/downloads/
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -requires Microsoft.Component.MSBuild -property installationPath`) do (
    set "VS_INSTALL=%%i"
)

if "%VS_INSTALL%"=="" (
    echo [!] Could not locate a Visual Studio 2022 installation with MSBuild.
    exit /b 1
)

echo Loading VS 2022 environment from:
echo   %VS_INSTALL%
call "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

echo.
echo === Configure ===
cmake --preset windows-msvc-release || exit /b 1

echo.
echo === Build ===
cmake --build --preset windows-msvc-release || exit /b 1

echo.
echo Build succeeded.
exit /b 0

REM ---- :test -----------------------------------------------------------------
:test
if not exist "%ULTIMA_EXE%" (
    echo [!] Not built yet.
    exit /b 1
)
ctest --preset windows-msvc-release
exit /b 0
