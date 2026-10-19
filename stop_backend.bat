@echo off
rem Stop only the backend python process; the distro (and your other WSL sessions) stay alive.
rem Double-click to run manually; the desktop app also calls this file.
rem Keep this file CRLF + ASCII only, otherwise cmd.exe will silently skip it.
setlocal
set "DISTRO=Ubuntu-22.04"

wsl.exe -d %DISTRO% -- bash -lc "pkill -f 'server_models.main'"
set "CODE=%ERRORLEVEL%"

echo ===== stop_backend.bat: pkill server_models.main, code %CODE% =====

if "%~1"=="--manual" pause
exit /b %CODE%
