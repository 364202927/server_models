@echo off
rem ??????????????????????????????????
rem Keep this file CRLF + ASCII only, otherwise cmd.exe will silently skip it.
setlocal
if /i not "%~1"=="SM-6f1a9c3e8b2d4705" exit /b 1
set "DISTRO=Ubuntu-22.04"

wsl.exe -d %DISTRO% -- bash -lc "pkill -f 'server_models.main'"
set "CODE=%ERRORLEVEL%"

echo ===== stop_backend.bat: pkill server_models.main, code %CODE% =====
exit /b %CODE%
