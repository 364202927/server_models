@echo off
rem ??????????????????????????????????
rem Keep this file CRLF + ASCII only, otherwise cmd.exe will silently skip it.
setlocal
if /i not "%~1"=="SM-6f1a9c3e8b2d4705" exit /b 1
set "DISTRO=Ubuntu-22.04"
set "CONDA_ENV=work"
set "WSL_CMD=conda activate %CONDA_ENV% && cd /mnt/f/code && exec python -u -m server_models.main"

echo ===== start_backend.bat invoked =====
wsl.exe -d %DISTRO% -- bash -lic "%WSL_CMD%"
exit /b %ERRORLEVEL%
