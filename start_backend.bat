@echo off
rem Start WSL backend: conda 'work' + python -m server_models.main
rem Double-click to run manually (output shows in this window);
rem the desktop app calls this file and captures the output into backend.log.
rem Keep this file CRLF + ASCII only, otherwise cmd.exe will silently skip it.
setlocal
set "DISTRO=Ubuntu-22.04"
set "CONDA_ENV=work"
set "WSL_CMD=conda activate %CONDA_ENV% && cd /mnt/f/code && exec python -u -m server_models.main"

echo ===== start_backend.bat invoked =====
wsl.exe -d %DISTRO% -- bash -lic "%WSL_CMD%"

if "%~1"=="--manual" pause
exit /b %ERRORLEVEL%
