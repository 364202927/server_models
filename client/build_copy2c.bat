@echo off
setlocal EnableExtensions
rem build_copy2c.bat - build the release exe and copy it to the installed location on C:.
rem Keep this file CRLF + ASCII only, otherwise cmd.exe may silently skip it.
rem Usage: build_copy2c.bat [-y]   (-y = do not pause at the end)

if /i "%~1"=="-y" set "NO_PAUSE=1"
set "PROJECT_DIR=%~dp0"
set "BUILD_EXE=%PROJECT_DIR%src-tauri\target\release\server-models-client.exe"
set "DEPLOY_DIR=%LOCALAPPDATA%\Programs\ServerModels"
set "DEPLOY_EXE=%DEPLOY_DIR%\SERVER MODELS.exe"

if not exist "%DEPLOY_DIR%" (
  echo [ERROR] install dir not found: %DEPLOY_DIR%
  goto :fail
)

rem A running app locks its own exe, so refuse before spending a minute on the build.
tasklist /fo csv 2>nul | findstr /i /c:"SERVER MODELS.exe" >nul
if not errorlevel 1 (
  echo [ERROR] SERVER MODELS.exe is running, close it first.
  goto :fail
)

if not exist "%PROJECT_DIR%node_modules\@tauri-apps\cli\tauri.js" (
  echo [ERROR] node_modules missing, run npm install in %PROJECT_DIR% first.
  goto :fail
)

rem Use a real Node.js binary: the PATH node may be a desktop shim that breaks the CLI argv.
set "NODE_BIN=C:\Program Files\nodejs\node.exe"
if not exist "%NODE_BIN%" set "NODE_BIN=node"

echo === build release exe ===
pushd "%PROJECT_DIR%"
"%NODE_BIN%" "%PROJECT_DIR%node_modules\@tauri-apps\cli\tauri.js" build --no-bundle
if errorlevel 1 (
  popd
  echo [ERROR] build failed
  goto :fail
)
popd

if not exist "%BUILD_EXE%" (
  echo [ERROR] build output missing: %BUILD_EXE%
  goto :fail
)

echo === copy to %DEPLOY_EXE% ===
copy /y "%BUILD_EXE%" "%DEPLOY_EXE%" >nul
if errorlevel 1 (
  echo [ERROR] copy failed, is the app still running?
  goto :fail
)

rem Byte-compare the copy so a half-written exe never passes as success.
fc /b "%BUILD_EXE%" "%DEPLOY_EXE%" >nul
if errorlevel 1 (
  echo [ERROR] deployed file differs from the build output.
  goto :fail
)

for %%F in ("%BUILD_EXE%") do echo [OK] build    %%~zF bytes  %%~tF
for %%F in ("%DEPLOY_EXE%") do echo [OK] deployed %%~zF bytes  %%~tF
echo Done. Backend changes still need a restart of the 8666 service.
goto :end

:fail
echo.
echo FAILED - nothing was deployed.
if not defined NO_PAUSE pause
exit /b 1

:end
if not defined NO_PAUSE pause
exit /b 0
