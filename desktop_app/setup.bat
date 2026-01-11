@echo off
REM Setup script for Financial Agent Desktop App
REM This script sets up the Electron environment

echo Setting up Financial Agent Desktop App...

REM Install dependencies in desktop_app
echo Installing desktop app dependencies...
call npm install
if errorlevel 1 goto error

REM Install dependencies in frontend if not already installed
echo Installing frontend dependencies...
cd ..\frontend
if not exist "node_modules" (
  call npm install
  if errorlevel 1 goto error
)

REM Build frontend for production
echo Building frontend...
call npm run build
if errorlevel 1 goto error

REM Go back to desktop_app
cd ..\desktop_app

echo.
echo Setup complete!
echo.
echo To start development:
echo   npm run dev
echo.
echo To build for distribution:
echo   npm run dist
goto end

:error
echo.
echo Error occurred during setup!
pause
exit /b 1

:end
pause
