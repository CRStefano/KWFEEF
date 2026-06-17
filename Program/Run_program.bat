@echo off
chcp 65001 >nul
title S-FEEF - Semantic Information Dashboard
cd /d "%~dp0"

echo ============================================================
echo    S-FEEF  -  Semantic Information ^& Foraging Dashboard
echo ============================================================
echo.

REM --- find Python (py launcher or python) ---
where py >nul 2>nul && (set "PY=py") || (set "PY=python")
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on this system.
  echo Install Python 3.11+ from https://www.python.org/downloads/
  echo during installation, tick "Add Python to PATH".
  echo.
  pause
  exit /b 1
)

echo [1/3] Checking and installing dependencies ^(nicegui, numpy, plotly, scipy^)...
%PY% -m pip install --quiet --disable-pip-version-check nicegui numpy plotly scipy

echo [2/3] Starting the server. The browser will open in a few seconds.
start "" /b cmd /c "timeout /t 5 >nul && start http://localhost:5000"

echo [3/3] Program running at http://localhost:5000
echo       To stop it: close this window or press Ctrl+C.
echo.
%PY% main.py

echo.
echo Program finished.
pause
