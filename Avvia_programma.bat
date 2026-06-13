@echo off
chcp 65001 >nul
title S-FEEF - Semantic Information Dashboard
cd /d "%~dp0"

echo ============================================================
echo    S-FEEF  -  Semantic Information ^& Foraging Dashboard
echo ============================================================
echo.

REM --- trova Python (py launcher oppure python) ---
where py >nul 2>nul && (set "PY=py") || (set "PY=python")
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo [ERRORE] Python non trovato sul sistema.
  echo Installa Python 3.11+ da https://www.python.org/downloads/
  echo durante l'installazione spunta "Add Python to PATH".
  echo.
  pause
  exit /b 1
)

echo [1/3] Controllo e installazione dipendenze ^(nicegui, numpy, plotly, scipy^)...
%PY% -m pip install --quiet --disable-pip-version-check nicegui numpy plotly scipy

echo [2/3] Avvio del server. Il browser si aprira' tra pochi secondi.
start "" /b cmd /c "timeout /t 5 >nul && start http://localhost:5000"

echo [3/3] Programma in esecuzione su http://localhost:5000
echo       Per fermarlo: chiudi questa finestra oppure premi Ctrl+C.
echo.
%PY% main.py

echo.
echo Programma terminato.
pause
