@echo off
REM ============================================================
REM  Tracker UAI - Lanzador para Windows
REM  Doble clic para iniciar. La primera vez instala todo solo.
REM ============================================================
cd /d "%~dp0"
title Tracker UAI

REM --- Verifica que Python este instalado ---
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] No se encontro Python.
    echo Instala Python 3.11 o superior desde https://www.python.org/downloads/
    echo marcando "Add Python to PATH", y vuelve a ejecutar este archivo.
    echo.
    pause
    exit /b 1
)

REM --- Entorno virtual FUERA de Dropbox (evita bloqueos de sincronizacion) ---
set "VENVDIR=%LOCALAPPDATA%\TrackerUAI\.venv"

if not exist "%VENVDIR%\Scripts\activate.bat" (
    echo Primera ejecucion: preparando el entorno. Puede tardar varios minutos...
    python -m venv "%VENVDIR%"
    if errorlevel 1 ( echo [ERROR] No se pudo crear el entorno virtual. & pause & exit /b 1 )
    call "%VENVDIR%\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 ( echo [ERROR] Fallo la instalacion de dependencias. Revisa tu conexion. & pause & exit /b 1 )
) else (
    call "%VENVDIR%\Scripts\activate.bat"
)

echo.
echo Iniciando Tracker UAI... se abrira en tu navegador.
echo Para cerrar el programa, cierra esta ventana.
echo.
python -m streamlit run app.py

pause
