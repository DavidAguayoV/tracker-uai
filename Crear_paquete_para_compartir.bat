@echo off
REM ============================================================
REM  Crea un ZIP limpio de Tracker UAI para enviar a estudiantes
REM  (excluye .venv, __pycache__, videos y archivos temporales)
REM ============================================================
cd /d "%~dp0"
title Crear paquete Tracker UAI

set "STAGE=%TEMP%\TrackerUAI_pkg"
set "OUT=%~dp0TrackerUAI_compartir.zip"

if exist "%STAGE%" rmdir /s /q "%STAGE%"
mkdir "%STAGE%\TrackerUAI"

REM --- Copia solo lo necesario ---
copy /y "app.py"            "%STAGE%\TrackerUAI\" >nul
copy /y "export.py"         "%STAGE%\TrackerUAI\" >nul
copy /y "requirements.txt"  "%STAGE%\TrackerUAI\" >nul
copy /y "README.md"         "%STAGE%\TrackerUAI\" >nul
copy /y "LEEME_ESTUDIANTES.txt" "%STAGE%\TrackerUAI\" >nul
copy /y "MANUAL_DE_USO.md"   "%STAGE%\TrackerUAI\" >nul
copy /y "Iniciar_Tracker_UAI.bat" "%STAGE%\TrackerUAI\" >nul
copy /y "iniciar_tracker.sh" "%STAGE%\TrackerUAI\" >nul
xcopy "core"    "%STAGE%\TrackerUAI\core\"    /e /i /q >nul
xcopy "models"  "%STAGE%\TrackerUAI\models\"  /e /i /q >nul
xcopy "tests"   "%STAGE%\TrackerUAI\tests\"   /e /i /q >nul
xcopy ".streamlit" "%STAGE%\TrackerUAI\.streamlit\" /e /i /q >nul

REM --- Limpia cachés que se hayan colado ---
for /d /r "%STAGE%" %%d in (__pycache__) do if exist "%%d" rmdir /s /q "%%d"

REM --- Comprime ---
if exist "%OUT%" del /q "%OUT%"
powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%\TrackerUAI' -DestinationPath '%OUT%' -Force"

rmdir /s /q "%STAGE%"
echo.
echo Listo. Paquete creado en:
echo   %OUT%
echo Enviaselo a tus estudiantes y que sigan LEEME_ESTUDIANTES.txt
echo.
pause
