@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Tracker UAI

REM ============================================================
REM  Tracker UAI - Lanzador para Windows
REM
REM  Doble clic para iniciar. La primera vez prepara todo solo.
REM
REM  NUNCA pide permisos de administrador: todo se instala dentro
REM  de tu cuenta de usuario, en %LOCALAPPDATA%\TrackerUAI.
REM  Si no encuentra Python, ofrece descargar una copia portable
REM  desde python.org (no toca el PATH ni el resto del sistema).
REM
REM  Uso avanzado:
REM    Iniciar_Tracker_UAI.bat --portable   fuerza la copia portable
REM    Iniciar_Tracker_UAI.bat --reparar    rehace el entorno desde cero
REM ============================================================

set "BASE=%LOCALAPPDATA%\TrackerUAI"
set "VENVDIR=%BASE%\.venv"
set "PORTDIR=%BASE%\python"
set "PYVER=3.11.9"
set "PY="
set "SYSPY="

REM Evita que variables heredadas rompan el interprete portable.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONSTARTUP="

echo.
echo   ==================================================
echo    Tracker UAI - Analisis de video para laboratorios
echo   ==================================================
echo.

if /i "%~1"=="--reparar" (
    echo Modo reparacion: borrando el entorno anterior...
    if exist "%VENVDIR%" rmdir /s /q "%VENVDIR%"
    if exist "%PORTDIR%" rmdir /s /q "%PORTDIR%"
    echo.
)

if /i "%~1"=="--portable" goto :instalar_portable

REM --- 1) Interprete ya preparado en una ejecucion anterior ----
if exist "%VENVDIR%\Scripts\python.exe" set "PY=%VENVDIR%\Scripts\python.exe"
if not defined PY if exist "%PORTDIR%\python.exe" set "PY=%PORTDIR%\python.exe"
if defined PY goto :dependencias

REM --- 2) Buscar un Python ya instalado en el equipo -----------
echo Buscando Python en tu equipo...
call :probar py -3
if not defined SYSPY call :probar python
if not defined SYSPY (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%~fD\python.exe" call :probar "%%~fD\python.exe"
    )
)
if defined SYSPY goto :crear_venv

echo.
echo No se encontro Python 3.9 o superior en tu equipo.
echo.
echo Tracker UAI puede instalar su propia copia de Python, portable
echo y solo para tu usuario:
echo    - NO necesita permisos de administrador
echo    - NO modifica el PATH ni afecta a otros programas
echo    - Se guarda en "%PORTDIR%" y se borra con esa carpeta
echo    - Descarga unos 11 MB desde el sitio oficial python.org
echo.
set "RESP="
set /p "RESP=Instalar la copia portable ahora? [S/n]: "
if /i "!RESP!"=="n" goto :err_sin_python

:instalar_portable
call :portable
if errorlevel 1 goto :err_portable
set "PY=%PORTDIR%\python.exe"
goto :dependencias

REM --- 3) Entorno virtual sobre el Python del sistema ----------
:crear_venv
echo Python encontrado: !SYSPY!
echo Preparando el entorno de trabajo (solo la primera vez)...
!SYSPY! -m venv "%VENVDIR%"
if errorlevel 1 goto :err_venv
set "PY=%VENVDIR%\Scripts\python.exe"
if not exist "%PY%" goto :err_venv

REM --- 4) Dependencias -----------------------------------------
:dependencias
"%PY%" -c "import streamlit, streamlit_image_coordinates, cv2, numpy, scipy, plotly, pandas" >nul 2>&1
if not errorlevel 1 goto :ejecutar

echo.
echo Instalando las librerias necesarias. Esto tarda varios minutos
echo la primera vez; despues el programa abre en segundos.
echo.
"%PY%" -m pip install --upgrade pip --disable-pip-version-check --no-warn-script-location
"%PY%" -m pip install -r requirements.txt --disable-pip-version-check --no-warn-script-location
if errorlevel 1 goto :err_dependencias
"%PY%" -c "import streamlit, cv2" >nul 2>&1
if errorlevel 1 goto :err_dependencias

REM --- 5) Arranque ---------------------------------------------
:ejecutar
echo.
echo Iniciando Tracker UAI... se abrira solo en tu navegador.
echo Para CERRAR el programa, cierra esta ventana.
echo.
"%PY%" -m streamlit run app.py
goto :fin


REM ============================================================
REM  Subrutinas
REM ============================================================

:probar
REM %* = comando candidato (con comillas si es una ruta con espacios).
if defined SYSPY exit /b 0
%* -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "SYSPY=%*"
exit /b 0


:portable
if /i not "%PROCESSOR_ARCHITECTURE%"=="AMD64" if /i not "%PROCESSOR_ARCHITEW6432%"=="AMD64" (
    echo [ERROR] La copia portable solo esta disponible para Windows de 64 bits.
    exit /b 1
)
set "ZIP=%TEMP%\trackeruai_python.zip"
set "GETPIP=%TEMP%\trackeruai_get-pip.py"
if not exist "%BASE%" mkdir "%BASE%"
if exist "%PORTDIR%" rmdir /s /q "%PORTDIR%"

echo.
echo   [1/4] Descargando Python %PYVER% desde python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip' -OutFile '%ZIP%'"
if errorlevel 1 exit /b 1

echo   [2/4] Descomprimiendo...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '%ZIP%' -DestinationPath '%PORTDIR%' -Force"
if errorlevel 1 exit /b 1
if not exist "%PORTDIR%\python.exe" exit /b 1

echo   [3/4] Habilitando la instalacion de librerias...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $f=Get-ChildItem -LiteralPath '%PORTDIR%' -Filter 'python*._pth' | Select-Object -First 1; if(-not $f){throw 'No se encontro el archivo ._pth'}; $c=Get-Content -LiteralPath $f.FullName; $c=$c -replace '^#\s*import site','import site'; if($c -notcontains 'Lib\site-packages'){$c=@($c)+'Lib\site-packages'}; Set-Content -LiteralPath $f.FullName -Value $c -Encoding ascii"
if errorlevel 1 exit /b 1

echo   [4/4] Instalando pip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%'"
if errorlevel 1 exit /b 1
"%PORTDIR%\python.exe" "%GETPIP%" --no-warn-script-location --disable-pip-version-check
if errorlevel 1 exit /b 1
"%PORTDIR%\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 exit /b 1

del /q "%ZIP%" >nul 2>&1
del /q "%GETPIP%" >nul 2>&1
echo   Copia portable lista en "%PORTDIR%".
echo.
exit /b 0


REM ============================================================
REM  Mensajes de error
REM ============================================================

:err_sin_python
echo.
echo [ERROR] No se puede continuar sin Python.
echo.
echo Tienes dos caminos:
echo   a^) Volver a ejecutar este archivo y aceptar la copia portable.
echo   b^) Instalar Python 3.11 o superior desde python.org.
echo      En el instalador, deja marcado "Install for me only"
echo      ^(o "Solo para mi"^); asi NO te pedira permisos de
echo      administrador. La casilla "Add python.exe to PATH" ayuda,
echo      pero no es obligatoria.
echo.
goto :fin

:err_portable
echo.
echo [ERROR] No se pudo preparar la copia portable de Python.
echo.
echo Causas mas frecuentes:
echo   - Sin conexion a internet, o la red del campus bloquea
echo     python.org / bootstrap.pypa.io.
echo   - Un antivirus corporativo bloqueo la descarga.
echo.
echo Prueba con otra red, o instala Python 3.11 desde python.org
echo eligiendo "Install for me only" ^(no requiere administrador^) y
echo vuelve a ejecutar este archivo.
echo.
goto :fin

:err_venv
echo.
echo [ERROR] No se pudo crear el entorno de trabajo con el Python
echo         encontrado: !SYSPY!
echo.
echo Solucion recomendada: ejecuta este archivo forzando la copia
echo portable, que es independiente de tu instalacion de Python.
echo   1. Abre la carpeta de Tracker UAI.
echo   2. Manten SHIFT y haz clic derecho en un espacio vacio.
echo   3. Elige "Abrir la ventana de PowerShell aqui".
echo   4. Escribe:   .\Iniciar_Tracker_UAI.bat --portable
echo.
goto :fin

:err_dependencias
echo.
echo [ERROR] Fallo la instalacion de las librerias.
echo.
echo Causas mas frecuentes:
echo   - Conexion a internet caida o inestable.
echo   - La red del campus bloquea pypi.org.
echo   - Una instalacion anterior quedo a medias.
echo.
echo Para empezar de cero ^(borra solo el entorno, no tus datos^):
echo   1. Manten SHIFT y haz clic derecho en esta carpeta.
echo   2. Elige "Abrir la ventana de PowerShell aqui".
echo   3. Escribe:   .\Iniciar_Tracker_UAI.bat --reparar
echo.
goto :fin

:fin
echo.
pause
endlocal
