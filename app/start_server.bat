@echo off
setlocal enabledelayedexpansion
title JT Segmenter - Mizo Sentence Segmentation

REM ============================================================
REM JT Segmenter launcher
REM Edit CONDA_ENV below if your environment is named differently.
REM ============================================================

set "CONDA_ENV=mizen"
set "APP_FILE=app.py"
set "PORT=5000"

REM ---- Move to this script's folder ----
cd /d "%~dp0"

echo.
echo ============================================================
echo   JT Segmenter - starting up
echo   Folder : %CD%
echo   Env    : %CONDA_ENV%
echo ============================================================
echo.

REM ---- Sanity check: app.py present? ----
if not exist "%APP_FILE%" (
    echo [ERROR] %APP_FILE% not found in this folder.
    echo         Make sure run_app.bat sits next to app.py.
    goto :pause_and_exit
)

REM ---- Sanity check: port already in use? ----
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [ERROR] Port %PORT% is already in use.
    echo         JT Segmenter may already be running in another window.
    echo         Open http://127.0.0.1:%PORT% in your browser to check,
    echo         or close the existing server and try again.
    goto :pause_and_exit
)

REM ---- Locate conda ----
set "CONDA_HOOK="
for %%P in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%USERPROFILE%\AppData\Local\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\AppData\Local\anaconda3\condabin\conda.bat"
    "%PROGRAMDATA%\miniconda3\condabin\conda.bat"
    "%PROGRAMDATA%\anaconda3\condabin\conda.bat"
    "C:\miniconda3\condabin\conda.bat"
    "C:\anaconda3\condabin\conda.bat"
) do (
    if exist "%%~P" (
        set "CONDA_HOOK=%%~P"
        goto :found_conda
    )
)

echo [ERROR] Could not locate conda.bat in any standard location.
echo         Edit run_app.bat and add the full path to your conda.bat
echo         to the search list, or open Anaconda Prompt manually and run:
echo             conda activate %CONDA_ENV%
echo             python %APP_FILE%
goto :pause_and_exit

:found_conda
echo Using conda at: %CONDA_HOOK%
echo.

REM ---- Activate the env ----
call "%CONDA_HOOK%" activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env "%CONDA_ENV%".
    echo         Check that the environment exists:  conda env list
    goto :pause_and_exit
)

REM ---- Open the browser after a short delay (server warm-up) ----
start "" /B cmd /c "timeout /t 8 >nul & start http://127.0.0.1:%PORT%"

REM ---- Run the app ----
echo Starting Flask server on http://127.0.0.1:%PORT%
echo (Stop the server with Ctrl+C, then close this window.)
echo.
python "%APP_FILE%"

REM ---- If python exits, pause so user can read any error ----
echo.
echo ============================================================
echo   Server stopped.
echo ============================================================

:pause_and_exit
echo.
pause
endlocal
exit /b