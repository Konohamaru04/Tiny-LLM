@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

echo.
echo  Tiny-LLM Training Dashboard
echo  ===========================
echo.

if not exist "%VENV_PYTHON%" (
    echo [1/3] Creating virtual environment at "%VENV_DIR%"...

    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo [ERROR] Python was not found on PATH.
            echo Install Python 3.10 or newer, then run this launcher again.
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    )

    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        exit /b 1
    )
) else (
    echo [1/3] Using virtual environment at "%VENV_DIR%".
)

if not exist "%VENV_ACTIVATE%" (
    echo [ERROR] Virtual-environment activation script is missing:
    echo         "%VENV_ACTIVATE%"
    exit /b 1
)

call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo [ERROR] Could not activate the virtual environment.
    exit /b 1
)

if /I "%TINY_LLM_SKIP_INSTALL%"=="1" (
    echo [2/3] Dependency installation skipped by TINY_LLM_SKIP_INSTALL.
) else (
    echo [2/3] Installing any missing or incompatible dependencies...
    python -m pip install --disable-pip-version-check -r "%CD%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        exit /b 1
    )
)

echo [3/3] Launching the Gradio training dashboard...
echo        http://127.0.0.1:7860
echo.

python "%CD%\scripts\training_dashboard.py" --inbrowser %*
set "DASHBOARD_EXIT_CODE=%ERRORLEVEL%"

if not "%DASHBOARD_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] The dashboard exited with code %DASHBOARD_EXIT_CODE%.
)

exit /b %DASHBOARD_EXIT_CODE%
