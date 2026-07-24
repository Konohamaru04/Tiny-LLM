@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
set "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu130"

if defined TINY_LLM_TORCH_INDEX_URL (
    set "TORCH_CUDA_INDEX_URL=%TINY_LLM_TORCH_INDEX_URL%"
)

echo.
echo  Tiny-LLM Training Dashboard
echo  ===========================
echo.

if not exist "%VENV_PYTHON%" (
    echo [1/4] Creating virtual environment at "%VENV_DIR%"...

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
    echo [1/4] Using virtual environment at "%VENV_DIR%".
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
    echo [2/4] CUDA PyTorch check skipped by TINY_LLM_SKIP_INSTALL.
) else (
    echo [2/4] Verifying CUDA-enabled PyTorch...

    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] NVIDIA driver tools were not found on PATH.
        echo This launcher requires an NVIDIA GPU and current NVIDIA driver.
        exit /b 1
    )

    python -c "import sys, torch; sys.exit(0 if torch.version.cuda and torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [CUDA] CPU-only or unusable PyTorch detected.
        echo [CUDA] Installing from "%TORCH_CUDA_INDEX_URL%"...
        python -m pip install --disable-pip-version-check --upgrade --force-reinstall --no-deps --index-url "%TORCH_CUDA_INDEX_URL%" "torch>=2.6.0,<3.0.0"
        if errorlevel 1 (
            echo [ERROR] CUDA-enabled PyTorch installation failed.
            exit /b 1
        )
    ) else (
        echo [CUDA] Existing CUDA-enabled PyTorch installation is ready.
    )
)

if /I "%TINY_LLM_SKIP_INSTALL%"=="1" (
    echo [3/4] Dependency installation skipped by TINY_LLM_SKIP_INSTALL.
) else (
    echo [3/4] Installing any missing or incompatible dependencies...
    python -m pip install --disable-pip-version-check -r "%CD%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        exit /b 1
    )

    python -c "import sys, torch; ok=bool(torch.version.cuda and torch.cuda.is_available()); print('[CUDA] torch={}  runtime={}  device={}'.format(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if ok else 'UNAVAILABLE')); sys.exit(0 if ok else 1)"
    if errorlevel 1 (
        echo [ERROR] PyTorch still cannot use CUDA after dependency installation.
        echo Refusing to start long-context training on CPU.
        exit /b 1
    )
)

echo [4/4] Launching the Gradio training dashboard...
echo        http://127.0.0.1:7860
echo.

python "%CD%\scripts\training_dashboard.py" --inbrowser %*
set "DASHBOARD_EXIT_CODE=%ERRORLEVEL%"

if not "%DASHBOARD_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] The dashboard exited with code %DASHBOARD_EXIT_CODE%.
)

exit /b %DASHBOARD_EXIT_CODE%
