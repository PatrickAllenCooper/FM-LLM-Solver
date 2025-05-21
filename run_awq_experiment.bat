@echo off
:: FM-LLM-Solver AWQ Experiment Launcher
:: This batch file activates conda and runs the AWQ experiment

echo === FM-LLM-Solver AWQ Experiment Launcher ===

:: Always skip KB check since we don't have the knowledge base yet
set SKIP_KB=--skip-kb
echo Will skip knowledge base check (using --skip-kb flag)

:: Define model choice
set MODEL_CHOICE=1
echo Will use model choice: %MODEL_CHOICE% (Qwen/Qwen2.5-7B-Instruct-AWQ)

:: Try to find conda in common locations
set CONDA_CMD=

if exist "C:\Users\patri\anaconda3\Scripts\activate.bat" (
    set CONDA_CMD=C:\Users\patri\anaconda3\Scripts\activate.bat
) else if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
    set CONDA_CMD=C:\ProgramData\Anaconda3\Scripts\activate.bat
) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set CONDA_CMD=C:\ProgramData\miniconda3\Scripts\activate.bat
)

if not defined CONDA_CMD (
    echo ERROR: Conda activation script not found. Please activate conda manually before running this script.
    exit /b 1
)

echo Conda found at: %CONDA_CMD%

:: Determine which environment to use
set CONDA_ENV=

if exist "C:\Users\patri\anaconda3\envs\llmfm_cuda" (
    set CONDA_ENV=llmfm_cuda
) else if exist "C:\Users\patri\anaconda3\envs\llmfm" (
    set CONDA_ENV=llmfm
) else (
    set CONDA_ENV=base
)

echo Using conda environment: %CONDA_ENV%

:: Activate conda and run the experiment
echo Activating conda environment...
call %CONDA_CMD% %CONDA_ENV%

:: Display environment version info
echo.
echo === Python Environment Information ===
python -V
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); import transformers; print(f'Transformers: {transformers.__version__}')"

:: Check if CUDA is available and install PyTorch with CUDA if it's not
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo === CUDA NOT DETECTED IN PYTORCH ===
    echo Reinstalling PyTorch with CUDA support...
    
    :: First uninstall current PyTorch
    pip uninstall -y torch torchvision torchaudio
    
    :: Install PyTorch with CUDA 11.8 support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    :: Check if installation was successful
    echo.
    echo === Checking new PyTorch installation ===
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"
    
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Failed to install PyTorch with CUDA support.
        echo Please try installing it manually with:
        echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        exit /b 1
    )
)

:: Get PyTorch version for compatibility check
for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__.split('+')[0])"') do set TORCH_VERSION=%%i
echo Current PyTorch version: %TORCH_VERSION%

:: Check for compatibility between PyTorch and AutoAWQ
echo.
echo === Checking AutoAWQ compatibility with PyTorch ===
python -c "import torch; import awq; print('Compatible')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Updating AutoAWQ for compatibility with PyTorch %TORCH_VERSION%...
    
    :: Uninstall current autoawq and reinstall with the latest version
    pip uninstall -y autoawq autoawq-kernels
    pip install autoawq --no-cache-dir
    
    :: Check if installation was successful
    python -c "import awq; print(f'AutoAWQ version: {awq.__version__}')" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: AutoAWQ might not be fully compatible with the current PyTorch version.
        echo The script will continue, but you might encounter issues.
    ) else (
        echo AutoAWQ successfully updated and is compatible with the current PyTorch version.
    )
)

echo.
echo === CUDA is available, proceeding with experiment ===

:: Direct execution with preset model choice parameter
echo Automatically selecting model option 1: Qwen/Qwen2.5-7B-Instruct-AWQ
call scripts\experiments\run_quantized_qwen.bat %MODEL_CHOICE% %SKIP_KB%

echo.
echo Experiment completed. 