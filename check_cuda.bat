@echo off
:: Check if CUDA is available
echo === Checking CUDA Availability ===

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

:: Activate conda and check CUDA
call %CONDA_CMD% %CONDA_ENV%

echo.
echo === Python Environment Information ===
python -V
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \"N/A\"}')"
echo.

echo === Running nvidia-smi ===
nvidia-smi
echo.

echo === CUDA Check Complete === 