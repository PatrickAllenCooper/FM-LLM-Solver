@echo off
:: FM-LLM-Solver Fine-tuning Diagnostic Tool
:: This batch script runs diagnostics and fixes for fine-tuning issues

echo === FM-LLM-Solver Fine-tuning Diagnostics ===
echo This tool will diagnose and fix issues with fine-tuning on your system.

:: Set working directory to project root
cd ..\..

:: Check if huggingface_hub is installed
python -c "import huggingface_hub" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    pip install huggingface_hub psutil
)

:: Check command-line arguments
set FIX=false
set TEST=false

if "%1"=="--fix" (
    set FIX=true
    shift
)

if "%1"=="--test" (
    set TEST=true
    shift
)

if "%1"=="--help" (
    echo.
    echo Usage: diagnose_finetune.bat [options]
    echo.
    echo Options:
    echo   --fix    Apply automatic fixes to config.yaml
    echo   --test   Run a minimal fine-tuning test
    echo   --all    Apply fixes and run diagnostic test
    echo   --help   Display this help message
    echo.
    exit /b 0
)

if "%1"=="--all" (
    set FIX=true
    set TEST=true
)

:: Run the diagnostic script with appropriate flags
echo.
echo Running diagnostics...

if "%FIX%"=="true" (
    if "%TEST%"=="true" (
        python scripts/optimization/diagnose_and_fix_finetune.py --fix --test
    ) else (
        python scripts/optimization/diagnose_and_fix_finetune.py --fix
    )
) else (
    if "%TEST%"=="true" (
        python scripts/optimization/diagnose_and_fix_finetune.py --test
    ) else (
        python scripts/optimization/diagnose_and_fix_finetune.py
    )
)

echo.
echo Diagnostics complete.

:: If no arguments were provided, suggest options
if "%FIX%"=="false" (
    if "%TEST%"=="false" (
        echo.
        echo To fix issues automatically, run: diagnose_finetune.bat --fix
        echo To run a minimal fine-tuning test, run: diagnose_finetune.bat --test
        echo To do both, run: diagnose_finetune.bat --all
    )
) 