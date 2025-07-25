@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver - Qwen 15B Model Comparison
ECHO ================================================================================
ECHO.
ECHO This script will compare the performance of the Qwen 15B base model against the
ECHO fine-tuned Qwen 15B model with RAG on barrier certificate generation.
ECHO.

REM Make sure Python's output is unbuffered
SET PYTHONUNBUFFERED=1

REM Create timestamped logs directory
FOR /F "tokens=2-4 delims=/ " %%a IN ('date /t') DO (SET date=%%c-%%a-%%b)
FOR /F "tokens=1-2 delims=: " %%a IN ('time /t') DO (SET time=%%a%%b)
SET LOGS_DIR=output\logs\comparison_qwen15b_%date%_%time%
MKDIR %LOGS_DIR% 2>nul
ECHO Log files will be saved to: %LOGS_DIR%
ECHO.

REM Set custom output directory for 15B model
SET MODEL_OUTPUT_DIR=output\model_comparison_qwen15b

REM Create output directory if it doesn't exist
MKDIR %MODEL_OUTPUT_DIR% 2>nul

REM Try to find Python in common locations
SET PYTHON_FOUND=false

REM Check if python is directly accessible
python --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON_CMD=python
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

REM Try py launcher
py --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON_CMD=py
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

REM Check common Python installation locations
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    SET PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

IF EXIST "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    SET PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

IF EXIST "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    SET PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

REM Try to check for Conda/Miniconda
IF EXIST "%USERPROFILE%\Miniconda3\python.exe" (
    SET PYTHON_CMD="%USERPROFILE%\Miniconda3\python.exe"
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

IF EXIST "%USERPROFILE%\Anaconda3\python.exe" (
    SET PYTHON_CMD="%USERPROFILE%\Anaconda3\python.exe"
    SET PYTHON_FOUND=true
    GOTO :RUN_COMPARISON
)

IF NOT %PYTHON_FOUND%==true (
    ECHO ERROR: Python not found. Please install Python or add it to your PATH.
    GOTO :END
)

:RUN_COMPARISON
ECHO Found Python: %PYTHON_CMD%
ECHO.

ECHO Running Qwen 15B model comparison...
ECHO.

REM Execute the comparison script with the logs directory and custom output
%PYTHON_CMD% compare_models.py --log-dir=%LOGS_DIR% --qwen15b

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ================================================================================
    ECHO ERROR: Model comparison failed with exit code %ERRORLEVEL%.
    ECHO ================================================================================
    GOTO :END
)

ECHO.
ECHO ================================================================================
ECHO Model comparison completed successfully.
ECHO Detailed logs are available in %LOGS_DIR%
ECHO ================================================================================

ECHO.
ECHO Running advanced analysis on Qwen 15B results...
ECHO.

REM Run the analysis script on the 15B model results
%PYTHON_CMD% analyze_comparison_results.py --comparison-dir=%MODEL_OUTPUT_DIR% --output-dir=%MODEL_OUTPUT_DIR%\analysis

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ================================================================================
    ECHO ERROR: Results analysis failed with exit code %ERRORLEVEL%.
    ECHO ================================================================================
    GOTO :END
)

ECHO.
ECHO ================================================================================
ECHO Results analysis completed successfully.
ECHO Analysis report saved to %MODEL_OUTPUT_DIR%\analysis
ECHO ================================================================================

:END
ECHO.
ECHO Press any key to exit...
pause > nul 