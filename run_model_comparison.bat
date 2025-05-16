@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver - Model Comparison
ECHO ================================================================================
ECHO.
ECHO This script will compare the performance of the base model against the
ECHO fine-tuned model with RAG on barrier certificate generation.
ECHO.

REM Make sure Python's output is unbuffered
SET PYTHONUNBUFFERED=1

REM Create timestamped logs directory
FOR /F "tokens=2-4 delims=/ " %%a IN ('date /t') DO (SET date=%%c-%%a-%%b)
FOR /F "tokens=1-2 delims=: " %%a IN ('time /t') DO (SET time=%%a%%b)
SET LOGS_DIR=output\logs\comparison_%date%_%time%
MKDIR %LOGS_DIR% 2>nul
ECHO Log files will be saved to: %LOGS_DIR%
ECHO.

REM Check if Python is in the path and get its path
FOR /F "tokens=*" %%i IN ('where python 2^>nul') DO (
    SET PYTHON_PATH=%%i
    GOTO :PYTHON_FOUND
)

REM Check for Python3 if Python was not found
FOR /F "tokens=*" %%i IN ('where py 2^>nul') DO (
    SET PYTHON_PATH=%%i -3
    GOTO :PYTHON_FOUND
)

ECHO Python executable not found in PATH.
ECHO Checking common installation locations...

REM Check common Python installation locations
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    SET PYTHON_PATH="%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    GOTO :PYTHON_FOUND
)

IF EXIST "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    SET PYTHON_PATH="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    GOTO :PYTHON_FOUND
)

IF EXIST "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    SET PYTHON_PATH="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    GOTO :PYTHON_FOUND
)

ECHO ERROR: Python not found. Please install Python or add it to your PATH.
GOTO :END

:PYTHON_FOUND
ECHO Found Python: %PYTHON_PATH%
ECHO.

ECHO Running model comparison...
ECHO.

REM Execute the comparison script with the logs directory
%PYTHON_PATH% compare_models.py --log-dir=%LOGS_DIR% %*

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ================================================================================
    ECHO ERROR: Model comparison failed with exit code %ERRORLEVEL%.
    ECHO ================================================================================
) ELSE (
    ECHO.
    ECHO ================================================================================
    ECHO Model comparison completed successfully.
    ECHO Detailed logs are available in %LOGS_DIR%
    ECHO ================================================================================
)

:END
ECHO.
ECHO Press any key to exit...
pause > nul 