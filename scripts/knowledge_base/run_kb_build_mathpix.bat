@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - MATHPIX VERSION
ECHO ================================================================================
ECHO.
ECHO This script will build the knowledge base using the Mathpix API.
ECHO The Mathpix credentials will be loaded from .env file.
ECHO.

REM Make sure Python's output is unbuffered
set PYTHONUNBUFFERED=1

REM Check if .env file exists
IF NOT EXIST .env (
    ECHO ERROR: .env file not found!
    ECHO Please create a .env file with your Mathpix credentials:
    ECHO.
    ECHO MATHPIX_APP_ID=your_app_id_here
    ECHO MATHPIX_APP_KEY=your_app_key_here
    ECHO.
    GOTO :END
)

ECHO Running knowledge base builder with Mathpix pipeline...
ECHO.

REM Run the Python script with debug mode and batch size of 1
python run_mathpix_kb.py --debug --batch-size 1

:END
ECHO.
ECHO ================================================================================
ECHO Process completed. Press any key to exit.
ECHO ================================================================================
pause > nul 