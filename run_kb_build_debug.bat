@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - DEBUG MODE
ECHO ================================================================================
ECHO.
ECHO This script will run the knowledge base builder in debugging mode.
ECHO - Processing only 1 PDF at a time
ECHO - Using detailed logging
ECHO - Running watchdog thread to detect hangs
ECHO - Displaying process information
ECHO.
ECHO This should help identify where the process is hanging.
ECHO.
set PYTHONUNBUFFERED=1

REM Install psutil for system monitoring if not already installed
pip install psutil --quiet

ECHO Running knowledge base builder in debug mode...
ECHO.

python kb_builder.py --batch-size 1 --debug

ECHO.
ECHO ================================================================================
ECHO Debugging completed. Check the logs for detailed information.
ECHO Press any key to exit.
ECHO ================================================================================
pause > nul 