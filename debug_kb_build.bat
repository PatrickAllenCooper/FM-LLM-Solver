@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - DEBUG MODE
ECHO ================================================================================
ECHO.
ECHO This script will run a diagnostic test of the knowledge base builder
ECHO It will process only one PDF and a few chunks to identify any issues.
ECHO.
set PYTHONUNBUFFERED=1
python run_debug_kb_build.py
ECHO.
ECHO ================================================================================
ECHO Process completed. Press any key to exit.
ECHO ================================================================================
pause > nul 