@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - GPU Mode
ECHO ================================================================================
ECHO.
ECHO This script will build the knowledge base using your GPU with memory limits.
ECHO Progress information will be displayed every 5 seconds.
ECHO.
set PYTHONUNBUFFERED=1
python run_kb_build_with_monitor.py --use-gpu
ECHO.
ECHO ================================================================================
ECHO Process completed. Press any key to exit.
ECHO ================================================================================
pause > nul 