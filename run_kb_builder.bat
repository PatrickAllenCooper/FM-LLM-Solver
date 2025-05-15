@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - Consolidated Version
ECHO ================================================================================
ECHO.
ECHO This script will build the knowledge base using your GPU with memory limits.
ECHO PDFs will be processed in small batches with automatic recovery.
ECHO Progress information will be displayed every 5 seconds.
ECHO.
ECHO You can customize this process with the following options:
ECHO   --batch-size N    : Process N PDFs at a time (default: 3)
ECHO   --force           : Force rebuild entire knowledge base
ECHO   --cpu-only        : Use CPU only (slower but more reliable)
ECHO.
set PYTHONUNBUFFERED=1
python kb_builder.py %*
ECHO.
ECHO ================================================================================
ECHO Process completed. Press any key to exit.
ECHO ================================================================================
pause > nul 