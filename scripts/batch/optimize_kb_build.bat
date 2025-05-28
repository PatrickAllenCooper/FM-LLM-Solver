@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - OPTIMIZED
ECHO ================================================================================
ECHO.
ECHO This script will run the knowledge base builder with optimized chunking.
ECHO The optimized chunker avoids regex patterns that cause hanging.
ECHO.
set PYTHONUNBUFFERED=1
python optimize_kb_build.py
ECHO.
ECHO ================================================================================
ECHO Process completed. Press any key to exit.
ECHO ================================================================================
pause > nul 