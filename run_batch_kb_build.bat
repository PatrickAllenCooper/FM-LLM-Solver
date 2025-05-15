@echo off
ECHO ================================================================================
ECHO FM-LLM-Solver Knowledge Base Builder - BATCH MODE
ECHO ================================================================================
ECHO.
ECHO This script will build the knowledge base by processing PDFs in small batches.
ECHO If it crashes, you can run it again to continue from where it left off.
ECHO.
set PYTHONUNBUFFERED=1
python run_batch_kb_build.py --batch-size 3
ECHO.
ECHO ================================================================================
ECHO Process completed. Press any key to exit.
ECHO ================================================================================
pause > nul 