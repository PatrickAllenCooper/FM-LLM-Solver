@echo off
:: FM-LLM-Solver AWQ-Quantized Model Runner
:: This script runs the fine-tuning and evaluation pipeline with AWQ-quantized Qwen2.5 models

echo === FM-LLM-Solver AWQ-Quantized Qwen2.5 Experiment Runner ===
echo This script will run experiments with AWQ-quantized Qwen2.5 models

:: Get the full path of the repository root directory
set REPO_ROOT=%~dp0..\..
echo Using repository root: %REPO_ROOT%

:: Set working directory to project root
cd /d %REPO_ROOT%
echo Current working directory: %CD%

:: Use the active Python from the current environment
set PYTHON_CMD=python

:: Check if model choice is passed as environment variable
if "%1"=="" (
    :: Interactive mode for model selection
    echo.
    echo === Select Model Size ===
    echo Available AWQ-quantized Qwen2.5 models:
    echo 1. Qwen/Qwen2.5-7B-Instruct-AWQ (Recommended for RTX 3080)
    echo 2. Qwen/Qwen2.5-14B-Instruct-AWQ (May require 16GB+ VRAM)
    echo 3. Qwen/Qwen2.5-72B-Instruct-AWQ (Requires 40GB+ VRAM, inference only)
    
    set /p MODEL_CHOICE="Enter your choice (1, 2, or 3): "
) else (
    :: Non-interactive mode, use passed parameter
    set MODEL_CHOICE=%1
    echo Using provided model choice: %MODEL_CHOICE%
)

:: Set model based on choice (using explicit equality comparison)
if "%MODEL_CHOICE%"=="1" (
    set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
    set CAN_FINETUNE=yes
    echo Selected model: 7B model
) else if "%MODEL_CHOICE%"=="2" (
    set MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
    set CAN_FINETUNE=maybe
    echo Selected model: 14B model
) else if "%MODEL_CHOICE%"=="3" (
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
    set CAN_FINETUNE=no
    echo Selected model: 72B model
) else (
    echo Invalid choice. Defaulting to Qwen/Qwen2.5-7B-Instruct-AWQ
    set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
    set CAN_FINETUNE=yes
)

echo Selected model: %MODEL_NAME%

:: Create necessary directories with absolute paths
mkdir "%REPO_ROOT%\output\knowledge_base" 2>nul
mkdir "%REPO_ROOT%\output\finetuning_results" 2>nul
mkdir "%REPO_ROOT%\kb_data" 2>nul

:: Define full paths for config files
set CONFIG_FILE=%REPO_ROOT%\config.yaml
set CONFIG_BAK=%REPO_ROOT%\config.yaml.bak

:: Check if knowledge base exists (skip if --skip-kb flag is set)
if "%2"=="--skip-kb" (
    echo Knowledge base check skipped (--skip-kb flag detected)
    
    :: Create empty placeholder files if they don't exist
    if not exist "%REPO_ROOT%\kb_data\paper_index_mathpix.faiss" (
        echo Creating placeholder knowledge base files for testing...
        copy nul "%REPO_ROOT%\kb_data\paper_index_mathpix.faiss" >nul
        echo [] > "%REPO_ROOT%\kb_data\paper_metadata_mathpix.jsonl"
    )
) else (
    if exist "%REPO_ROOT%\kb_data\paper_index_mathpix.faiss" (
        echo Knowledge base found, continuing with fine-tuning
    ) else (
        echo Knowledge base files not found. Please run knowledge base creation first.
        echo You can build the knowledge base using: python knowledge_base/knowledge_base_builder.py
        echo.
        echo Alternatively, use --skip-kb flag to bypass this check for testing:
        echo scripts\experiments\run_quantized_qwen.bat --skip-kb
        exit /b 1
    )
)

:: Check if AWQ dependencies are installed
python -c "import awq" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing AutoAWQ...
    python -m pip install autoawq
)

echo.
echo === Setting Memory Optimization Environment Variables ===
echo 1. Setting PyTorch to release memory efficiently
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo 2. Limiting CUDA visible devices to single GPU
set CUDA_VISIBLE_DEVICES=0

echo 3. Setting TF memory growth
set TF_FORCE_GPU_ALLOW_GROWTH=true

echo 4. Enabling garbage collection
set PYTHONMALLOC=debug

echo.
echo === Creating AWQ-Quantized Model Configuration ===

:: Create a backup of the original config
copy "%CONFIG_FILE%" "%CONFIG_BAK%" /Y
echo Config backup created as config.yaml.bak

:: Create temporary configuration file for the quantized model
echo Creating modified config for AWQ-quantized model...

:: Try to use yaml module for more reliable config modification
python -c "import yaml; config = yaml.safe_load(open('%CONFIG_FILE%', 'r')); config['fine_tuning']['base_model_name'] = '%MODEL_NAME%'; config['fine_tuning']['quantization']['quantization_method'] = 'awq'; config['fine_tuning']['quantization']['use_4bit'] = True; config['fine_tuning']['quantization']['use_nested_quant'] = True; config['fine_tuning']['lora']['r'] = 8; config['fine_tuning']['lora']['alpha'] = 8; config['fine_tuning']['training']['gradient_accumulation_steps'] = 8; config['fine_tuning']['training']['max_seq_length'] = 1024; yaml.dump(config, open('%CONFIG_FILE%', 'w'))" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo Failed to create optimized config. Using simple text replacement instead.
    
    :: Update the model using PowerShell (more reliable than batch replacements)
    powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'base_model_name: \".*\"', 'base_model_name: \"%MODEL_NAME%\"' | Set-Content '%CONFIG_FILE%'"
    
    :: Add AWQ method if it doesn't exist
    powershell -Command "if (-not (Select-String -Path '%CONFIG_FILE%' -Pattern 'quantization_method:')) { (Get-Content '%CONFIG_FILE%') -replace 'quantization:', 'quantization:`n    quantization_method: awq' | Set-Content '%CONFIG_FILE%' }"
    
    :: Update other parameters
    powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'r: \d+', 'r: 8' | Set-Content '%CONFIG_FILE%'"
    powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'alpha: \d+', 'alpha: 8' | Set-Content '%CONFIG_FILE%'"
    powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'use_4bit: false', 'use_4bit: true' | Set-Content '%CONFIG_FILE%'"
    powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'use_nested_quant: false', 'use_nested_quant: true' | Set-Content '%CONFIG_FILE%'"
    powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'gradient_accumulation_steps: \d+', 'gradient_accumulation_steps: 8' | Set-Content '%CONFIG_FILE%'"
)

echo Config updated with AWQ-quantized model settings

if "%CAN_FINETUNE%"=="no" (
    echo.
    echo NOTICE: The selected model (%MODEL_NAME%) is too large for fine-tuning on most consumer GPUs.
    echo This script will only run inference/evaluation with the pre-trained model.
    echo For fine-tuning, please select a smaller model (7B or 14B).
    
    echo.
    echo === Running Evaluation Only ===
    python "%REPO_ROOT%\evaluation\evaluate_pipeline.py" --config "%CONFIG_FILE%" --skip_finetuning
    
    echo.
    echo === Restoring Original Configuration ===
    copy "%CONFIG_BAK%" "%CONFIG_FILE%" /Y
    echo Original config restored
    exit /b 0
)

if "%CAN_FINETUNE%"=="maybe" (
    echo.
    echo WARNING: The selected model (%MODEL_NAME%) may be too large for fine-tuning on your GPU.
    echo If you encounter memory errors, try the 7B model instead.
    echo.
    set /p CONTINUE_CHOICE="Do you want to continue with fine-tuning? (Y/N): "
    if /i not "%CONTINUE_CHOICE%"=="Y" (
        echo.
        echo === Running Evaluation Only ===
        python "%REPO_ROOT%\evaluation\evaluate_pipeline.py" --config "%CONFIG_FILE%" --skip_finetuning
        
        echo.
        echo === Restoring Original Configuration ===
        copy "%CONFIG_BAK%" "%CONFIG_FILE%" /Y
        echo Original config restored
        exit /b 0
    )
)

echo.
echo === Running QLoRA Fine-tuning with AWQ-Quantized Model ===
echo This will use QLoRA to fine-tune the AWQ-quantized model.
echo.
python "%REPO_ROOT%\fine_tuning\finetune_llm.py" --config "%CONFIG_FILE%"

:: Check if fine-tuning was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Fine-tuning failed with error level %ERRORLEVEL%
    echo Check error messages above for details.
    echo.
    echo Troubleshooting tips:
    echo 1. If you're getting CUDA out of memory errors:
    echo    - Try using the smaller 7B model instead
    echo    - Reduce the LoRA rank further (to 4)
    echo    - Increase gradient accumulation steps to 16
    echo 2. If you're getting loading errors:
    echo    - Make sure you have the latest transformers library
    echo    - Make sure you have autoawq installed
    echo.
    echo Restoring original config...
    copy "%CONFIG_BAK%" "%CONFIG_FILE%" /Y
    exit /b %ERRORLEVEL%
)

echo.
echo === Running Evaluation and Verification ===
python "%REPO_ROOT%\evaluation\evaluate_pipeline.py" --config "%CONFIG_FILE%"

:: Check if evaluation was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Evaluation failed with error level %ERRORLEVEL%
    echo Check error messages above for details.
    echo.
    echo Restoring original config...
    copy "%CONFIG_BAK%" "%CONFIG_FILE%" /Y
    exit /b %ERRORLEVEL%
)

echo.
echo === Restoring Original Configuration ===
copy "%CONFIG_BAK%" "%CONFIG_FILE%" /Y
echo Original config restored

echo.
echo === All Done! ===
echo AWQ-quantized model experiment completed successfully.
echo Results can be found in the output directory. 