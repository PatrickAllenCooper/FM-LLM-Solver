@echo off
:: FM-LLM-Solver 7B Model with Optimized Settings
:: This script runs an optimized fine-tuning and evaluation pipeline with Qwen2.5-7B-Instruct

echo === FM-LLM-Solver 7B Model with Optimized Settings ===
echo This script will run fine-tuning and evaluation with Qwen2.5-7B-Instruct
echo using optimized parameters for reliable performance on RTX 3080

:: Set working directory to project root
cd ..\..

:: Check CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" > cuda_check.txt
set /p CUDA_CHECK=<cuda_check.txt
del cuda_check.txt

if not "%CUDA_CHECK%"=="CUDA Available: True" (
    echo ERROR: CUDA is not available. This script requires a CUDA-compatible GPU.
    exit /b 1
)

:: Create necessary directories 
mkdir output\knowledge_base 2>nul
mkdir output\finetuning_results 2>nul
mkdir kb_data 2>nul

:: Check if knowledge base exists
if exist kb_data\paper_index_mathpix.faiss (
    echo Knowledge base found, continuing with fine-tuning
) else (
    echo Knowledge base files not found. Please run knowledge base creation first.
    echo You can build the knowledge base using: python knowledge_base/knowledge_base_builder.py
    exit /b 1
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
echo === Creating 7B Model Configuration ===

:: Create a backup of the original config
copy config.yaml config.yaml.bak /Y
echo Config backup created as config.yaml.bak

:: Create temporary modifications for 7B model with optimized settings
python -c "import yaml; config = yaml.safe_load(open('config.yaml', 'r')); config['fine_tuning']['base_model_name'] = 'Qwen/Qwen2.5-7B-Instruct'; config['fine_tuning']['lora']['r'] = 8; config['fine_tuning']['lora']['alpha'] = 8; config['fine_tuning']['quantization']['use_4bit'] = True; config['fine_tuning']['quantization']['use_nested_quant'] = True; config['fine_tuning']['training']['gradient_accumulation_steps'] = 4; config['fine_tuning']['training']['max_seq_length'] = 1024; yaml.dump(config, open('config.yaml', 'w'))" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo Failed to create optimized config. Using simple text replacement instead.
    
    :: Update the model using PowerShell (more reliable than batch replacements)
    powershell -Command "(Get-Content config.yaml) -replace 'base_model_name: \".*\"', 'base_model_name: \"Qwen/Qwen2.5-7B-Instruct\"' | Set-Content config.yaml"
    powershell -Command "(Get-Content config.yaml) -replace 'r: \d+', 'r: 8' | Set-Content config.yaml"
    powershell -Command "(Get-Content config.yaml) -replace 'alpha: \d+', 'alpha: 8' | Set-Content config.yaml"
    powershell -Command "(Get-Content config.yaml) -replace 'use_4bit: false', 'use_4bit: true' | Set-Content config.yaml"
    powershell -Command "(Get-Content config.yaml) -replace 'use_nested_quant: false', 'use_nested_quant: true' | Set-Content config.yaml"
    powershell -Command "(Get-Content config.yaml) -replace 'gradient_accumulation_steps: \d+', 'gradient_accumulation_steps: 4' | Set-Content config.yaml"
)

echo Config updated with optimized 7B model settings

echo.
echo === Running Fine-tuning with Memory-Optimized Settings ===
python fine_tuning/finetune_llm.py --config config.yaml

:: Check if fine-tuning was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Fine-tuning failed with error level %ERRORLEVEL%
    echo Check error messages above for details.
    echo.
    echo Restoring original config...
    copy config.yaml.bak config.yaml /Y
    exit /b %ERRORLEVEL%
)

echo.
echo === Running Evaluation and Verification ===
python evaluation/evaluate_pipeline.py --config config.yaml

:: Check if evaluation was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Evaluation failed with error level %ERRORLEVEL%
    echo Check error messages above for details.
    echo.
    echo Restoring original config...
    copy config.yaml.bak config.yaml /Y
    exit /b %ERRORLEVEL%
)

echo.
echo === Restoring Original Configuration ===
copy config.yaml.bak config.yaml /Y
echo Original config restored

echo.
echo === All Done! ===
echo 7B model experiment completed successfully.
echo Results can be found in the output directory.
echo. 