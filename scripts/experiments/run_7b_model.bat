@echo off
:: FM-LLM-Solver Experiment Runner for 7B Model
:: This script runs the fine-tuning and evaluation pipeline with 7B model settings

echo === FM-LLM-Solver 7B Model Experiment Runner ===
echo This script will run experiments with Qwen2.5-7B-Instruct model

:: Set working directory to project root
cd ..\..

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
echo === Setting Model Parameters ===
echo 1. Setting PyTorch to release memory efficiently
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo 2. Limiting CUDA visible devices to single GPU
set CUDA_VISIBLE_DEVICES=0

echo 3. Setting TF memory growth (for potential TF dependencies)
set TF_FORCE_GPU_ALLOW_GROWTH=true

echo.
echo === Setting Model to 7B Variant ===
echo Temporarily modifying config to use Qwen2.5-7B-Instruct

:: Create a backup of the original config
copy config.yaml config.yaml.bak /Y
echo Config backup created as config.yaml.bak

:: Update the model in config
powershell -Command "(Get-Content config.yaml) -replace 'base_model_name: Qwen/Qwen2.5-15B-Instruct', 'base_model_name: Qwen/Qwen2.5-7B-Instruct' | Set-Content config.yaml"
echo Config updated to use 7B model

echo.
echo === Starting Fine-tuning with 7B Model ===
echo This will use the 7B model with default LoRA settings

:: Run fine-tuning directly to skip unnecessary steps in the pipeline
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
echo === Starting Evaluation and Verification ===
echo Running evaluation on the fine-tuned model

:: Run evaluation pipeline
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
echo === Restoring Original Config ===
copy config.yaml.bak config.yaml /Y
echo Original config restored

echo.
echo === All Done! ===
echo Experiment with 7B model completed successfully.
echo Results can be found in the output directory.
echo. 