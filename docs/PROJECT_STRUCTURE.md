# FM-LLM-Solver Project Structure

This document outlines the reorganized directory structure of the FM-LLM-Solver project. The codebase has been restructured to improve organization, maintainability, and clarity.

## Directory Organization

### Top-Level Directories

- **data_fetching/**: Scripts for downloading papers and fetching research data
- **docs/**: Documentation files
- **evaluation/**: Barrier certificate verification and pipeline evaluation code
- **fine_tuning/**: Scripts for fine-tuning language models
- **inference/**: Inference code for generating barrier certificates
- **kb_data/**: Knowledge base data files (FAISS index and metadata)
- **knowledge_base/**: Knowledge base construction code
- **scripts/**: Organized script files by purpose
- **utils/**: Utility functions and helpers
- **data/**: Input data and fetched raw data
- **output/**: Generated outputs and results

### Scripts Organization

The `scripts/` directory has been organized into multiple subdirectories by purpose:

- **scripts/comparison/**: Tools for comparing different models' performance
  - `compare_model_sizes.py`: Compare models of different sizes
  - `compare_models.py`: Compare base model vs fine-tuned model performance
  - Various batch files for running comparisons

- **scripts/experiments/**: Experiment execution scripts
  - `run_experiments.py`: Unified experiment runner
  - `run_parameterized_experiments.py`: Run experiments with different parameters
  - `run_barrier_certificate_experiments.sh`: Linux shell script for running barrier certificate experiments
  - `analyze_experiment_results.py`: Analyze and visualize experiment results
  - `run_optimized_experiments.bat`: Windows batch file for running optimized 15B model experiments
  - `run_optimized_experiments.sh`: Linux shell script for running optimized 15B model experiments
  - `run_7b_model.bat`: Windows batch file for running experiments with 7B model
  - `run_inference.bat`: Windows batch file for single inference test
  - `run_inference.sh`: Linux shell script for single inference test

- **scripts/knowledge_base/**: Knowledge base utilities
  - Various scripts for building and testing knowledge bases
  - PDF processing utilities and tests

- **scripts/optimization/**: Scripts for optimizing the system
  - `optimize_kb_build.py`: Optimize knowledge base building process
  - Various batch files for running optimizations

- **scripts/setup/**: Setup and installation scripts
  - `setup_environment.py`: Main setup script for the project
  - `install_deps.py`: Install dependencies
  - Environment configuration examples and templates

### Documentation Organization

The `docs/` directory contains various documentation files:

- **EXPERIMENTS.md**: Documentation on experiments and results
- **OPTIMIZED_README.md**: Guide for optimized execution on limited hardware
- **PROJECT_STRUCTURE.md**: This file explaining the project structure

## Data Files Location

- **Knowledge Base Files**:
  - Primary location: `kb_data/` directory (paper_index_mathpix.faiss, paper_metadata_mathpix.jsonl)
  - Alternative location: `output/knowledge_base/` (for backward compatibility)

- **Model Files**:
  - Fine-tuned models: `output/finetuning_results/final_adapter/`

- **Input Data**:
  - Benchmark systems: `data/benchmark_systems.json`
  - User IDs: `data/user_ids.csv`
  - Fine-tuning data: Various JSONL files in `data/`

- **Output Data**:
  - Evaluation results: `output/evaluation_results.csv`
  - Model comparison: Files in `output/model_comparison/`
  - Logs: `output/logs/`

## Running Scripts With the New Structure

With the reorganized structure, scripts can be run using paths relative to the project root. For example:

```bash
# Run the unified experiment runner
python scripts/experiments/run_experiments.py

# Run model comparison
python scripts/comparison/compare_models.py

# Set up the environment
python scripts/setup/setup_environment.py

# Run fine-tuning directly
python fine_tuning/finetune_llm.py
```

### Running Model Experiments

To run experiments with different model sizes:

```bash
# For the 15B model (optimized for RTX 3080)
.\scripts\experiments\run_optimized_experiments.bat  # Windows
./scripts/experiments/run_optimized_experiments.sh   # Linux/Mac

# For the 7B model
.\scripts\experiments\run_7b_model.bat  # Windows
```

All scripts have been updated to work with the new directory structure while maintaining backward compatibility with the previous structure. 