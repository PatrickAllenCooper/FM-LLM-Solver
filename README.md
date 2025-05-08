# FMLLMSolver: Barrier Certificate Generation using LLMs

## Recent Code Improvements

This codebase has recently undergone significant refactoring and improvements to enhance usability, clarity, and maintainability:

1. **Enhanced Documentation**: 
   - Improved docstrings in all core functions following NumPy style
   - More comprehensive in-line comments explaining complex logic
   - Better structured section headings in code files

2. **Improved Error Handling**:
   - More robust error handling and validation in critical functions
   - Better reporting of error conditions with detailed messages
   - Graceful failure modes with meaningful exit codes

3. **Code Organization**:
   - Standardized function structures with clear input/output documentation
   - Better separation of concerns between data loading, processing, and evaluation
   - Clearer step-by-step progression in main execution blocks

4. **Configuration System**:
   - More detailed comments in configuration file
   - Improved validation and logging of configuration parameters
   - Better organization of configuration sections

5. **Data Processing**:
   - Enhanced regex patterns for barrier certificate extraction
   - More robust CSV processing for user IDs
   - Expanded benchmark system examples

These changes should make the codebase more accessible, easier to understand, and simpler to extend with new features.

---

**FMLLMSolver** explores the use of Large Language Models (LLMs), enhanced by Retrieval-Augmented Generation (RAG) and fine-tuning, to assist in or automate the generation of **barrier certificates** for autonomous systems. The core idea is to leverage a knowledge base built from relevant research papers to improve the LLM's ability to propose valid barrier functions for given system dynamics.

---

## Table of Contents

*   [Overview](#overview)
*   [Project Structure](#project-structure)
*   [Setup](#setup)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
*   [Workflow / Usage](#workflow--usage)
    *   [1. Data Fetching](#1-data-fetching)
    *   [2. Build Knowledge Base](#2-build-knowledge-base)
    *   [3. Test Knowledge Base (Optional)](#3-test-knowledge-base-optional)
    *   [4. Create/Prepare Fine-tuning Data](#4-createprepare-fine-tuning-data)
    *   [5. Fine-tune the LLM](#5-fine-tune-the-llm)
    *   [6. Generate Certificate (Inference)](#6-generate-certificate-inference)
    *   [7. Evaluate the Pipeline](#7-evaluate-the-pipeline)
    *   [First Battery of Experiments: Complete Instructions](#first-battery-of-experiments-complete-instructions)
*   [Unified Experiment Runner](#unified-experiment-runner)
*   [Configuration](#configuration)
*   [Verification Limitations](#verification-limitations)
*   [Author / Context](#author--context)
*   [Future Work / Enhancements](#future-work--enhancements)

---

## Overview

Formal verification provides essential safety and correctness guarantees for complex autonomous systems, with techniques like barrier certificates offering powerful tools for proving set invariance and reach-avoid properties. However, the synthesis of suitable barrier functions remains a significant bottleneck, often demanding considerable domain expertise or relying on computationally intensive methods like Sum-of-Squares (SOS) programming, which can struggle with non-polynomial dynamics or high dimensionality. This project investigates the potential of Large Language Models (LLMs) to address the *candidate generation* challenge within the barrier certificate synthesis workflow.

We propose leveraging LLMs, augmented by domain knowledge extracted from a curated corpus of relevant research literature via Retrieval-Augmented Generation (RAG), to propose plausible barrier certificate candidates for given system dynamics. The core idea is to fine-tune an LLM specifically on the task of mapping system descriptions (dynamics, initial/unsafe sets) to potential barrier function structures, learning heuristics and patterns from existing published examples. The goal is not to supplant rigorous verification but to accelerate the overall process by providing formally-inclined researchers with high-quality, structured hypotheses for barrier functions, thereby narrowing the search space for subsequent analysis.

The implemented pipeline includes modules for automated paper fetching, knowledge base construction using text embedding and vector indexing (FAISS), efficient LLM fine-tuning (QLoRA), and an inference engine combining RAG with the specialized model. Crucially, the evaluation module incorporates symbolic differentiation (`sympy`) for Lie derivative calculation and numerical sampling checks for basic validation of proposed certificates. While this preliminary verification helps filter candidates, the framework explicitly acknowledges the need for integration with established formal methods tools (e.g., SOS solvers, theorem provers, robust numerical verification techniques) to provide the necessary soundness guarantees for the generated barrier certificates before they can be formally certified.

---

## Project Structure

The project is organized into modules based on functionality:

```
./
├── data_fetching/              # Scripts for downloading papers
│   ├── __init__.py
│   └── paper_fetcher.py
|
├── knowledge_base/             # Scripts & data for the RAG knowledge base
│   ├── __init__.py
│   ├── knowledge_base_builder.py
│   └── test_knowledge_base.py
|
├── fine_tuning/                # Scripts & data for fine-tuning the LLM
│   ├── __init__.py
│   ├── create_finetuning_data.py
│   ├── finetune_llm.py
│   ├── generate_synthetic_data.py # (Example)
│   ├── extract_from_papers.py
│   └── combine_datasets.py
|
├── inference/                  # Scripts for running inference
│   ├── __init__.py
│   └── generate_certificate.py
|
├── evaluation/                 # Scripts & data for pipeline evaluation
│   ├── __init__.py
│   ├── evaluate_pipeline.py
│   └── verify_certificate.py
|
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── config_loader.py
|
├── data/                       # Input data & fetched raw data
│   ├── fetched_papers/         # Default location for downloaded PDFs
│   ├── benchmark_systems.json  # Evaluation benchmarks
│   ├── user_ids.csv            # Input for data_fetching
│   ├── ft_manual_data.jsonl    # Example fine-tuning data file
│   ├── ft_extracted_data_verified.jsonl # Example
│   └── ft_data_combined.jsonl  # Example combined data file
|
├── output/                     # Generated outputs
│   ├── knowledge_base/         # Default location for FAISS index & metadata
│   │   ├── paper_index_mathpix.faiss
│   │   └── paper_metadata_mathpix.jsonl
│   ├── finetuning_results/     # Default location for model checkpoints/adapter
│   │   └── final_adapter/
│   └── evaluation_results.csv  # Default location for evaluation CSV
|
├── config.yaml                   # Central configuration file
├── requirements.txt            # Python package dependencies
├── .gitignore
└── README.md                     # This file
```

---

## Setup

### Prerequisites

*   **Python:** Version 3.10 required.
*   **Git:** For cloning the repository.
*   **API Credentials & Email (Environment Variables):**
    ```bash
    export MATHPIX_APP_ID='your_app_id'         # Required for knowledge_base_builder.py
    export MATHPIX_APP_KEY='your_app_key'         # Required for knowledge_base_builder.py
    export UNPAYWALL_EMAIL='your-email@example.com' # Required for data_fetching/paper_fetcher.py
    # Optional:
    # export SEMANTIC_SCHOLAR_API_KEY='your_key'
    ```
*   **CUDA Toolkit:** Required for GPU acceleration.
*   **SDP Solver (Optional):** MOSEK (recommended) or SCS required for SOS verification in `evaluation/verify_certificate.py`.

### Installation

1.  **Clone:**
    ```bash
    git clone https://your-repository-url/FMLLMSolver.git
    cd FMLLMSolver
    ```

2.  **Create Environment (Recommended):**
    ```bash
    conda create -n fmllm python=3.10
    conda activate fmllm
    ```

3.  **Setup Environment with CUDA Support:**
    ```bash
    # This script automatically sets up the environment with proper CUDA support
    python setup_environment.py
    ```
    The setup script will:
    - Check for CUDA-compatible GPU
    - Install PyTorch with the correct CUDA support
    - Install other dependencies from requirements.txt
    - Verify the installation

    If you need to reinstall PyTorch with CUDA support later:
    ```bash
    python setup_environment.py --force-reinstall
    ```

4.  **Alternative Manual Installation (NOT RECOMMENDED):**
    ```bash
    # WARNING: May result in CPU-only PyTorch which cannot be used for fine-tuning
    pip install -r requirements.txt
    # Install PyTorch with CUDA support separately
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Install SCS if not using MOSEK for SOS:
    # pip install scs
    ```

---

## Workflow / Usage

Execute the steps in the following order from the project root directory (`FMLLMSolver/`). Scripts primarily use settings from `config.yaml`.

### 1. Data Fetching

*   **(Optional)** Create/Modify `data/user_ids.csv`.
*   **Set Env Var:** `export UNPAYWALL_EMAIL='...'`
*   Run:
    ```bash
    python data_fetching/paper_fetcher.py
    ```
*   Downloads PDFs to `data/fetched_papers/` (default).

### 2. Build Knowledge Base

*   **Set Env Vars:** `export MATHPIX_APP_ID='...' MATHPIX_APP_KEY='...'`
*   Run:
    ```bash
    python knowledge_base/knowledge_base_builder.py
    ```
*   Creates KB files in `output/knowledge_base/` (default).

### 3. Test Knowledge Base (Optional)

*   Run:
    ```bash
    python knowledge_base/test_knowledge_base.py "What is a barrier certificate?" -k 3
    ```

### 4. Create/Prepare Fine-tuning Data

Place or generate fine-tuning data files (e.g., `.jsonl`) in the `data/` directory. Update `config.yaml` paths (`paths.ft_manual_data_file`, etc.) if using different filenames.

*   **Manual:** `python fine_tuning/create_finetuning_data.py`
*   **Extraction:** `python fine_tuning/extract_from_papers.py` (Requires manual LLM step & review)
*   **Combine:**
    ```bash
    # Uses paths specified in config.yaml by default to find inputs & determine output
    python fine_tuning/combine_datasets.py
    ```

### 5. Fine-tune the LLM

*   ⚠️ **REQUIRES CUDA GPU with PyTorch CUDA support!**
*   Ensure the environment is set up correctly:
    ```bash
    # Verify PyTorch CUDA support
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    # Should output: CUDA available: True
    
    # If False, run the setup script
    python setup_environment.py --force-reinstall
    ```
*   Run:
    ```bash
    python fine_tuning/finetune_llm.py
    ```
*   Uses `data/ft_data_combined.jsonl` (default) and saves adapter to `output/finetuning_results/` (default).
*   If you encounter errors related to CUDA, check [Troubleshooting](#troubleshooting) section.

### 6. Generate Certificate (Inference)

*   Run:
    ```bash
    python inference/generate_certificate.py \
      "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5"
    ```
*   Uses adapter from `output/finetuning_results/` and KB from `output/knowledge_base/` (defaults).

### 7. Evaluate the Pipeline

*   **Populate Benchmark:** Modify `data/benchmark_systems.json`.
*   Run:
    ```bash
    python evaluation/evaluate_pipeline.py
    ```
*   Uses benchmark from `data/`, adapter/KB from `output/`, saves results to `output/evaluation_results.csv` (defaults).

---

## First Battery of Experiments: Complete Instructions

This section provides detailed instructions for setting up and running the first set of experiments to evaluate the barrier certificate generation capabilities.

### Prerequisites

1. **Environment Setup**
   - Ensure Python 3.8-3.12 is installed
   - Set up a virtual environment:
     ```bash
     conda create -n fmllm python=3.12
     conda activate fmllm
     pip install -r requirements.txt
     ```
   - Ensure you have a CUDA-compatible GPU with appropriate drivers installed

2. **API Keys Configuration**
   - Export required environment variables:
     ```bash
     export MATHPIX_APP_ID='your_app_id'
     export MATHPIX_APP_KEY='your_app_key'
     export UNPAYWALL_EMAIL='your-email@example.com'
     ```

### Experiment Setup Steps

1. **Data Preparation**
   - Review the benchmark systems in `data/benchmark_systems.json`
   - The default file includes 5 systems of varying complexity (2D nonlinear, linear stable, coupled linear, and 3D)
   - You can add additional systems by following the same JSON structure

2. **Knowledge Base Construction**
   - Fetch papers to build the knowledge base:
     ```bash
     # Make sure data directory exists
     mkdir -p data/fetched_papers
     # Run the paper fetcher with default settings
     python data_fetching/paper_fetcher.py
     ```
   - Build the knowledge base from the fetched papers:
     ```bash
     # Make sure output directory exists
     mkdir -p output/knowledge_base
     # Build the knowledge base
     python knowledge_base/knowledge_base_builder.py
     ```
   - Verify the knowledge base works correctly:
     ```bash
     python knowledge_base/test_knowledge_base.py "What is a barrier certificate?" -k 3
     ```

3. **Fine-tuning Preparation**
   - Prepare the training data:
     ```bash
     # Make sure directories exist
     mkdir -p output/finetuning_results
     # Combine existing datasets (or create a new one if needed)
     python fine_tuning/combine_datasets.py
     ```
   - Fine-tune the LLM:
     ```bash
     python fine_tuning/finetune_llm.py
     ```
   - This process will take some time depending on your GPU capabilities
   - The fine-tuned model adapter will be saved to `output/finetuning_results/final_adapter/`

### Running the First Battery of Experiments

1. **Initial Test Run with a Single Example**
   - Test certificate generation on a single example:
     ```bash
     python inference/generate_certificate.py "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5"
     ```
   - Review the output to ensure the model is generating reasonable barrier certificates

2. **Full Benchmark Evaluation**
   - Run the complete evaluation pipeline on all benchmark systems:
     ```bash
     python evaluation/evaluate_pipeline.py
     ```
   - This will:
     - Process each system in `data/benchmark_systems.json`
     - Use the RAG-enhanced fine-tuned model to generate barrier certificates
     - Verify each certificate using numerical sampling and symbolic checks
     - Attempt SOS verification when applicable
     - Save detailed results to `output/evaluation_results.csv`

3. **Evaluation with Different Parameters**
   - Try varying the RAG context size:
     ```bash
     python evaluation/evaluate_pipeline.py -k 5
     ```
   - Adjust other parameters by modifying `config.yaml` or using command-line overrides:
     ```bash
     # Example: Save results to a different file for comparison
     python evaluation/evaluate_pipeline.py --results output/eval_results_experiment1.csv
     ```

### Analyzing Experiment Results

1. **Review the Summary Output**
   - After evaluation completes, a summary will be displayed showing:
     - Total systems evaluated
     - Successful generations percentage
     - Successfully parsed certificates percentage
     - Verification success rate
     - Breakdown by verification type (numerical, symbolic, SOS)

2. **Examine Detailed Results**
   - Open `output/evaluation_results.csv` to view detailed metrics for each system:
     - Generated certificate expressions
     - Verification outcomes for different checks
     - Timing information
     - Full LLM outputs

3. **Result Interpretation**
   - Key performance indicators:
     - **Generation Rate**: Percentage of systems where the LLM successfully generated output
     - **Parse Rate**: Percentage of outputs where a valid certificate could be extracted
     - **Verification Rate**: Percentage of certificates that passed verification
     - **Specific Verification Types**: Success rates for numerical, symbolic, and SOS checks

### Experiment Variations to Try

1. **Knowledge Base Variations**
   - Try with different subsets of papers to see how knowledge base quality affects results
   - Modify the `kb_vector_store_filename` in `config.yaml` to switch between different knowledge bases

2. **Model Variations**
   - Experiment with different base models by changing `base_model_name` in `config.yaml`
   - Try different fine-tuning parameters (e.g., number of epochs, learning rate)

3. **System Complexity Tests**
   - Add more complex systems to `data/benchmark_systems.json` to test the limits of the approach
   - Try systems with more state variables or more complex dynamics

All experiment results will be saved to CSV files for later analysis and comparison.

---

## Unified Experiment Runner

A unified experiment runner script has been added to simplify the process of running experiments.

### Overview

The `run_experiments.py` script provides a single command to execute the entire pipeline or specific parts of it. It handles:

- Setting up directories
- Loading environment variables from .env file or config
- Running each step in sequence
- Logging progress and errors
- Saving experiment configurations for reproducibility

### Basic Usage

To run the complete experiment pipeline:

```bash
python run_experiments.py
```

This will execute all steps in sequence using settings from `config.yaml`.

### Environment Variables

You can manage environment variables in three ways:

1. **Using a .env file** (recommended):
   ```bash
   # Create a template .env file
   python run_experiments.py --create-env-template
   
   # Edit the file with your credentials
   nano .env
   ```

2. **Using config.yaml**:
   - Add your credentials to the `env_vars` section in `config.yaml`
   - Run with the `--env-from-config` flag:
     ```bash
     python run_experiments.py --env-from-config
     ```

3. **Using system environment variables** (traditional method):
   ```bash
   export MATHPIX_APP_ID="your_id"
   export MATHPIX_APP_KEY="your_key"
   export UNPAYWALL_EMAIL="your_email"
   ```

### Running Specific Steps

You can run specific parts of the pipeline:

```bash
# Run only data fetching
python run_experiments.py --only-data-fetch

# Run only knowledge base building
python run_experiments.py --only-kb-build

# Run only fine-tuning
python run_experiments.py --only-finetune

# Run only evaluation
python run_experiments.py --only-evaluate

# Run a single test example
python run_experiments.py --test-example
```

Or skip specific steps:

```bash
# Skip data fetching and KB building (if already done)
python run_experiments.py --skip-data-fetching --skip-kb-building
```

### Experiment Variations

Control experiment parameters directly:

```bash
# Change RAG context size
python run_experiments.py --rag-k 5

# Use a different benchmark file
python run_experiments.py --benchmark-file data/custom_benchmarks.json

# Save results to a specific file
python run_experiments.py --results-file output/experiment1_results.csv
```

### Experiment Tracking

Track and save experiment configurations:

```bash
# Name and describe your experiment for tracking
python run_experiments.py --experiment-name "experiment1" --description "Testing RAG with k=5"
```

This saves a copy of the configuration and metadata in the `experiments/experiment1/` directory.

### Help

For a full list of options:

```bash
python run_experiments.py --help
```

---

## Configuration

This project uses a central configuration file (`config.yaml`) located at the project root, managed using the OmegaConf library.

### `config.yaml`

Contains parameters for all pipeline stages. Relative paths are resolved based on project root. Edit this file to change defaults.

**Key Sections:**

*   `paths`: Defines input/output directories and key file locations.
*   `data_fetching`: Parameters for `data_fetching/paper_fetcher.py`.
*   `knowledge_base`: Parameters for `knowledge_base/knowledge_base_builder.py`.
*   `fine_tuning`: Parameters for `fine_tuning/finetune_llm.py`.
*   `inference`: Parameters for `inference/generate_certificate.py`.
*   `evaluation`: Parameters for `evaluation/evaluate_pipeline.py` and `evaluation/verify_certificate.py`.

### Environment Variables

Set required API keys/emails as environment variables (see [Prerequisites](#prerequisites)).

### Running Scripts with Custom Config / Overrides

Most executable scripts accept `--config /path/to/custom_config.yaml`. Some allow further overrides (check script `--help`).

---

## Verification Limitations

⚠️ **The current verification (`evaluation/verify_certificate.py`) has limitations.**

*   **SOS:** Formal verification for polynomial systems only (requires MOSEK/SCS). Experimental.
*   **Symbolic Checks:** Basic, often inconclusive.
*   **Numerical Checks:** Sampling/Optimization find counterexamples but do not provide formal proof.

**SOS (if applicable/successful) provides the strongest guarantee.**

---

## Author / Context

Developed by **Patrick Cooper** at **CU Boulder**.

---

## Future Work / Enhancements

*   Refine SOS implementation in `verify_certificate.py`.
*   Improve optimization-based falsification.
*   Explore alternative PDF parsing (e.g., GROBID + MathPix).
*   Semi-automate fine-tuning data extraction using MMD structure.
*   Improve robustness of LLM output parsing.
*   Experiment with different models, embeddings, etc.
*   Add command-line overrides for more config parameters.
*   Develop a UI.

---

## Troubleshooting

### CUDA Issues

**Problem: "AssertionError: Torch not compiled with CUDA enabled"**

This error occurs when PyTorch was installed without CUDA support but the fine-tuning script attempts to use CUDA.

**Solution:**
1. Run the setup script to reinstall PyTorch with CUDA support:
   ```bash
   python setup_environment.py --force-reinstall
   ```
2. Verify the installation:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda)"
   ```
   You should see `CUDA available: True` and a non-empty CUDA version.

**Problem: "RuntimeError: CUDA error: no kernel image is available for execution on the device"**

This typically means your CUDA toolkit version doesn't match the PyTorch build.

**Solution:**
1. Check your GPU capabilities:
   ```bash
   nvidia-smi
   ```
2. Based on your GPU's CUDA capability, reinstall PyTorch with the appropriate CUDA version.
   For newer GPUs (RTX 30xx, 40xx series):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   For older GPUs, stick with CUDA 11.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### General Issues

**Problem: Library or module not found errors**

**Solution:**
1. Ensure all dependencies are installed:
   ```bash
   python setup_environment.py
   ```
2. If specific packages are still missing, install them manually:
   ```bash
   pip install <package-name>
   ```

**Problem: API-related errors in data fetching or knowledge base construction**

**Solution:**
1. Verify your environment variables are set correctly:
   ```bash
   echo $MATHPIX_APP_ID
   echo $MATHPIX_APP_KEY
   echo $UNPAYWALL_EMAIL
   ```
2. Try using the .env file approach:
   ```bash
   python run_experiments.py --create-env-template
   # Edit the .env file with your credentials
   ```

**Problem: "Out of memory" errors during fine-tuning**

**Solution:**
1. Reduce batch size in `config.yaml`:
   ```yaml
   fine_tuning:
     training:
       per_device_train_batch_size: 1  # Try a smaller value
       gradient_accumulation_steps: 8  # Increase this value
   ```
2. Enable gradient checkpointing (already enabled by default)
3. Use a smaller model or increase quantization (e.g., enable 4-bit) 