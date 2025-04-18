# FMLLMSolver: Barrier Certificate Generation using LLMs

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
    *   [5. Run an Initial Training Test (Optional)](#5-run-an-initial-training-test-optional)
    *   [6. Fine-tune the LLM (Full)](#6-fine-tune-the-llm-full)
    *   [7. Generate Certificate (Inference)](#7-generate-certificate-inference)
    *   [8. Evaluate the Pipeline](#8-evaluate-the-pipeline)
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
paper_population/
|
├── data_fetching/              # Scripts for downloading papers
│   ├── __init__.py
│   ├── paper_fetcher.py        # Main script to fetch papers
│   └── user_ids.csv            # Example author IDs (optional input)
|
├── knowledge_base/             # Scripts & data for the RAG knowledge base
│   ├── __init__.py
│   ├── knowledge_base_builder.py # Script to build vector index & metadata
│   ├── test_knowledge_base.py    # Script to test KB retrieval
│   └── knowledge_base_enhanced/  # Default output dir for KB data
│       ├── paper_index_enhanced.faiss
│       └── paper_metadata_enhanced.json
|
├── fine_tuning/                # Scripts & data for fine-tuning the LLM
│   ├── __init__.py
│   ├── create_finetuning_data.py # Interactive script for manual data creation
│   ├── finetune_llm.py         # Script to run QLoRA fine-tuning
│   ├── generate_synthetic_data.py # Generates simple synthetic examples
│   ├── extract_from_papers.py  # Generates prompts for LLM-based extraction
│   └── combine_datasets.py     # Utility to merge datasets
│   # *.jsonl files are example outputs/inputs for datasets
|
├── inference/                  # Scripts for running inference
│   ├── __init__.py
│   └── generate_certificate.py # Generates certificate using RAG + Fine-tuned LLM
|
├── evaluation/                 # Scripts & data for pipeline evaluation
│   ├── __init__.py
│   ├── benchmark_systems.json    # Sample benchmark systems (incl. sampling bounds)
│   ├── evaluate_pipeline.py    # Main script to run evaluation
│   ├── verify_certificate.py     # Script for symbolic & numerical checks (basic)
│   └── evaluation_results.csv    # (Output) Example evaluation results
|
├── recent_papers_all_sources_v2/ # Default output directory for fetched papers
|
├── results_barrier_certs/        # Default output directory for fine-tuning results
│   └── final_adapter/            # Saved LoRA adapter weights
|
├── install.sh                  # Original install script (review needed)
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

---

## Setup

### Prerequisites

*   **Python:** Version 3.8 - 3.12 recommended. (Versions >= 3.13 may have compatibility issues with dependencies like `spacy`).
*   **Git:** For cloning the repository.
*   **Tesseract OCR Engine:** Required by `pytesseract` for OCR.
    *   *Debian/Ubuntu:* `sudo apt update && sudo apt install tesseract-ocr`
    *   *macOS:* `brew install tesseract`
    *   *Windows:* Download installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) or build from source.
*   **CUDA Toolkit:** Required for GPU acceleration.
    *   Ensure compatibility with `torch` and `bitsandbytes` versions in `requirements.txt`. Check NVIDIA's documentation for installation.
*   **SDP Solver (for SOS Verification):** To use the Sum-of-Squares verification functionality (for polynomial systems), you need to install a compatible Semidefinite Programming solver.
    *   **MOSEK:** Recommended (high-performance, commercial, free academic licenses available). Follow [MOSEK installation instructions](https://docs.mosek.com/latest/install/installation.html) and ensure `cvxpy` can find it.
    *   **SCS:** Good open-source alternative. Install via pip: `pip install scs`. `cvxpy` should detect it automatically.

### Installation

1.  **Clone:**
    ```bash
    # Replace with your repository URL
    git clone https://your-repository-url/FMLLMSolver.git
    cd FMLLMSolver/paper_population
    ```

2.  **Create Environment (Recommended):**
    ```bash
    # Use a recommended Python version (e.g., 3.12)
    conda create -n fmllm python=3.12 # Or python -m venv venv
    conda activate fmllm             # Or source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Install SCS if not using MOSEK for SOS:
    # pip install scs
    ```

4.  **Download SpaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

---

## Workflow / Usage

Execute the steps in the following order. Ensure you are in the `FMLLMSolver/paper_population` directory and your environment is activated.

### 1. Data Fetching

*   **(Optional)** Modify `data_fetching/user_ids.csv` if needed.
*   Run the script:
    ```bash
    python data_fetching/paper_fetcher.py
    ```
*   Downloads PDFs to `recent_papers_all_sources_v2/` (by default).

### 2. Build Knowledge Base

*   Processes downloaded PDFs, performs OCR, chunks, embeds, and indexes the text.
    ```bash
    python knowledge_base/knowledge_base_builder.py
    ```
*   Creates the index (`.faiss`) and metadata (`.json`) in `knowledge_base/knowledge_base_enhanced/`.

### 3. Test Knowledge Base (Optional)

*   Perform a quick check on the knowledge base retrieval.
    ```bash
    python knowledge_base/test_knowledge_base.py "What is a barrier certificate?" -k 3
    ```

### 4. Create/Prepare Fine-tuning Data

This is the most critical step for model performance. Choose one or more methods:

*   **Option A: Manual Creation** (Requires domain expertise, Recommended for high quality)
    ```bash
    python fine_tuning/create_finetuning_data.py --output_file fine_tuning/manual_data.jsonl
    ```
    Follow prompts to enter verified (System, Certificate) pairs.
*   **Option B: Synthetic Generation** (Good for basic examples, limited scope)
    ```bash
    python fine_tuning/generate_synthetic_data.py --output_file fine_tuning/synthetic_data.jsonl
    ```
*   **Option C: LLM-Assisted Extraction** (Requires powerful external LLM + **Mandatory Manual Review**)
    1.  `python fine_tuning/extract_from_papers.py --output_instructions_file prompts_to_run.txt`
    2.  Manually run prompts from `prompts_to_run.txt` with an external LLM (e.g., GPT-4, Claude 3).
    3.  **CRITICALLY REVIEW** LLM outputs for correctness against source papers.
    4.  Save **verified** JSON objects (one per line) to `fine_tuning/extracted_data_verified.jsonl`.
*   **Combine Datasets:** Merge desired sources into a final training file. The `combine_datasets.py` script attempts to read `metadata.source` or infer source from filenames.
    ```bash
    # Example: Combine manual, synthetic, and verified extracted data
    python fine_tuning/combine_datasets.py \\\
        fine_tuning/manual_data.jsonl \\\
        fine_tuning/synthetic_data.jsonl \\\
        fine_tuning/extracted_data_verified.jsonl \\\
        --output_file fine_tuning/finetuning_data_combined.jsonl

    # Example: Create dataset with only synthetic data
    # python fine_tuning/combine_datasets.py fine_tuning/synthetic_data.jsonl --output_file fine_tuning/finetuning_data_synthetic_only.jsonl
    ```

### 5. Run an Initial Training Test (Optional)

This section guides you through performing a minimal run of the fine-tuning pipeline using only synthetic data to ensure the scripts execute correctly. **Note:** This is likely insufficient for producing an effective model.

*   **Prerequisites:** Ensure steps 1 and 2 (Fetching, Building KB) are done. Ensure [Setup](#setup) is complete.
*   **Generate Synthetic Data:**
    ```bash
    python fine_tuning/generate_synthetic_data.py --output_file fine_tuning/synthetic_data.jsonl
    ```
*   **Prepare Synthetic-Only Dataset for Trainer:**
    ```bash
    python fine_tuning/combine_datasets.py \\\
        fine_tuning/synthetic_data.jsonl \\\
        --output_file fine_tuning/finetuning_data_initial_test.jsonl \\\
    ```
*   **Run Fine-tuning (Requires GPU):**
    ```bash
    python fine_tuning/finetune_llm.py \\\
        --data_path fine_tuning/finetuning_data_initial_test.jsonl \\\
        --output_dir ./results_initial_test \\\
        --num_train_epochs 1 # Keep epochs low for initial test
    ```
*   Monitor output. If successful, adapter weights appear in `./results_initial_test/final_adapter`.

### 6. Fine-tune the LLM (Full)

*   ⚠️ **Requires a CUDA-enabled GPU with sufficient VRAM** (e.g., >16GB for 7B/8B models with QLoRA).
*   Execute the fine-tuning script, pointing to your **high-quality combined dataset** created in step 4.
    ```bash
    python fine_tuning/finetune_llm.py \\\
        --data_path fine_tuning/finetuning_data_combined.jsonl \\\
        --output_dir ./results_full_run1 \\\
        --num_train_epochs 3 # Adjust as needed
    ```
*   Saves the LoRA adapter weights to the specified output directory (e.g., `./results_full_run1/final_adapter`).

### 7. Generate Certificate (Inference)

*   Uses the RAG pipeline with the fine-tuned adapter to propose a certificate.
    ```bash
    python inference/generate_certificate.py \\\
      "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5" \\\
      --adapter ./results_full_run1/final_adapter # Use adapter from full run
    ```
*   Replace the system description text. Ensure the `--adapter` path is correct.

### 8. Evaluate the Pipeline

*   **Populate Benchmark:** Add diverse systems to `evaluation/benchmark_systems.json`. Ensure `sampling_bounds` are defined. For SOS checks, ensure systems, certificates, and set definitions are polynomial and defined using `>= 0` or `<= 0` inequalities (parsed to `>=0` form by `relationals_to_polynomials`).
*   Run the full evaluation pipeline using your trained adapter:
    ```bash
    python evaluation/evaluate_pipeline.py \\\
      --adapter ./results_full_run1/final_adapter \\\
      --benchmark evaluation/benchmark_systems.json \\\
      --results evaluation/evaluation_full_run1.csv \\\
      # Optional: Disable SOS checks if solver not installed or for non-poly benchmarks
      # --no_sos \\\
      # Optional: Disable optimization checks
      # --no_opt \
    ```
*   This script runs generation, attempts parsing, performs verification (SOS if applicable, symbolic, numerical sampling, optimization), and saves results.

---

## Configuration

Default paths, model names, and hyperparameters are set as constants near the top of each relevant Python script. Many of these can be overridden using command-line arguments.

Run scripts with `-h` or `--help` to see available arguments, for example:

```bash
python fine_tuning/finetune_llm.py --help
python evaluation/evaluate_pipeline.py --help
```

---

## Verification Limitations

⚠️ **The current verification (`evaluation/verify_certificate.py`) attempts multiple methods but still has limitations, especially regarding formal guarantees.**

*   **Sum-of-Squares (SOS):**
    *   Uses `cvxpy` to formulate and solve SOS conditions as SDPs for **polynomial systems only**. Provides **formal verification** if the solver returns an optimal solution.
    *   Requires a separate SDP solver installation (**MOSEK** recommended, **SCS** alternative).
    *   **Current Implementation Note:** The logic for translating SymPy polynomials and SOS constraints into the specific CVXPY format (`calculate_sos_poly_coeffs`, `add_sos_constraints_poly`) is **complex and experimental**. It may require debugging or refinement for robust use across diverse polynomial forms.
    *   Failure (`infeasible` status) means the SOS relaxation (at the chosen degree) failed, but does not formally disprove the property.
*   **Symbolic Checks:** Basic checks for trivial cases (e.g., \(\dot{B} = 0\)) using `sympy`. Generally **inconclusive** for complex expressions.
*   **Numerical Checks (Sampling & Optimization):**
    *   Uses `numpy` and `scipy` for random sampling and `differential_evolution` based optimization (falsification).
    *   Can effectively find **counterexamples** for both polynomial and non-polynomial systems if they exist within the search bounds.
    *   These methods **do not provide formal proof** of validity; they only demonstrate the absence of violations within the tested samples/optimization search.
    *   Set membership checks rely on `sympy` parsing and numerical evaluation, which is more robust than `eval` but may face precision issues.

For the highest confidence, **SOS verification (when applicable and successful) is preferred**. Numerical checks serve as a fallback and are the primary method for non-polynomial systems.

---

## Author / Context

This project was developed by **Patrick Cooper** as part of graduate work at the **University of Colorado Boulder (CU Boulder)**.

---

## Future Work / Enhancements

*   **SOS Implementation:** Refine and rigorously test the SymPy-to-CVXPY conversion and SOS constraint formulation in `verify_certificate.py`. Consider using dedicated SOS libraries (like `SumOfSquares.jl` via Python interface, or others) if the current approach proves too slow or difficult to maintain.
*   **Robust Verification:** Implement optimization-based falsification fully (currently structure is there, needs testing/refinement). Replace any remaining risky parsing/evaluation with safer methods.
*   **PDF Parsing:** Integrate GROBID (structure) or MathPix API (equations) into `knowledge_base_builder.py`.
*   **Fine-tuning Data:** Explore semi-automated methods for extracting (System, Certificate) pairs.
*   **Experimentation:** Test different base LLMs, embedding models, vector databases, and fine-tuning strategies.
*   **UI:** Develop a simple graphical or web interface.
 