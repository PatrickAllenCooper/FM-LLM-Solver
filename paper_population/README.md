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

This project implements a pipeline consisting of several distinct phases:

1.  **Data Fetching:** Acquiring relevant research papers (PDFs).
2.  **Knowledge Base Construction:** Processing papers (including OCR) into a searchable vector database (FAISS index + metadata) for RAG.
3.  **Fine-tuning:** Specializing an LLM (using QLoRA for efficiency) on examples of system dynamics and corresponding barrier certificates.
4.  **Inference:** Combining the knowledge base (RAG) and the fine-tuned LLM to generate barrier certificate candidates for new systems.
5.  **Evaluation:** Assessing the validity and quality of the generated certificates using basic symbolic checks (`sympy`) **and numerical sampling** (`numpy`, `scipy`) against a benchmark dataset.

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

*   **Python:** >= 3.8 recommended.
*   **Git:** For cloning the repository.
*   **Tesseract OCR Engine:** Required by `pytesseract` for OCR in Phase 2.
    *   *Debian/Ubuntu:* `sudo apt update && sudo apt install tesseract-ocr`
    *   *macOS:* `brew install tesseract`
    *   *Windows:* Download installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) or build from source.
*   **CUDA Toolkit:** Required for GPU acceleration (fine-tuning, inference). Ensure compatibility with `torch` and `bitsandbytes` versions in `requirements.txt`. Check NVIDIA's documentation for installation.

### Installation

1.  **Clone:**
    ```bash
    # Replace with your repository URL
    git clone https://your-repository-url/FMLLMSolver.git
    cd FMLLMSolver/paper_population
    ```

2.  **Create Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows (cmd/powershell)
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download SpaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

---

## Workflow / Usage

Execute the steps in the following order. Ensure you are in the `FMLLMSolver/paper_population` directory.

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

*   **Populate Benchmark:** Add diverse systems to `evaluation/benchmark_systems.json`. **Ensure `sampling_bounds` are defined** for each system.
*   Run the full evaluation pipeline using your trained adapter:
    ```bash
    python evaluation/evaluate_pipeline.py \\\
      --adapter ./results_full_run1/final_adapter \\\
      --benchmark evaluation/benchmark_systems.json \\\
      --results evaluation/evaluation_full_run1.csv
    ```
*   This script runs generation, attempts parsing, performs symbolic **and numerical** verification, and saves results.

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

⚠️ **The current verification (`evaluation/verify_certificate.py`) provides basic symbolic checks and more extensive numerical sampling checks, but it is NOT sufficient for formal safety guarantees.**

*   **Symbolic Checks:** Uses `sympy` to calculate the Lie derivative (\(\dot{B}\)). Checks if \(\dot{B}\) is identically zero, but other symbolic checks for \(\dot{B} \le 0\) are heuristic and limited.
*   **Numerical Checks:** Uses `numpy` and `scipy` (via `sympy.lambdify`) to sample points within specified `sampling_bounds`.
    *   Checks if \(\dot{B}(x) \le \epsilon\) for samples within the safe set.
    *   Checks boundary conditions (e.g., \(B(x) \le \epsilon\) in initial set, \(B(x) \ge -\epsilon\) outside unsafe set) on samples.
    *   **Provides empirical evidence but not formal proof.** A counterexample might be missed.
    *   Set membership checks currently use `eval` on condition strings, which requires trusted input in the benchmark file.
*   **Boundary Conditions:** Formal verification of boundary conditions remains challenging, especially symbolically.

For reliable verification, especially for publication or deployment, integration with more advanced methods like **Sum-of-Squares (SOS) programming** (for polynomial systems) or **robust optimization-based falsification** is necessary.

---

## Author / Context

This project was developed by **Patrick Cooper** as part of graduate work at the **University of Colorado Boulder (CU Boulder)**.

---

## Future Work / Enhancements

*   **PDF Parsing:** Integrate GROBID (structure) or MathPix API (equations) into `knowledge_base_builder.py`.
*   **Fine-tuning Data:** Explore semi-automated methods for extracting (System, Certificate) pairs.
*   **Robust Verification:** Implement optimization-based falsification or interface with SOS solvers in `verify_certificate.py`. Replace `eval`-based set checks with safer parsing.
*   **LLM Output Parsing:** Improve the robustness of `extract_certificate_from_llm_output`.
*   **Experimentation:** Test different base LLMs, embedding models, vector databases, and fine-tuning strategies.
*   **UI:** Develop a simple graphical or web interface.
 