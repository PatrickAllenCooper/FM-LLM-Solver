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
    *   [4. Create Fine-tuning Data](#4-create-fine-tuning-data)
    *   [5. Fine-tune the LLM](#5-fine-tune-the-llm)
    *   [6. Generate Certificate (Inference)](#6-generate-certificate-inference)
    *   [7. Evaluate the Pipeline](#7-evaluate-the-pipeline)
*   [Configuration](#configuration)
*   [Verification Limitations](#verification-limitations)
*   [Author / Context](#author--context)
*   [Future Work / Enhancements](#future-work--enhancements)

---

## Overview

This project implements a pipeline consisting of several distinct phases:

1.  **Data Fetching:** Acquiring relevant research papers (PDFs).
2.  **Knowledge Base Construction:** Processing papers (including OCR) into a searchable vector database (FAISS index + metadata) for RAG.
3.  **Fine-tuning:** Specializing an LLM (using QLoRA for efficiency) on manually created examples of system dynamics and corresponding barrier certificates.
4.  **Inference:** Combining the knowledge base (RAG) and the fine-tuned LLM to generate barrier certificate candidates for new systems.
5.  **Evaluation:** Assessing the validity and quality of the generated certificates using symbolic math (`sympy`) and a benchmark dataset (with limitations on verification rigor).

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
│   └── finetuning_data.jsonl     # (Output) Manually created training data
|
├── inference/                  # Scripts for running inference
│   ├── __init__.py
│   └── generate_certificate.py # Generates certificate using RAG + Fine-tuned LLM
|
├── evaluation/                 # Scripts & data for pipeline evaluation
│   ├── __init__.py
│   ├── benchmark_systems.json    # Sample benchmark systems for testing
│   ├── evaluate_pipeline.py    # Main script to run evaluation
│   ├── verify_certificate.py     # Script for symbolic verification (basic)
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

### 4. Create Fine-tuning Data

*   ⚠️ **Requires significant manual effort and domain expertise.**
*   Launch the interactive data creation tool:
    ```bash
    python fine_tuning/create_finetuning_data.py
    ```
*   Follow the prompts. Input system dynamics and corresponding **known, valid** barrier certificates. Use **LaTeX** for mathematical expressions where possible for consistency.
*   Saves data in JSON Lines format to `fine_tuning/finetuning_data.jsonl`.

### 5. Fine-tune the LLM

*   ⚠️ **Requires a CUDA-enabled GPU with sufficient VRAM** (e.g., >16GB for 7B/8B models with QLoRA).
*   Execute the fine-tuning script. Default uses Llama 3 8B Instruct.
    ```bash
    # Use defaults
    python fine_tuning/finetune_llm.py

    # Example: Specify different model, data, output dir, epochs
    # python fine_tuning/finetune_llm.py \
    #   --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    #   --data_path ./fine_tuning/my_custom_data.jsonl \
    #   --output_dir ./results_mistral_run1 \
    #   --num_train_epochs 3
    ```
*   Saves the LoRA adapter weights to the specified output directory (default: `results_barrier_certs/final_adapter`).

### 6. Generate Certificate (Inference)

*   Uses the RAG pipeline with the fine-tuned adapter to propose a certificate.
    ```bash
    python inference/generate_certificate.py \
      "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5" \
      --adapter ./results_barrier_certs/final_adapter
    ```
*   Replace the system description text. Ensure the `--adapter` path points to your trained adapter.

### 7. Evaluate the Pipeline

*   **Populate Benchmark:** Add diverse and representative systems to `evaluation/benchmark_systems.json`.
*   Run the full evaluation pipeline:
    ```bash
    python evaluation/evaluate_pipeline.py \
      --adapter ./results_barrier_certs/final_adapter \
      --benchmark evaluation/benchmark_systems.json \
      --results evaluation/my_run_results.csv
    ```
*   This script runs generation for each benchmark case, attempts parsing, calls the basic verifier, and saves results to a CSV file.

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

⚠️ **Crucially, the current symbolic verification (`evaluation/verify_certificate.py`) is basic and not sufficient for formal safety guarantees.**

*   It uses `sympy` to calculate the Lie derivative (\(\dot{B}\)) symbolically.
*   The check for \(\dot{B} \le 0\) within the safe region is **heuristic** (looks for simple non-positive forms) and not mathematically rigorous for general cases.
*   Checks for boundary conditions (e.g., \(B(x) \le 0\) on the initial set, \(B(x) > 0\) outside the unsafe set) are currently **placeholders** due to the difficulty of symbolic inequality proving.

For reliable verification, especially for publication or deployment, integration with more advanced methods like **Sum-of-Squares (SOS) programming** (for polynomial systems) or **robust numerical sampling/optimization** is necessary.

---

## Author / Context

This project was developed by **Patrick Cooper** as part of graduate work at the **University of Colorado Boulder (CU Boulder)**.

---

## Future Work / Enhancements

*   **PDF Parsing:** Integrate GROBID (structure) or MathPix API (equations) into `knowledge_base_builder.py`.
*   **Fine-tuning Data:** Explore semi-automated methods for extracting (System, Certificate) pairs from papers.
*   **Verification:** Implement numerical sampling within safe/initial/unsafe sets and/or interface with SOS solvers (e.g., via CVXPY/PySOS + MOSEK/SDPA) in `verify_certificate.py`.
*   **LLM Output Parsing:** Improve the robustness of `extract_certificate_from_llm_output` in `evaluate_pipeline.py`.
*   **Experimentation:** Test different base LLMs, embedding models, vector databases, and fine-tuning strategies.
*   **UI:** Develop a simple graphical or web interface.
 