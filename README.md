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

*   **Python:** Version 3.8 - 3.12 recommended.
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
    conda create -n fmllm python=3.12
    conda activate fmllm
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
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

*   ⚠️ **Requires CUDA GPU.**
*   Run:
    ```bash
    python fine_tuning/finetune_llm.py
    ```
*   Uses `data/ft_data_combined.jsonl` (default) and saves adapter to `output/finetuning_results/` (default).

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