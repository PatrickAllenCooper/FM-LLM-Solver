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
├── paper_population/
│   ├── data_fetching/              # Scripts for downloading papers
│   │   ├── __init__.py
│   │   ├── paper_fetcher.py        # Main script to fetch papers
│   │   └── user_ids.csv            # Example author IDs (optional input)
│   |
│   ├── knowledge_base/             # Scripts & data for the RAG knowledge base
│   │   ├── __init__.py
│   │   ├── knowledge_base_builder.py # Script to build vector index & metadata (uses MathPix API)
│   │   ├── test_knowledge_base.py    # Script to test KB retrieval
│   │   └── knowledge_base_mathpix/ # Default output dir for MathPix-based KB data
│   │       ├── paper_index_mathpix.faiss
│   │       └── paper_metadata_mathpix.jsonl
│   |
│   ├── fine_tuning/                # Scripts & data for fine-tuning the LLM
│   │   ├── __init__.py
│   │   ├── create_finetuning_data.py # Interactive script for manual data creation
│   │   ├── finetune_llm.py         # Script to run QLoRA fine-tuning
│   │   ├── generate_synthetic_data.py # Generates simple synthetic examples
│   │   ├── extract_from_papers.py  # Generates prompts for LLM-based extraction
│   │   └── combine_datasets.py     # Utility to merge datasets
│   │   # *.jsonl files are example outputs/inputs for datasets
│   |
│   ├── inference/                  # Scripts for running inference
│   │   ├── __init__.py
│   │   └── generate_certificate.py # Generates certificate using RAG + Fine-tuned LLM
│   |
│   ├── evaluation/                 # Scripts & data for pipeline evaluation
│   │   ├── __init__.py
│   │   ├── benchmark_systems.json    # Sample benchmark systems (incl. sampling bounds)
│   │   ├── evaluate_pipeline.py    # Main script to run evaluation
│   │   ├── verify_certificate.py     # Script for SOS, symbolic & numerical checks
│   │   └── evaluation_results.csv    # (Output) Example evaluation results
│   |
│   ├── utils/                      # Utility functions (e.g., config loader)
│   │   ├── __init__.py
│   │   └── config_loader.py
│   |
│   ├── requirements.txt            # Python package dependencies
│   └── README.md                   # (This file - will be moved)
|
├── recent_papers_all_sources_v2/ # Default output directory for fetched papers
|
├── results_barrier_certs/        # Default output directory for fine-tuning results
│   └── final_adapter/            # Saved LoRA adapter weights
|
├── config.yaml                   # Central configuration file
├── .gitignore
└── README.md                     # This file (Project root)
```

---

## Setup

### Prerequisites

*   **Python:** Version 3.8 - 3.12 recommended. (Versions >= 3.13 may have compatibility issues with dependencies like `spacy`).
*   **Git:** For cloning the repository.
*   **MathPix API Credentials:** Required for the default knowledge base builder (`knowledge_base_builder.py`). Obtain an App ID and App Key from [MathPix](https://mathpix.com/) and set them as environment variables:
    ```bash
    export MATHPIX_APP_ID='your_app_id'
    export MATHPIX_APP_KEY='your_app_key'
    ```
*   **UNPAYWALL Email:** Required for the data fetcher (`paper_fetcher.py`) for API politeness. Set as an environment variable:
    ```bash
    export UNPAYWALL_EMAIL='your-email@example.com'
    ```
*   **Tesseract OCR Engine:** Required by `pytesseract` if using OCR-based PDF extraction (not the default MathPix path).
    *   *Debian/Ubuntu:* `sudo apt update && sudo apt install tesseract-ocr`
    *   *macOS:* `brew install tesseract`
    *   *Windows:* Download installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) or build from source.
*   **CUDA Toolkit:** Required for GPU acceleration.
    *   Ensure compatibility with `torch` and `bitsandbytes` versions in `paper_population/requirements.txt`. Check NVIDIA's documentation for installation.
*   **SDP Solver (for SOS Verification):** To use the Sum-of-Squares verification functionality (for polynomial systems), you need to install a compatible Semidefinite Programming solver.
    *   **MOSEK:** Recommended (high-performance, commercial, free academic licenses available). Follow [MOSEK installation instructions](https://docs.mosek.com/latest/install/installation.html) and ensure `cvxpy` can find it.
    *   **SCS:** Good open-source alternative. Install via pip: `pip install scs`. `cvxpy` should detect it automatically.

### Installation

1.  **Clone:**
    ```bash
    # Replace with your repository URL
    git clone https://your-repository-url/FMLLMSolver.git
    cd FMLLMSolver
    ```

2.  **Create Environment (Recommended):**
    ```bash
    # Use a recommended Python version (e.g., 3.12)
    conda create -n fmllm python=3.12 # Or python -m venv venv
    conda activate fmllm             # Or source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r paper_population/requirements.txt
    # Install SCS if not using MOSEK for SOS:
    # pip install scs
    ```

4.  **Download SpaCy Model (Optional):** Only needed if `knowledge_base_builder.py` relies on it (current MathPix version primarily uses paragraph splitting).
    ```bash
    # python -m spacy download en_core_web_sm
    ```

---

## Workflow / Usage

Execute the steps in the following order. Ensure you are in the project root directory (`FMLLMSolver/`) and your environment is activated.

### 1. Data Fetching

*   **(Optional)** Modify `paper_population/data_fetching/user_ids.csv` if needed.
*   **Set Environment Variable:** `export UNPAYWALL_EMAIL='your-email@example.com'`
*   Run the script:
    ```bash
    python paper_population/data_fetching/paper_fetcher.py
    ```
*   Downloads PDFs based on author IDs to the directory specified in `config.yaml` (`paths.pdf_input_dir`, default: `recent_papers_all_sources_v2/`).

### 2. Build Knowledge Base

*   **Set MathPix Credentials:** Ensure `MATHPIX_APP_ID` and `MATHPIX_APP_KEY` environment variables are set.
*   Processes downloaded PDFs using the MathPix API.
    ```bash
    # Uses paths defined in config.yaml by default
    python paper_population/knowledge_base/knowledge_base_builder.py
    # Or specify a custom config
    # python paper_population/knowledge_base/knowledge_base_builder.py --config path/to/your_config.yaml
    ```
*   Creates the index (`.faiss`) and metadata (`.jsonl`) in the directory specified in `config.yaml` (`paths.kb_output_dir`, default: `paper_population/knowledge_base/knowledge_base_mathpix/`).

### 3. Test Knowledge Base (Optional)

*   Perform a quick check using the MathPix-generated knowledge base.
    ```bash
    python paper_population/knowledge_base/test_knowledge_base.py "What is a barrier certificate?" -k 3
    ```

### 4. Create/Prepare Fine-tuning Data

Choose one or more methods. Output paths are configurable via `config.yaml` or CLI overrides.

*   **Option A: Manual Creation**
    ```bash
    python paper_population/fine_tuning/create_finetuning_data.py
    ```
*   **Option B: Synthetic Generation**
    ```bash
    # (Assuming a synthetic generator script exists)
    # python paper_population/fine_tuning/generate_synthetic_data.py
    ```
*   **Option C: LLM-Assisted Extraction**
    1.  Generate prompts:
        ```bash
        python paper_population/fine_tuning/extract_from_papers.py
        ```
    2.  Manually run prompts (e.g., from `llm_extraction_prompts_mmd.txt`) with an external LLM.
    3.  **CRITICALLY REVIEW** LLM outputs.
    4.  Save **verified** JSON objects to the file specified in `config.yaml` (`paths.ft_extracted_data_file`).
*   **Combine Datasets:**
    ```bash
    # Uses default input/output paths from config.yaml
    python paper_population/fine_tuning/combine_datasets.py
    # Or specify patterns/output via CLI
    # python paper_population/fine_tuning/combine_datasets.py --input_patterns "path/to/manual.jsonl" "path/to/extracted.jsonl" --output_file path/to/combined.jsonl
    ```

### 5. Fine-tune the LLM

*   ⚠️ **Requires a CUDA-enabled GPU.**
*   Execute the fine-tuning script. It will use the combined data file and output directory specified in `config.yaml` by default.
    ```bash
    python paper_population/fine_tuning/finetune_llm.py
    # Or specify a custom config
    # python paper_population/fine_tuning/finetune_llm.py --config path/to/your_config.yaml
    ```
*   Saves the LoRA adapter weights to the directory specified in `config.yaml` (`paths.ft_output_dir`).

### 6. Generate Certificate (Inference)

*   Runs inference using the RAG pipeline and the fine-tuned adapter.
    ```bash
    python paper_population/inference/generate_certificate.py \
      "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5"
    ```
*   Uses model, adapter, and KB paths specified in `config.yaml` by default.

### 7. Evaluate the Pipeline

*   **Populate Benchmark:** Add diverse systems to the file specified in `config.yaml` (`paths.eval_benchmark_file`).
*   Run the evaluation script:
    ```bash
    python paper_population/evaluation/evaluate_pipeline.py
    # Or specify a custom config
    # python paper_population/evaluation/evaluate_pipeline.py --config path/to/your_config.yaml
    ```
*   Uses benchmark file, adapter, KB paths, and saves results to paths specified in `config.yaml` by default.

---

## Configuration

This project uses a central configuration file (`config.yaml`) located at the project root, managed using the OmegaConf library.

### `config.yaml`

This file contains parameters for all stages of the pipeline: data fetching, knowledge base creation, fine-tuning, inference, and evaluation. Paths defined within the `paths` section are typically relative to the project root and are resolved automatically by the loading script.

**Key Sections:**

*   `paths`: Defines input/output directories and key file locations.
*   `data_fetching`: Parameters for `paper_fetcher.py`.
*   `knowledge_base`: Parameters for `knowledge_base_builder.py`.
*   `fine_tuning`: Parameters for `finetune_llm.py`, including model choice, LoRA, quantization, and training settings.
*   `inference`: Parameters for `generate_certificate.py`.
*   `evaluation`: Parameters for `evaluate_pipeline.py` and `verify_certificate.py`.

You can modify this file directly to change default behaviors and settings.

### Environment Variables

Sensitive information, such as API keys, should **not** be stored directly in `config.yaml`. Instead, set them as environment variables:

*   `UNPAYWALL_EMAIL`: Your email address for the Unpaywall API (used in data fetching).
*   `MATHPIX_APP_ID`: Your Mathpix App ID (used in knowledge base building).
*   `MATHPIX_APP_KEY`: Your Mathpix App Key (used in knowledge base building).
*   `SEMANTIC_SCHOLAR_API_KEY`: (Optional) Your Semantic Scholar API key for potentially higher rate limits.

The scripts that require these variables will check for their presence in the environment.

### Running Scripts with Custom Config / Overrides

Most executable scripts (e.g., `knowledge_base_builder.py`, `finetune_llm.py`) now accept an optional `--config` argument to specify the path to the configuration file:

```bash
python paper_population/fine_tuning/finetune_llm.py --config /path/to/your/custom_config.yaml
```

If omitted, the default `config.yaml` at the project root will be used. Some scripts may also allow overriding specific configuration values via additional command-line arguments (check the script's `--help` message).

---

## Verification Limitations

⚠️ **The current verification (`paper_population/evaluation/verify_certificate.py`) attempts multiple methods but still has limitations, especially regarding formal guarantees.**

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
*   **Optimization Falsification:** Finish testing and refinement of the optimization-based checks in `verify_certificate.py`.
*   **PDF Parsing:** Explore integrating GROBID for document structure analysis *in addition* to MathPix for math/text extraction.
*   **Fine-tuning Data:** Explore semi-automated methods for extracting (System, Certificate) pairs, **leveraging the structured MathPix MMD output** to potentially guide LLM extraction more effectively.
*   **LLM Output Parsing:** Improve the robustness of `extract_certificate_from_llm_output`.
*   **Experimentation:** Test different base LLMs, embedding models, vector databases, and fine-tuning strategies.
*   **Configuration:** Potentially allow command-line overrides for more parameters using OmegaConf's CLI support.
*   **UI:** Develop a simple graphical or web interface. 