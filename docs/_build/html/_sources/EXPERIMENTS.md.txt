# Barrier Certificate Experimentation Framework

This document provides instructions for running parameterized experiments with the barrier certificate generation pipeline. The framework allows systematic testing of different configurations across multiple dimensions:

1. **LLM Models**: Compare different model architectures and sizes
2. **Knowledge Base Configuration**: Test different embedding models and retrieval settings
3. **System Types**: Evaluate performance across different dynamical system categories
4. **Verification Methods**: Compare different verification approaches

## Prerequisites

Before running experiments, ensure you have:

1. Set up the environment as described in the main README.md
2. The required API keys set in your environment variables or .env file
3. Completed at least one successful run of the basic pipeline

## Quick Start

For a simple experiment testing different RAG retrieval settings:

```bash
./run_barrier_certificate_experiments.sh --type rag --rag-k "3,5,7,10"
```

This will run experiments varying only the RAG k parameter, using the default model and other settings.

## Running Experiments

The experiment runner script provides a convenient interface for running various experiment configurations.

### Basic Usage

```bash
./run_barrier_certificate_experiments.sh [options]
```

### Common Options

- `--type TYPE`: Predefined experiment type (full, quick, model, rag, system, verification)
- `--models MODELS`: Comma-separated list of models to test
- `--rag-k VALUES`: Comma-separated list of RAG k values to test
- `--embeddings MODELS`: Comma-separated list of embedding models to test
- `--dimensions DIMS`: Comma-separated list of specific dimensions to vary
- `--limit N`: Limit number of experiments to run
- `--random-sample`: Randomly sample experiments when using limit
- `--no-combinations`: Run each dimension independently (no cross-product)

### Predefined Experiment Types

- `full`: All combinations of parameters (may generate many experiments)
- `quick`: Single dimension variation (model OR rag OR system OR verification)
- `model`: Test different LLM models only
- `rag`: Test different RAG settings only
- `system`: Test different system types only
- `verification`: Test different verification methods only

### Examples

**Test different models**:
```bash
./run_barrier_certificate_experiments.sh --type model --models "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-2-7b-chat-hf"
```

**Compare RAG settings**:
```bash
./run_barrier_certificate_experiments.sh --type rag --rag-k "3,5,7,10"
```

**Test system categories**:
```bash
./run_barrier_certificate_experiments.sh --type system
```

**Multi-dimensional experiment with limit**:
```bash
./run_barrier_certificate_experiments.sh --dimensions "model,knowledge_base" --limit 10 --random-sample
```

**Quick experiment across all dimensions (no combinations)**:
```bash
./run_barrier_certificate_experiments.sh --type quick
```

## Understanding Experiment Results

After running experiments, results are saved in the `experiments` directory (or custom directory if specified). Each experiment batch gets its own timestamped directory.

### Raw Results Structure

A typical experiment directory contains:
- `experiment_plan.json`: Details of the experiment setup
- `all_experiments_summary.csv`: Summary of all experiment results
- Individual experiment subdirectories with:
  - `config.yaml`: Configuration used for this experiment
  - `results.csv`: Detailed results for each system
  - `summary.json`: Summary metrics for this experiment
  - Log files

### Analyzing Results

To analyze experiment results, use the provided analysis script:

```bash
python analyze_experiment_results.py --experiment-dir experiments/experiment_batch_TIMESTAMP
```

This generates comprehensive analysis including:
- Success rates across different dimensions
- Certificate complexity analysis
- Verification method effectiveness
- Cross-dimensional correlations
- System-specific performance
- Visualization plots and charts

The analysis outputs are saved to an `analysis` subdirectory within the experiment directory, including:
- Visualization plots in `analysis_plots/`
- A comprehensive markdown report `comprehensive_analysis_report.md`

## Advanced: Custom Experiment Dimensions

For more advanced experimentation, you can modify:

1. **System subsets**: Edit `define_system_filters()` in `run_parameterized_experiments.py`
2. **Verification settings**: Edit `create_verification_variants()` in `run_parameterized_experiments.py`
3. **Model variants**: Add to the model list in the script or command line

## Best Practices

1. **Start small**: Begin with quick experiments on a single dimension
2. **Use limits**: For multi-dimensional experiments, use `--limit` to avoid excessive runtime
3. **Skip when possible**: Use `--skip-data-fetching` and `--skip-kb-building` to avoid redundant operations
4. **Analyze incrementally**: Run analysis after each batch to guide subsequent experiments
5. **Document configurations**: Keep notes on which configurations perform best for your use case

## Troubleshooting

If experiments fail:
- Check log files in the experiment directories
- Ensure all prerequisites are met
- Verify API keys are correctly set
- Check for CUDA/GPU availability if using models that require it

For more help:
```bash
./run_barrier_certificate_experiments.sh --help
``` 