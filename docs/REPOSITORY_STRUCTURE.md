# Repository Structure

This document describes the organized structure of the FM-LLM-Solver repository after reorganization.

## Overview

The repository has been reorganized to reduce clutter at the root level and improve maintainability. Files are now logically grouped into subdirectories based on their purpose.

## Directory Structure

```
FM-LLM-Solver/
├── config/                           # Configuration files
│   ├── config.yaml                   # Main configuration file
│   ├── config.yaml.bak              # Backup configuration
│   └── env.example                  # Environment variables template
│
├── scripts/                          # All executable scripts
│   ├── analysis/                     # Analysis and comparison scripts
│   │   ├── analyze_comparison_results.py
│   │   ├── analyze_experiment_results.py
│   │   ├── compare_models.py
│   │   └── compare_model_sizes.py
│   │
│   ├── batch/                        # Batch and shell scripts
│   │   ├── *.bat                     # Windows batch files
│   │   └── *.sh                      # Unix shell scripts
│   │
│   ├── build/                        # Build and knowledge base scripts
│   │   ├── kb_builder.py             # Main knowledge base builder
│   │   ├── build_*.py                # Build scripts
│   │   ├── rebuild_*.py              # Rebuild scripts
│   │   ├── optimize_*.py             # Optimization scripts
│   │   └── *debug*.py                # Debug scripts
│   │
│   ├── run/                          # Main execution scripts
│   │   ├── run_experiments.py        # Main experiment runner
│   │   ├── run_parameterized_experiments.py
│   │   ├── run_mathpix_kb.py
│   │   ├── run_kb_build*.py
│   │   └── run_batch_kb_build.py
│   │
│   └── setup/                        # Setup and installation scripts
│       ├── setup_environment.py      # Environment setup
│       ├── install_deps.py           # Dependency installation
│       └── install_tqdm.py           # TQDM installation
│
├── tests/                            # Test files
│   └── test_*.py                     # Test scripts
│
├── docs/                             # Documentation
│   ├── README.md                     # Main documentation
│   ├── REPOSITORY_STRUCTURE.md       # This file
│   ├── EXPERIMENTS.md                # Experiment documentation
│   ├── KB_BUILDER_README.md          # Knowledge base builder docs
│   ├── OPTIMIZED_README.md           # Optimization documentation
│   ├── AWQ_MODELS.md                 # AWQ models documentation
│   ├── MEMORY_OPTIMIZATION.md        # Memory optimization guide
│   ├── PROJECT_STRUCTURE.md          # Project structure overview
│   └── DISCRETE_CONTINUOUS_BARRIER_CERTIFICATES.md
│
├── requirements/                     # Requirements files
│   ├── requirements.txt              # Main requirements
│   └── open_source_kb_requirements.txt # Open source KB requirements
│
├── logs/                             # Log files
│   └── experiment_run.log            # Experiment logs
│
├── data/                             # Data directory
├── output/                           # Output directory
├── kb_data/                          # Knowledge base data
├── kb_data_discrete/                 # Discrete barrier certificate KB
├── knowledge_base/                   # Knowledge base modules
├── fine_tuning/                      # Fine-tuning modules
├── inference/                        # Inference modules
├── evaluation/                       # Evaluation modules
├── utils/                            # Utility modules
├── data_fetching/                    # Data fetching modules
│
├── config.yaml                       # Config file (copy for compatibility)
├── .gitignore                        # Git ignore file
└── LICENSE                           # License file
```

## Key Changes

### Files Moved

1. **Configuration Files** → `config/`
   - `config.yaml`, `config.yaml.bak`, `env.example`

2. **Scripts** → `scripts/` (with subdirectories)
   - Analysis scripts → `scripts/analysis/`
   - Batch/Shell files → `scripts/batch/`
   - Build scripts → `scripts/build/`
   - Run scripts → `scripts/run/`
   - Setup scripts → `scripts/setup/`

3. **Test Files** → `tests/`
   - All `test_*.py` files

4. **Documentation** → `docs/`
   - All `.md` files (except README.md which is kept at root)

5. **Requirements** → `requirements/`
   - `requirements.txt` and related files

6. **Logs** → `logs/`
   - Log files

### Backward Compatibility

- A copy of `config.yaml` is maintained at the root level for scripts that expect it there
- All relative paths in configuration files remain valid
- Existing import statements should continue to work

## Usage Notes

### Running Scripts

Scripts can still be run from the root directory using their new paths:

```bash
# Analysis
python scripts/analysis/analyze_experiment_results.py

# Experiments
python scripts/run/run_experiments.py

# Knowledge base building
python scripts/build/kb_builder.py

# Batch scripts
scripts/batch/run_experiments.bat
```

### Configuration

The main configuration file is now at `config/config.yaml`, but a copy exists at the root level for compatibility.

### Adding New Files

When adding new files, please follow this organization:
- Scripts go in appropriate `scripts/` subdirectories
- Documentation goes in `docs/`
- Tests go in `tests/`
- Configuration in `config/`
- Requirements in `requirements/`

## Benefits

1. **Cleaner Root Directory**: Reduced from 50+ files to essential directories and key files
2. **Logical Organization**: Files grouped by purpose and functionality
3. **Better Maintainability**: Easier to find and manage related files
4. **Improved Navigation**: Clear directory structure for developers
5. **Scalability**: Easy to add new files in appropriate locations

## Migration Guide

If you have local scripts or configurations that reference the old file locations, update the paths as follows:

- `config.yaml` → `config/config.yaml` (or use the root copy)
- `run_*.py` → `scripts/run/run_*.py`
- `analyze_*.py` → `scripts/analysis/analyze_*.py`
- `*.bat`, `*.sh` → `scripts/batch/*.bat`, `scripts/batch/*.sh`
- `requirements.txt` → `requirements/requirements.txt`

The repository structure now follows best practices for Python projects and provides a solid foundation for continued development. 