# FM-LLM Solver

A research project that uses Large Language Models (LLMs) to generate barrier certificates for autonomous systems, combining Retrieval-Augmented Generation (RAG) with fine-tuning techniques.

## Overview

FM-LLM Solver accelerates formal verification by using LLMs to propose barrier certificate candidates for dynamical systems. Instead of replacing rigorous verification, it helps researchers by generating high-quality hypotheses that can be formally verified using established methods.

## Key Features

- **RAG-Enhanced Generation**: Leverages a knowledge base built from research papers
- **Fine-Tuned Models**: Specialized LLMs trained on barrier certificate examples
- **Verification Pipeline**: Symbolic and numerical validation of generated certificates
- **Web Interface**: Secure, rate-limited web app with authentication
- **Cost-Effective Deployment**: Hybrid architecture for 80-95% cost savings
- **Comprehensive Monitoring**: Usage tracking, cost analysis, and performance metrics

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for fine-tuning)
- API credentials for Mathpix and Unpaywall

### Installation

```bash
# Clone repository
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# Set up environment
conda create -n fmllm python=3.10
conda activate fmllm

# Install with CUDA support
python scripts/setup/setup_environment.py
```

### Basic Usage

1. **Generate a barrier certificate:**
```bash
python inference/generate_certificate.py \
  "System: dx/dt = -x^3 - y, dy/dt = x - y^3. Initial: x^2+y^2 <= 0.1. Unsafe: x >= 1.5"
```

2. **Run web interface:**
```bash
python scripts/init_security.py  # First time only
python run_web_interface.py
```
Access at http://localhost:5000

3. **Evaluate on benchmarks:**
```bash
python evaluation/evaluate_pipeline.py
```

## Documentation

### Core Guides
- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions
- [User Guide](docs/USER_GUIDE.md) - Complete usage documentation
- [API Reference](docs/API_REFERENCE.md) - REST and Python APIs
- [Development Guide](docs/DEVELOPMENT.md) - Contributing guidelines

### Feature Documentation
- [Security](docs/SECURITY.md) - Authentication and protection features
- [Monitoring](docs/MONITORING.md) - Usage and cost tracking
- [Optimization](docs/OPTIMIZATION.md) - Performance tuning for limited hardware
- [Verification](docs/VERIFICATION.md) - Certificate validation methods
- [Features Overview](docs/FEATURES.md) - Complete feature list

### Additional Resources
- [Experiments](docs/EXPERIMENTS.md) - Benchmark configurations
- [Hybrid Deployment](HYBRID_DEPLOYMENT.md) - Cloud deployment guide

## Project Structure

```
FM-LLM-Solver/
├── inference/          # Certificate generation
├── evaluation/         # Verification and benchmarking
├── fine_tuning/        # Model training
├── knowledge_base/     # RAG implementation
├── web_interface/      # Flask application
├── scripts/            # Utility scripts
├── config/             # Configuration files
└── docs/               # Documentation
```

## Configuration

Edit `config/config.yaml` to customize:
- Model selection (base or fine-tuned)
- RAG parameters
- Deployment mode (local/hybrid/cloud)
- Security settings

## Citation

If you use this work in your research, please cite:
```bibtex
@software{fmllmsolver2024,
  title = {FM-LLM Solver: Barrier Certificate Generation using Large Language Models},
  author = {Cooper, Patrick Allen},
  year = {2024},
  institution = {University of Colorado Boulder}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Developed at the University of Colorado Boulder with support from the Autonomous Systems research group. 