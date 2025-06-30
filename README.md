# FM-LLM Solver

<p align="center">
  <img src="docs/assets/logo.png" alt="FM-LLM Solver Logo" width="200">
</p>

<p align="center">
  <a href="https://github.com/yourusername/FM-LLM-Solver/actions"><img src="https://github.com/yourusername/FM-LLM-Solver/workflows/CI/badge.svg" alt="CI Status"></a>
  <a href="https://codecov.io/gh/yourusername/FM-LLM-Solver"><img src="https://codecov.io/gh/yourusername/FM-LLM-Solver/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/fm-llm-solver/"><img src="https://img.shields.io/pypi/v/fm-llm-solver.svg" alt="PyPI"></a>
  <a href="https://fm-llm-solver.readthedocs.io/"><img src="https://readthedocs.org/projects/fm-llm-solver/badge/?version=latest" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

**FM-LLM Solver** is a cutting-edge system for generating and verifying barrier certificates for dynamical systems using Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) and fine-tuning capabilities.

## 🌟 Features

- **LLM-Powered Generation**: Generate barrier certificates using state-of-the-art language models
- **Multi-System Support**: Handle continuous, discrete, and stochastic dynamical systems
- **RAG Integration**: Leverage academic papers and examples for improved generation
- **Comprehensive Verification**: Multiple verification methods (numerical, symbolic, SOS)
- **Web Interface**: User-friendly interface for system input and visualization
- **API Access**: RESTful API for programmatic access
- **Extensible Architecture**: Modular design for easy extension and customization

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (coming soon)
pip install fm-llm-solver

# Install from source
git clone https://github.com/yourusername/FM-LLM-Solver.git
cd FM-LLM-Solver
pip install -e ".[all]"
```

### Basic Usage

```python
from fm_llm_solver import CertificateGenerator, SystemDescription

# Define your dynamical system
system = SystemDescription(
    dynamics={"x": "-x + y", "y": "x - y"},
    initial_set="x**2 + y**2 <= 0.5",
    unsafe_set="x**2 + y**2 >= 2.0"
)

# Generate a barrier certificate
generator = CertificateGenerator.from_config()
result = generator.generate(system)

if result.success:
    print(f"Certificate: {result.certificate}")
    print(f"Confidence: {result.confidence:.2%}")
else:
    print(f"Generation failed: {result.error}")
```

### Running the Web Interface

```bash
# Start the web interface
fm-llm-solver web

# Or with custom configuration
fm-llm-solver web --config config/production.yaml --host 0.0.0.0 --port 8080
```

### Running the API Server

```bash
# Start the inference API
fm-llm-solver api

# Run both web interface and API
fm-llm-solver both
```

## 📁 Project Structure

```
FM-LLM-Solver/
├── fm_llm_solver/          # Main package
│   ├── core/               # Core components (config, logging, types)
│   ├── services/           # Business logic services
│   ├── web/                # Flask web interface
│   ├── api/                # FastAPI inference API
│   └── utils/              # Utility functions
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── benchmarks/         # Performance benchmarks
├── config/                 # Configuration files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── data/                   # Data files
```

## 🔧 Configuration

Create a `config/config.yaml` file:

```yaml
model:
  provider: qwen
  name: Qwen/Qwen2.5-14B-Instruct
  temperature: 0.7
  device: cuda

rag:
  enabled: true
  k_retrieved: 3
  chunk_size: 1000

verification:
  methods: [numerical, symbolic]
  numerical:
    num_samples: 1000
    
security:
  rate_limit:
    requests_per_day: 50
```

## 🧪 Testing

```bash
# Run all tests
fm-llm-solver test

# Run with coverage
fm-llm-solver test --coverage

# Run specific tests
fm-llm-solver test tests/unit/test_generator.py
```

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Mathematical Primer](docs/MATHEMATICAL_PRIMER.md)
- [Development Guide](docs/DEVELOPMENT.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Set up development environment
pip install -e ".[dev]"
pre-commit install

# Run code quality checks
black fm_llm_solver tests
isort fm_llm_solver tests
flake8 fm_llm_solver tests
mypy fm_llm_solver
```

## 🏗️ Architecture

FM-LLM Solver follows a modular architecture:

- **Core Layer**: Configuration, logging, exceptions, and type definitions
- **Service Layer**: Certificate generation, verification, knowledge base management
- **Interface Layer**: Web UI and REST API
- **Infrastructure Layer**: Caching, monitoring, and deployment utilities

## 🚀 Deployment

### Docker

```bash
# Build and run with Docker
docker build -t fm-llm-solver .
docker run -p 5000:5000 -p 8000:8000 fm-llm-solver
```

### Cloud Deployment

See [Deployment Guide](docs/DEPLOYMENT.md) for cloud deployment options (AWS, GCP, Azure).

## 📊 Performance

- Certificate generation: ~2-5 seconds per system
- Verification: <1 second for numerical, 2-10 seconds for symbolic
- Supports batch processing for multiple systems
- GPU acceleration available for LLM inference

## 🔒 Security

- Authentication and authorization for web interface
- Rate limiting to prevent abuse
- API key management for programmatic access
- Secure session handling
- Input validation and sanitization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the Qwen team for the excellent language models
- Inspired by research in formal methods and neural certificate generation
- Built with support from the University of Colorado

## 📞 Contact

- **Author**: Patrick Allen Cooper
- **Email**: patrick.cooper@colorado.edu
- **Website**: [fm-llm-solver.ai](https://fm-llm-solver.ai)

---

<p align="center">Made with ❤️ by researchers, for researchers</p> 