# FM-LLM Solver Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# Install dependencies
pip install -r requirements.txt
pip install -r web_requirements.txt

# Install in development mode
pip install -e .
```

## Entry Points

### 1. Unified Application Entry (`run_application.py`)

```bash
# Run web interface (default)
python run_application.py web

# Run inference API
python run_application.py api

# Run both web and API
python run_application.py both

# Build knowledge base
python run_application.py build-kb

# Run tests
python run_application.py test
```

### 2. Direct Web Interface (`run_web_interface.py`)

```bash
# Start web interface with defaults
python run_web_interface.py

# Custom configuration
python run_web_interface.py --config my_config.yaml --port 8080
```

### 3. CLI Tool (`fm-llm`)

```bash
# After installation, use the unified CLI
fm-llm --help

# Common commands
fm-llm config show              # Show configuration
fm-llm kb build                 # Build knowledge base
fm-llm web start                # Start web interface
fm-llm train finetune           # Fine-tune model
fm-llm experiment run           # Run experiments
```

## Core Capabilities

### Certificate Generation

**Web Interface**:
1. Navigate to http://localhost:5000
2. Enter system dynamics, initial set, and unsafe set
3. Click "Generate Certificate"

**CLI**:
```bash
fm-llm generate --system "x'=-x+y, y'=x-y" --initial "x^2+y^2<=0.5" --unsafe "x^2+y^2>=2"
```

**Python API**:
```python
from fm_llm_solver.services.certificate_generator import CertificateGenerator
from fm_llm_solver.core.config_manager import ConfigurationManager

config_manager = ConfigurationManager()
generator = CertificateGenerator(config_manager)

result = generator.generate({
    "dynamics": {"x": "-x + y", "y": "x - y"},
    "initial_set": "x**2 + y**2 <= 0.5",
    "unsafe_set": "x**2 + y**2 >= 2.0"
})
```

### Knowledge Base Building

```bash
# Build from papers directory
fm-llm kb build --papers-dir data/papers

# Force rebuild
fm-llm kb build --force

# Build specific type
fm-llm kb build --type continuous
```

### Fine-tuning

```bash
# Create training data
fm-llm train create-data

# Fine-tune model
fm-llm train finetune --model qwen2.5-7b --epochs 3

# Evaluate fine-tuned model
fm-llm train evaluate
```

### Monitoring

**Web Dashboard**:
- Navigate to http://localhost:5000/monitoring/dashboard
- View real-time metrics and usage statistics

**Prometheus Metrics**:
- Available at http://localhost:5000/metrics

### Deployment

**Docker**:
```bash
# Build and run with Docker Compose
docker-compose up

# Run specific service
docker-compose up web
docker-compose up api
```

**Kubernetes**:
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/
```

## Configuration

Main configuration file: `config.yaml`

Key settings:
- `model.name`: LLM model to use
- `model.device`: CPU/GPU selection
- `web_interface.port`: Web server port
- `knowledge_base.enabled`: Enable/disable RAG
- `monitoring.enabled`: Enable/disable metrics

## Testing

```bash
# Run all tests
python run_application.py test

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/benchmarks/

# Run with coverage
pytest --cov=fm_llm_solver

# Run performance tests
python tests/performance/test_performance.py
```

## System Types Supported

1. **Continuous-Time**: `x' = f(x)`
2. **Discrete-Time**: `x[k+1] = f(x[k])`
3. **Stochastic**: `dx = f(x)dt + g(x)dW`
4. **Domain-Bounded**: Certificates valid in specific regions

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install dependencies with `pip install -r requirements.txt`
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Knowledge base not found**: Run `fm-llm kb build` first
4. **Port already in use**: Change port in config.yaml or use --port flag

### Logs

- Application logs: `logs/fm_llm_solver.log`
- Web interface logs: `web_interface.log`
- Error details: Check console output with --debug flag

## Additional Resources

- Full documentation: `docs/`
- API Reference: `docs/API_REFERENCE.md`
- Architecture: `docs/ARCHITECTURE.md`
- Contributing: `CONTRIBUTING.md`

## Support

- GitHub Issues: https://github.com/PatrickAllenCooper/FM-LLM-Solver/issues
- Documentation: See `docs/` directory 