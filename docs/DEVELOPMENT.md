# Development Guide

This guide is for developers who want to contribute to or extend FM-LLM Solver.

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Web UI    │────▶│  Inference   │────▶│ Verification│
└─────────────┘     │    Engine    │     └─────────────┘
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     RAG      │
                    │ (Knowledge   │
                    │    Base)     │
                    └──────────────┘
```

### Core Components

1. **Inference Engine** (`inference/`): Handles certificate generation
2. **RAG System** (`knowledge_base/`): Retrieval-augmented generation
3. **Verification** (`evaluation/`): Validates generated certificates
4. **Fine-Tuning** (`fine_tuning/`): Model training pipeline
5. **Web Interface** (`web_interface/`): Flask application
6. **Deployment** (`deployment/`): Cloud deployment scripts

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/FM-LLM-Solver.git
cd FM-LLM-Solver
git remote add upstream https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
```

### 2. Development Environment

```bash
# Create dev environment
conda create -n fmllm-dev python=3.10
conda activate fmllm-dev

# Install in development mode
pip install -e .
pip install -r requirements/dev-requirements.txt  # If available
```

### 3. Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Document functions with NumPy-style docstrings
- Keep functions focused and modular

Example:
```python
def verify_barrier_certificate(
    dynamics: Dict[str, str],
    certificate: str,
    initial_set: str,
    unsafe_set: str
) -> Dict[str, Any]:
    """
    Verify a barrier certificate for a given system.
    
    Parameters
    ----------
    dynamics : dict
        System dynamics as variable -> expression mapping
    certificate : str
        Barrier certificate expression
    initial_set : str
        Initial set constraint
    unsafe_set : str
        Unsafe set constraint
        
    Returns
    -------
    dict
        Verification results with 'valid' key and details
    """
    # Implementation
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_verification.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Writing Tests

Create test files in `tests/unit/` or `tests/integration/`:

```python
import pytest
from inference.generate_certificate import generate_barrier_certificate

def test_basic_generation():
    """Test basic certificate generation."""
    result = generate_barrier_certificate(
        "dx/dt = -x, Initial: x <= 1, Unsafe: x >= 2"
    )
    assert "certificate" in result
    assert result["certificate"] is not None
```

## Adding Features

### 1. New System Types

To support new system types (e.g., hybrid systems):

1. Update parser in `inference/generate_certificate.py`
2. Add verification logic in `evaluation/verify_certificate.py`
3. Create examples in `data/benchmark_systems.json`
4. Update documentation

### 2. New Models

To add support for a new LLM:

1. Add provider in `config/config.yaml`:
   ```yaml
   model:
     provider: "new_provider"
     name: "model-name"
   ```

2. Update model loading in `inference/generate_certificate.py`
3. Add any provider-specific logic

### 3. New Verification Methods

To add verification methods:

1. Create new module in `evaluation/`
2. Integrate into `verify_certificate.py`
3. Add configuration options
4. Document the method

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **CUDA Errors**: Check GPU memory and compatibility
2. **API Rate Limits**: Implement exponential backoff
3. **Memory Leaks**: Use memory profiling tools

### Performance Profiling

```bash
# Profile certificate generation
python -m cProfile -o profile.stats inference/generate_certificate.py "..."

# Analyze results
python -m pstats profile.stats
```

## Contributing

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Test Thoroughly

```bash
# Run tests
python -m pytest tests/

# Check code style
flake8 .

# Type checking (if using mypy)
mypy .
```

### 4. Submit Pull Request

1. Push to your fork
2. Create PR against main branch
3. Describe changes clearly
4. Link any related issues

## Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish packages

## Project Structure Best Practices

- Keep modules focused and cohesive
- Use dependency injection where appropriate
- Avoid circular imports
- Maintain backward compatibility

## Security Considerations

- Never commit API keys or credentials
- Validate all user inputs
- Use parameterized queries for databases
- Keep dependencies updated

## Performance Optimization

- Use batch processing where possible
- Implement caching for expensive operations
- Profile before optimizing
- Consider async operations for I/O

## Documentation

When adding features:
1. Update relevant .md files in `docs/`
2. Add docstrings to all functions
3. Include usage examples
4. Update API reference if needed

## Getting Help

- Check existing issues on GitHub
- Join discussions in issue threads
- Contact maintainers for guidance
- Review similar PRs for examples 