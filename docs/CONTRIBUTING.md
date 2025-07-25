# Contributing to FM-LLM Solver

Thank you for your interest in contributing to FM-LLM Solver! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/FM-LLM-Solver.git
   cd FM-LLM-Solver
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original/FM-LLM-Solver.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Git

### Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Configure git**:
   ```bash
   git config --local user.name "Your Name"
   git config --local user.email "your.email@example.com"
   ```

## Project Structure

```
fm_llm_solver/
├── core/           # Core components (config, logging, exceptions, types)
├── services/       # Business logic services
├── web/            # Flask web interface
├── api/            # FastAPI inference API
└── utils/          # Utility functions

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── benchmarks/     # Performance benchmarks
```

### Key Principles

1. **Modularity**: Each component should have a single, well-defined responsibility
2. **Testability**: Write code that is easy to test
3. **Documentation**: Document all public APIs
4. **Type Safety**: Use type hints throughout the codebase

## Making Changes

### Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**:
   ```bash
   pytest tests/
   ```

4. **Run code quality checks**:
   ```bash
   pre-commit run --all-files
   ```

### Types of Contributions

- **Bug Fixes**: Fix issues reported in GitHub Issues
- **Features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add missing tests or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

### Creating New Components

When adding new services or components:

1. **Define interfaces first** in `core/interfaces.py`
2. **Create type definitions** in `core/types.py`
3. **Implement the service** in the appropriate module
4. **Add comprehensive tests**
5. **Update documentation**

Example:
```python
# In core/interfaces.py
class MyNewService(ABC):
    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        """Process the input data."""
        pass

# In services/my_service.py
class MyServiceImpl(MyNewService):
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
    
    def process(self, data: InputType) -> OutputType:
        # Implementation
        pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fm_llm_solver --cov-report=html

# Run specific test file
pytest tests/unit/test_generator.py

# Run tests matching pattern
pytest -k "test_certificate"

# Run tests with markers
pytest -m "not slow"
```

### Writing Tests

1. **Unit Tests**: Test individual components in isolation
   ```python
   def test_certificate_parser():
       parser = SystemParser()
       cert = parser.parse_certificate("B(x,y) = x^2 + y^2", ["x", "y"])
       assert cert.expression == "x**2 + y**2"
   ```

2. **Integration Tests**: Test component interactions
   ```python
   @pytest.mark.integration
   def test_full_pipeline():
       # Test complete generation and verification flow
   ```

3. **Use fixtures** for common test data:
   ```python
   @pytest.fixture
   def sample_system():
       return SystemDescription(
           dynamics={"x": "-x + y", "y": "x - y"},
           initial_set="x**2 + y**2 <= 0.5",
           unsafe_set="x**2 + y**2 >= 2.0"
       )
   ```

### Test Coverage

We aim for at least 80% test coverage. Check coverage reports:
```bash
pytest --cov=fm_llm_solver --cov-report=term-missing
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for all public functions
- Use docstrings (Google style) for all public APIs

### Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks for automatic checks

### Example

```python
from typing import List, Optional

from fm_llm_solver.core.types import SystemDescription
from fm_llm_solver.core.logging import get_logger


class ExampleService:
    """Example service demonstrating code style.
    
    This service shows how to structure code according to
    our style guidelines.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the service.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def process_systems(
        self,
        systems: List[SystemDescription],
        parallel: bool = True
    ) -> List[ProcessResult]:
        """Process multiple systems.
        
        Args:
            systems: List of systems to process
            parallel: Whether to process in parallel
            
        Returns:
            List of processing results
            
        Raises:
            ProcessingError: If processing fails
        """
        self.logger.info(f"Processing {len(systems)} systems")
        
        if parallel:
            return self._process_parallel(systems)
        else:
            return self._process_sequential(systems)
```

## Documentation

### Docstrings

All public modules, classes, and functions must have docstrings:

```python
def calculate_lie_derivative(
    expression: str,
    dynamics: Dict[str, str],
    variables: List[str]
) -> str:
    """Calculate the Lie derivative of an expression.
    
    Computes L_f h where f is the vector field defined by
    the dynamics and h is the given expression.
    
    Args:
        expression: The expression h(x)
        dynamics: Dictionary mapping variables to their derivatives
        variables: List of state variables
        
    Returns:
        The Lie derivative as a string expression
        
    Example:
        >>> calculate_lie_derivative("x**2 + y**2", {"x": "-x", "y": "-y"}, ["x", "y"])
        "-2*x**2 - 2*y**2"
    """
```

### API Documentation

- Update `docs/API_REFERENCE.md` when adding new public APIs
- Include examples in documentation
- Document breaking changes

### README Updates

Update README.md if you:
- Add new features
- Change installation requirements
- Modify the basic usage

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Rebase your feature branch**:
   ```bash
   git checkout feature/your-feature
   git rebase main
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature
   ```

4. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what changes you made and why
- **Link Issues**: Reference any related issues
- **Tests**: Ensure all tests pass
- **Documentation**: Update relevant documentation

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Changes
- List of specific changes
- Another change

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Fixes #123
```

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Steps

1. Update version in:
   - `fm_llm_solver/__init__.py`
   - `pyproject.toml`
   - `setup.py`

2. Update CHANGELOG.md

3. Create release PR

4. After merge, tag release:
   ```bash
   git tag -a v1.2.3 -m "Release version 1.2.3"
   git push upstream v1.2.3
   ```

## Questions?

If you have questions:
1. Check existing documentation
2. Search GitHub issues
3. Ask in discussions
4. Contact maintainers

Thank you for contributing to FM-LLM Solver! 🎉 