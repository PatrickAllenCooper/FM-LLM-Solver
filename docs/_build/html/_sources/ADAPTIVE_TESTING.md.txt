# Adaptive Testing System

The FM-LLM Solver now includes an intelligent adaptive testing system that automatically detects your environment and runs the most appropriate test suite based on your system's capabilities and constraints.

## Overview

The adaptive testing system supports three primary environments:

1. **MacBook** - Local development on MacBook (optimized for battery life and thermal management)
2. **Desktop** - Local development on high-powered desktop (comprehensive testing with GPU acceleration)  
3. **Deployed** - Production/staging environments (focused on production readiness)

## Quick Start

### Basic Usage

```bash
# Auto-detect environment and run appropriate tests
python test_runner.py

# Preview what will be run without executing
python test_runner.py --dry-run

# Show detailed environment detection results
python test_runner.py --info
```

### Environment-Specific Testing

```bash
# Force MacBook mode (fast, essential tests)
python test_runner.py --environment macbook

# Force desktop mode (comprehensive tests)
python test_runner.py --environment desktop

# Force deployed mode (production-focused tests)  
python test_runner.py --environment deployed
```

### Test Scope Control

```bash
# Run essential tests only (fastest)
python test_runner.py --scope essential

# Run comprehensive test suite
python test_runner.py --scope comprehensive

# Run production-focused tests
python test_runner.py --scope production
```

### Category Selection

```bash
# Include specific test categories
python test_runner.py --include unit_tests security_tests

# Exclude specific test categories
python test_runner.py --exclude load_tests gpu_tests

# Combine with environment and scope
python test_runner.py --environment desktop --scope comprehensive --exclude load_tests
```

## Environment Detection

The system automatically detects your environment using multiple indicators:

### MacBook Detection
- Apple Silicon (M1/M2/M3) architecture
- Intel Mac with laptop battery
- Limited GPU capabilities
- Conservative resource usage

### Desktop Detection  
- High-end dedicated GPU (>8GB VRAM)
- Many CPU cores (≥16)
- Large RAM (≥32GB)
- Unrestricted resource usage

### Deployed Environment Detection
- Cloud platform environment variables (AWS, GCP, Azure, etc.)
- Container environments (Docker, Kubernetes)
- CI/CD environments (GitHub Actions, Jenkins, etc.)  
- Headless/server configurations
- Cloud instance metadata

## Test Scopes

### Essential Scope
**Best for**: MacBook development, quick feedback loops

**Includes**:
- Unit tests
- Core integration tests
- Basic security tests

**Characteristics**:
- Fast execution (< 5 minutes)
- Conservative resource usage
- Essential functionality validation

### Comprehensive Scope  
**Best for**: Desktop development, pre-commit validation

**Includes**:
- Unit tests
- Full integration tests
- Security tests
- Performance tests
- GPU tests (if available)
- Load tests (if supported)
- End-to-end tests

**Characteristics**:
- Thorough validation (10-30 minutes)
- Uses available hardware acceleration
- Complete feature coverage

### Production Scope
**Best for**: Deployed environments, production validation

**Includes**:
- Unit tests
- Integration tests  
- Security tests
- Deployment tests
- Monitoring tests
- GPU tests (if available)
- Load tests (if supported)

**Characteristics**:
- Production-readiness focus
- Conservative resource usage
- Reliability and security emphasis

## Test Categories

### Core Categories
- **unit_tests** - Unit test suite
- **core_integration** - Essential integration tests (subset)
- **integration_tests** - Full integration test suite
- **security_tests** - Complete security test suite
- **basic_security** - Essential security tests (subset)

### Performance Categories
- **performance_tests** - Performance benchmarking
- **load_tests** - Load testing with K6 (requires K6 installation)
- **gpu_tests** - GPU-accelerated tests (requires CUDA/MPS)

### Infrastructure Categories
- **deployment_tests** - Deployment configuration tests
- **monitoring_tests** - Monitoring and metrics tests
- **end_to_end_tests** - Full end-to-end workflow tests

## Environment-Specific Optimizations

### MacBook Optimizations
- **Parallel Jobs**: Limited to ≤4 to prevent overheating
- **Memory**: 30% of total RAM max per test (≤4GB)
- **Timeouts**: 2-3x longer to account for thermal throttling
- **Test Selection**: Essential scope by default
- **GPU**: Uses MPS if available, otherwise CPU-only

### Desktop Optimizations  
- **Parallel Jobs**: Up to CPU cores - 1 (≤8)
- **Memory**: 50% of total RAM max per test (≤16GB)  
- **Timeouts**: Normal or 1.5x if no GPU
- **Test Selection**: Comprehensive scope by default
- **GPU**: Full CUDA acceleration when available

### Deployed Optimizations
- **Parallel Jobs**: Very conservative (≤2)
- **Memory**: 20% of total RAM max per test (≤2GB)
- **Timeouts**: 2x longer for network/disk variability  
- **Test Selection**: Production scope by default
- **Monitoring**: Memory and resource monitoring enabled

## Legacy Mode

The original test runner is still available for backward compatibility:

```bash
# Use legacy test runner
python test_runner.py --legacy --unit
python test_runner.py --legacy --integration

# Or directly
python tests/run_tests.py --no-adaptive --unit
```

## Advanced Usage

### Direct Adaptive Runner

```bash
# Use adaptive test runner directly
python tests/adaptive_test_runner.py

# With specific options
python tests/adaptive_test_runner.py --environment desktop --scope comprehensive --verbose
```

### Integration with CI/CD

```yaml
# GitHub Actions example
- name: Run Adaptive Tests
  run: |
    python test_runner.py --environment deployed --scope production
```

### Environment Override

You can override environment detection by setting environment variables:

```bash
# Force specific environment type
export FM_LLM_TEST_ENVIRONMENT=macbook
python test_runner.py

# Force specific scope
export FM_LLM_TEST_SCOPE=essential  
python test_runner.py
```

## Troubleshooting

### Common Issues

**Adaptive testing not available**
- Ensure `psutil` is installed: `pip install psutil`
- Check that environment detector module exists

**GPU tests not running**
- Verify CUDA/PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- For Apple Silicon: Ensure MPS is available

**Tests timing out**
- Try increasing timeout multiplier: `--environment macbook` (uses 3x timeouts)
- Check system resources with: `python test_runner.py --info`

**Memory issues**
- Use essential scope: `--scope essential`
- Force MacBook environment for conservative memory usage

### Debug Information

```bash
# Show detailed environment detection
python test_runner.py --info

# Preview test strategy
python test_runner.py --dry-run --verbose

# Test specific categories
python test_runner.py --include unit_tests --dry-run
```

## Configuration

The adaptive testing system reads configuration from:

1. Command line arguments (highest priority)
2. Environment variables
3. Detected hardware capabilities
4. Default scope for environment type

### Environment Variables

- `FM_LLM_TEST_ENVIRONMENT` - Force environment type
- `FM_LLM_TEST_SCOPE` - Force test scope  
- `FM_LLM_TEST_PARALLEL_JOBS` - Override parallel job count
- `FM_LLM_TEST_TIMEOUT_MULTIPLIER` - Override timeout multiplier

## Performance

### Typical Execution Times

| Environment | Scope | Estimated Time | Parallel Jobs |
|-------------|-------|----------------|---------------|
| MacBook | Essential | 2-5 minutes | 2-4 |
| MacBook | Comprehensive | 10-20 minutes | 2-4 |
| Desktop | Essential | 1-3 minutes | 4-8 |
| Desktop | Comprehensive | 5-15 minutes | 4-8 |
| Deployed | Production | 3-10 minutes | 1-2 |

### Memory Usage

| Environment | Max Memory/Test | Monitoring | OOM Protection |
|-------------|-----------------|------------|----------------|
| MacBook | 4GB | Enabled | Enabled |
| Desktop | 16GB | Disabled | Disabled |
| Deployed | 2GB | Enabled | Enabled |

## Contributing

When adding new tests:

1. **Categorize appropriately** - Use existing categories or propose new ones
2. **Consider all environments** - Tests should work on MacBook, desktop, and deployed
3. **Add environment-specific variations** - Use different parameters based on capabilities
4. **Update documentation** - Document any new test categories or requirements

### Adding New Test Categories

```python
# In adaptive_test_runner.py
def _run_new_category_tests(self, timeout_multiplier: float) -> Dict:
    """Run new category tests."""
    # Implement test execution
    # Adjust based on self.environment_type
    # Return standardized result dict
```

### Environment-Specific Test Logic

```python
if self.environment_type == "macbook":
    # Use conservative settings
    batch_size = 1
    max_workers = 2
elif self.environment_type == "desktop":
    # Use full capabilities  
    batch_size = 4
    max_workers = 8
else:  # deployed
    # Use production-safe settings
    batch_size = 1  
    max_workers = 1
``` 