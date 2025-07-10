# FM-LLM-Solver Testing System

## Overview

The FM-LLM-Solver testing system provides a unified, comprehensive approach to testing that adapts to your environment and maximizes coverage. The system is designed to work seamlessly across Windows, macOS, and Linux with GPU acceleration support.

## 🚀 Quick Start

### Basic System Check
```bash
python tests/run_tests.py --quick
```

### Run All Tests
```bash
python tests/run_tests.py --all
```

### Run Specific Test Categories
```bash
# Unit tests only
python tests/run_tests.py --unit

# GPU-accelerated tests
python tests/run_tests.py --gpu

# Unified test suite (comprehensive)
python tests/run_tests.py --unified

# Show test summary
python tests/run_tests.py --summary
```

## 🧪 Test Categories

### 1. Unit Tests (`--unit`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Core utilities, data structures, helper functions
- **Duration**: ~1-2 minutes
- **Examples**: Certificate extraction, verification helpers, numerical checks

### 2. GPU Tests (`--gpu`)
- **Purpose**: Test GPU acceleration and memory management
- **Coverage**: CUDA operations, memory allocation, batch processing
- **Duration**: ~2-3 minutes
- **Requirements**: NVIDIA GPU with CUDA support

### 3. Unified Test Suite (`--unified`)
- **Purpose**: Comprehensive end-to-end testing
- **Coverage**: Full pipeline from certificate generation to verification
- **Duration**: ~5-10 minutes
- **Features**: Environment detection, adaptive testing, performance benchmarks

### 4. Quick System Check (`--quick`)
- **Purpose**: Verify system readiness
- **Coverage**: Module availability, GPU detection, basic functionality
- **Duration**: ~10 seconds
- **Use Case**: Pre-flight check before development

## 🎯 Testing Flywheel

The testing system implements a **testing flywheel** that ensures:

1. **Environment Adaptation**: Automatically detects your system (Windows/macOS/Linux, GPU/CPU)
2. **Maximal Coverage**: Tests all components with appropriate depth for your environment
3. **GPU Acceleration**: Leverages your RTX 3080 for faster certificate generation and validation
4. **Robust Validation**: Tests certificate extraction, cleaning, and verification pipelines
5. **Performance Monitoring**: Tracks memory usage, processing speed, and GPU utilization

## 📊 Test Results

### Result Files
- `test_results/unified_test_results.json` - Comprehensive test results
- `test_results/certificate_pipeline_results.json` - Pipeline-specific results
- `test_results/comprehensive_test_results.json` - Legacy results

### Success Criteria
- **High Success**: ≥80% pass rate (green)
- **Moderate Success**: ≥60% pass rate (yellow)
- **Low Success**: <60% pass rate (red)

## 🔧 Environment Detection

The system automatically detects:

- **Platform**: Windows, macOS, Linux
- **Python Version**: 3.8+
- **GPU**: NVIDIA GPUs with CUDA support
- **Memory**: Available RAM and GPU memory
- **CPU**: Core count and processing capability

## 🚀 GPU Acceleration

### Supported GPUs
- NVIDIA RTX series (3080, 3090, 4080, 4090)
- NVIDIA GTX series (1660, 2060, 3060)
- Any CUDA-compatible GPU

### GPU Features Tested
- Memory allocation and deallocation
- Batch processing capabilities
- Tensor operations performance
- Memory management efficiency

## 🛠️ Development Workflow

### 1. Pre-Development Check
```bash
python tests/run_tests.py --quick
```

### 2. During Development
```bash
# Run unit tests for quick feedback
python tests/run_tests.py --unit

# Run GPU tests if working on GPU features
python tests/run_tests.py --gpu
```

### 3. Before Committing
```bash
# Run comprehensive test suite
python tests/run_tests.py --unified
```

### 4. Continuous Integration
```bash
# Run all tests with summary
python tests/run_tests.py --all
```

## 📈 Performance Monitoring

The testing system monitors:

- **Certificate Generation Speed**: Time to generate barrier certificates
- **Verification Performance**: Time to verify certificate validity
- **Memory Efficiency**: RAM and GPU memory usage
- **GPU Utilization**: CUDA memory allocation and computation speed

## 🔍 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Ensure PyTorch is installed with CUDA support
   - Check NVIDIA drivers are up to date
   - Verify CUDA toolkit installation

2. **Module Import Errors**
   - Run `python tests/run_tests.py --quick` to identify missing modules
   - Install required dependencies: `pip install torch pydantic pytest`

3. **Test Timeouts**
   - Reduce test scope: `python tests/run_tests.py --unit`
   - Check system resources (CPU, memory)
   - Consider running tests individually

### Debug Mode
```bash
# Run with verbose output
python -m pytest tests/unit/ -v

# Run specific test file
python tests/unit/test_certificate_pipeline.py
```

## 🎯 Best Practices

1. **Regular Testing**: Run quick checks before development sessions
2. **GPU Utilization**: Use GPU tests when working on performance-critical features
3. **Comprehensive Validation**: Run unified suite before major changes
4. **Result Analysis**: Review test summaries to identify areas for improvement

## 📋 Test Architecture

```
tests/
├── run_tests.py              # Main test runner
├── unified_test_suite.py     # Comprehensive test suite
├── test_summary.py           # Results summary
├── unit/                     # Unit tests
│   ├── test_certificate_pipeline.py
│   ├── test_gpu_accelerated_generation.py
│   └── test_core_components.py
├── integration/              # Integration tests
└── test_results/            # Test output files
```

## 🚀 Advanced Usage

### Custom Test Configuration
```python
# In your test file
from tests.unified_test_suite import UnifiedTestSuite

suite = UnifiedTestSuite()
results = suite.run_comprehensive_suite()
```

### Environment-Specific Testing
```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Run with specific GPU
export CUDA_VISIBLE_DEVICES="0"
```

## 📊 Metrics and Monitoring

The testing system provides detailed metrics:

- **Success Rates**: Per-category and overall pass rates
- **Performance Benchmarks**: Processing times and speedups
- **Resource Usage**: Memory and GPU utilization
- **Error Analysis**: Detailed failure information

## 🎉 Success Indicators

Your testing flywheel is working optimally when:

- ✅ Quick system check passes all components
- ✅ GPU tests show significant speedup (>5x)
- ✅ Unified test suite achieves >80% success rate
- ✅ Memory usage remains stable during tests
- ✅ Certificate extraction and validation work reliably

---

**Ready to test?** Start with `python tests/run_tests.py --quick` to verify your system is ready for development! 