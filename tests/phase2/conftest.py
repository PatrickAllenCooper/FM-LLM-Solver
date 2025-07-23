"""
Phase 2 Test Configuration and Fixtures
======================================

Shared test configuration and fixtures specifically for Phase 2 components:
- Validation Strategies
- Validation Orchestrator
- Performance Monitoring
- Caching Systems
- Parallel Processing
"""

import pytest
import numpy as np
import sympy as sp
from omegaconf import DictConfig
from typing import Dict, List, Any
import tempfile
import os
import json

# Import Phase 2 components
from utils.validation_strategies import (
    SamplingValidationStrategy,
    SymbolicValidationStrategy,
    IntervalValidationStrategy,
    SMTValidationStrategy,
    ValidationResult,
    StrategyPerformance
)
from utils.validation_orchestrator import ValidationOrchestrator, OrchestratedResult


@pytest.fixture
def phase2_config():
    """Phase 2 specific configuration"""
    return DictConfig({
        'confidence_threshold': 0.8,
        'max_strategies': 3,
        'parallel_execution': True,
        'num_samples_boundary': 100,
        'num_samples_lie': 200,
        'numerical_tolerance': 1e-6,
        'cache_enabled': True,
        'cache_size': 1000,
        'performance_monitoring': True
    })


@pytest.fixture
def simple_system_info():
    """Simple 2D system for testing"""
    return {
        'variables': ['x', 'y'],
        'dynamics': ['-x', '-y'],
        'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
        'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
        'safe_set_conditions': [],
        'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
    }


@pytest.fixture
def complex_system_info():
    """Complex 3D system for testing"""
    return {
        'variables': ['x', 'y', 'z'],
        'dynamics': ['-x', '-y', '-z'],
        'initial_set_conditions': ['x**2 + y**2 + z**2 <= 0.25'],
        'unsafe_set_conditions': ['x**2 + y**2 + z**2 >= 4.0'],
        'safe_set_conditions': [],
        'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3), 'z': (-3, 3)}
    }


@pytest.fixture
def polynomial_system_info():
    """Polynomial system for symbolic testing"""
    return {
        'variables': ['x', 'y'],
        'dynamics': ['-x**2', '-y**2'],
        'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
        'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
        'safe_set_conditions': [],
        'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
    }


@pytest.fixture
def valid_certificates():
    """Collection of valid barrier certificates for testing"""
    return {
        'simple_2d': "x**2 + y**2 - 1.0",
        'complex_2d': "x**2 + y**2 - 2.0",
        'polynomial_2d': "x**4 + y**4 - 1.0",
        'simple_3d': "x**2 + y**2 + z**2 - 1.0",
        'complex_3d': "x**2 + y**2 + z**2 - 2.0"
    }


@pytest.fixture
def invalid_certificates():
    """Collection of invalid barrier certificates for testing"""
    return {
        'too_small': "x**2 + y**2 - 0.1",  # Too small barrier
        'too_large': "x**2 + y**2 - 5.0",  # Too large barrier
        'wrong_sign': "x**2 + y**2 + 1.0",  # Wrong sign
        'non_polynomial': "sin(x) + cos(y) - 1.0"  # Non-polynomial
    }


@pytest.fixture
def sampling_strategy(phase2_config):
    """Sampling validation strategy fixture"""
    return SamplingValidationStrategy(phase2_config)


@pytest.fixture
def symbolic_strategy(phase2_config):
    """Symbolic validation strategy fixture"""
    return SymbolicValidationStrategy(phase2_config)


@pytest.fixture
def interval_strategy(phase2_config):
    """Interval validation strategy fixture"""
    return IntervalValidationStrategy(phase2_config)


@pytest.fixture
def smt_strategy(phase2_config):
    """SMT validation strategy fixture"""
    return SMTValidationStrategy(phase2_config)


@pytest.fixture
def orchestrator(phase2_config):
    """Validation orchestrator fixture"""
    return ValidationOrchestrator(phase2_config)


@pytest.fixture
def all_strategies(sampling_strategy, symbolic_strategy, interval_strategy, smt_strategy):
    """All validation strategies fixture"""
    return {
        'sampling': sampling_strategy,
        'symbolic': symbolic_strategy,
        'interval': interval_strategy,
        'smt': smt_strategy
    }


@pytest.fixture
def performance_benchmark_data():
    """Performance benchmark data for testing"""
    return {
        'baseline_times': {
            'sampling': 1.0,
            'symbolic': 0.3,
            'interval': 0.8,
            'smt': 0.5
        },
        'baseline_samples': {
            'sampling': 1000,
            'symbolic': 0,
            'interval': 0,
            'smt': 0
        },
        'target_speedup': 3.0,
        'target_sample_reduction': 0.5
    }


@pytest.fixture
def mock_validation_result():
    """Mock validation result for testing"""
    return ValidationResult(
        is_valid=True,
        confidence=0.9,
        execution_time=1.0,
        samples_used=100,
        violations_found=[],
        strategy_name='test_strategy',
        details={'test': True}
    )


@pytest.fixture
def mock_orchestrated_result():
    """Mock orchestrated result for testing"""
    return OrchestratedResult(
        is_valid=True,
        confidence=0.95,
        execution_time=1.5,
        strategies_used=['sampling', 'symbolic'],
        individual_results={
            'sampling': ValidationResult(
                is_valid=True, confidence=0.9, execution_time=1.0,
                samples_used=100, violations_found=[], strategy_name='sampling',
                details={}
            ),
            'symbolic': ValidationResult(
                is_valid=True, confidence=0.95, execution_time=0.3,
                samples_used=0, violations_found=[], strategy_name='symbolic',
                details={'method': 'symbolic'}
            )
        },
        consensus_achieved=True,
        details={'consensus': True, 'avg_confidence': 0.925}
    )


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def cache_test_data():
    """Test data for caching tests"""
    return {
        'certificate': "x**2 + y**2 - 1.0",
        'system_hash': "abc123",
        'result_hash': "def456",
        'cached_result': {
            'is_valid': True,
            'confidence': 0.9,
            'execution_time': 1.0,
            'strategy_used': 'sampling'
        }
    }


@pytest.fixture
def parallel_test_config():
    """Configuration for parallel processing tests"""
    return {
        'max_workers': 4,
        'chunk_size': 100,
        'timeout': 30,
        'memory_limit': '1GB'
    }


@pytest.fixture
def gpu_test_config():
    """Configuration for GPU acceleration tests"""
    return {
        'gpu_enabled': True,
        'gpu_memory_limit': '2GB',
        'batch_size': 1000,
        'fallback_to_cpu': True
    }


@pytest.fixture
def adaptive_sampling_config():
    """Configuration for adaptive sampling tests"""
    return {
        'initial_samples': 100,
        'max_samples': 1000,
        'refinement_threshold': 0.1,
        'critical_region_ratio': 0.2,
        'convergence_tolerance': 1e-6
    }


@pytest.fixture
def performance_monitoring_config():
    """Configuration for performance monitoring tests"""
    return {
        'metrics_enabled': True,
        'sampling_interval': 1.0,
        'retention_hours': 24,
        'alert_thresholds': {
            'execution_time': 10.0,
            'memory_usage': 0.8,
            'error_rate': 0.05
        }
    }


def create_test_certificate(expression: str, variables: List[str]) -> str:
    """Helper function to create test certificates"""
    return expression


def create_test_system(variables: List[str], dynamics: List[str], 
                      initial_set: List[str], unsafe_set: List[str]) -> Dict[str, Any]:
    """Helper function to create test systems"""
    bounds = {}
    for var in variables:
        bounds[var] = (-3.0, 3.0)
    
    return {
        'variables': variables,
        'dynamics': dynamics,
        'initial_set_conditions': initial_set,
        'unsafe_set_conditions': unsafe_set,
        'safe_set_conditions': [],
        'sampling_bounds': bounds
    }


def assert_validation_result(result: ValidationResult, expected_valid: bool = None):
    """Helper function to assert validation result properties"""
    assert isinstance(result, ValidationResult)
    assert isinstance(result.is_valid, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert result.execution_time >= 0.0
    assert isinstance(result.samples_used, int)
    assert isinstance(result.violations_found, list)
    assert isinstance(result.strategy_name, str)
    assert isinstance(result.details, dict)
    
    if expected_valid is not None:
        assert result.is_valid == expected_valid


def assert_orchestrated_result(result: OrchestratedResult, expected_valid: bool = None):
    """Helper function to assert orchestrated result properties"""
    assert isinstance(result, OrchestratedResult)
    assert isinstance(result.is_valid, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert result.execution_time >= 0.0
    assert isinstance(result.strategies_used, list)
    assert isinstance(result.individual_results, dict)
    assert isinstance(result.consensus_achieved, bool)
    assert isinstance(result.details, dict)
    
    if expected_valid is not None:
        assert result.is_valid == expected_valid


def assert_performance_improvement(baseline_time: float, current_time: float, 
                                 min_improvement: float = 0.0):
    """Helper function to assert performance improvements"""
    if baseline_time > 0:
        improvement = (baseline_time - current_time) / baseline_time
        assert improvement >= min_improvement, f"Expected improvement >= {min_improvement}, got {improvement}"


def create_mock_violation(point: List[float], condition: str, severity: str = 'high'):
    """Helper function to create mock violations"""
    return {
        'point': point,
        'condition': condition,
        'severity': severity,
        'timestamp': 1234567890.0
    }


def create_mock_performance_metrics():
    """Helper function to create mock performance metrics"""
    return {
        'execution_time': 1.5,
        'memory_usage': 0.3,
        'cpu_usage': 0.4,
        'samples_processed': 1000,
        'cache_hits': 50,
        'cache_misses': 50
    } 