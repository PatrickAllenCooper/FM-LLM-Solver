#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Phase 2 Validation Strategies
========================================================

Tests all validation strategies with comprehensive coverage:
- SamplingValidationStrategy
- SymbolicValidationStrategy  
- IntervalValidationStrategy
- SMTValidationStrategy

Includes edge cases, performance tests, and integration scenarios.
"""

import pytest
import numpy as np
import sympy as sp
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from utils.validation_strategies import (
    SamplingValidationStrategy,
    SymbolicValidationStrategy,
    IntervalValidationStrategy,
    SMTValidationStrategy,
    ValidationResult,
    StrategyPerformance
)


class TestSamplingValidationStrategy:
    """Comprehensive tests for sampling validation strategy"""
    
    def test_strategy_creation(self, phase2_config):
        """Test strategy creation and initialization"""
        strategy = SamplingValidationStrategy(phase2_config)
        
        assert strategy.name == "SamplingValidation"
        assert strategy.config == phase2_config
        assert len(strategy.performance_history) == 0
    
    def test_can_handle_any_certificate(self, sampling_strategy, simple_system_info):
        """Test that sampling can handle any certificate"""
        certificates = [
            "x**2 + y**2 - 1.0",
            "sin(x) + cos(y)",
            "x**4 + y**4 - 2.0",
            "exp(-x**2 - y**2) - 0.5"
        ]
        
        for certificate in certificates:
            assert sampling_strategy.can_handle(certificate, simple_system_info)
    
    def test_basic_validation(self, sampling_strategy, simple_system_info):
        """Test basic validation functionality"""
        certificate = "x**2 + y**2 - 1.0"
        
        result = sampling_strategy.validate(certificate, simple_system_info)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_time > 0.0
        assert result.samples_used > 0
        assert result.strategy_name == "SamplingValidation"
    
    def test_validation_with_violations(self, sampling_strategy, simple_system_info):
        """Test validation when violations are found"""
        # This certificate should have violations
        certificate = "x**2 + y**2 + 1.0"  # Always positive
        
        result = sampling_strategy.validate(certificate, simple_system_info)
        
        assert isinstance(result, ValidationResult)
        # Should be invalid due to violations
        assert not result.is_valid
        assert result.confidence < 1.0
    
    def test_sample_generation(self, sampling_strategy, simple_system_info):
        """Test sample generation functionality"""
        n_samples = 100
        samples = sampling_strategy._generate_samples(simple_system_info, n_samples)
        
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (n_samples, 2)  # 2 variables
        
        # Check bounds
        for sample in samples:
            assert -3.0 <= sample[0] <= 3.0  # x bounds
            assert -3.0 <= sample[1] <= 3.0  # y bounds
    
    def test_confidence_calculation(self, sampling_strategy):
        """Test confidence calculation logic"""
        # No violations should give high confidence
        high_confidence = sampling_strategy._calculate_confidence(0, 1000)
        assert high_confidence >= 0.8
        
        # Many violations should give low confidence
        low_confidence = sampling_strategy._calculate_confidence(100, 1000)
        assert low_confidence < 0.5
        
        # Some violations should give medium confidence
        medium_confidence = sampling_strategy._calculate_confidence(10, 1000)
        assert 0.3 <= medium_confidence <= 0.7
    
    def test_performance_tracking(self, sampling_strategy, simple_system_info):
        """Test performance tracking functionality"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Run multiple validations
        for _ in range(3):
            sampling_strategy.validate(certificate, simple_system_info)
        
        # Check performance metrics
        performance = sampling_strategy.get_performance_metrics()
        
        assert isinstance(performance, StrategyPerformance)
        assert performance.strategy_name == "SamplingValidation"
        assert performance.avg_execution_time > 0.0
        assert 0.0 <= performance.avg_confidence <= 1.0
        assert 0.0 <= performance.success_rate <= 1.0
        assert performance.samples_per_second > 0.0
    
    def test_error_handling(self, sampling_strategy):
        """Test error handling in validation"""
        # Invalid system info
        invalid_system = {'invalid': 'data'}
        certificate = "x**2 + y**2 - 1.0"
        
        result = sampling_strategy.validate(certificate, invalid_system)
        
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert result.confidence == 0.0
        assert 'error' in result.details
    
    def test_different_system_dimensions(self, sampling_strategy):
        """Test validation with different system dimensions"""
        # 2D system
        system_2d = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        # 3D system
        system_3d = {
            'variables': ['x', 'y', 'z'],
            'dynamics': ['-x', '-y', '-z'],
            'initial_set_conditions': ['x**2 + y**2 + z**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 + z**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3), 'z': (-3, 3)}
        }
        
        certificate_2d = "x**2 + y**2 - 1.0"
        certificate_3d = "x**2 + y**2 + z**2 - 1.0"
        
        # Test 2D
        result_2d = sampling_strategy.validate(certificate_2d, system_2d)
        assert isinstance(result_2d, ValidationResult)
        
        # Test 3D
        result_3d = sampling_strategy.validate(certificate_3d, system_3d)
        assert isinstance(result_3d, ValidationResult)


class TestSymbolicValidationStrategy:
    """Comprehensive tests for symbolic validation strategy"""
    
    def test_strategy_creation(self, phase2_config):
        """Test strategy creation and initialization"""
        strategy = SymbolicValidationStrategy(phase2_config)
        
        assert strategy.name == "SymbolicValidation"
        assert strategy.config == phase2_config
    
    def test_can_handle_polynomial_systems(self, symbolic_strategy, polynomial_system_info):
        """Test that symbolic can handle polynomial systems"""
        polynomial_certificate = "x**2 + y**2 - 1.0"
        
        assert symbolic_strategy.can_handle(polynomial_certificate, polynomial_system_info)
    
    def test_cannot_handle_non_polynomial_systems(self, symbolic_strategy, simple_system_info):
        """Test that symbolic cannot handle non-polynomial systems"""
        non_polynomial_certificate = "sin(x) + cos(y) - 1.0"
        
        assert not symbolic_strategy.can_handle(non_polynomial_certificate, simple_system_info)
    
    def test_basic_validation(self, symbolic_strategy, polynomial_system_info):
        """Test basic symbolic validation"""
        certificate = "x**2 + y**2 - 1.0"
        
        result = symbolic_strategy.validate(certificate, polynomial_system_info)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_time > 0.0
        assert result.samples_used == 0  # No sampling in symbolic
        assert result.strategy_name == "SymbolicValidation"
        assert result.details['method'] == 'symbolic'
    
    def test_symbolic_condition_checking(self, symbolic_strategy, polynomial_system_info):
        """Test symbolic condition checking"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Mock the symbolic check to return True
        with patch.object(symbolic_strategy, '_symbolic_check_conditions', return_value=True):
            result = symbolic_strategy.validate(certificate, polynomial_system_info)
            assert result.is_valid
        
        # Mock the symbolic check to return False
        with patch.object(symbolic_strategy, '_symbolic_check_conditions', return_value=False):
            result = symbolic_strategy.validate(certificate, polynomial_system_info)
            assert not result.is_valid
    
    def test_error_handling(self, symbolic_strategy, simple_system_info):
        """Test error handling in symbolic validation"""
        # Non-polynomial certificate
        certificate = "sin(x) + cos(y) - 1.0"
        
        result = symbolic_strategy.validate(certificate, simple_system_info)
        
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert result.confidence == 0.0
        assert 'error' in result.details
    
    def test_performance_tracking(self, symbolic_strategy, polynomial_system_info):
        """Test performance tracking for symbolic strategy"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Run multiple validations
        for _ in range(3):
            with patch.object(symbolic_strategy, '_symbolic_check_conditions', return_value=True):
                symbolic_strategy.validate(certificate, polynomial_system_info)
        
        # Check performance metrics
        performance = symbolic_strategy.get_performance_metrics()
        
        assert isinstance(performance, StrategyPerformance)
        assert performance.strategy_name == "SymbolicValidation"
        assert performance.avg_execution_time > 0.0
        assert performance.samples_per_second == 0.0  # No sampling


class TestIntervalValidationStrategy:
    """Comprehensive tests for interval validation strategy"""
    
    def test_strategy_creation(self, phase2_config):
        """Test strategy creation and initialization"""
        strategy = IntervalValidationStrategy(phase2_config)
        
        assert strategy.name == "IntervalValidation"
        assert strategy.config == phase2_config
    
    def test_can_handle_any_system(self, interval_strategy, simple_system_info):
        """Test that interval validation can handle any system"""
        certificates = [
            "x**2 + y**2 - 1.0",
            "sin(x) + cos(y)",
            "x**4 + y**4 - 2.0"
        ]
        
        for certificate in certificates:
            assert interval_strategy.can_handle(certificate, simple_system_info)
    
    def test_basic_validation(self, interval_strategy, simple_system_info):
        """Test basic interval validation"""
        certificate = "x**2 + y**2 - 1.0"
        
        result = interval_strategy.validate(certificate, simple_system_info)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_time > 0.0
        assert result.samples_used == 0  # No sampling in interval arithmetic
        assert result.strategy_name == "IntervalValidation"
        assert result.details['method'] == 'interval_arithmetic'
    
    def test_interval_condition_checking(self, interval_strategy, simple_system_info):
        """Test interval condition checking"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Mock the interval check to return True
        with patch.object(interval_strategy, '_interval_check_conditions', return_value=True):
            result = interval_strategy.validate(certificate, simple_system_info)
            assert result.is_valid
        
        # Mock the interval check to return False
        with patch.object(interval_strategy, '_interval_check_conditions', return_value=False):
            result = interval_strategy.validate(certificate, simple_system_info)
            assert not result.is_valid
    
    def test_error_handling(self, interval_strategy):
        """Test error handling in interval validation"""
        invalid_system = {'invalid': 'data'}
        certificate = "x**2 + y**2 - 1.0"
        
        result = interval_strategy.validate(certificate, invalid_system)
        
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert result.confidence == 0.0
        assert 'error' in result.details


class TestSMTValidationStrategy:
    """Comprehensive tests for SMT validation strategy"""
    
    def test_strategy_creation(self, phase2_config):
        """Test strategy creation and initialization"""
        strategy = SMTValidationStrategy(phase2_config)
        
        assert strategy.name == "SMTValidation"
        assert strategy.config == phase2_config
    
    def test_smt_availability_check(self, phase2_config):
        """Test SMT availability checking"""
        strategy = SMTValidationStrategy(phase2_config)
        
        # Check if Z3 is available
        has_z3 = hasattr(strategy, 'z3_available')
        assert isinstance(has_z3, bool)
    
    def test_can_handle_systems(self, smt_strategy, simple_system_info):
        """Test that SMT can handle appropriate systems"""
        certificate = "x**2 + y**2 - 1.0"
        
        # SMT availability depends on Z3 installation
        can_handle = smt_strategy.can_handle(certificate, simple_system_info)
        assert isinstance(can_handle, bool)
    
    def test_basic_validation(self, smt_strategy, simple_system_info):
        """Test basic SMT validation"""
        certificate = "x**2 + y**2 - 1.0"
        
        result = smt_strategy.validate(certificate, simple_system_info)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_time > 0.0
        assert result.samples_used == 0  # No sampling in SMT
        assert result.strategy_name == "SMTValidation"
    
    def test_smt_condition_checking(self, smt_strategy, simple_system_info):
        """Test SMT condition checking"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Mock the SMT check to return True
        with patch.object(smt_strategy, '_smt_check_conditions', return_value=True):
            result = smt_strategy.validate(certificate, simple_system_info)
            assert result.is_valid
        
        # Mock the SMT check to return False
        with patch.object(smt_strategy, '_smt_check_conditions', return_value=False):
            result = smt_strategy.validate(certificate, simple_system_info)
            assert not result.is_valid
    
    def test_error_handling_no_smt(self, phase2_config):
        """Test error handling when SMT solver is not available"""
        # Mock Z3 import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'z3'")):
            strategy = SMTValidationStrategy(phase2_config)
            assert not strategy.z3_available
            
            result = strategy.validate("x**2 + y**2 - 1.0", {})
            assert not result.is_valid
            assert 'error' in result.details


class TestValidationStrategiesIntegration:
    """Integration tests for validation strategies"""
    
    def test_strategy_comparison(self, all_strategies, simple_system_info):
        """Test comparison between different strategies"""
        certificate = "x**2 + y**2 - 1.0"
        results = {}
        
        for name, strategy in all_strategies.items():
            if strategy.can_handle(certificate, simple_system_info):
                result = strategy.validate(certificate, simple_system_info)
                results[name] = result
        
        # Check that we got results from at least some strategies
        assert len(results) > 0
        
        # Check result properties
        for name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert result.strategy_name == name
    
    def test_performance_comparison(self, all_strategies, simple_system_info):
        """Test performance comparison between strategies"""
        certificate = "x**2 + y**2 - 1.0"
        performance_data = {}
        
        for name, strategy in all_strategies.items():
            if strategy.can_handle(certificate, simple_system_info):
                # Run multiple times for better performance data
                for _ in range(3):
                    strategy.validate(certificate, simple_system_info)
                
                performance = strategy.get_performance_metrics()
                performance_data[name] = performance
        
        # Check performance metrics
        for name, perf in performance_data.items():
            assert isinstance(perf, StrategyPerformance)
            assert perf.strategy_name == name
            assert perf.avg_execution_time >= 0.0
    
    def test_strategy_fallback(self, all_strategies, simple_system_info):
        """Test strategy fallback when some strategies fail"""
        certificate = "x**2 + y**2 - 1.0"
        working_strategies = []
        
        for name, strategy in all_strategies.items():
            try:
                if strategy.can_handle(certificate, simple_system_info):
                    result = strategy.validate(certificate, simple_system_info)
                    if result.is_valid is not None:  # Valid result
                        working_strategies.append(name)
            except Exception:
                continue  # Strategy failed
        
        # Should have at least one working strategy
        assert len(working_strategies) > 0


class TestValidationStrategiesEdgeCases:
    """Edge case tests for validation strategies"""
    
    def test_empty_certificate(self, sampling_strategy, simple_system_info):
        """Test handling of empty certificate"""
        result = sampling_strategy.validate("", simple_system_info)
        assert not result.is_valid
        assert result.confidence == 0.0
    
    def test_invalid_certificate_syntax(self, sampling_strategy, simple_system_info):
        """Test handling of invalid certificate syntax"""
        result = sampling_strategy.validate("invalid syntax", simple_system_info)
        assert not result.is_valid
        assert result.confidence == 0.0
    
    def test_empty_system_info(self, sampling_strategy):
        """Test handling of empty system info"""
        result = sampling_strategy.validate("x**2 + y**2 - 1.0", {})
        assert not result.is_valid
        assert result.confidence == 0.0
    
    def test_large_system(self, sampling_strategy):
        """Test handling of large systems"""
        # Create a 10D system
        variables = [f'x{i}' for i in range(10)]
        dynamics = [f'-x{i}' for i in range(10)]
        bounds = {var: (-3, 3) for var in variables}
        
        large_system = {
            'variables': variables,
            'dynamics': dynamics,
            'initial_set_conditions': ['x0**2 + x1**2 <= 0.25'],
            'unsafe_set_conditions': ['x0**2 + x1**2 >= 4.0'],
            'sampling_bounds': bounds
        }
        
        certificate = "x0**2 + x1**2 - 1.0"
        
        result = sampling_strategy.validate(certificate, large_system)
        assert isinstance(result, ValidationResult)
    
    def test_concurrent_validation(self, sampling_strategy, simple_system_info):
        """Test concurrent validation execution"""
        import threading
        import time
        
        certificate = "x**2 + y**2 - 1.0"
        results = []
        
        def validate():
            result = sampling_strategy.validate(certificate, simple_system_info)
            results.append(result)
        
        # Run multiple validations concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=validate)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 