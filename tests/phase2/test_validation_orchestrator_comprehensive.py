#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Phase 2 Validation Orchestrator
===========================================================

Tests the validation orchestrator with comprehensive coverage:
- Strategy selection and scoring
- Parallel and sequential execution
- Result combination and consensus
- Performance monitoring
- Error handling and fallbacks

Includes integration tests and edge cases.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from utils.validation_orchestrator import ValidationOrchestrator, OrchestratedResult
from utils.validation_strategies import ValidationResult


class TestValidationOrchestrator:
    """Comprehensive tests for validation orchestrator"""
    
    def test_orchestrator_creation(self, phase2_config):
        """Test orchestrator creation and initialization"""
        orchestrator = ValidationOrchestrator(phase2_config)
        
        assert orchestrator.config == phase2_config
        assert orchestrator.confidence_threshold == 0.8
        assert orchestrator.max_strategies == 3
        assert orchestrator.parallel_execution is True
        assert len(orchestrator.performance_history) == 0
    
    def test_strategy_selection(self, orchestrator, simple_system_info):
        """Test intelligent strategy selection"""
        certificate = "x**2 + y**2 - 1.0"
        
        selected_strategies = orchestrator._select_strategies(certificate, simple_system_info)
        
        assert isinstance(selected_strategies, list)
        assert len(selected_strategies) <= orchestrator.max_strategies
        
        # Check that selected strategies can handle the problem
        for strategy in selected_strategies:
            assert strategy.can_handle(certificate, simple_system_info)
    
    def test_strategy_scoring(self, orchestrator, simple_system_info):
        """Test strategy scoring algorithm"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Get all available strategies
        available_strategies = []
        for name, strategy in orchestrator.strategies.items():
            if strategy.can_handle(certificate, simple_system_info):
                available_strategies.append((name, strategy))
        
        # Score each strategy
        scored_strategies = []
        for name, strategy in available_strategies:
            score = orchestrator._score_strategy(strategy, certificate, simple_system_info)
            scored_strategies.append((score, name, strategy))
        
        # Check scoring results
        assert len(scored_strategies) > 0
        
        for score, name, strategy in scored_strategies:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_problem_specific_scoring(self, orchestrator):
        """Test problem-specific scoring logic"""
        # Test polynomial system
        polynomial_system = {
            'variables': ['x', 'y'],
            'dynamics': ['-x**2', '-y**2'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        polynomial_certificate = "x**2 + y**2 - 1.0"
        
        # Get symbolic strategy
        symbolic_strategy = orchestrator.strategies['symbolic']
        
        # Score should be higher for polynomial systems
        score = orchestrator._calculate_problem_score(
            symbolic_strategy, polynomial_certificate, polynomial_system
        )
        assert score > 0.5  # Should be higher for polynomial systems
    
    def test_basic_validation(self, orchestrator, simple_system_info):
        """Test basic orchestrated validation"""
        certificate = "x**2 + y**2 - 1.0"
        
        result = orchestrator.validate(certificate, simple_system_info)
        
        assert isinstance(result, OrchestratedResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_time > 0.0
        assert isinstance(result.strategies_used, list)
        assert isinstance(result.individual_results, dict)
        assert isinstance(result.consensus_achieved, bool)
        assert isinstance(result.details, dict)
    
    def test_parallel_execution(self, orchestrator, simple_system_info):
        """Test parallel execution of strategies"""
        orchestrator.parallel_execution = True
        certificate = "x**2 + y**2 - 1.0"
        
        result = orchestrator.validate(certificate, simple_system_info)
        
        assert isinstance(result, OrchestratedResult)
        assert len(result.strategies_used) > 0
        assert len(result.individual_results) > 0
    
    def test_sequential_execution(self, orchestrator, simple_system_info):
        """Test sequential execution of strategies"""
        orchestrator.parallel_execution = False
        certificate = "x**2 + y**2 - 1.0"
        
        result = orchestrator.validate(certificate, simple_system_info)
        
        assert isinstance(result, OrchestratedResult)
        assert len(result.strategies_used) > 0
        assert len(result.individual_results) > 0
    
    def test_result_combination_consensus(self, orchestrator):
        """Test result combination when strategies agree"""
        # Create mock results that agree
        mock_results = {
            'strategy1': ValidationResult(
                is_valid=True, confidence=0.9, execution_time=1.0,
                samples_used=100, violations_found=[], strategy_name='strategy1',
                details={}
            ),
            'strategy2': ValidationResult(
                is_valid=True, confidence=0.8, execution_time=1.5,
                samples_used=150, violations_found=[], strategy_name='strategy2',
                details={}
            )
        }
        
        result = orchestrator._combine_results(mock_results)
        
        assert isinstance(result, OrchestratedResult)
        assert result.is_valid is True  # Both agree
        assert result.consensus_achieved is True
        assert result.confidence > 0.8  # Should be boosted for consensus
    
    def test_result_combination_disagreement(self, orchestrator):
        """Test result combination when strategies disagree"""
        # Create mock results that disagree
        mock_results = {
            'strategy1': ValidationResult(
                is_valid=True, confidence=0.9, execution_time=1.0,
                samples_used=100, violations_found=[], strategy_name='strategy1',
                details={}
            ),
            'strategy2': ValidationResult(
                is_valid=False, confidence=0.8, execution_time=1.5,
                samples_used=150, violations_found=[], strategy_name='strategy2',
                details={}
            )
        }
        
        result = orchestrator._combine_results(mock_results)
        
        assert isinstance(result, OrchestratedResult)
        assert result.consensus_achieved is False
        assert result.confidence < 0.9  # Should be reduced for disagreement
    
    def test_early_termination(self, orchestrator, simple_system_info):
        """Test early termination when high confidence is achieved"""
        orchestrator.confidence_threshold = 0.9
        orchestrator.parallel_execution = False
        certificate = "x**2 + y**2 - 1.0"
        
        # Mock a strategy to return high confidence
        with patch.object(orchestrator.strategies['sampling'], 'validate') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True, confidence=0.95, execution_time=0.5,
                samples_used=50, violations_found=[], strategy_name='sampling',
                details={}
            )
            
            result = orchestrator.validate(certificate, simple_system_info)
            
            # Should terminate early due to high confidence
            assert result.confidence >= 0.9
    
    def test_error_handling(self, orchestrator, simple_system_info):
        """Test error handling in orchestrator"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Mock a strategy to raise an exception
        with patch.object(orchestrator.strategies['sampling'], 'validate') as mock_validate:
            mock_validate.side_effect = Exception("Test error")
            
            result = orchestrator.validate(certificate, simple_system_info)
            
            # Should still get a result, but with error details
            assert isinstance(result, OrchestratedResult)
            assert 'error' in result.individual_results['sampling'].details
    
    def test_performance_summary(self, orchestrator, simple_system_info):
        """Test performance summary generation"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Run multiple validations
        for _ in range(3):
            orchestrator.validate(certificate, simple_system_info)
        
        summary = orchestrator.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert 'total_validations' in summary
        assert 'avg_execution_time' in summary
        assert 'avg_confidence' in summary
        assert 'consensus_rate' in summary
        assert 'strategy_usage' in summary
        
        assert summary['total_validations'] == 3
        assert summary['avg_execution_time'] > 0.0
    
    def test_strategy_performance(self, orchestrator):
        """Test strategy performance metrics"""
        performance = orchestrator.get_strategy_performance()
        
        assert isinstance(performance, dict)
        assert 'sampling' in performance
        assert 'symbolic' in performance
        assert 'interval' in performance
        assert 'smt' in performance
        
        for strategy_name, metrics in performance.items():
            assert isinstance(metrics, StrategyPerformance)
            assert metrics.strategy_name == strategy_name


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with real strategies"""
    
    def test_end_to_end_validation(self, orchestrator, simple_system_info):
        """Test end-to-end validation with real strategies"""
        certificate = "x**2 + y**2 - 1.0"
        
        result = orchestrator.validate(certificate, simple_system_info)
        
        # Check basic result structure
        assert isinstance(result, OrchestratedResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        
        # Check that strategies were used
        assert len(result.strategies_used) > 0
        assert len(result.individual_results) > 0
        
        # Check individual results
        for strategy_name, individual_result in result.individual_results.items():
            assert isinstance(individual_result, ValidationResult)
            assert individual_result.strategy_name == strategy_name
    
    def test_different_certificate_types(self, orchestrator, simple_system_info):
        """Test validation with different certificate types"""
        certificates = {
            'simple': "x**2 + y**2 - 1.0",
            'complex': "x**4 + y**4 - 2.0",
            'polynomial': "x**2 + y**2 - 1.5"
        }
        
        results = {}
        for name, certificate in certificates.items():
            result = orchestrator.validate(certificate, simple_system_info)
            results[name] = result
        
        # Check that all validations completed
        for name, result in results.items():
            assert isinstance(result, OrchestratedResult)
            assert result.is_valid is not None
    
    def test_different_system_complexities(self, orchestrator):
        """Test validation with different system complexities"""
        systems = {
            'simple_2d': {
                'variables': ['x', 'y'],
                'dynamics': ['-x', '-y'],
                'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
                'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
                'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
            },
            'complex_3d': {
                'variables': ['x', 'y', 'z'],
                'dynamics': ['-x', '-y', '-z'],
                'initial_set_conditions': ['x**2 + y**2 + z**2 <= 0.25'],
                'unsafe_set_conditions': ['x**2 + y**2 + z**2 >= 4.0'],
                'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3), 'z': (-3, 3)}
            }
        }
        
        certificate = "x**2 + y**2 - 1.0"
        
        for name, system in systems.items():
            result = orchestrator.validate(certificate, system)
            assert isinstance(result, OrchestratedResult)
    
    def test_performance_comparison(self, orchestrator, simple_system_info):
        """Test performance comparison between parallel and sequential"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Test parallel execution
        orchestrator.parallel_execution = True
        start_time = time.time()
        parallel_result = orchestrator.validate(certificate, simple_system_info)
        parallel_time = time.time() - start_time
        
        # Test sequential execution
        orchestrator.parallel_execution = False
        start_time = time.time()
        sequential_result = orchestrator.validate(certificate, simple_system_info)
        sequential_time = time.time() - start_time
        
        # Both should produce valid results
        assert isinstance(parallel_result, OrchestratedResult)
        assert isinstance(sequential_result, OrchestratedResult)
        
        # Performance comparison (parallel might be faster)
        print(f"Parallel time: {parallel_time:.3f}s")
        print(f"Sequential time: {sequential_time:.3f}s")


class TestOrchestratorEdgeCases:
    """Edge case tests for orchestrator"""
    
    def test_no_available_strategies(self, orchestrator):
        """Test handling when no strategies can handle the problem"""
        # Create a system that no strategy can handle
        impossible_system = {
            'variables': ['x'],
            'dynamics': ['invalid_dynamics'],
            'initial_set_conditions': ['invalid_condition'],
            'unsafe_set_conditions': ['invalid_condition'],
            'sampling_bounds': {'x': (-1, 1)}
        }
        
        certificate = "invalid_certificate"
        
        result = orchestrator.validate(certificate, impossible_system)
        
        assert isinstance(result, OrchestratedResult)
        assert not result.is_valid
        assert result.confidence == 0.0
        assert len(result.strategies_used) == 0
    
    def test_empty_certificate(self, orchestrator, simple_system_info):
        """Test handling of empty certificate"""
        result = orchestrator.validate("", simple_system_info)
        
        assert isinstance(result, OrchestratedResult)
        assert not result.is_valid
        assert result.confidence == 0.0
    
    def test_empty_system_info(self, orchestrator):
        """Test handling of empty system info"""
        result = orchestrator.validate("x**2 + y**2 - 1.0", {})
        
        assert isinstance(result, OrchestratedResult)
        assert not result.is_valid
        assert result.confidence == 0.0
    
    def test_concurrent_orchestrator_usage(self, phase2_config):
        """Test concurrent usage of orchestrator"""
        import threading
        import time
        
        def validate_with_orchestrator():
            orchestrator = ValidationOrchestrator(phase2_config)
            system_info = {
                'variables': ['x', 'y'],
                'dynamics': ['-x', '-y'],
                'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
                'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
                'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
            }
            certificate = "x**2 + y**2 - 1.0"
            return orchestrator.validate(certificate, system_info)
        
        # Run multiple validations concurrently
        threads = []
        results = []
        
        for _ in range(3):
            thread = threading.Thread(target=lambda: results.append(validate_with_orchestrator()))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, OrchestratedResult)
    
    def test_large_number_of_strategies(self, phase2_config):
        """Test orchestrator with large number of strategies"""
        # Create orchestrator with many strategies
        config = DictConfig({
            'confidence_threshold': 0.8,
            'max_strategies': 10,  # Large number
            'parallel_execution': True,
            'num_samples_boundary': 100,
            'num_samples_lie': 200,
            'numerical_tolerance': 1e-6
        })
        
        orchestrator = ValidationOrchestrator(config)
        
        # Test validation
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        certificate = "x**2 + y**2 - 1.0"
        
        result = orchestrator.validate(certificate, system_info)
        
        assert isinstance(result, OrchestratedResult)
        assert len(result.strategies_used) <= config.max_strategies


class TestOrchestratorPerformance:
    """Performance tests for orchestrator"""
    
    def test_execution_time_tracking(self, orchestrator, simple_system_info):
        """Test execution time tracking"""
        certificate = "x**2 + y**2 - 1.0"
        
        start_time = time.time()
        result = orchestrator.validate(certificate, simple_system_info)
        actual_time = time.time() - start_time
        
        # Check that execution time is tracked
        assert result.execution_time > 0.0
        assert abs(result.execution_time - actual_time) < 0.1  # Within 100ms
    
    def test_memory_usage(self, orchestrator, simple_system_info):
        """Test memory usage during validation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        certificate = "x**2 + y**2 - 1.0"
        
        # Run multiple validations
        for _ in range(5):
            orchestrator.validate(certificate, simple_system_info)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
    
    def test_strategy_selection_performance(self, orchestrator, simple_system_info):
        """Test performance of strategy selection"""
        certificate = "x**2 + y**2 - 1.0"
        
        # Measure strategy selection time
        start_time = time.time()
        selected_strategies = orchestrator._select_strategies(certificate, simple_system_info)
        selection_time = time.time() - start_time
        
        # Strategy selection should be fast (less than 1 second)
        assert selection_time < 1.0
        assert len(selected_strategies) > 0
    
    def test_result_combination_performance(self, orchestrator):
        """Test performance of result combination"""
        # Create many mock results
        mock_results = {}
        for i in range(10):
            mock_results[f'strategy{i}'] = ValidationResult(
                is_valid=True, confidence=0.8 + i * 0.02, execution_time=1.0,
                samples_used=100, violations_found=[], strategy_name=f'strategy{i}',
                details={}
            )
        
        # Measure combination time
        start_time = time.time()
        result = orchestrator._combine_results(mock_results)
        combination_time = time.time() - start_time
        
        # Result combination should be fast (less than 0.1 second)
        assert combination_time < 0.1
        assert isinstance(result, OrchestratedResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 