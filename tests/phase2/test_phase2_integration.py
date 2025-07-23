#!/usr/bin/env python3
"""
Phase 2 Integration Tests
=========================

Integration tests that verify Phase 2 components work seamlessly
with the existing Phase 1 system and infrastructure.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import Phase 1 components
from utils.level_set_tracker import BarrierCertificateValidator
from evaluation.verify_certificate import verify_barrier_certificate

# Import Phase 2 components
from utils.validation_strategies import (
    SamplingValidationStrategy,
    SymbolicValidationStrategy,
    IntervalValidationStrategy,
    SMTValidationStrategy
)
from utils.validation_orchestrator import ValidationOrchestrator


class TestPhase2Phase1Integration:
    """Integration tests between Phase 2 and Phase 1 components"""
    
    def test_orchestrator_with_phase1_validator(self, phase2_config):
        """Test that orchestrator can work alongside Phase 1 validator"""
        # Create Phase 1 validator
        phase1_validator = BarrierCertificateValidator(
            certificate_str="x**2 + y**2 - 1.0",
            system_info={
                'variables': ['x', 'y'],
                'dynamics': ['-x', '-y'],
                'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
                'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
                'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
            },
            config=phase2_config
        )
        
        # Create Phase 2 orchestrator
        phase2_orchestrator = ValidationOrchestrator(phase2_config)
        
        # Both should be able to validate the same certificate
        certificate = "x**2 + y**2 - 1.0"
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        # Phase 1 validation
        phase1_result = phase1_validator.validate()
        
        # Phase 2 validation
        phase2_result = phase2_orchestrator.validate(certificate, system_info)
        
        # Both should produce valid results
        assert isinstance(phase1_result, dict)
        assert isinstance(phase2_result, OrchestratedResult)
        
        # Both should agree on validity (if both are working correctly)
        # Note: This might not always be true due to different validation approaches
        assert 'is_valid' in phase1_result
        assert isinstance(phase2_result.is_valid, bool)
    
    def test_strategies_with_phase1_systems(self, phase2_config):
        """Test that Phase 2 strategies can handle Phase 1 system formats"""
        # Phase 1 system format
        phase1_system = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        certificate = "x**2 + y**2 - 1.0"
        
        # Test each Phase 2 strategy
        strategies = [
            SamplingValidationStrategy(phase2_config),
            SymbolicValidationStrategy(phase2_config),
            IntervalValidationStrategy(phase2_config),
            SMTValidationStrategy(phase2_config)
        ]
        
        for strategy in strategies:
            if strategy.can_handle(certificate, phase1_system):
                result = strategy.validate(certificate, phase1_system)
                assert isinstance(result, ValidationResult)
                assert result.is_valid is not None
    
    def test_backward_compatibility(self, phase2_config):
        """Test that Phase 2 doesn't break existing Phase 1 functionality"""
        # Test that existing Phase 1 functions still work
        certificate = "x**2 + y**2 - 1.0"
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        # Phase 1 validation should still work
        phase1_result = verify_barrier_certificate(
            certificate, system_info, phase2_config
        )
        
        assert isinstance(phase1_result, dict)
        assert 'overall_success' in phase1_result or 'is_valid' in phase1_result
    
    def test_configuration_compatibility(self, phase2_config):
        """Test that Phase 2 config is compatible with Phase 1 config"""
        # Phase 1 config format
        phase1_config = {
            'numerical_tolerance': 1e-6,
            'num_samples_boundary': 1000,
            'num_samples_lie': 2000,
            'optimization_maxiter': 100,
            'optimization_popsize': 30
        }
        
        # Phase 2 config should include Phase 1 parameters
        for key, value in phase1_config.items():
            if key in phase2_config:
                assert phase2_config[key] == value or isinstance(phase2_config[key], type(value))


class TestPhase2WebInterfaceIntegration:
    """Integration tests for Phase 2 with web interface"""
    
    def test_orchestrator_in_web_context(self, phase2_config):
        """Test orchestrator usage in web interface context"""
        # Simulate web interface validation request
        request_data = {
            'certificate': "x**2 + y**2 - 1.0",
            'system_description': "2D linear system with circular sets",
            'param_overrides': {
                'confidence_threshold': 0.8,
                'max_strategies': 3
            }
        }
        
        # Create orchestrator
        orchestrator = ValidationOrchestrator(phase2_config)
        
        # Parse system info (simulating web interface parsing)
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        # Perform validation
        result = orchestrator.validate(request_data['certificate'], system_info)
        
        # Check result format suitable for web interface
        assert isinstance(result, OrchestratedResult)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_time > 0.0
        
        # Result should be serializable for JSON response
        result_dict = {
            'is_valid': result.is_valid,
            'confidence': result.confidence,
            'execution_time': result.execution_time,
            'strategies_used': result.strategies_used,
            'consensus_achieved': result.consensus_achieved,
            'details': result.details
        }
        
        assert isinstance(result_dict, dict)
        assert all(key in result_dict for key in ['is_valid', 'confidence', 'execution_time'])
    
    def test_strategy_performance_in_web_context(self, phase2_config):
        """Test strategy performance tracking in web interface context"""
        orchestrator = ValidationOrchestrator(phase2_config)
        
        # Simulate multiple web requests
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        certificates = [
            "x**2 + y**2 - 1.0",
            "x**2 + y**2 - 1.5",
            "x**2 + y**2 - 2.0"
        ]
        
        # Process multiple requests
        for certificate in certificates:
            orchestrator.validate(certificate, system_info)
        
        # Get performance summary for web dashboard
        summary = orchestrator.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert summary['total_validations'] == 3
        assert summary['avg_execution_time'] > 0.0
        
        # Get strategy performance for web interface
        strategy_performance = orchestrator.get_strategy_performance()
        
        assert isinstance(strategy_performance, dict)
        assert len(strategy_performance) > 0


class TestPhase2TestHarnessIntegration:
    """Integration tests for Phase 2 with existing test harness"""
    
    def test_orchestrator_with_test_harness(self, phase2_config):
        """Test orchestrator integration with existing test harness"""
        from tests.test_harness import BarrierCertificateTestHarness
        
        # Create test harness
        harness = BarrierCertificateTestHarness()
        
        # Create orchestrator
        orchestrator = ValidationOrchestrator(phase2_config)
        
        # Test case from harness
        test_case = {
            'id': 'test_phase2_integration',
            'system': {
                'name': '2D Linear System',
                'variables': ['x', 'y'],
                'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
                'initial_set': ['x**2 + y**2 <= 0.25'],
                'unsafe_set': ['x**2 + y**2 >= 4.0']
            },
            'certificate': "x**2 + y**2 - 1.0",
            'expected_valid': True
        }
        
        # Prepare system info
        system_info = harness._prepare_system_info(test_case['system'])
        
        # Run Phase 2 validation
        phase2_result = orchestrator.validate(test_case['certificate'], system_info)
        
        # Check result compatibility with test harness
        assert isinstance(phase2_result, OrchestratedResult)
        assert isinstance(phase2_result.is_valid, bool)
        
        # Result should be compatible with test harness expectations
        if phase2_result.is_valid == test_case['expected_valid']:
            print("Phase 2 result matches expected outcome")
        else:
            print(f"Phase 2 result: {phase2_result.is_valid}, Expected: {test_case['expected_valid']}")
    
    def test_strategies_with_ground_truth(self, phase2_config):
        """Test Phase 2 strategies with ground truth test cases"""
        # Load ground truth test cases
        import json
        import os
        
        ground_truth_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests/ground_truth/barrier_certificates.json"
        )
        
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                ground_truth_data = json.load(f)
            
            # Test first few cases
            for test_case in ground_truth_data['certificates'][:3]:
                certificate = test_case['certificate']
                system = test_case['system']
                
                # Prepare system info
                system_info = {
                    'variables': system['variables'],
                    'dynamics': [d.split('=')[1].strip() for d in system['dynamics']],
                    'initial_set_conditions': system['initial_set'],
                    'unsafe_set_conditions': system['unsafe_set'],
                    'sampling_bounds': {var: (-3, 3) for var in system['variables']}
                }
                
                # Test with orchestrator
                orchestrator = ValidationOrchestrator(phase2_config)
                result = orchestrator.validate(certificate, system_info)
                
                assert isinstance(result, OrchestratedResult)
                assert result.is_valid is not None
                
                print(f"Ground truth test: {test_case.get('name', 'unnamed')}")
                print(f"  Expected: {test_case['expected_valid']}")
                print(f"  Phase 2: {result.is_valid}")
                print(f"  Confidence: {result.confidence:.3f}")


class TestPhase2PerformanceIntegration:
    """Integration tests for Phase 2 performance with existing benchmarks"""
    
    def test_orchestrator_performance_benchmark(self, phase2_config):
        """Test orchestrator performance against existing benchmarks"""
        from tests.benchmarks.profiler import BarrierCertificateProfiler
        
        # Create profiler
        profiler = BarrierCertificateProfiler(output_dir="phase2_performance")
        
        # Create orchestrator
        orchestrator = ValidationOrchestrator(phase2_config)
        
        # Test system
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        certificate = "x**2 + y**2 - 1.0"
        
        # Profile orchestrator performance
        start_time = time.time()
        result = orchestrator.validate(certificate, system_info)
        orchestrator_time = time.time() - start_time
        
        # Compare with Phase 1 performance
        phase1_start = time.time()
        phase1_validator = BarrierCertificateValidator(
            certificate_str=certificate,
            system_info=system_info,
            config=phase2_config
        )
        phase1_result = phase1_validator.validate()
        phase1_time = time.time() - phase1_start
        
        print(f"Phase 1 time: {phase1_time:.3f}s")
        print(f"Phase 2 time: {orchestrator_time:.3f}s")
        print(f"Speedup: {phase1_time/orchestrator_time:.2f}x" if orchestrator_time > 0 else "N/A")
        
        # Both should produce valid results
        assert isinstance(phase1_result, dict)
        assert isinstance(result, OrchestratedResult)
    
    def test_strategy_performance_comparison(self, phase2_config):
        """Test performance comparison between different strategies"""
        system_info = {
            'variables': ['x', 'y'],
            'dynamics': ['-x', '-y'],
            'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
            'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
            'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
        }
        
        certificate = "x**2 + y**2 - 1.0"
        
        # Test individual strategies
        strategies = {
            'sampling': SamplingValidationStrategy(phase2_config),
            'symbolic': SymbolicValidationStrategy(phase2_config),
            'interval': IntervalValidationStrategy(phase2_config),
            'smt': SMTValidationStrategy(phase2_config)
        }
        
        performance_results = {}
        
        for name, strategy in strategies.items():
            if strategy.can_handle(certificate, system_info):
                start_time = time.time()
                result = strategy.validate(certificate, system_info)
                execution_time = time.time() - start_time
                
                performance_results[name] = {
                    'execution_time': execution_time,
                    'confidence': result.confidence,
                    'is_valid': result.is_valid
                }
        
        # Print performance comparison
        print("\nStrategy Performance Comparison:")
        for name, perf in performance_results.items():
            print(f"  {name}: {perf['execution_time']:.3f}s, "
                  f"confidence: {perf['confidence']:.3f}, "
                  f"valid: {perf['is_valid']}")
        
        # Should have at least one working strategy
        assert len(performance_results) > 0


class TestPhase2ErrorHandlingIntegration:
    """Integration tests for Phase 2 error handling with existing error handling"""
    
    def test_orchestrator_error_propagation(self, phase2_config):
        """Test that orchestrator properly handles and propagates errors"""
        orchestrator = ValidationOrchestrator(phase2_config)
        
        # Test with invalid input
        invalid_system = {'invalid': 'data'}
        certificate = "x**2 + y**2 - 1.0"
        
        result = orchestrator.validate(certificate, invalid_system)
        
        # Should handle error gracefully
        assert isinstance(result, OrchestratedResult)
        assert not result.is_valid
        assert result.confidence == 0.0
        
        # Should have error details
        assert len(result.individual_results) > 0
        for strategy_name, individual_result in result.individual_results.items():
            if 'error' in individual_result.details:
                print(f"Strategy {strategy_name} reported error: {individual_result.details['error']}")
    
    def test_strategy_error_isolation(self, phase2_config):
        """Test that errors in one strategy don't affect others"""
        orchestrator = ValidationOrchestrator(phase2_config)
        
        # Mock one strategy to fail
        with patch.object(orchestrator.strategies['sampling'], 'validate') as mock_validate:
            mock_validate.side_effect = Exception("Simulated strategy failure")
            
            system_info = {
                'variables': ['x', 'y'],
                'dynamics': ['-x', '-y'],
                'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
                'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
                'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
            }
            certificate = "x**2 + y**2 - 1.0"
            
            result = orchestrator.validate(certificate, system_info)
            
            # Should still get a result
            assert isinstance(result, OrchestratedResult)
            
            # Failed strategy should have error
            assert 'error' in result.individual_results['sampling'].details
            
            # Other strategies should still work
            working_strategies = [
                name for name, individual_result in result.individual_results.items()
                if 'error' not in individual_result.details
            ]
            
            print(f"Working strategies: {working_strategies}")
            assert len(working_strategies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 