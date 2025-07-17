#!/usr/bin/env python3
"""
Tests for Phase 2 Validation Strategies and Orchestrator
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omegaconf import DictConfig
from utils.validation_strategies import (
    SamplingValidationStrategy,
    SymbolicValidationStrategy,
    IntervalValidationStrategy,
    SMTValidationStrategy,
    ValidationResult,
    StrategyPerformance,
)
from utils.validation_orchestrator import ValidationOrchestrator, OrchestratedResult


class TestValidationStrategies(unittest.TestCase):
    """Test suite for validation strategies"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DictConfig(
            {"num_samples_boundary": 100, "num_samples_lie": 200, "numerical_tolerance": 1e-6}
        )

        self.system_info = {
            "variables": ["x", "y"],
            "dynamics": ["-x", "-y"],
            "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
            "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
            "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
        }

        self.valid_certificate = "x**2 + y**2 - 1.0"
        self.invalid_certificate = "x**2 + y**2 + 1.0"

    def test_sampling_strategy_creation(self):
        """Test sampling strategy creation"""
        strategy = SamplingValidationStrategy(self.config)
        self.assertEqual(strategy.name, "SamplingValidation")
        self.assertTrue(strategy.can_handle(self.valid_certificate, self.system_info))

    def test_sampling_strategy_validation(self):
        """Test sampling strategy validation"""
        strategy = SamplingValidationStrategy(self.config)
        result = strategy.validate(self.valid_certificate, self.system_info)

        self.assertIsInstance(result, ValidationResult)
        self.assertIn("violation_count", result.details)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreater(result.execution_time, 0.0)

    def test_symbolic_strategy_creation(self):
        """Test symbolic strategy creation"""
        strategy = SymbolicValidationStrategy(self.config)
        self.assertEqual(strategy.name, "SymbolicValidation")
        self.assertTrue(strategy.can_handle(self.valid_certificate, self.system_info))

    def test_symbolic_strategy_validation(self):
        """Test symbolic strategy validation"""
        strategy = SymbolicValidationStrategy(self.config)
        result = strategy.validate(self.valid_certificate, self.system_info)

        self.assertIsInstance(result, ValidationResult)
        self.assertIn("method", result.details)
        self.assertEqual(result.details["method"], "symbolic")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_interval_strategy_creation(self):
        """Test interval strategy creation"""
        strategy = IntervalValidationStrategy(self.config)
        self.assertEqual(strategy.name, "IntervalValidation")
        self.assertTrue(strategy.can_handle(self.valid_certificate, self.system_info))

    def test_interval_strategy_validation(self):
        """Test interval strategy validation"""
        strategy = IntervalValidationStrategy(self.config)
        result = strategy.validate(self.valid_certificate, self.system_info)

        self.assertIsInstance(result, ValidationResult)
        self.assertIn("method", result.details)
        self.assertEqual(result.details["method"], "interval_arithmetic")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_smt_strategy_creation(self):
        """Test SMT strategy creation"""
        strategy = SMTValidationStrategy(self.config)
        self.assertEqual(strategy.name, "SMTValidation")
        # SMT availability depends on Z3 installation
        can_handle = strategy.can_handle(self.valid_certificate, self.system_info)
        self.assertIsInstance(can_handle, bool)

    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking"""
        strategy = SamplingValidationStrategy(self.config)

        # Run multiple validations
        for _ in range(3):
            strategy.validate(self.valid_certificate, self.system_info)

        # Check performance metrics
        performance = strategy.get_performance_metrics()
        self.assertIsInstance(performance, StrategyPerformance)
        self.assertEqual(performance.strategy_name, "SamplingValidation")
        self.assertGreater(performance.avg_execution_time, 0.0)
        self.assertGreaterEqual(performance.avg_confidence, 0.0)
        self.assertLessEqual(performance.avg_confidence, 1.0)


class TestValidationOrchestrator(unittest.TestCase):
    """Test suite for validation orchestrator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DictConfig(
            {
                "confidence_threshold": 0.8,
                "max_strategies": 3,
                "parallel_execution": True,
                "num_samples_boundary": 100,
                "num_samples_lie": 200,
                "numerical_tolerance": 1e-6,
            }
        )

        self.system_info = {
            "variables": ["x", "y"],
            "dynamics": ["-x", "-y"],
            "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
            "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
            "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
        }

        self.valid_certificate = "x**2 + y**2 - 1.0"
        self.orchestrator = ValidationOrchestrator(self.config)

    def test_orchestrator_creation(self):
        """Test orchestrator creation"""
        self.assertIsInstance(self.orchestrator, ValidationOrchestrator)
        self.assertEqual(self.orchestrator.confidence_threshold, 0.8)
        self.assertEqual(self.orchestrator.max_strategies, 3)
        self.assertTrue(self.orchestrator.parallel_execution)

    def test_strategy_selection(self):
        """Test strategy selection"""
        strategies = self.orchestrator._select_strategies(self.valid_certificate, self.system_info)

        self.assertIsInstance(strategies, list)
        self.assertLessEqual(len(strategies), self.orchestrator.max_strategies)

        # Check that selected strategies can handle the problem
        for strategy in strategies:
            self.assertTrue(strategy.can_handle(self.valid_certificate, self.system_info))

    def test_orchestrated_validation(self):
        """Test orchestrated validation"""
        result = self.orchestrator.validate(self.valid_certificate, self.system_info)

        self.assertIsInstance(result, OrchestratedResult)
        self.assertIsInstance(result.is_valid, bool)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreater(result.execution_time, 0.0)
        self.assertIsInstance(result.strategies_used, list)
        self.assertIsInstance(result.individual_results, dict)
        self.assertIsInstance(result.consensus_achieved, bool)

    def test_parallel_execution(self):
        """Test parallel execution"""
        self.orchestrator.parallel_execution = True
        result = self.orchestrator.validate(self.valid_certificate, self.system_info)

        self.assertIsInstance(result, OrchestratedResult)
        self.assertGreater(len(result.strategies_used), 0)

    def test_sequential_execution(self):
        """Test sequential execution"""
        self.orchestrator.parallel_execution = False
        result = self.orchestrator.validate(self.valid_certificate, self.system_info)

        self.assertIsInstance(result, OrchestratedResult)
        self.assertGreater(len(result.strategies_used), 0)

    def test_result_combination(self):
        """Test result combination logic"""
        # Create mock results
        mock_results = {
            "strategy1": ValidationResult(
                is_valid=True,
                confidence=0.9,
                execution_time=1.0,
                samples_used=100,
                violations_found=[],
                strategy_name="strategy1",
                details={},
            ),
            "strategy2": ValidationResult(
                is_valid=True,
                confidence=0.8,
                execution_time=1.5,
                samples_used=150,
                violations_found=[],
                strategy_name="strategy2",
                details={},
            ),
        }

        result = self.orchestrator._combine_results(mock_results)

        self.assertIsInstance(result, OrchestratedResult)
        self.assertTrue(result.consensus_achieved)  # Both agree
        self.assertTrue(result.is_valid)  # Should be valid
        self.assertGreater(result.confidence, 0.8)  # Should be boosted

    def test_performance_summary(self):
        """Test performance summary generation"""
        # Run some validations first
        for _ in range(2):
            self.orchestrator.validate(self.valid_certificate, self.system_info)

        summary = self.orchestrator.get_performance_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn("total_validations", summary)
        self.assertIn("avg_execution_time", summary)
        self.assertIn("avg_confidence", summary)
        self.assertIn("consensus_rate", summary)
        self.assertIn("strategy_usage", summary)

        self.assertEqual(summary["total_validations"], 2)
        self.assertGreater(summary["avg_execution_time"], 0.0)

    def test_strategy_performance(self):
        """Test strategy performance metrics"""
        performance = self.orchestrator.get_strategy_performance()

        self.assertIsInstance(performance, dict)
        self.assertIn("sampling", performance)
        self.assertIn("symbolic", performance)
        self.assertIn("interval", performance)
        self.assertIn("smt", performance)

        for strategy_name, metrics in performance.items():
            self.assertIsInstance(metrics, StrategyPerformance)
            self.assertEqual(metrics.strategy_name, strategy_name)


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2 components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DictConfig(
            {
                "confidence_threshold": 0.7,
                "max_strategies": 2,
                "parallel_execution": True,
                "num_samples_boundary": 50,
                "num_samples_lie": 100,
                "numerical_tolerance": 1e-6,
            }
        )

        self.system_info = {
            "variables": ["x", "y"],
            "dynamics": ["-x", "-y"],
            "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
            "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
            "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
        }

        self.orchestrator = ValidationOrchestrator(self.config)

    def test_end_to_end_validation(self):
        """Test end-to-end validation with orchestrator"""
        certificate = "x**2 + y**2 - 1.0"

        result = self.orchestrator.validate(certificate, self.system_info)

        # Check basic result structure
        self.assertIsInstance(result, OrchestratedResult)
        self.assertIsInstance(result.is_valid, bool)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

        # Check that strategies were used
        self.assertGreater(len(result.strategies_used), 0)
        self.assertGreater(len(result.individual_results), 0)

        # Check individual results
        for strategy_name, individual_result in result.individual_results.items():
            self.assertIsInstance(individual_result, ValidationResult)
            self.assertEqual(individual_result.strategy_name, strategy_name)

    def test_performance_comparison(self):
        """Test performance comparison between strategies"""
        certificate = "x**2 + y**2 - 1.0"

        # Run validation multiple times to build performance data
        for _ in range(3):
            self.orchestrator.validate(certificate, self.system_info)

        # Get performance summary
        summary = self.orchestrator.get_performance_summary()
        strategy_performance = self.orchestrator.get_strategy_performance()

        # Check that we have performance data
        self.assertGreater(summary["total_validations"], 0)
        self.assertGreater(summary["avg_execution_time"], 0.0)

        # Check strategy usage
        self.assertGreater(len(summary["strategy_usage"]), 0)

        # Check individual strategy performance
        for strategy_name, metrics in strategy_performance.items():
            self.assertIsInstance(metrics.avg_execution_time, float)
            self.assertIsInstance(metrics.avg_confidence, float)
            self.assertIsInstance(metrics.success_rate, float)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
