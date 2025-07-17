#!/usr/bin/env python3
"""
FM-LLM Solver Web Interface Testbench

A comprehensive testing framework for validating and improving all web interface components
including certificate generation, conversation handling, verification, and RAG integration.

This testbench allows for isolated component testing, integration testing, performance
benchmarking, and systematic quality improvement without running the full web application.

Author: Professional Development Team
Date: 2024
License: Academic Use
"""

import sys
import json
import time
import logging
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from unittest.mock import Mock, patch

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import web interface components
from utils.config_loader import load_config
from web_interface.certificate_generator import CertificateGenerator
from web_interface.verification_service import VerificationService
from web_interface.conversation_service import ConversationService

# Test data and utilities
from omegaconf import DictConfig

# Configure logging for testbench
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("testbench.log")],
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Structured test result data."""

    component: str
    test_name: str
    status: str  # 'PASS', 'FAIL', 'ERROR', 'SKIP'
    duration: float
    details: Dict[str, Any]
    timestamp: datetime
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class BenchmarkMetrics:
    """Performance benchmark metrics."""

    avg_duration: float
    min_duration: float
    max_duration: float
    std_deviation: float
    success_rate: float
    total_tests: int
    memory_usage_mb: float
    cpu_usage_percent: float


class TestDataProvider:
    """Provides comprehensive test data for various scenarios."""

    @staticmethod
    def get_system_descriptions() -> List[Dict[str, Any]]:
        """Get diverse system descriptions for testing."""
        return [
            {
                "name": "simple_continuous",
                "description": """System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x >= 1.5
Safe Set: x < 1.5""",
                "expected_type": "continuous",
                "complexity": "low",
                "known_certificate": "x**2 + y**2",
            },
            {
                "name": "linear_stable",
                "description": """System Dynamics: dx/dt = -0.5*x, dy/dt = -0.5*y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 1.0
Safe Set: x**2 + y**2 < 1.0""",
                "expected_type": "continuous",
                "complexity": "low",
                "known_certificate": "x**2 + y**2 - 0.25",
            },
            {
                "name": "discrete_linear",
                "description": """System Dynamics: x{k+1} = 0.8*x{k}, y{k+1} = 0.9*y{k}
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: abs(x) >= 2.0 or abs(y) >= 2.0
Safe Set: abs(x) < 2.0 and abs(y) < 2.0""",
                "expected_type": "discrete",
                "complexity": "low",
                "known_certificate": "x**2 + y**2",
            },
            {
                "name": "nonlinear_complex",
                "description": """System Dynamics: dx/dt = x*(1-x**2-y**2) - y, dy/dt = y*(1-x**2-y**2) + x
Initial Set: x**2 + y**2 <= 0.01
Unsafe Set: x**2 + y**2 >= 4.0
Safe Set: x**2 + y**2 < 4.0""",
                "expected_type": "continuous",
                "complexity": "high",
                "known_certificate": None,
            },
            {
                "name": "malformed_input",
                "description": "This is not a proper system description",
                "expected_type": "invalid",
                "complexity": "error",
                "known_certificate": None,
            },
            {
                "name": "incomplete_system",
                "description": "System Dynamics: dx/dt = x + y",
                "expected_type": "continuous",
                "complexity": "incomplete",
                "known_certificate": None,
            },
        ]

    @staticmethod
    def get_domain_bounds_test_cases() -> List[Dict[str, Any]]:
        """Get domain bounds test cases."""
        return [
            {
                "name": "simple_bounds",
                "bounds": {"x": [-2, 2], "y": [-2, 2]},
                "description": "Simple rectangular domain",
            },
            {
                "name": "asymmetric_bounds",
                "bounds": {"x": [-1, 3], "y": [-2, 1]},
                "description": "Asymmetric domain bounds",
            },
            {"name": "empty_bounds", "bounds": {}, "description": "No domain bounds specified"},
            {
                "name": "invalid_bounds",
                "bounds": {"x": [2, -1], "y": [1, 1]},  # min > max
                "description": "Invalid bounds for error testing",
            },
        ]

    @staticmethod
    def get_conversation_scenarios() -> List[Dict[str, Any]]:
        """Get conversation test scenarios."""
        return [
            {
                "name": "guided_discovery",
                "messages": [
                    "I have a robot that needs to avoid obstacles",
                    "The robot moves according to dx/dt = -x + u, dy/dt = -y + v",
                    "It starts in a circle of radius 0.5 around the origin",
                    "It must avoid the region x >= 2",
                ],
                "expected_extraction": True,
            },
            {
                "name": "direct_specification",
                "messages": [
                    "System: dx/dt = -x**3 - y, dy/dt = x - y**3\nInitial: x**2 + y**2 <= 0.1\nUnsafe: x >= 1.5"
                ],
                "expected_extraction": True,
            },
            {
                "name": "vague_description",
                "messages": ["I have a system", "It needs to be safe", "Can you help?"],
                "expected_extraction": False,
            },
        ]


class ComponentTestSuite:
    """Base class for component test suites."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.results: List[TestResult] = []

    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Execute a test function and record results."""
        start_time = time.time()
        timestamp = datetime.now()

        try:
            details = test_func(*args, **kwargs)
            duration = time.time() - start_time

            # Determine status based on details
            if isinstance(details, dict) and details.get("success", True):
                status = "PASS"
            else:
                status = "FAIL"

            return TestResult(
                component=self.__class__.__name__,
                test_name=test_name,
                status=status,
                duration=duration,
                details=details,
                timestamp=timestamp,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Test {test_name} failed with exception: {str(e)}")

            return TestResult(
                component=self.__class__.__name__,
                test_name=test_name,
                status="ERROR",
                duration=duration,
                details={"error": str(e), "traceback": traceback.format_exc()},
                timestamp=timestamp,
                error_message=str(e),
            )

    def run_all_tests(self) -> List[TestResult]:
        """Run all tests for this component."""
        raise NotImplementedError("Subclasses must implement run_all_tests")


class CertificateGeneratorTestSuite(ComponentTestSuite):
    """Test suite for CertificateGenerator component."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.generator = None

    def setup(self) -> TestResult:
        """Setup the certificate generator."""

        def _setup():
            try:
                self.generator = CertificateGenerator(self.config)
                available_models = self.generator.get_available_models()
                return {
                    "success": True,
                    "available_models": available_models,
                    "model_count": len(available_models),
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        return self.run_test("setup", _setup)

    def test_model_availability(self) -> TestResult:
        """Test model availability and loading."""

        def _test():
            if not self.generator:
                return {"success": False, "error": "Generator not initialized"}

            available_models = self.generator.get_available_models()
            results = {}

            for model in available_models:
                model_key = model["key"]
                availability = self.generator.test_model_availability(model_key)
                results[model_key] = availability

            return {
                "success": all(r.get("available", False) for r in results.values()),
                "model_results": results,
            }

        return self.run_test("model_availability", _test)

    def test_certificate_generation(self) -> TestResult:
        """Test certificate generation with various inputs."""

        def _test():
            if not self.generator:
                return {"success": False, "error": "Generator not initialized"}

            test_systems = TestDataProvider.get_system_descriptions()
            available_models = self.generator.get_available_models()

            if not available_models:
                return {"success": False, "error": "No models available for testing"}

            results = {}
            model_key = available_models[0]["key"]  # Use first available model

            for system in test_systems:
                if system["complexity"] == "error":
                    continue  # Skip intentionally malformed inputs for this test

                result = self.generator.generate_certificate(
                    system["description"], model_key, rag_k=0  # Disable RAG for basic testing
                )

                results[system["name"]] = {
                    "generation_success": result["success"],
                    "certificate_extracted": result["certificate"] is not None,
                    "certificate": result["certificate"],
                    "llm_output_length": len(result.get("llm_output", "")),
                    "prompt_length": result.get("prompt_length", 0),
                }

            successful_generations = sum(1 for r in results.values() if r["generation_success"])
            certificate_extractions = sum(1 for r in results.values() if r["certificate_extracted"])

            return {
                "success": successful_generations > 0,
                "total_systems": len(results),
                "successful_generations": successful_generations,
                "successful_extractions": certificate_extractions,
                "extraction_rate": certificate_extractions / len(results) if results else 0,
                "results": results,
            }

        return self.run_test("certificate_generation", _test)

    def test_domain_bounds_integration(self) -> TestResult:
        """Test certificate generation with domain bounds."""

        def _test():
            if not self.generator:
                return {"success": False, "error": "Generator not initialized"}

            available_models = self.generator.get_available_models()
            if not available_models:
                return {"success": False, "error": "No models available"}

            model_key = available_models[0]["key"]
            test_system = TestDataProvider.get_system_descriptions()[0]  # Simple system
            domain_bounds_cases = TestDataProvider.get_domain_bounds_test_cases()

            results = {}

            for bounds_case in domain_bounds_cases:
                if bounds_case["name"] == "invalid_bounds":
                    continue  # Skip invalid for this test

                result = self.generator.generate_certificate(
                    test_system["description"],
                    model_key,
                    rag_k=0,
                    domain_bounds=bounds_case["bounds"],
                )

                results[bounds_case["name"]] = {
                    "success": result["success"],
                    "bounds_passed": bounds_case["bounds"] == result.get("domain_bounds"),
                    "certificate": result["certificate"],
                }

            return {"success": all(r["success"] for r in results.values()), "results": results}

        return self.run_test("domain_bounds_integration", _test)

    def test_rag_integration(self) -> TestResult:
        """Test RAG (Retrieval-Augmented Generation) integration."""

        def _test():
            if not self.generator:
                return {"success": False, "error": "Generator not initialized"}

            available_models = self.generator.get_available_models()
            if not available_models:
                return {"success": False, "error": "No models available"}

            model_key = available_models[0]["key"]
            test_system = TestDataProvider.get_system_descriptions()[0]

            results = {}
            rag_k_values = [0, 1, 3, 5]

            for rag_k in rag_k_values:
                result = self.generator.generate_certificate(
                    test_system["description"], model_key, rag_k=rag_k
                )

                results[f"rag_k_{rag_k}"] = {
                    "success": result["success"],
                    "context_chunks": result.get("context_chunks", 0),
                    "certificate": result["certificate"],
                    "prompt_length": result.get("prompt_length", 0),
                }

            # Verify that higher RAG k values retrieve more context
            context_increasing = True
            prev_chunks = -1
            for rag_k in rag_k_values:
                current_chunks = results[f"rag_k_{rag_k}"]["context_chunks"]
                if rag_k > 0 and current_chunks <= prev_chunks and prev_chunks >= 0:
                    context_increasing = False
                prev_chunks = current_chunks

            return {
                "success": all(r["success"] for r in results.values()),
                "context_retrieval_working": any(r["context_chunks"] > 0 for r in results.values()),
                "context_scaling_properly": context_increasing,
                "results": results,
            }

        return self.run_test("rag_integration", _test)

    def run_all_tests(self) -> List[TestResult]:
        """Run all certificate generator tests."""
        tests = [
            self.setup(),
            self.test_model_availability(),
            self.test_certificate_generation(),
            self.test_domain_bounds_integration(),
            self.test_rag_integration(),
        ]

        self.results.extend(tests)
        return tests


class VerificationServiceTestSuite(ComponentTestSuite):
    """Test suite for VerificationService component."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.verification_service = None

    def setup(self) -> TestResult:
        """Setup the verification service."""

        def _setup():
            try:
                self.verification_service = VerificationService(self.config)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return self.run_test("setup", _setup)

    def test_system_parsing(self) -> TestResult:
        """Test system description parsing."""

        def _test():
            if not self.verification_service:
                return {"success": False, "error": "Service not initialized"}

            test_systems = TestDataProvider.get_system_descriptions()
            results = {}

            for system in test_systems:
                parsed = self.verification_service.parse_system_description(system["description"])

                results[system["name"]] = {
                    "parsing_success": len(parsed.get("variables", [])) > 0,
                    "has_dynamics": len(parsed.get("dynamics", [])) > 0,
                    "has_initial_set": len(parsed.get("initial_set", [])) > 0,
                    "has_unsafe_set": len(parsed.get("unsafe_set", [])) > 0,
                    "parsed_data": parsed,
                }

            successful_parsing = sum(1 for r in results.values() if r["parsing_success"])

            return {
                "success": successful_parsing > 0,
                "total_systems": len(results),
                "successful_parsing": successful_parsing,
                "parsing_rate": successful_parsing / len(results),
                "results": results,
            }

        return self.run_test("system_parsing", _test)

    def test_certificate_verification(self) -> TestResult:
        """Test certificate verification with known good certificates."""

        def _test():
            if not self.verification_service:
                return {"success": False, "error": "Service not initialized"}

            test_systems = TestDataProvider.get_system_descriptions()
            results = {}

            for system in test_systems:
                if system["known_certificate"] is None or system["complexity"] == "error":
                    continue

                verification_result = self.verification_service.verify_certificate(
                    system["known_certificate"], system["description"]
                )

                results[system["name"]] = {
                    "verification_attempted": True,
                    "overall_success": verification_result["overall_success"],
                    "numerical_passed": verification_result["numerical_passed"],
                    "symbolic_passed": verification_result["symbolic_passed"],
                    "sos_passed": verification_result["sos_passed"],
                    "verification_time": verification_result["verification_time"],
                }

            if not results:
                return {"success": False, "error": "No systems with known certificates to test"}

            successful_verifications = sum(1 for r in results.values() if r["overall_success"])

            return {
                "success": successful_verifications > 0,
                "total_verifications": len(results),
                "successful_verifications": successful_verifications,
                "verification_rate": successful_verifications / len(results),
                "results": results,
            }

        return self.run_test("certificate_verification", _test)

    def test_bounds_creation(self) -> TestResult:
        """Test sampling bounds creation."""

        def _test():
            if not self.verification_service:
                return {"success": False, "error": "Service not initialized"}

            test_systems = TestDataProvider.get_system_descriptions()
            results = {}

            for system in test_systems:
                if system["complexity"] == "error":
                    continue

                parsed_system = self.verification_service.parse_system_description(
                    system["description"]
                )
                bounds = self.verification_service.create_sampling_bounds(parsed_system)

                results[system["name"]] = {
                    "bounds_created": len(bounds) > 0,
                    "all_variables_bounded": all(
                        isinstance(b, tuple) and len(b) == 2 and b[0] < b[1]
                        for b in bounds.values()
                    ),
                    "bounds": bounds,
                }

            successful_bounds = sum(
                1 for r in results.values() if r["bounds_created"] and r["all_variables_bounded"]
            )

            return {
                "success": successful_bounds > 0,
                "total_systems": len(results),
                "successful_bounds": successful_bounds,
                "results": results,
            }

        return self.run_test("bounds_creation", _test)

    def run_all_tests(self) -> List[TestResult]:
        """Run all verification service tests."""
        tests = [
            self.setup(),
            self.test_system_parsing(),
            self.test_certificate_verification(),
            self.test_bounds_creation(),
        ]

        self.results.extend(tests)
        return tests


class ConversationServiceTestSuite(ComponentTestSuite):
    """Test suite for ConversationService component."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.conversation_service = None

    def setup(self) -> TestResult:
        """Setup the conversation service."""

        def _setup():
            try:
                # Mock database for testing
                with patch("web_interface.conversation_service.db"):
                    self.conversation_service = ConversationService(self.config)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return self.run_test("setup", _setup)

    def test_conversation_extraction(self) -> TestResult:
        """Test system description extraction from conversations."""

        def _test():
            if not self.conversation_service:
                return {"success": False, "error": "Service not initialized"}

            conversation_scenarios = TestDataProvider.get_conversation_scenarios()
            results = {}

            for scenario in conversation_scenarios:
                # Mock conversation object
                mock_conversation = Mock()
                mock_conversation.messages = []

                for i, message in enumerate(scenario["messages"]):
                    mock_msg = Mock()
                    mock_msg.role = "user"
                    mock_msg.content = message
                    mock_conversation.messages.append(mock_msg)

                # Test extraction
                extracted = self.conversation_service._extract_system_description_from_conversation(
                    mock_conversation
                )

                results[scenario["name"]] = {
                    "extraction_attempted": True,
                    "description_extracted": extracted is not None,
                    "expected_extraction": scenario["expected_extraction"],
                    "correct_prediction": (extracted is not None)
                    == scenario["expected_extraction"],
                    "extracted_content": extracted,
                }

            correct_predictions = sum(1 for r in results.values() if r["correct_prediction"])

            return {
                "success": correct_predictions == len(results),
                "total_scenarios": len(results),
                "correct_predictions": correct_predictions,
                "accuracy": correct_predictions / len(results),
                "results": results,
            }

        return self.run_test("conversation_extraction", _test)

    def test_domain_bounds_extraction(self) -> TestResult:
        """Test domain bounds extraction from conversations."""

        def _test():
            if not self.conversation_service:
                return {"success": False, "error": "Service not initialized"}

            test_messages = [
                "domain: x ∈ [-2, 2], y ∈ [-1, 1]",
                "x in [-2, 2] and y in [-1, 1]",
                "bounds: x [-2, 2], y [-1, 1]",
                "valid for x between -2 and 2, y between -1 and 1",
            ]

            results = {}

            for i, message in enumerate(test_messages):
                mock_conversation = Mock()
                mock_msg = Mock()
                mock_msg.role = "user"
                mock_msg.content = message
                mock_conversation.messages = [mock_msg]

                extracted_bounds = (
                    self.conversation_service._extract_domain_bounds_from_conversation(
                        mock_conversation
                    )
                )

                results[f"message_{i}"] = {
                    "bounds_extracted": extracted_bounds is not None and len(extracted_bounds) > 0,
                    "x_bound_correct": (
                        extracted_bounds.get("x") == [-2, 2] if extracted_bounds else False
                    ),
                    "y_bound_correct": (
                        extracted_bounds.get("y") == [-1, 1] if extracted_bounds else False
                    ),
                    "extracted_bounds": extracted_bounds,
                    "original_message": message,
                }

            successful_extractions = sum(
                1 for r in results.values() if r["bounds_extracted"] and r["x_bound_correct"]
            )

            return {
                "success": successful_extractions > 0,
                "total_messages": len(results),
                "successful_extractions": successful_extractions,
                "extraction_rate": successful_extractions / len(results),
                "results": results,
            }

        return self.run_test("domain_bounds_extraction", _test)

    def run_all_tests(self) -> List[TestResult]:
        """Run all conversation service tests."""
        tests = [
            self.setup(),
            self.test_conversation_extraction(),
            self.test_domain_bounds_extraction(),
        ]

        self.results.extend(tests)
        return tests


class IntegrationTestSuite(ComponentTestSuite):
    """Integration tests for combined component functionality."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.generator = None
        self.verification_service = None

    def setup(self) -> TestResult:
        """Setup all components for integration testing."""

        def _setup():
            try:
                self.generator = CertificateGenerator(self.config)
                self.verification_service = VerificationService(self.config)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return self.run_test("setup", _setup)

    def test_end_to_end_workflow(self) -> TestResult:
        """Test complete workflow from generation to verification."""

        def _test():
            if not self.generator or not self.verification_service:
                return {"success": False, "error": "Components not initialized"}

            test_systems = TestDataProvider.get_system_descriptions()
            available_models = self.generator.get_available_models()

            if not available_models:
                return {"success": False, "error": "No models available"}

            model_key = available_models[0]["key"]
            results = {}

            for system in test_systems[:3]:  # Test first 3 systems
                if system["complexity"] == "error":
                    continue

                # Generate certificate
                generation_result = self.generator.generate_certificate(
                    system["description"], model_key, rag_k=1
                )

                verification_result = None
                if generation_result["success"] and generation_result["certificate"]:
                    # Verify generated certificate
                    verification_result = self.verification_service.verify_certificate(
                        generation_result["certificate"], system["description"]
                    )

                results[system["name"]] = {
                    "generation_success": generation_result["success"],
                    "certificate_extracted": generation_result["certificate"] is not None,
                    "verification_attempted": verification_result is not None,
                    "verification_success": (
                        verification_result["overall_success"] if verification_result else False
                    ),
                    "workflow_complete": (
                        generation_result["success"]
                        and generation_result["certificate"] is not None
                        and verification_result is not None
                    ),
                }

            completed_workflows = sum(1 for r in results.values() if r["workflow_complete"])

            return {
                "success": completed_workflows > 0,
                "total_systems": len(results),
                "completed_workflows": completed_workflows,
                "completion_rate": completed_workflows / len(results) if results else 0,
                "results": results,
            }

        return self.run_test("end_to_end_workflow", _test)

    def run_all_tests(self) -> List[TestResult]:
        """Run all integration tests."""
        tests = [self.setup(), self.test_end_to_end_workflow()]

        self.results.extend(tests)
        return tests


class WebInterfaceTestbench:
    """Main testbench orchestrator for web interface testing."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the testbench with configuration."""
        self.config_path = config_path or str(PROJECT_ROOT / "config.yaml")
        self.config = load_config(self.config_path)
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

        # Initialize test suites
        self.test_suites = {
            "certificate_generator": CertificateGeneratorTestSuite(self.config),
            "verification_service": VerificationServiceTestSuite(self.config),
            "conversation_service": ConversationServiceTestSuite(self.config),
            "integration": IntegrationTestSuite(self.config),
        }

        logger.info("Web Interface Testbench initialized")

    def run_component_tests(self, component: str) -> List[TestResult]:
        """Run tests for a specific component."""
        if component not in self.test_suites:
            raise ValueError(f"Unknown component: {component}")

        logger.info(f"Running tests for component: {component}")
        suite = self.test_suites[component]
        return suite.run_all_tests()

    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all test suites."""
        self.start_time = datetime.now()
        logger.info("Starting comprehensive test suite")

        all_results = {}

        for component_name, suite in self.test_suites.items():
            logger.info(f"Running {component_name} test suite...")
            try:
                component_results = suite.run_all_tests()
                all_results[component_name] = component_results
                self.results.extend(component_results)

                passed = sum(1 for r in component_results if r.status == "PASS")
                total = len(component_results)
                logger.info(f"{component_name}: {passed}/{total} tests passed")

            except Exception as e:
                logger.error(f"Error running {component_name} tests: {str(e)}")
                all_results[component_name] = []

        self.end_time = datetime.now()
        logger.info("Test suite completed")

        return all_results

    def run_performance_benchmark(self, iterations: int = 10) -> BenchmarkMetrics:
        """Run performance benchmarks."""
        logger.info(f"Running performance benchmark with {iterations} iterations")

        if not self.test_suites["certificate_generator"].generator:
            self.test_suites["certificate_generator"].setup()

        generator = self.test_suites["certificate_generator"].generator
        if not generator:
            raise RuntimeError("Certificate generator not available for benchmarking")

        available_models = generator.get_available_models()
        if not available_models:
            raise RuntimeError("No models available for benchmarking")

        model_key = available_models[0]["key"]
        test_system = TestDataProvider.get_system_descriptions()[0]

        durations = []
        successes = 0

        for i in range(iterations):
            start_time = time.time()
            result = generator.generate_certificate(test_system["description"], model_key, rag_k=1)
            duration = time.time() - start_time
            durations.append(duration)

            if result["success"]:
                successes += 1

        return BenchmarkMetrics(
            avg_duration=np.mean(durations),
            min_duration=np.min(durations),
            max_duration=np.max(durations),
            std_deviation=np.std(durations),
            success_rate=successes / iterations,
            total_tests=iterations,
            memory_usage_mb=0.0,  # Could implement actual memory monitoring
            cpu_usage_percent=0.0,  # Could implement actual CPU monitoring
        )

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.results:
            logger.warning("No test results available for report generation")
            return {}

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        error_tests = sum(1 for r in self.results if r.status == "ERROR")

        # Component breakdown
        component_stats = {}
        for component in self.test_suites.keys():
            component_results = [
                r for r in self.results if r.component.lower().replace("testsuite", "") == component
            ]
            if component_results:
                component_stats[component] = {
                    "total": len(component_results),
                    "passed": sum(1 for r in component_results if r.status == "PASS"),
                    "failed": sum(1 for r in component_results if r.status == "FAIL"),
                    "errors": sum(1 for r in component_results if r.status == "ERROR"),
                    "avg_duration": np.mean([r.duration for r in component_results]),
                }

        # Overall timing
        total_duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else 0
        )

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat(),
            },
            "component_breakdown": component_stats,
            "detailed_results": [r.to_dict() for r in self.results],
        }

        # Save report if output path specified
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Test report saved to: {output_file}")

        return report

    def run_quality_improvement_cycle(self, max_iterations: int = 5) -> Dict[str, Any]:
        """Run iterative quality improvement cycle."""
        logger.info(f"Starting quality improvement cycle (max {max_iterations} iterations)")

        improvement_history = []
        baseline_score = None

        for iteration in range(max_iterations):
            logger.info(f"Quality improvement iteration {iteration + 1}")

            # Run tests
            results = self.run_all_tests()

            # Calculate current score
            all_results = []
            for component_results in results.values():
                all_results.extend(component_results)

            passed = sum(1 for r in all_results if r.status == "PASS")
            total = len(all_results)
            current_score = passed / total if total > 0 else 0

            # Record iteration
            iteration_data = {
                "iteration": iteration + 1,
                "score": current_score,
                "total_tests": total,
                "passed_tests": passed,
                "improvements_made": [],
            }

            if baseline_score is None:
                baseline_score = current_score
                logger.info(f"Baseline score: {baseline_score:.2%}")
            else:
                improvement = current_score - baseline_score
                logger.info(f"Current score: {current_score:.2%} (change: {improvement:+.2%})")

            # Analyze failures and suggest improvements
            failed_results = [r for r in all_results if r.status != "PASS"]
            if failed_results:
                suggestions = self._analyze_failures_and_suggest_improvements(failed_results)
                iteration_data["improvement_suggestions"] = suggestions

                # Apply automatic fixes if possible
                applied_fixes = self._apply_automatic_fixes(failed_results)
                iteration_data["improvements_made"] = applied_fixes

            improvement_history.append(iteration_data)

            # Check for convergence
            if current_score >= 0.95:  # 95% success rate threshold
                logger.info("Quality threshold achieved, stopping improvement cycle")
                break

        return {
            "baseline_score": baseline_score,
            "final_score": current_score,
            "improvement": current_score - baseline_score if baseline_score else 0,
            "iterations_completed": len(improvement_history),
            "history": improvement_history,
        }

    def _analyze_failures_and_suggest_improvements(
        self, failed_results: List[TestResult]
    ) -> List[str]:
        """Analyze failed tests and suggest improvements."""
        suggestions = []

        # Group failures by component
        failure_patterns = {}
        for result in failed_results:
            component = result.component
            if component not in failure_patterns:
                failure_patterns[component] = []
            failure_patterns[component].append(result)

        for component, failures in failure_patterns.items():
            error_messages = [f.error_message for f in failures if f.error_message]

            # Analyze common error patterns
            if any("model" in msg.lower() for msg in error_messages):
                suggestions.append(f"Consider adjusting model configuration for {component}")

            if any("timeout" in msg.lower() for msg in error_messages):
                suggestions.append(f"Increase timeout values for {component} operations")

            if any("memory" in msg.lower() for msg in error_messages):
                suggestions.append(
                    f"Reduce batch sizes or enable memory optimization for {component}"
                )

            if any("connection" in msg.lower() for msg in error_messages):
                suggestions.append(
                    f"Check network connectivity and service availability for {component}"
                )

        return suggestions

    def _apply_automatic_fixes(self, failed_results: List[TestResult]) -> List[str]:
        """Apply automatic fixes for common failure patterns."""
        applied_fixes = []

        # This is where you would implement automatic fixes
        # For now, we'll just log potential fixes
        for result in failed_results:
            if result.error_message and "timeout" in result.error_message.lower():
                applied_fixes.append(f"Would increase timeout for {result.test_name}")

            if result.error_message and "memory" in result.error_message.lower():
                applied_fixes.append(f"Would enable memory optimization for {result.test_name}")

        return applied_fixes


def main():
    """Main entry point for the testbench."""
    import argparse

    parser = argparse.ArgumentParser(description="FM-LLM Solver Web Interface Testbench")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--component",
        type=str,
        choices=[
            "certificate_generator",
            "verification_service",
            "conversation_service",
            "integration",
        ],
        help="Run tests for specific component only",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--improve", action="store_true", help="Run quality improvement cycle")
    parser.add_argument(
        "--max-improve-iterations", type=int, default=5, help="Maximum improvement iterations"
    )
    parser.add_argument("--output", type=str, help="Output file for test report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize testbench
    testbench = WebInterfaceTestbench(args.config)

    try:
        if args.component:
            # Run specific component tests
            results = testbench.run_component_tests(args.component)
            logger.info(
                f"Component {args.component}: {len([r for r in results if r.status == 'PASS'])}/{len(results)} tests passed"
            )

        elif args.benchmark:
            # Run performance benchmark
            metrics = testbench.run_performance_benchmark(args.iterations)
            logger.info("Benchmark Results:")
            logger.info(f"  Average Duration: {metrics.avg_duration:.3f}s")
            logger.info(f"  Success Rate: {metrics.success_rate:.1%}")
            logger.info(
                f"  Min/Max Duration: {metrics.min_duration:.3f}s / {metrics.max_duration:.3f}s"
            )

        elif args.improve:
            # Run quality improvement cycle
            improvement_results = testbench.run_quality_improvement_cycle(
                args.max_improve_iterations
            )
            logger.info("Quality Improvement Results:")
            logger.info(f"  Baseline Score: {improvement_results['baseline_score']:.1%}")
            logger.info(f"  Final Score: {improvement_results['final_score']:.1%}")
            logger.info(f"  Improvement: {improvement_results['improvement']:+.1%}")

        else:
            # Run all tests
            results = testbench.run_all_tests()
            total_passed = sum(
                len([r for r in component_results if r.status == "PASS"])
                for component_results in results.values()
            )
            total_tests = sum(len(component_results) for component_results in results.values())
            logger.info(
                f"Overall Results: {total_passed}/{total_tests} tests passed ({total_passed/total_tests:.1%})"
            )

        # Generate report
        testbench.generate_report(args.output)

        if args.output:
            logger.info(f"Detailed report saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Testbench execution failed: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
