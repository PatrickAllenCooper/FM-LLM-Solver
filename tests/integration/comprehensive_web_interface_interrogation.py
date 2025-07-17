#!/usr/bin/env python3
"""
Comprehensive Web Interface Interrogation Testbench

Deep analysis of every web interface element:
1. SOS vs Numerical vs Symbolic verification consistency
2. Known correct barrier certificate validation
3. Qwen LLM prompting optimization analysis
4. LLM output parsing accuracy and symbolic representation
5. Cross-component consistency verification
"""

import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from unittest.mock import Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from web_interface.verification_service import VerificationService
from web_interface.certificate_generator import CertificateGenerator
from inference.generate_certificate import format_prompt_with_context

logger = logging.getLogger(__name__)


@dataclass
class VerificationConsistencyResult:
    """Result from verification consistency analysis."""

    certificate: str
    system: str
    sos_result: Optional[bool]
    numerical_result: Optional[bool]
    symbolic_result: Optional[bool]
    overall_result: bool
    consistency_score: float
    discrepancies: List[str]
    verification_time: float
    theoretical_expectation: Optional[bool]


@dataclass
class LLMPromptAnalysis:
    """Analysis of LLM prompting effectiveness."""

    model_type: str
    prompt_length: int
    context_chunks: int
    template_used: str
    domain_bounds_included: bool
    system_complexity: str
    expected_output_format: str
    qwen_specific_optimizations: List[str]


@dataclass
class ParsingAccuracyResult:
    """Result from LLM output parsing analysis."""

    raw_output: str
    parsed_certificate: Optional[str]
    parsing_successful: bool
    symbolic_representation: Optional[str]
    syntax_errors: List[str]
    semantic_correctness: bool
    template_detected: bool
    confidence_score: float


class ComprehensiveWebInterfaceInterrogator:
    """Deep interrogation of all web interface components."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize comprehensive interrogator."""
        self.config = load_config(config_path or "config.yaml")
        self.verification_service = VerificationService(self.config)
        self.certificate_generator = None
        self.results = {
            "verification_consistency": [],
            "llm_prompt_analysis": [],
            "parsing_accuracy": [],
            "known_certificate_validation": [],
            "cross_component_consistency": [],
        }

    def _initialize_certificate_generator(self):
        """Initialize certificate generator with mock for testing."""
        if self.certificate_generator is None:
            # Create minimal mock to test parsing without loading full models
            mock_config = Mock()
            mock_config.fine_tuning = Mock()
            mock_config.fine_tuning.base_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Actual model
            mock_config.paths = Mock()
            mock_config.paths.ft_output_dir = "/mock/path"
            mock_config.knowledge_base = Mock()
            mock_config.knowledge_base.barrier_certificate_type = "unified"
            mock_config.inference = Mock()
            mock_config.inference.max_new_tokens = 512
            mock_config.inference.temperature = 0.1
            mock_config.inference.top_p = 0.9

            self.certificate_generator = CertificateGenerator.__new__(CertificateGenerator)
            self.certificate_generator.config = mock_config
            self.certificate_generator.models = {}
            self.certificate_generator.knowledge_bases = {}
            self.certificate_generator.embedding_model = None

    def create_known_correct_certificates(self) -> List[Dict[str, Any]]:
        """Create mathematically proven correct barrier certificates."""
        return [
            {
                "name": "simple_stable_linear",
                "certificate": "x**2 + y**2",
                "system": """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "theoretical_result": True,
                "theory": "Quadratic Lyapunov function for stable linear system",
                "expected_sos": True,
                "expected_numerical": True,
                "complexity": "simple",
            },
            {
                "name": "elliptical_barrier_scaled",
                "certificate": "2*x**2 + y**2",
                "system": """System Dynamics: dx/dt = -2*x, dy/dt = -y
Initial Set: 2*x**2 + y**2 <= 0.5
Unsafe Set: x**2 + y**2 >= 4.0""",
                "theoretical_result": True,
                "theory": "Scaled elliptical barrier matching system dynamics",
                "expected_sos": True,
                "expected_numerical": True,
                "complexity": "medium",
            },
            {
                "name": "polynomial_barrier_nonlinear",
                "certificate": "x**4 + y**4",
                "system": """System Dynamics: dx/dt = -x**3, dy/dt = -y**3
Initial Set: x**4 + y**4 <= 0.0625
Unsafe Set: x**4 + y**4 >= 1.0""",
                "theoretical_result": True,
                "theory": "Higher-order Lyapunov for polynomial dynamics",
                "expected_sos": True,
                "expected_numerical": True,
                "complexity": "complex",
            },
            {
                "name": "known_failing_certificate",
                "certificate": "x + y",
                "system": """System Dynamics: dx/dt = x + y, dy/dt = -x + y
Initial Set: x <= -1 and y <= -1
Unsafe Set: x >= 1 or y >= 1""",
                "theoretical_result": False,
                "theory": "Linear barrier for unstable system (should fail)",
                "expected_sos": False,
                "expected_numerical": False,
                "complexity": "simple",
            },
        ]

    def test_verification_consistency(self) -> List[VerificationConsistencyResult]:
        """Test consistency between SOS, numerical, and symbolic verification."""
        known_certificates = self.create_known_correct_certificates()
        consistency_results = []

        logger.info("🔍 Testing verification method consistency...")

        for cert_info in known_certificates:
            start_time = time.time()

            try:
                # Run full verification
                result = self.verification_service.verify_certificate(
                    cert_info["certificate"],
                    cert_info["system"],
                    param_overrides={
                        "num_samples_lie": 300,
                        "num_samples_boundary": 150,
                        "numerical_tolerance": 1e-6,
                        "enable_sos": True,
                        "sos_degree": 2,
                    },
                )

                verification_time = time.time() - start_time

                # Extract individual verification results
                sos_result = result.get("details", {}).get("sos", {}).get("success")
                numerical_result = result.get("numerical_passed", False)
                symbolic_result = result.get("symbolic_passed")
                overall_result = result.get("overall_success", False)

                # Calculate consistency score
                expected_result = cert_info["theoretical_result"]
                results = [
                    r for r in [sos_result, numerical_result, symbolic_result] if r is not None
                ]

                if len(results) > 1:
                    # Check if all non-None results agree
                    agreement = len(set(results)) == 1
                    consistency_score = 1.0 if agreement else 0.0

                    # Check agreement with theoretical expectation
                    if expected_result is not None:
                        theory_agreement = all(
                            r == expected_result for r in results if r is not None
                        )
                        if theory_agreement:
                            consistency_score = min(consistency_score + 0.5, 1.0)
                else:
                    consistency_score = 0.5  # Insufficient data

                # Identify discrepancies
                discrepancies = []
                if sos_result is not None and numerical_result is not None:
                    if sos_result != numerical_result:
                        discrepancies.append(
                            f"SOS ({sos_result}) vs Numerical ({numerical_result})"
                        )

                if expected_result is not None:
                    if overall_result != expected_result:
                        discrepancies.append(
                            f"Overall ({overall_result}) vs Expected ({expected_result})"
                        )

                consistency_results.append(
                    VerificationConsistencyResult(
                        certificate=cert_info["certificate"],
                        system=cert_info["name"],
                        sos_result=sos_result,
                        numerical_result=numerical_result,
                        symbolic_result=symbolic_result,
                        overall_result=overall_result,
                        consistency_score=consistency_score,
                        discrepancies=discrepancies,
                        verification_time=verification_time,
                        theoretical_expectation=expected_result,
                    )
                )

                status = (
                    "✅" if consistency_score >= 0.8 else "⚠️" if consistency_score >= 0.5 else "❌"
                )
                logger.info(f"   {status} {cert_info['name']}: Consistency {consistency_score:.1%}")

            except Exception as e:
                logger.error(f"   ❌ {cert_info['name']}: Verification failed - {str(e)}")
                consistency_results.append(
                    VerificationConsistencyResult(
                        certificate=cert_info["certificate"],
                        system=cert_info["name"],
                        sos_result=None,
                        numerical_result=None,
                        symbolic_result=None,
                        overall_result=False,
                        consistency_score=0.0,
                        discrepancies=[f"Verification error: {str(e)}"],
                        verification_time=time.time() - start_time,
                        theoretical_expectation=cert_info["theoretical_result"],
                    )
                )

        return consistency_results

    def analyze_qwen_llm_prompting(self) -> List[LLMPromptAnalysis]:
        """Analyze Qwen LLM prompting strategy and optimization."""
        logger.info("🤖 Analyzing Qwen LLM prompting strategy...")

        test_systems = [
            {
                "name": "simple_system",
                "description": """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "complexity": "simple",
                "domain_bounds": {"x": [-3, 3], "y": [-3, 3]},
            },
            {
                "name": "complex_system",
                "description": """System Dynamics: dx/dt = -x**3 + 0.1*y**2, dy/dt = -y**3 - 0.1*x*y
Initial Set: (x-0.5)**2 + (y+0.3)**2 <= 0.1 and x >= 0
Unsafe Set: x**2 + y**2 >= 4.0 or x <= -2.0
Domain: x ∈ [-3, 3], y ∈ [-3, 3]""",
                "complexity": "complex",
                "domain_bounds": {"x": [-3, 3], "y": [-3, 3]},
            },
        ]

        analyses = []

        for system_info in test_systems:
            try:
                # Analyze prompt generation
                context = "Example context from knowledge base..."  # Mock context
                barrier_type = "unified"

                # Test prompt generation
                prompt = format_prompt_with_context(
                    system_info["description"], context, barrier_type, system_info["domain_bounds"]
                )

                # Analyze Qwen-specific optimizations
                qwen_optimizations = []

                # Check for Qwen-specific prompt patterns
                if "[INST]" in prompt and "[/INST]" in prompt:
                    qwen_optimizations.append("Qwen instruction format")

                if len(prompt) < 2048:  # Qwen 7B optimal context length
                    qwen_optimizations.append("Optimal context length for Qwen 7B")

                if "barrier certificate" in prompt.lower():
                    qwen_optimizations.append("Clear task specification")

                if "BARRIER_CERTIFICATE_START" in prompt:
                    qwen_optimizations.append("Output format markers")

                # Check for mathematical notation clarity
                if any(symbol in prompt for symbol in ["dx/dt", "**", "<=", ">="]):
                    qwen_optimizations.append("Clear mathematical notation")

                # Analyze domain bounds integration
                domain_bounds_included = (
                    system_info["domain_bounds"] is not None and "Domain" in prompt
                )

                analyses.append(
                    LLMPromptAnalysis(
                        model_type="Qwen/Qwen2.5-7B-Instruct",
                        prompt_length=len(prompt),
                        context_chunks=context.count("Context Chunk") if context else 0,
                        template_used="Instruction template with context",
                        domain_bounds_included=domain_bounds_included,
                        system_complexity=system_info["complexity"],
                        expected_output_format="BARRIER_CERTIFICATE_START/END markers",
                        qwen_specific_optimizations=qwen_optimizations,
                    )
                )

                logger.info(
                    f"   📝 {system_info['name']}: {len(qwen_optimizations)} Qwen optimizations detected"
                )

            except Exception as e:
                logger.error(f"   ❌ {system_info['name']}: Prompt analysis failed - {str(e)}")

        return analyses

    def test_llm_parsing_accuracy(self) -> List[ParsingAccuracyResult]:
        """Test accuracy of LLM output parsing into symbolic representation."""
        self._initialize_certificate_generator()

        logger.info("📝 Testing LLM output parsing accuracy...")

        test_outputs = [
            {
                "name": "clean_standard_output",
                "output": """Based on the system dynamics, I propose the following barrier certificate:

BARRIER_CERTIFICATE_START
B(x, y) = 0.5*x**2 + 0.8*y**2 - 0.1
BARRIER_CERTIFICATE_END

This quadratic function ensures safety by...""",
                "expected_certificate": "0.5*x**2 + 0.8*y**2 - 0.1",
                "semantic_correctness": True,
            },
            {
                "name": "latex_artifacts",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = x^2 + y^2 \\]
BARRIER_CERTIFICATE_END""",
                "expected_certificate": "x**2 + y**2",
                "semantic_correctness": True,
            },
            {
                "name": "complex_polynomial",
                "output": """For this nonlinear system, I suggest:

BARRIER_CERTIFICATE_START
B(x, y) = 0.2*x**4 + 0.3*y**4 + 0.1*x**2*y**2 - 0.05
BARRIER_CERTIFICATE_END

The higher-order terms help handle the nonlinearity.""",
                "expected_certificate": "0.2*x**4 + 0.3*y**4 + 0.1*x**2*y**2 - 0.05",
                "semantic_correctness": True,
            },
            {
                "name": "rational_coefficients",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = (1/2)*x**2 + (3/4)*y**2 - 1/10
BARRIER_CERTIFICATE_END""",
                "expected_certificate": "(1/2)*x**2 + (3/4)*y**2 - 1/10",
                "semantic_correctness": True,
            },
            {
                "name": "malformed_output",
                "output": """I think the barrier certificate could be something like x^2 + y^2 but I'm not sure about the exact form...""",
                "expected_certificate": None,
                "semantic_correctness": False,
            },
            {
                "name": "template_expression",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = ax**2 + bxy + cy**2 + dx + ey + f
BARRIER_CERTIFICATE_END""",
                "expected_certificate": None,  # Should be rejected as template
                "semantic_correctness": False,
            },
        ]

        parsing_results = []

        for test in test_outputs:
            try:
                # Test extraction
                extracted = self.certificate_generator.extract_certificate_from_output(
                    test["output"]
                )
                parsing_successful = extracted is not None

                # Test template detection
                template_detected = False
                if extracted:
                    template_detected = self.certificate_generator._is_template_expression(
                        extracted
                    )
                    if template_detected:
                        extracted = None  # Should be rejected
                        parsing_successful = False

                # Calculate confidence score
                confidence_score = 0.0
                if test["expected_certificate"] is None:
                    # Expecting failure
                    confidence_score = 1.0 if not parsing_successful else 0.0
                else:
                    # Expecting success
                    if parsing_successful and extracted:
                        # Check if extracted matches expected (approximately)
                        expected_clean = test["expected_certificate"].replace(" ", "")
                        extracted_clean = extracted.replace(" ", "")
                        if expected_clean == extracted_clean:
                            confidence_score = 1.0
                        else:
                            # Partial match
                            confidence_score = 0.5
                    else:
                        confidence_score = 0.0

                # Identify syntax errors
                syntax_errors = []
                if parsing_successful and extracted:
                    try:
                        # Test symbolic parsing
                        import sympy

                        variables = [sympy.Symbol("x"), sympy.Symbol("y")]
                        local_dict = {var.name: var for var in variables}
                        symbolic_expr = sympy.parse_expr(extracted, local_dict=local_dict)
                        symbolic_representation = str(symbolic_expr)
                    except Exception as e:
                        syntax_errors.append(f"Symbolic parsing error: {str(e)}")
                        symbolic_representation = None
                else:
                    symbolic_representation = None

                parsing_results.append(
                    ParsingAccuracyResult(
                        raw_output=(
                            test["output"][:100] + "..."
                            if len(test["output"]) > 100
                            else test["output"]
                        ),
                        parsed_certificate=extracted,
                        parsing_successful=parsing_successful,
                        symbolic_representation=symbolic_representation,
                        syntax_errors=syntax_errors,
                        semantic_correctness=test["semantic_correctness"],
                        template_detected=template_detected,
                        confidence_score=confidence_score,
                    )
                )

                status = (
                    "✅" if confidence_score >= 0.8 else "⚠️" if confidence_score >= 0.5 else "❌"
                )
                logger.info(f"   {status} {test['name']}: Confidence {confidence_score:.1%}")

            except Exception as e:
                logger.error(f"   ❌ {test['name']}: Parsing test failed - {str(e)}")
                parsing_results.append(
                    ParsingAccuracyResult(
                        raw_output=test["output"][:100] + "...",
                        parsed_certificate=None,
                        parsing_successful=False,
                        symbolic_representation=None,
                        syntax_errors=[f"Test error: {str(e)}"],
                        semantic_correctness=False,
                        template_detected=False,
                        confidence_score=0.0,
                    )
                )

        return parsing_results

    def test_cross_component_consistency(self) -> Dict[str, Any]:
        """Test consistency across all web interface components."""
        logger.info("🔗 Testing cross-component consistency...")

        test_certificate = "x**2 + y**2"
        test_system = """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0"""

        consistency_checks = {}

        try:
            # Test 1: System parsing consistency
            parsed_system1 = self.verification_service.parse_system_description(test_system)
            parsed_system2 = self.verification_service.parse_system_description(test_system)

            parsing_consistent = parsed_system1.get("variables") == parsed_system2.get(
                "variables"
            ) and parsed_system1.get("dynamics") == parsed_system2.get("dynamics")
            consistency_checks["system_parsing"] = parsing_consistent

            # Test 2: Bounds generation consistency
            bounds1 = self.verification_service.create_sampling_bounds(parsed_system1)
            bounds2 = self.verification_service.create_sampling_bounds(parsed_system2)

            bounds_consistent = bounds1 == bounds2
            consistency_checks["bounds_generation"] = bounds_consistent

            # Test 3: Certificate cleaning consistency
            test_cert_dirty = "x**2 + y**2 \\]"
            cleaned1 = self.verification_service._clean_certificate_string(test_cert_dirty)
            cleaned2 = self.verification_service._clean_certificate_string(test_cert_dirty)

            cleaning_consistent = cleaned1 == cleaned2
            consistency_checks["certificate_cleaning"] = cleaning_consistent

            # Test 4: Verification determinism (run same verification twice)
            verification1 = self.verification_service.verify_certificate(
                test_certificate,
                test_system,
                param_overrides={"num_samples_lie": 100, "num_samples_boundary": 50},
            )

            verification2 = self.verification_service.verify_certificate(
                test_certificate,
                test_system,
                param_overrides={"num_samples_lie": 100, "num_samples_boundary": 50},
            )

            # Check if main results are consistent (allowing for numerical sampling variation)
            verification_consistent = verification1.get("overall_success") == verification2.get(
                "overall_success"
            ) and verification1.get("sos_passed") == verification2.get("sos_passed")
            consistency_checks["verification_determinism"] = verification_consistent

            # Test 5: Certificate generator parsing consistency
            self._initialize_certificate_generator()
            test_output = """BARRIER_CERTIFICATE_START
B(x, y) = x**2 + y**2
BARRIER_CERTIFICATE_END"""

            extracted1 = self.certificate_generator.extract_certificate_from_output(test_output)
            extracted2 = self.certificate_generator.extract_certificate_from_output(test_output)

            extraction_consistent = extracted1 == extracted2
            consistency_checks["certificate_extraction"] = extraction_consistent

        except Exception as e:
            logger.error(f"Cross-component consistency test failed: {str(e)}")
            consistency_checks["error"] = str(e)

        return consistency_checks

    def run_comprehensive_interrogation(self) -> Dict[str, Any]:
        """Run complete comprehensive interrogation of web interface."""
        print("🔬 COMPREHENSIVE WEB INTERFACE INTERROGATION")
        print("=" * 70)

        start_time = time.time()

        # Configure logging for clear output
        logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

        # 1. Verification Consistency Analysis
        print("\n🔍 VERIFICATION METHOD CONSISTENCY ANALYSIS")
        verification_consistency = self.test_verification_consistency()
        self.results["verification_consistency"] = verification_consistency

        # 2. Qwen LLM Prompting Analysis
        print("\n🤖 QWEN LLM PROMPTING STRATEGY ANALYSIS")
        llm_prompt_analysis = self.analyze_qwen_llm_prompting()
        self.results["llm_prompt_analysis"] = llm_prompt_analysis

        # 3. LLM Parsing Accuracy
        print("\n📝 LLM OUTPUT PARSING ACCURACY ANALYSIS")
        parsing_accuracy = self.test_llm_parsing_accuracy()
        self.results["parsing_accuracy"] = parsing_accuracy

        # 4. Cross-Component Consistency
        print("\n🔗 CROSS-COMPONENT CONSISTENCY ANALYSIS")
        cross_component = self.test_cross_component_consistency()
        self.results["cross_component_consistency"] = cross_component

        total_time = time.time() - start_time

        # Generate comprehensive analysis report
        return self._generate_interrogation_report(total_time)

    def _generate_interrogation_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive interrogation report."""
        report = {
            "execution_time": total_time,
            "summary": {},
            "detailed_analysis": {},
            "recommendations": [],
            "overall_assessment": {},
        }

        # Analyze verification consistency
        verification_results = self.results["verification_consistency"]
        if verification_results:
            consistency_scores = [r.consistency_score for r in verification_results]
            avg_consistency = np.mean(consistency_scores)
            report["summary"]["verification_consistency"] = {
                "average_consistency": avg_consistency,
                "tests_run": len(verification_results),
                "high_consistency_count": sum(1 for s in consistency_scores if s >= 0.8),
            }

        # Analyze LLM prompting
        llm_analysis = self.results["llm_prompt_analysis"]
        if llm_analysis:
            total_optimizations = sum(len(a.qwen_specific_optimizations) for a in llm_analysis)
            avg_optimizations = total_optimizations / len(llm_analysis) if llm_analysis else 0
            report["summary"]["llm_prompting"] = {
                "average_qwen_optimizations": avg_optimizations,
                "prompt_analyses_run": len(llm_analysis),
                "domain_bounds_integration": sum(
                    1 for a in llm_analysis if a.domain_bounds_included
                ),
            }

        # Analyze parsing accuracy
        parsing_results = self.results["parsing_accuracy"]
        if parsing_results:
            confidence_scores = [r.confidence_score for r in parsing_results]
            avg_confidence = np.mean(confidence_scores)
            report["summary"]["parsing_accuracy"] = {
                "average_confidence": avg_confidence,
                "tests_run": len(parsing_results),
                "high_confidence_count": sum(1 for s in confidence_scores if s >= 0.8),
            }

        # Analyze cross-component consistency
        cross_component = self.results["cross_component_consistency"]
        if cross_component:
            consistent_components = sum(1 for v in cross_component.values() if v is True)
            total_components = len([v for v in cross_component.values() if isinstance(v, bool)])
            report["summary"]["cross_component_consistency"] = {
                "consistent_components": consistent_components,
                "total_components": total_components,
                "consistency_rate": (
                    consistent_components / total_components if total_components > 0 else 0
                ),
            }

        # Generate overall assessment
        scores = []
        if "verification_consistency" in report["summary"]:
            scores.append(report["summary"]["verification_consistency"]["average_consistency"])
        if "parsing_accuracy" in report["summary"]:
            scores.append(report["summary"]["parsing_accuracy"]["average_confidence"])
        if "cross_component_consistency" in report["summary"]:
            scores.append(report["summary"]["cross_component_consistency"]["consistency_rate"])

        overall_score = np.mean(scores) if scores else 0.0

        if overall_score >= 0.9:
            assessment = "EXCELLENT"
            status = "🏆"
        elif overall_score >= 0.8:
            assessment = "VERY_GOOD"
            status = "✅"
        elif overall_score >= 0.7:
            assessment = "GOOD"
            status = "⚡"
        elif overall_score >= 0.6:
            assessment = "ADEQUATE"
            status = "⚠️"
        else:
            assessment = "NEEDS_IMPROVEMENT"
            status = "🔧"

        report["overall_assessment"] = {
            "score": overall_score,
            "assessment": assessment,
            "status": status,
        }

        # Store detailed results
        report["detailed_analysis"] = {
            "verification_consistency": [asdict(r) for r in verification_results],
            "llm_prompt_analysis": [asdict(a) for a in llm_analysis],
            "parsing_accuracy": [asdict(r) for r in parsing_results],
            "cross_component_consistency": cross_component,
        }

        return report


def main():
    """Run comprehensive web interface interrogation."""
    try:
        interrogator = ComprehensiveWebInterfaceInterrogator()
        results = interrogator.run_comprehensive_interrogation()

        # Display summary results
        print("\n" + "=" * 70)
        print("🏆 COMPREHENSIVE INTERROGATION SUMMARY")
        print("=" * 70)

        overall = results["overall_assessment"]
        print(f"\n{overall['status']} OVERALL ASSESSMENT: {overall['assessment']}")
        print(f"📊 Overall Score: {overall['score']:.1%}")

        summary = results["summary"]

        if "verification_consistency" in summary:
            vc = summary["verification_consistency"]
            print(f"\n🔍 Verification Consistency: {vc['average_consistency']:.1%}")
            print(f"   High consistency: {vc['high_consistency_count']}/{vc['tests_run']} tests")

        if "llm_prompting" in summary:
            lp = summary["llm_prompting"]
            print(f"\n🤖 LLM Prompting: {lp['average_qwen_optimizations']:.1f} avg optimizations")
            print(
                f"   Domain bounds integration: {lp['domain_bounds_integration']}/{lp['prompt_analyses_run']} prompts"
            )

        if "parsing_accuracy" in summary:
            pa = summary["parsing_accuracy"]
            print(f"\n📝 Parsing Accuracy: {pa['average_confidence']:.1%}")
            print(f"   High confidence: {pa['high_confidence_count']}/{pa['tests_run']} tests")

        if "cross_component_consistency" in summary:
            cc = summary["cross_component_consistency"]
            print(f"\n🔗 Cross-Component: {cc['consistency_rate']:.1%}")
            print(
                f"   Consistent components: {cc['consistent_components']}/{cc['total_components']}"
            )

        print(f"\n⏱️ Total execution time: {results['execution_time']:.2f}s")

        # Save detailed results
        with open("comprehensive_interrogation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Detailed results saved to comprehensive_interrogation_results.json")

        return 0 if overall["score"] >= 0.8 else 1

    except Exception as e:
        print(f"❌ Comprehensive interrogation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
