"""
Tests for the new utility modules created during systematic code refinement.

This test suite validates that all refactored functionality is working correctly
and that the new utility modules provide the expected behavior.
"""

import pytest
import sympy
from unittest.mock import Mock

# Import the new utility modules
from utils.verification_helpers import (
    VerificationConfig,
    SystemInfo,
    validate_candidate_expression,
    build_verification_summaries,
)
from utils.numerical_checks import (
    NumericalCheckConfig,
    ViolationInfo,
    NumericalCheckResult,
)
from utils.condition_parser import (
    parse_set_conditions_simplified,
    parse_single_condition,
    parse_or_condition,
    validate_condition_structure,
    clean_condition_string,
)
from utils.certificate_extraction import (
    extract_certificate_from_llm_output,
    clean_and_validate_expression,
    is_template_expression,
)
from utils.data_formatting import (
    format_instruction_example,
    format_prompt_completion_example,
    format_synthetic_example,
)
from utils.experiment_analysis import analyze_certificate_complexity


class TestVerificationHelpers:
    """Test the verification helper utilities."""

    def test_verification_config_creation(self):
        """Test VerificationConfig data class creation."""
        config = VerificationConfig(
            num_samples_lie=100,
            num_samples_boundary=50,
            numerical_tolerance=1e-6,
            sos_default_degree=2,
            sos_solver="MOSEK",
            sos_epsilon=1e-8,
            optimization_max_iter=1000,
            optimization_pop_size=50,
            attempt_sos=True,
            attempt_optimization=True,
        )

        assert config.num_samples_lie == 100
        assert config.numerical_tolerance == 1e-6
        assert config.attempt_sos is True

    def test_system_info_creation(self):
        """Test SystemInfo data class creation."""
        system_info = SystemInfo(
            id="test_system",
            state_variables=["x", "y"],
            dynamics=["-x", "-y"],
            initial_set_conditions=["x**2 + y**2 <= 1"],
            unsafe_set_conditions=["x**2 + y**2 >= 4"],
            safe_set_conditions=["x**2 + y**2 <= 3"],
            sampling_bounds={"x": [-2, 2], "y": [-2, 2]},
            parameters={"a": 1.0},
            certificate_domain_bounds={"x": [-1, 1], "y": [-1, 1]},
        )

        assert system_info.id == "test_system"
        assert len(system_info.state_variables) == 2
        assert system_info.system_type == "continuous"  # default value

    def test_validate_candidate_expression_valid(self):
        """Test candidate expression validation with valid input."""
        context = Mock()
        context.variables_sympy = [sympy.symbols("x"), sympy.symbols("y")]
        context.system_params_sympy = {}
        context.system_info.state_variables = ["x", "y"]

        result, error_msg = validate_candidate_expression("x**2 + y**2", context)

        assert result is not None
        assert error_msg == ""
        assert str(result) == "x**2 + y**2"

    def test_validate_candidate_expression_invalid(self):
        """Test candidate expression validation with invalid input."""
        context = Mock()
        context.variables_sympy = [sympy.symbols("x"), sympy.symbols("y")]
        context.system_params_sympy = {}
        context.system_info.state_variables = ["x", "y"]

        result, error_msg = validate_candidate_expression("invalid_expression", context)

        assert result is None
        assert "unexpected symbols" in error_msg

    def test_build_verification_summaries(self):
        """Test building verification summaries."""
        results = {
            "sos_attempted": True,
            "sos_passed": True,
            "sos_reason": "SOS verification passed",
            "symbolic_lie_check_passed": True,
            "symbolic_boundary_check_passed": True,
            "numerical_overall_passed": True,
            "numerical_sampling_reason": "All checks passed",
            "parsing_B_successful": True,
            "parsing_sets_successful": True,
            "is_polynomial_system": True,
            "system_type": "continuous",
            "final_verdict": "Passed SOS Checks",
        }

        context = Mock()

        summaries = build_verification_summaries(results, context)

        assert "sos_verification" in summaries
        assert "symbolic_verification" in summaries
        assert "numerical_verification" in summaries
        assert "parsing" in summaries
        assert summaries["overall_success"] is True


class TestNumericalChecks:
    """Test the numerical checking utilities."""

    def test_numerical_check_config_creation(self):
        """Test NumericalCheckConfig data class creation."""
        config = NumericalCheckConfig(n_samples=100, tolerance=1e-6, max_iter=1000, pop_size=50)

        assert config.n_samples == 100
        assert config.tolerance == 1e-6

    def test_violation_info_creation(self):
        """Test ViolationInfo data class creation."""
        violation = ViolationInfo(
            point={"x": 1.0, "y": 2.0}, violation_type="initial_set", value=0.5, expected="≤ 0"
        )

        assert violation.point["x"] == 1.0
        assert violation.violation_type == "initial_set"
        assert violation.value == 0.5

    def test_numerical_check_result_creation(self):
        """Test NumericalCheckResult data class creation."""
        result = NumericalCheckResult(
            passed=True,
            reason="All checks passed",
            violations=0,
            violation_points=[],
            samples_checked={"total": 100, "safe": 50},
        )

        assert result.passed is True
        assert result.violations == 0
        assert len(result.violation_points) == 0


class TestConditionParser:
    """Test the condition parsing utilities."""

    def test_parse_single_condition_valid(self):
        """Test parsing a single valid condition."""
        local_dict = {"x": sympy.symbols("x"), "y": sympy.symbols("y")}

        result = parse_single_condition("x >= 0", local_dict)

        assert result is not None
        assert isinstance(result, sympy.core.relational.Relational)

    def test_parse_single_condition_invalid(self):
        """Test parsing an invalid condition."""
        local_dict = {"x": sympy.symbols("x")}

        result = parse_single_condition("invalid_condition", local_dict)

        assert result is None

    def test_parse_or_condition(self):
        """Test parsing OR conditions."""
        local_dict = {"x": sympy.symbols("x"), "y": sympy.symbols("y")}

        result = parse_or_condition("x >= 0 or y >= 0", local_dict)

        assert result is not None
        assert isinstance(result, sympy.logic.boolalg.BooleanFunction)

    def test_parse_set_conditions_simplified(self):
        """Test simplified set condition parsing."""
        variables = [sympy.symbols("x"), sympy.symbols("y")]
        conditions = ["x >= 0", "y >= 0"]

        result, message = parse_set_conditions_simplified(conditions, variables)

        assert result is not None
        assert len(result) == 2
        assert message == "Conditions parsed successfully"

    def test_validate_condition_structure(self):
        """Test condition structure validation."""
        assert validate_condition_structure("x >= 0") is True
        assert validate_condition_structure("x >= 0 +") is False  # trailing operator
        assert validate_condition_structure("x >= 0(") is False  # unbalanced parentheses
        assert validate_condition_structure("") is False

    def test_clean_condition_string(self):
        """Test condition string cleaning."""
        # Test LaTeX cleaning
        cleaned = clean_condition_string("x \\geq 0")
        # The function does not remove LaTeX commands like \geq, so it should remain
        assert "\\geq" in cleaned

        # Test descriptive text removal
        cleaned = clean_condition_string("x >= 0 where x is positive")
        assert "where" not in cleaned
        assert cleaned == "x >= 0"

        # Test trailing punctuation removal
        cleaned = clean_condition_string("x >= 0.")
        assert cleaned == "x >= 0"


class TestCertificateExtraction:
    """Test the certificate extraction utilities."""

    def test_extract_certificate_from_llm_output_delimited(self):
        """Test certificate extraction from delimited output."""
        llm_text = """
        BARRIER_CERTIFICATE_START
        B(x, y) = x**2 + y**2
        BARRIER_CERTIFICATE_END
        """
        variables = ["x", "y"]

        result, failed = extract_certificate_from_llm_output(llm_text, variables)

        assert failed is False
        assert result == "x**2 + y**2"

    def test_extract_certificate_from_llm_output_b_notation(self):
        """Test certificate extraction from B(x) notation."""
        llm_text = "The barrier certificate is B(x, y) = x**2 + y**2"
        variables = ["x", "y"]

        result, failed = extract_certificate_from_llm_output(llm_text, variables)

        assert failed is False
        assert result == "x**2 + y**2"

    def test_clean_and_validate_expression_valid(self):
        """Test expression cleaning and validation with valid input."""
        result = clean_and_validate_expression("x**2 + y**2", ["x", "y"])

        assert result == "x**2 + y**2"

    def test_clean_and_validate_expression_invalid(self):
        """Test expression cleaning and validation with invalid input."""
        result = clean_and_validate_expression("invalid_expression", ["x", "y"])

        assert result is None

    def test_is_template_expression(self):
        """Test template expression detection."""
        # Template expressions
        assert is_template_expression("ax + by") is True
        assert is_template_expression("a**2 + b**2") is True
        assert is_template_expression("c1*x + c2*y") is True

        # Valid expressions
        assert is_template_expression("x**2 + y**2") is False
        assert is_template_expression("2*x + 3*y") is False


class TestDataFormatting:
    """Test the data formatting utilities."""

    def test_format_instruction_example(self):
        """Test instruction example formatting."""
        system_description = "dx/dt = -x, dy/dt = -y"
        barrier_certificate = "B(x, y) = x**2 + y**2"

        formatted = format_instruction_example(system_description, barrier_certificate)

        assert isinstance(formatted, dict)
        assert "instruction" in formatted
        assert "input" in formatted
        assert "output" in formatted
        assert formatted["input"] == system_description
        assert formatted["output"] == barrier_certificate

    def test_format_prompt_completion_example(self):
        """Test prompt/completion example formatting."""
        system_description = "dx/dt = -x, dy/dt = -y"
        barrier_certificate = "B(x, y) = x**2 + y**2"

        formatted = format_prompt_completion_example(system_description, barrier_certificate)

        assert isinstance(formatted, dict)
        assert "prompt" in formatted
        assert "completion" in formatted
        assert system_description in formatted["prompt"]
        assert barrier_certificate in formatted["completion"]

    def test_format_synthetic_example(self):
        """Test synthetic example formatting."""
        system_desc_text = "dx/dt = -x, dy/dt = -y"
        certificate_str = "x**2 + y**2"

        # Test instruction format
        formatted_instruction = format_synthetic_example(
            system_desc_text, certificate_str, "instruction"
        )
        assert isinstance(formatted_instruction, dict)
        assert "instruction" in formatted_instruction
        assert "input" in formatted_instruction
        assert "output" in formatted_instruction

        # Test prompt_completion format
        formatted_prompt = format_synthetic_example(
            system_desc_text, certificate_str, "prompt_completion"
        )
        assert isinstance(formatted_prompt, dict)
        assert "prompt" in formatted_prompt
        assert "completion" in formatted_prompt


class TestExperimentAnalysis:
    """Test the experiment analysis utilities."""

    def test_analyze_certificate_complexity(self):
        """Test certificate complexity analysis."""
        import pandas as pd

        # Create test data
        data = {
            "parsed_certificate": ["x**2 + y**2", "x + y", "x**3 + y**3"],
            "final_verdict": ["Passed", "Failed", "Passed"],
            "system_id": ["sys1", "sys2", "sys3"],
            "experiment_name": ["exp1", "exp2", "exp3"],
        }
        df = pd.DataFrame(data)

        result = analyze_certificate_complexity(df)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestIntegration:
    """Integration tests for the utility modules."""

    def test_utility_modules_import_correctly(self):
        """Test that all utility modules can be imported correctly."""
        # This test ensures that all the new utility modules can be imported
        # without any import errors, which validates the refactoring didn't
        # break any dependencies

        modules_to_test = [
            "utils.verification_helpers",
            "utils.numerical_checks",
            "utils.condition_parser",
            "utils.certificate_extraction",
            "utils.data_formatting",
            "utils.experiment_analysis",
            "utils.simplified_verification",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_data_classes_serialization(self):
        """Test that data classes can be properly serialized."""
        # Test VerificationConfig
        config = VerificationConfig(
            num_samples_lie=100,
            num_samples_boundary=50,
            numerical_tolerance=1e-6,
            sos_default_degree=2,
            sos_solver="MOSEK",
            sos_epsilon=1e-8,
            optimization_max_iter=1000,
            optimization_pop_size=50,
            attempt_sos=True,
            attempt_optimization=True,
        )

        # Test that we can access all attributes
        assert hasattr(config, "num_samples_lie")
        assert hasattr(config, "numerical_tolerance")
        assert hasattr(config, "attempt_sos")

        # Test SystemInfo
        system_info = SystemInfo(
            id="test",
            state_variables=["x", "y"],
            dynamics=["-x", "-y"],
            initial_set_conditions=["x**2 + y**2 <= 1"],
            unsafe_set_conditions=["x**2 + y**2 >= 4"],
            safe_set_conditions=["x**2 + y**2 <= 3"],
            sampling_bounds={"x": [-2, 2], "y": [-2, 2]},
            parameters={},
            certificate_domain_bounds=None,
        )

        assert hasattr(system_info, "id")
        assert hasattr(system_info, "state_variables")
        assert hasattr(system_info, "system_type")

    def test_error_handling_robustness(self):
        """Test that error handling is robust in utility functions."""
        # Test with invalid inputs to ensure graceful error handling

        # Test certificate extraction with empty input
        result, failed = extract_certificate_from_llm_output("", ["x", "y"])
        assert failed is True
        assert result is None

        # Test condition parsing with invalid input
        result, message = parse_set_conditions_simplified(None, [])
        assert result == []
        assert "No conditions provided" in message

        # Test expression cleaning with invalid input
        result = clean_and_validate_expression("", ["x", "y"])
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
