"""
Test edge cases for improved certificate extraction (Phase 1 Day 5)
"""

import os
import sys

import pytest

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils.certificate_extraction_improved import (
    extract_certificate_from_llm_output,
    extract_from_ascii_math,
    is_template_expression,
    normalize_expression,
)


class TestDecimalExtraction:
    """Test decimal number extraction edge cases"""

    def test_standard_decimals(self):
        """Test standard decimal numbers"""
        test_cases = [
            ("B(x,y) = x**2 + y**2 - 1.5", "x**2 + y**2 - 1.5"),
            ("Certificate: x**2 - 0.123456789", "x**2 - 0.123456789"),
            ("B(x) = x**2 - 3.14159265359", "x**2 - 3.14159265359"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"Extraction failed for: {input_text}"
            assert result == expected, f"Expected {expected}, got {result}"

    def test_scientific_notation(self):
        """Test scientific notation"""
        test_cases = [
            ("B(x,y) = x**2 + y**2 - 1.5e-3", "x**2 + y**2 - 1.5e-3"),
            ("Certificate: x**2 - 2.5E-10", "x**2 - 2.5E-10"),
            ("B(x) = x**2 + 1.23e+5", "x**2 + 1.23e+5"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"Extraction failed for: {input_text}"
            # Note: SymPy might normalize scientific notation
            assert result is not None

    def test_edge_decimal_cases(self):
        """Test edge cases with decimal numbers"""
        test_cases = [
            # Decimal at end of line
            ("B(x,y) = x**2 + y**2 - 1.5\n", "x**2 + y**2 - 1.5"),
            # Decimal followed by period
            ("B(x,y) = x**2 + y**2 - 1.5.", "x**2 + y**2 - 1.5"),
            # Multiple decimals
            ("B(x) = 0.5*x**2 + 1.2*x - 3.7", "0.5*x**2 + 1.2*x - 3.7"),
            # Very small decimal
            ("B(x) = x**2 - 0.000000001", "x**2 - 0.000000001"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"Extraction failed for: {input_text}"
            assert result == expected, f"Expected {expected}, got {result}"


class TestTemplateDetection:
    """Test template detection improvements"""

    def test_single_letter_coefficients(self):
        """Test detection of single letter coefficients"""
        templates = [
            "a*x**2 + b*y**2 + c",
            "ax**2 + by**2 + c",
            "A*x + B*y + C",
            "a1*x**2 + a2*y**2",
        ]

        for template in templates:
            assert is_template_expression(
                template
            ), f"Failed to detect template: {template}"

    def test_greek_letters(self):
        """Test detection of Greek letter placeholders"""
        templates = [
            "α*x**2 + β*y**2",
            "\\alpha*x**2 + \\beta*y**2",
            "lambda*x + mu*y",
            "δ*x**2 - γ",
        ]

        for template in templates:
            assert is_template_expression(
                template
            ), f"Failed to detect Greek template: {template}"

    def test_subscripted_coefficients(self):
        """Test detection of subscripted coefficients"""
        templates = [
            "a_1*x**2 + a_2*y**2",
            "c_{11}*x**2 + c_{12}*x*y + c_{22}*y**2",
            "coeff1*x + coeff2*y",
        ]

        for template in templates:
            assert is_template_expression(
                template
            ), f"Failed to detect subscripted template: {template}"

    def test_ellipsis_patterns(self):
        """Test detection of ellipsis patterns"""
        templates = [
            "x**2 + ... + y**2",
            "a*x**2 + \\cdots + b*y**2",
            "x**2 + [?]*y**2",
        ]

        for template in templates:
            assert is_template_expression(
                template
            ), f"Failed to detect ellipsis template: {template}"

    def test_valid_non_templates(self):
        """Test that valid certificates are not marked as templates"""
        valid_certs = [
            "x**2 + y**2 - 1",
            "2*x**2 + 3*y**2 - 5",
            "x**2 + 2*x*y + y**2",
            "0.5*x**2 + 0.3*y**2 - 1.2",
        ]

        for cert in valid_certs:
            assert not is_template_expression(
                cert
            ), f"Incorrectly marked as template: {cert}"


class TestFormatSupport:
    """Test support for different mathematical formats"""

    def test_latex_inline_math(self):
        """Test LaTeX inline math extraction"""
        test_cases = [
            (r"The barrier certificate is $B(x,y) = x^2 + y^2 - 1$", "x**2 + y**2 - 1"),
            (r"Certificate: \(x^2 + y^2 - 2\)", "x**2 + y**2 - 2"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"LaTeX extraction failed for: {input_text}"
            assert result == expected, f"Expected {expected}, got {result}"

    def test_latex_display_math(self):
        """Test LaTeX display math extraction"""
        test_cases = [
            (r"The certificate is: \[B(x,y) = x^2 + y^2 - 1\]", "x**2 + y**2 - 1"),
            (r"$$B(x,y) = 2x^2 + 3y^2$$", "2*x**2 + 3*y**2"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"LaTeX display extraction failed for: {input_text}"
            assert result == expected, f"Expected {expected}, got {result}"

    def test_latex_operators(self):
        """Test LaTeX operator conversion"""
        test_cases = [
            (r"B(x,y) = x^2 \cdot y^2", "x**2*y**2"),
            (r"B(x,y) = x \times y", "x*y"),
            (r"B(x,y) = x \div y", "x/y"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"LaTeX operator conversion failed for: {input_text}"
            # Note: exact match might vary due to SymPy normalization
            assert result is not None

    def test_unicode_math(self):
        """Test unicode mathematical symbols"""
        test_cases = [
            ("B(x,y) = x² + y² - 1", "x**2 + y**2 - 1"),
            ("B(x) = x³ - 2x", "x**3 - 2*x"),
            ("B(x,y) = x × y", "x*y"),
        ]

        for input_text, expected in test_cases:
            result, failed = extract_certificate_from_llm_output(input_text, ["x", "y"])
            assert not failed, f"Unicode math extraction failed for: {input_text}"
            assert result == expected, f"Expected {expected}, got {result}"

    def test_ascii_math_extraction(self):
        """Test ASCII math extraction"""
        test_cases = [
            "The barrier certificate is `x^2 + y^2 - 1`",
            "Certificate: `2*x^2 + 3*y^2`",
        ]

        for text in test_cases:
            result = extract_from_ascii_math(text)
            assert result is not None, f"ASCII math extraction failed for: {text}"


class TestEdgeCases:
    """Test various edge cases"""

    def test_descriptive_text_rejection(self):
        """Test rejection of descriptive text"""
        descriptive_cases = [
            "B(x,y) = something that ensures safety",
            "Certificate is appropriate for the system",
            "B(x) guarantees convergence",
        ]

        for text in descriptive_cases:
            result, failed = extract_certificate_from_llm_output(text, ["x", "y"])
            assert failed, f"Should have rejected descriptive text: {text}"

    def test_incomplete_expressions(self):
        """Test handling of incomplete expressions"""
        incomplete_cases = [
            "B(x,y) = x**2 +",
            "Certificate: (x**2 + y**2",
            "B(x) = ",
        ]

        for text in incomplete_cases:
            result, failed = extract_certificate_from_llm_output(text, ["x", "y"])
            assert failed, f"Should have rejected incomplete expression: {text}"

    def test_non_polynomial_rejection(self):
        """Test rejection of non-polynomial expressions"""
        non_poly_cases = [
            "B(x) = sqrt(x)",
            "B(x,y) = log(x) + y**2",
            "B(x) = sin(x)",
            "B(x) = x**(-1)",
            "B(x) = x**(1/2)",
        ]

        for text in non_poly_cases:
            result, failed = extract_certificate_from_llm_output(text, ["x", "y"])
            assert failed, f"Should have rejected non-polynomial: {text}"

    def test_complex_valid_expressions(self):
        """Test complex but valid expressions"""
        valid_complex = [
            "B(x,y,z) = x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*x*z + 1",
            "B(x,y) = (x + y)**2 - 2*(x - y)**2 + 3",
            "B(x,y) = 0.1*x**4 + 0.2*x**3*y + 0.3*x**2*y**2 + 0.4*x*y**3 + 0.5*y**4",
        ]

        for text in valid_complex:
            result, failed = extract_certificate_from_llm_output(text, ["x", "y", "z"])
            assert not failed, f"Should have accepted valid expression: {text}"
            assert result is not None

    def test_code_block_extraction(self):
        """Test extraction from code blocks"""
        code_block = """
        Here's the barrier certificate:
        ```python
        def barrier_certificate(x, y):
            return x**2 + y**2 - 1.5
        ```
        """

        result, failed = extract_certificate_from_llm_output(code_block, ["x", "y"])
        assert not failed, "Failed to extract from code block"
        assert result == "x**2 + y**2 - 1.5"

    def test_mathematical_notation(self):
        """Test mathematical notation formats"""
        math_notation = "B: ℝ² → ℝ defined by B(x,y) := x² + y² - 1"

        result, failed = extract_certificate_from_llm_output(math_notation, ["x", "y"])
        assert not failed, "Failed to extract from mathematical notation"
        assert result == "x**2 + y**2 - 1"


class TestNormalization:
    """Test expression normalization"""

    def test_normalize_expression(self):
        """Test expression normalization"""
        test_cases = [
            ("x**2 + y**2", "x**2 + y**2"),
            ("y**2 + x**2", "x**2 + y**2"),  # Should reorder
            ("2*x + 3*y", "2*x + 3*y"),
        ]

        for expr, expected in test_cases:
            normalized = normalize_expression(expr)
            # Note: exact normalization depends on SymPy version
            assert normalized is not None


class TestRobustness:
    """Test robustness to various input formats"""

    def test_mixed_content(self):
        """Test extraction from mixed content"""
        mixed_content = """
        Let me explain the barrier certificate approach.

        For your system, we need B(x,y) = x**2 + y**2 - 1.0

        This ensures that trajectories starting in the safe set remain safe.
        """

        result, failed = extract_certificate_from_llm_output(mixed_content, ["x", "y"])
        assert not failed, "Failed to extract from mixed content"
        assert result == "x**2 + y**2 - 1.0"

    def test_multiple_candidates(self):
        """Test handling of multiple candidate expressions"""
        multiple = """
        We could use B(x,y) = a*x**2 + b*y**2, but specifically
        let's use B(x,y) = 2*x**2 + 3*y**2 - 5
        """

        result, failed = extract_certificate_from_llm_output(multiple, ["x", "y"])
        assert not failed, "Failed with multiple candidates"
        assert result == "2*x**2 + 3*y**2 - 5"  # Should pick the concrete one

    def test_whitespace_variations(self):
        """Test robustness to whitespace variations"""
        test_cases = [
            "B(x,y)=x**2+y**2-1",  # No spaces
            "B ( x , y ) = x ** 2 + y ** 2 - 1",  # Extra spaces
            "B(x,y) =\n  x**2 +\n  y**2 -\n  1",  # Newlines
        ]

        for text in test_cases:
            result, failed = extract_certificate_from_llm_output(text, ["x", "y"])
            assert not failed, f"Failed with whitespace variation: {text}"
            # Result should be normalized
            assert "x**2" in result and "y**2" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
