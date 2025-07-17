"""
Parser service for FM-LLM Solver.

Handles parsing of system descriptions and barrier certificates.
"""

import re
from typing import List, Dict
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from fm_llm_solver.core.interfaces import Parser
from fm_llm_solver.core.types import SystemDescription, BarrierCertificate, SystemType, DomainBounds
from fm_llm_solver.core.exceptions import ValidationError
from fm_llm_solver.core.logging import get_logger


class SystemParser(Parser):
    """Parses system descriptions and certificates."""

    def __init__(self):
        """Initialize the parser."""
        self.logger = get_logger(__name__)

        # Common patterns
        self.var_pattern = re.compile(r"\b[a-zA-Z_]\w*\b")
        self.cert_pattern = re.compile(
            r"B\s*\([^)]+\)\s*=\s*(.+)|"
            r"B\s*=\s*(.+)|"
            r"(?:barrier\s+)?certificate\s*[:=]\s*(.+)",
            re.IGNORECASE,
        )

    def parse_system(self, text: str) -> SystemDescription:
        """
        Parse a system description from text.

        Args:
            text: Text description of the system

        Returns:
            Parsed SystemDescription

        Raises:
            ValidationError: If parsing fails
        """
        try:
            lines = text.strip().split("\n")

            dynamics = {}
            initial_set = None
            unsafe_set = None
            system_type = SystemType.CONTINUOUS
            domain_bounds = {}

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse dynamics
                if "d" in line and "/dt" in line:
                    # Continuous dynamics: dx/dt = ...
                    match = re.match(r"d(\w+)/dt\s*=\s*(.+)", line)
                    if match:
                        var, expr = match.groups()
                        dynamics[var] = expr.strip()
                        system_type = SystemType.CONTINUOUS

                elif "[k+1]" in line:
                    # Discrete dynamics: x[k+1] = ...
                    match = re.match(r"(\w+)\[k\+1\]\s*=\s*(.+)", line)
                    if match:
                        var, expr = match.groups()
                        dynamics[var] = expr.strip()
                        system_type = SystemType.DISCRETE

                # Parse sets
                elif "initial" in line.lower():
                    match = re.search(r":\s*(.+)", line)
                    if match:
                        initial_set = match.group(1).strip()

                elif "unsafe" in line.lower():
                    match = re.search(r":\s*(.+)", line)
                    if match:
                        unsafe_set = match.group(1).strip()

                # Parse domain bounds
                elif "∈" in line or "in [" in line:
                    match = re.match(r"(\w+)\s*[∈in]\s*\[([^,]+),\s*([^\]]+)\]", line)
                    if match:
                        var, low, high = match.groups()
                        domain_bounds[var] = (float(low), float(high))

            # Validate
            if not dynamics:
                raise ValidationError("No dynamics found in system description")
            if not initial_set:
                raise ValidationError("No initial set found in system description")
            if not unsafe_set:
                raise ValidationError("No unsafe set found in system description")

            # Create domain bounds if specified
            bounds = DomainBounds(domain_bounds) if domain_bounds else None

            return SystemDescription(
                dynamics=dynamics,
                initial_set=initial_set,
                unsafe_set=unsafe_set,
                system_type=system_type,
                domain_bounds=bounds,
            )

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse system: {e}")
            raise ValidationError(f"Failed to parse system description: {e}")

    def parse_certificate(self, text: str, variables: List[str]) -> BarrierCertificate:
        """
        Parse a barrier certificate from text.

        Args:
            text: Text containing the certificate
            variables: Expected variable names

        Returns:
            Parsed BarrierCertificate

        Raises:
            ValidationError: If parsing fails
        """
        try:
            # Extract certificate expression
            expression = None

            # Try different patterns
            match = self.cert_pattern.search(text)
            if match:
                # Get the first non-None group
                expression = next(g for g in match.groups() if g)

            if not expression:
                # Try to find a mathematical expression
                # Look for lines that contain mathematical operators
                for line in text.split("\n"):
                    if any(op in line for op in ["+", "-", "*", "/", "^", "**"]):
                        if not any(skip in line.lower() for skip in ["example", "note", "where"]):
                            expression = line.strip()
                            break

            if not expression:
                raise ValidationError("No certificate expression found")

            # Clean expression
            expression = self._clean_expression(expression)

            # Validate expression
            self._validate_expression(expression, variables)

            # Determine certificate type
            cert_type = self._determine_certificate_type(expression)

            return BarrierCertificate(
                expression=expression, variables=variables, certificate_type=cert_type
            )

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse certificate: {e}")
            raise ValidationError(f"Failed to parse certificate: {e}")

    def extract_variables(self, dynamics: Dict[str, str]) -> List[str]:
        """
        Extract variable names from dynamics.

        Args:
            dynamics: Dictionary of dynamics equations

        Returns:
            List of variable names
        """
        # Start with the keys (state variables)
        variables = set(dynamics.keys())

        # Also extract from expressions to catch any parameters
        for expr in dynamics.values():
            # Parse as sympy expression
            try:
                parsed = parse_expr(expr)
                symbols = parsed.free_symbols
                variables.update(str(s) for s in symbols)
            except Exception:
                # Fallback to regex
                found_vars = self.var_pattern.findall(expr)
                variables.update(found_vars)

        # Filter out common functions and constants
        excluded = {
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "abs",
            "pi",
            "e",
            "in",
            "nan",
            "True",
            "False",
        }

        variables = [v for v in variables if v not in excluded]

        # Sort for consistency
        return sorted(list(variables))

    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize an expression."""
        # Remove leading/trailing whitespace
        expr = expr.strip()

        # Remove equation markers
        expr = re.sub(r"^[Bb]\s*\([^)]*\)\s*=\s*", "", expr)
        expr = re.sub(r"^[Bb]\s*=\s*", "", expr)

        # Convert to Python syntax
        expr = expr.replace("^", "**")

        # Remove trailing punctuation
        expr = re.sub(r"[.,;]+$", "", expr)

        return expr

    def _validate_expression(self, expr: str, variables: List[str]) -> None:
        """Validate that an expression is parseable and uses expected variables."""
        try:
            # Create symbol dictionary
            symbols = {var: sp.Symbol(var) for var in variables}

            # Try to parse
            parsed = parse_expr(expr, symbols)

            # Check that it uses at least one variable
            expr_vars = {str(s) for s in parsed.free_symbols}
            if not any(v in variables for v in expr_vars):
                raise ValidationError(
                    "Expression does not use any expected variables. "
                    f"Expected: {variables}, Found: {list(expr_vars)}"
                )

        except Exception as e:
            raise ValidationError(f"Invalid expression '{expr}': {e}")

    def _determine_certificate_type(self, expr: str) -> str:
        """Determine the type of certificate based on its form."""
        if "/" in expr and not "//" in expr:
            return "rational"
        elif "exp" in expr:
            return "exponential"
        else:
            return "standard"
