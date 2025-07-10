"""
Improved certificate extraction module for Phase 1 Day 5
Fixes decimal number extraction, template detection, and adds format support
"""

import re
import logging
import sympy
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


def extract_certificate_from_llm_output(llm_text: str, variables: List[str]) -> Tuple[Optional[str], bool]:
    """
    Extract barrier certificate B(x) string from LLM output with improved handling.
    
    Key improvements:
    1. Better decimal number preservation
    2. Enhanced template detection
    3. Support for LaTeX, MathML, and ASCII math formats
    4. More robust expression validation
    
    Parameters
    ----------
    llm_text : str
        The raw text output from the LLM
    variables : list
        List of string names representing the variables in the system (e.g. ["x", "y"])
        
    Returns
    -------
    tuple (str or None, bool)
        (extracted_expression, True_if_extraction_failed_else_False)
    """
    if not llm_text:
        logger.warning("Empty LLM output provided to extraction function")
        return None, True

    # Try extraction methods in order of reliability
    extraction_methods = [
        _extract_from_delimited_block,
        _extract_from_code_block,
        _extract_from_latex_format,
        _extract_from_mathml_format,
        _extract_from_mathematical_notation,
        _extract_from_b_notation,
        _extract_from_certificate_patterns,
        _extract_from_general_patterns
    ]
    
    for method in extraction_methods:
        result = method(llm_text, variables)
        if result:
            # Validate the extracted expression
            cleaned_expr = clean_and_validate_expression(result, variables)
            if cleaned_expr and not is_template_expression(cleaned_expr):
                logger.info(f"Successfully extracted certificate: {cleaned_expr}")
                return cleaned_expr, False
            elif cleaned_expr and is_template_expression(cleaned_expr):
                logger.warning(f"Rejected template expression: {cleaned_expr}")
                continue
    
    logger.warning(f"Could not extract valid certificate from LLM output: {llm_text[:100]}...")
    return None, True


def _extract_from_delimited_block(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from BARRIER_CERTIFICATE_START/END blocks"""
    # Build variable pattern
    vars_pattern = r",\s*".join(map(re.escape, variables)) if variables else r"[\w\s,]+"
    
    # Match delimited blocks with or without B(...) =
    patterns = [
        # With B(...) = 
        (r"BARRIER_CERTIFICATE_START\s*\n"
         r"B\s*\(\s*" + vars_pattern + r"\s*\)\s*=\s*(.*?)\s*\n"
         r"BARRIER_CERTIFICATE_END", 1),
        # Without B(...) =
        (r"BARRIER_CERTIFICATE_START\s*\n"
         r"(.*?)\s*\n"
         r"BARRIER_CERTIFICATE_END", 1),
    ]
    
    for pattern, group_idx in patterns:
        match = re.search(pattern, llm_text, re.DOTALL | re.IGNORECASE)
        if match:
            expr = match.group(group_idx).strip()
            logger.debug(f"Found delimited certificate: {expr}")
            return expr
    
    return None


def _extract_from_code_block(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from code blocks (```python or ``` blocks)"""
    patterns = [
        # Function definition with return
        r'```(?:python)?\s*\n.*?def\s+barrier_certificate\s*\([^)]*\)\s*:\s*\n\s*return\s+([^\n]+)\s*\n.*?```',
        # Just return statement
        r'```(?:python)?\s*\n.*?return\s+([^\n]+)\s*\n.*?```',
        # Assignment in code block
        r'```(?:python)?\s*\n.*?B\s*=\s*([^\n]+)\s*\n.*?```',
        r'```(?:python)?\s*\n.*?barrier\s*=\s*([^\n]+)\s*\n.*?```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, llm_text, re.DOTALL | re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            logger.debug(f"Found certificate in code block: {expr}")
            return expr
    
    return None


def _extract_from_latex_format(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from LaTeX formatted expressions"""
    # LaTeX math environments
    patterns = [
        # Display math
        r'\\\[\s*(.*?)\s*\\\]',
        r'\$\$\s*(.*?)\s*\$\$',
        # Inline math
        r'\\\(\s*(.*?)\s*\\\)',
        r'\$\s*(.*?)\s*\$',
        # Equation environment
        r'\\begin\{equation\}\s*(.*?)\s*\\end\{equation\}',
        r'\\begin\{align\}\s*(.*?)\s*\\end\{align\}',
    ]
    
    # Look for barrier certificate context
    context_patterns = [
        r'[Bb]arrier\s+[Cc]ertificate.*?(?:is|=|:)\s*',
        r'B\s*\([^)]*\)\s*(?:=|:)\s*',
        r'[Cc]ertificate.*?(?:is|=|:)\s*',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, llm_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            expr = match.group(1).strip()
            
            # Check if this is in a certificate context
            start_pos = max(0, match.start() - 100)
            context = llm_text[start_pos:match.start()]
            
            if any(re.search(ctx_pattern, context, re.IGNORECASE) for ctx_pattern in context_patterns):
                logger.debug(f"Found certificate in LaTeX format: {expr}")
                # Clean LaTeX-specific notation
                expr = _clean_latex_expression(expr)
                return expr
    
    return None


def _extract_from_mathml_format(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from MathML format (basic support)"""
    # Simple MathML extraction - look for <math> tags
    pattern = r'<math[^>]*>(.*?)</math>'
    match = re.search(pattern, llm_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        mathml_content = match.group(1)
        # Basic MathML to expression conversion (simplified)
        expr = _convert_mathml_to_expression(mathml_content)
        if expr:
            logger.debug(f"Found certificate in MathML format: {expr}")
            return expr
    
    return None


def _extract_from_mathematical_notation(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from mathematical notation (B: ℝ² → ℝ, etc.)"""
    patterns = [
        # B: ℝⁿ → ℝ defined by B(x,y) := expression
        r'B\s*:\s*ℝ[²2³3ⁿn]\s*→\s*ℝ\s+defined\s+by\s+B\s*\([^)]*\)\s*:=\s*([^\n]+)',
        # B(x,y) := expression
        r'B\s*\([^)]*\)\s*:=\s*([^\n]+)',
        # B = expression (in mathematical context)
        r'[Bb]arrier\s+[Cc]ertificate\s+B\s*=\s*([^\n]+)',
        # Mathematical definition style
        r'[Dd]efine\s+B\s*\([^)]*\)\s*=\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, llm_text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            # Convert unicode superscripts
            expr = _convert_unicode_math(expr)
            logger.debug(f"Found certificate in mathematical notation: {expr}")
            return expr
    
    return None


def _extract_from_b_notation(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from B(x) = expression notation with improved decimal handling"""
    # FIXED: Better regex for decimal preservation
    patterns = [
        # Standard B(...) = expression
        r'B\s*\([^)]*\)\s*=\s*([^;\n]+?)(?=\s*(?:$|\n|;|\.(?:\s|$)))',
        # With colon
        r'B\s*\([^)]*\)\s*:\s*([^;\n]+?)(?=\s*(?:$|\n|;|\.(?:\s|$)))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, llm_text)
        if match:
            expr = match.group(1).strip()
            
            # Skip if it's descriptive text
            if _is_descriptive_text(expr):
                continue
                
            logger.debug(f"Found certificate in B notation: {expr}")
            return expr
    
    return None


def _extract_from_certificate_patterns(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract from patterns like 'certificate is:' etc."""
    patterns = [
        r'[Cc]ertificate\s*(?:is|=|:)\s*([^\n]+?)(?=\s*(?:$|\n|\.(?:\s|$)))',
        r'[Bb]arrier\s+[Cc]ertificate\s*(?:is|=|:)\s*([^\n]+?)(?=\s*(?:$|\n|\.(?:\s|$)))',
        r'[Tt]he\s+certificate\s*(?:is|=|:)\s*([^\n]+?)(?=\s*(?:$|\n|\.(?:\s|$)))',
        r'[Uu]se\s+(?:the\s+)?certificate\s*(?:is|=|:)?\s*([^\n]+?)(?=\s*(?:$|\n|\.(?:\s|$)))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, llm_text)
        if match:
            expr = match.group(1).strip()
            
            # Must contain mathematical operators
            if not any(op in expr for op in ['**', '+', '-', '*', '/', '^']):
                continue
                
            if _is_descriptive_text(expr):
                continue
                
            logger.debug(f"Found certificate from pattern: {expr}")
            return expr
    
    return None


def _extract_from_general_patterns(llm_text: str, variables: List[str]) -> Optional[str]:
    """Extract using general patterns as last resort"""
    # Build variable regex
    vars_regex = '|'.join(map(re.escape, variables)) if variables else r'[xyz]'
    
    # General patterns that might contain certificates
    patterns = [
        # Quadratic forms
        rf'({vars_regex})\s*\*\*\s*2.*?(?:{vars_regex})\s*\*\*\s*2[^;\n]*',
        # Expressions with variables and operators
        rf'[^;\n]*(?:{vars_regex})[^;\n]*[\+\-\*/][^;\n]*(?:{vars_regex})[^;\n]*',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, llm_text, re.IGNORECASE)
        for match in matches:
            expr = match.group(0).strip()
            
            # Clean up the expression
            expr = _clean_general_expression(expr)
            
            if expr and not _is_descriptive_text(expr):
                logger.debug(f"Found certificate from general pattern: {expr}")
                return expr
    
    return None


def clean_and_validate_expression(candidate_str: str, system_variables: List[str]) -> Optional[str]:
    """
    Clean and validate a potential barrier certificate expression.
    
    Improvements:
    1. Better decimal number handling
    2. More robust polynomial checking
    3. Enhanced template detection
    """
    if not candidate_str:
        return None
    
    candidate_str = str(candidate_str).strip()
    
    # Remove B(x) = prefix if present
    candidate_str = re.sub(r'^B\s*\([^)]*\)\s*=\s*', '', candidate_str, flags=re.IGNORECASE)
    
    # Basic structure validation
    if candidate_str.count('(') != candidate_str.count(')'):
        logger.debug(f"Invalid: Unbalanced parentheses in '{candidate_str}'")
        return None
    
    if candidate_str.endswith(('+', '-', '*', '/', '**', '^', '(')):
        logger.debug(f"Invalid: Trailing operator in '{candidate_str}'")
        return None
    
    # Clean the expression
    cleaned_str = _clean_expression(candidate_str)
    
    if not cleaned_str:
        logger.debug(f"Expression became empty after cleaning")
        return None
    
    # Validate with SymPy
    try:
        # Create symbol dictionary
        local_dict = {var: sympy.Symbol(var, real=True) for var in system_variables}
        
        # Parse expression
        parsed_expr = sympy.parse_expr(cleaned_str, local_dict=local_dict, transformations='all')
        
        if parsed_expr is None or parsed_expr is sympy.S.EmptySet:
            logger.debug(f"Expression parsed to None or EmptySet")
            return None
        
        # Check if it's a polynomial
        if system_variables:
            symbols = [sympy.Symbol(var, real=True) for var in system_variables]
            if not parsed_expr.is_polynomial(*symbols):
                logger.debug(f"Expression is not a polynomial: {cleaned_str}")
                return None
        
        # Check if it contains system variables
        expr_vars = {str(s) for s in parsed_expr.free_symbols}
        if system_variables and not any(var in expr_vars for var in system_variables):
            # Allow constants
            if not parsed_expr.is_number:
                logger.debug(f"Expression doesn't contain system variables: {cleaned_str}")
                return None
    
    except Exception as e:
        logger.warning(f"Failed to parse expression '{cleaned_str}': {e}")
        return None
    
    logger.debug(f"Successfully validated expression: {cleaned_str}")
    return cleaned_str


def is_template_expression(expression: str) -> bool:
    """
    Enhanced template detection with more comprehensive patterns.
    """
    if not expression:
        return True
    
    # Check for placeholder variables
    if _has_placeholder_variables(expression):
        return True
    
    # Enhanced template patterns
    template_patterns = [
        # Single letter coefficients (excluding i, j, k which might be indices)
        r'\b[a-hA-H]\s*\*?\s*[xyz]',  # a*x, ax, A*y, etc.
        r'\b[a-hA-H]\d*\s*\*?\s*[xyz]',  # a1*x, a2x, etc.
        
        # Subscripted coefficients
        r'\b[a-zA-Z]_\d+',  # a_1, b_2, etc.
        r'\b[a-zA-Z]_\{[^}]+\}',  # a_{ij}, etc.
        
        # Common template words
        r'\b(?:coeff|param|const|alpha|beta|gamma|delta|lambda|mu|theta)',
        
        # Ellipsis indicating incomplete expression
        r'\.\.\.',
        r'\\(?:dots|cdots|ldots|vdots)',
        
        # Placeholder brackets
        r'<[^>]+>',  # <expression>
        r'\[[^\]]*\?\s*[^\]]*\]',  # [?] or [something?]
        
        # "Some" or generic descriptions
        r'\b(?:some|any|certain|appropriate|suitable)\s+(?:constant|coefficient|parameter)',
    ]
    
    for pattern in template_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            logger.debug(f"Template pattern '{pattern}' found in '{expression}'")
            return True
    
    # Check for too many undefined symbols
    try:
        # Parse to find all symbols
        parsed = sympy.parse_expr(expression)
        symbols = {str(s) for s in parsed.free_symbols}
        
        # Count non-state variables
        non_state_vars = symbols - {'x', 'y', 'z', 't'}
        
        if len(non_state_vars) >= 3:  # Too many undefined parameters
            logger.debug(f"Too many undefined symbols: {non_state_vars}")
            return True
            
    except:
        pass  # If parsing fails, continue with other checks
    
    return False


# Helper functions

def _is_descriptive_text(text: str) -> bool:
    """Check if text is descriptive rather than mathematical"""
    descriptive_keywords = [
        'penalizes', 'guarantees', 'ensures', 'maintains', 'establishes',
        'represents', 'captures', 'describes', 'measures', 'provides',
        'implements', 'achieves', 'prevents', 'avoids', 'limits',
        'restricts', 'controls', 'monitors', 'sufficient', 'necessary',
        'required', 'appropriate', 'suitable', 'the', 'this', 'that',
        'these', 'those', 'can', 'will', 'should', 'could', 'would',
        'might', 'must', 'shall', 'may'
    ]
    
    text_lower = text.lower()
    return any(word in text_lower.split() for word in descriptive_keywords)


def _clean_expression(expr: str) -> str:
    """Clean mathematical expression"""
    # Remove descriptive suffixes
    expr = re.sub(r'\s+(?:where|for|such that|on|ensuring|if|assuming|denotes|represents).*$', 
                  '', expr, flags=re.IGNORECASE)
    
    # Clean LaTeX
    expr = _clean_latex_expression(expr)
    
    # Convert common notations
    expr = expr.replace('^', '**')
    
    # Remove trailing punctuation
    expr = expr.rstrip('.,;:')
    
    # Normalize whitespace
    expr = re.sub(r'\s+', ' ', expr).strip()
    
    return expr


def _clean_latex_expression(expr: str) -> str:
    """Clean LaTeX-specific notation"""
    # Remove LaTeX delimiters
    expr = re.sub(r'\\[\[\]()]', '', expr)
    expr = re.sub(r'\\[\{\}]', '', expr)
    
    # Convert LaTeX operators
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\times', '*')
    expr = expr.replace('\\div', '/')
    expr = expr.replace('\\frac', '/')  # Simplified
    
    # Remove LaTeX commands
    expr = re.sub(r'\\[a-zA-Z]+', '', expr)
    
    return expr


def _convert_unicode_math(expr: str) -> str:
    """Convert unicode mathematical symbols"""
    replacements = {
        '²': '**2',
        '³': '**3',
        '⁴': '**4',
        '×': '*',
        '÷': '/',
        '·': '*',
        '−': '-',
        '∗': '*',
    }
    
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    
    return expr


def _convert_mathml_to_expression(mathml: str) -> Optional[str]:
    """Basic MathML to expression conversion (simplified)"""
    # This is a very basic implementation
    # Remove tags and try to extract content
    content = re.sub(r'<[^>]+>', ' ', mathml)
    content = re.sub(r'\s+', ' ', content).strip()
    
    if content and any(c in content for c in 'xyz+-*/'):
        return content
    
    return None


def _clean_general_expression(expr: str) -> str:
    """Clean expression extracted from general patterns"""
    # Remove leading/trailing non-mathematical content
    expr = re.sub(r'^[^a-zA-Z0-9\(\)\+\-\*/\^]+', '', expr)
    expr = re.sub(r'[^a-zA-Z0-9\(\)\+\-\*/\^]+$', '', expr)
    
    # Remove inline descriptive text
    expr = re.sub(r'\s+(?:and|or|with|where|for)\s+.*$', '', expr, flags=re.IGNORECASE)
    
    return expr.strip()


def _has_placeholder_variables(expression: str) -> bool:
    """Check for placeholder variables"""
    placeholders = [
        # Greek letters
        'α', 'β', 'γ', 'δ', 'λ', 'μ', 'θ', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω',
        '\\alpha', '\\beta', '\\gamma', '\\delta', '\\lambda', '\\mu',
        '\\theta', '\\rho', '\\sigma', '\\tau', '\\phi', '\\chi', '\\psi', '\\omega',
        
        # Common placeholders
        'eps', 'epsilon', 'delta', 'sigma',
    ]
    
    for placeholder in placeholders:
        if placeholder in expression:
            return True
    
    return False


# Additional utility functions for format support

def extract_from_ascii_math(text: str) -> Optional[str]:
    """Extract from ASCII math notation"""
    # ASCII math uses backticks
    pattern = r'`([^`]+)`'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        expr = match.group(1)
        # Check if it's in a certificate context
        context = text[max(0, match.start()-50):match.start()]
        if any(word in context.lower() for word in ['barrier', 'certificate', 'b(']):
            return expr
    
    return None


def normalize_expression(expr: str) -> str:
    """Normalize expression to standard form"""
    try:
        # Parse and reformat
        parsed = sympy.parse_expr(expr)
        # Convert to standard string representation
        return str(parsed)
    except:
        return expr


# Test the improved extraction
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Decimal numbers
        "B(x,y) = x**2 + y**2 - 1.234567890",
        "The certificate is x**2 + y**2 - 0.0001",
        
        # LaTeX format
        r"The barrier certificate is $B(x,y) = x^2 + y^2 - 1.5$",
        r"\[B(x,y) = x^2 + y^2 - 2.0\]",
        
        # Templates (should be rejected)
        "B(x,y) = a*x**2 + b*y**2 + c",
        "B(x,y) = alpha*x**2 + beta*y**2 - gamma",
        
        # Scientific notation
        "B(x,y) = x**2 + y**2 - 1.5e-3",
        "Certificate: x**2 + y**2 - 2E-10",
    ]
    
    for test in test_cases:
        result, failed = extract_certificate_from_llm_output(test, ['x', 'y'])
        print(f"Input: {test}")
        print(f"Result: {result}")
        print(f"Failed: {failed}")
        print("-" * 50) 