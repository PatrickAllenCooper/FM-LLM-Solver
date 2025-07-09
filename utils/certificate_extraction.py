"""
Shared utilities for certificate extraction and cleaning.

This module consolidates common certificate extraction and cleaning functions
used across multiple modules to eliminate code duplication.
"""

import re
import logging
import sympy
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def extract_certificate_from_llm_output(llm_text: str, variables: List[str]) -> Tuple[Optional[str], bool]:
    """
    Extract barrier certificate B(x) string from LLM output.
    Prioritizes delimited blocks, then falls back to regex patterns.
    
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

    # Primary extraction: Look for the delimited block
    vars_for_b_func_str = r",\s*".join(map(re.escape, variables)) if variables else r"[\w\s,]+"
    delimited_pattern_str = (
        r"BARRIER_CERTIFICATE_START\s*\n"
        r"B\s*\(\s*" + vars_for_b_func_str + r"\s*\)\s*=\s*(.*?)\s*\n"
        r"BARRIER_CERTIFICATE_END"
    )
    
    match = re.search(delimited_pattern_str, llm_text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate_expr = match.group(1).strip()
        logger.info(f"Found delimited certificate: B(...) = {candidate_expr}")
        cleaned_expr = clean_and_validate_expression(candidate_expr, variables)
        if cleaned_expr:
            logger.info(f"Extracted and validated B(x) from delimited block: {cleaned_expr}")
            return cleaned_expr, False
        logger.warning("Delimited certificate found but content was invalid/unparsable. Trying regex patterns.")

    # Fallback regex patterns
    vars_regex_part = '|'.join(map(re.escape, variables)) if variables else 'x|y'
    
    # Special pattern to extract the expression part from "B(x) = expression"
    b_func_pattern = r'B\s*\([^)]*\)\s*=\s*([^;\.]+)'
    match = re.search(b_func_pattern, llm_text)
    if match:
        expr_part = match.group(1).strip()
        # Check if the expression contains a descriptive phrase
        descriptive_keywords = [
            'penalizes', 'guarantees', 'ensures', 'maintains', 'establishes', 'represents', 
            'captures', 'describes', 'measures', 'provides', 'implements', 'achieves',
            'prevents', 'avoids', 'limits', 'restricts', 'controls', 'monitors',
            'sufficient', 'necessary', 'required', 'appropriate', 'suitable',
            'the', 'this', 'that', 'these', 'those', 'can', 'will', 'should', 'could'
        ]
        if any(word in expr_part.lower() for word in descriptive_keywords):
            logger.warning(f"Skipping B(x) notation candidate '{expr_part}' because it appears to be a descriptive phrase.")
        else:
            cleaned_expr = clean_and_validate_expression(expr_part, variables)
            if cleaned_expr:
                logger.info(f"Extracted and validated B(x) from B(x) notation: {cleaned_expr}")
                return cleaned_expr, False
    
    # Other standard patterns
    patterns = [
        r'B\s*\(\s*(?:" + vars_regex_part + r"(?:\s*,\s*" + vars_regex_part + r")*\s*)\)\s*=\s*([^{};\n\.]+)',
        r'Barrier\s+Certificate\s*[:=]\s*([^{};\n\.]+)',
        r'(?:is|certificate is|given by|function is)\s*[:=]?\s*([^{};\n\.]+)',
        r'(?:conditions|function|certificate|propose)\s+(?:is|for|that|as)\s+([^{};\n\.]+)',
        r'([^\n;:]+\*\*2[^\n;:]+)',
        r'([^\n;:]*)(?:{vars_regex_part})(?:[+\-*/^()])+(?:[^\n;:]*)'.format(vars_regex_part=vars_regex_part)
    ]

    # Words that indicate an extracted text is a descriptive phrase and not an actual certificate
    descriptive_keywords = [
        'penalizes', 'guarantees', 'ensures', 'maintains', 'establishes', 'represents', 
        'captures', 'describes', 'measures', 'provides', 'implements', 'achieves',
        'prevents', 'avoids', 'limits', 'restricts', 'controls', 'monitors',
        'sufficient', 'necessary', 'required', 'appropriate', 'suitable',
        'the', 'this', 'that', 'these', 'those', 'can', 'will', 'should', 'could'
    ]

    for i, pattern_str in enumerate(patterns):
        try:
            # For the general pattern, ensure it can find at least one variable if variables are specified
            if i == len(patterns) - 1 and variables and not any(v in llm_text for v in variables):
                continue

            match = re.search(pattern_str, llm_text, re.IGNORECASE | (re.DOTALL if i == 3 else 0))
            if match:
                candidate_text = match.group(1) if match.groups() and match.group(1) else match.group(0)
                
                # Skip this candidate if it contains descriptive keywords
                contains_descriptive_word = any(word in candidate_text.lower() for word in descriptive_keywords)
                if contains_descriptive_word:
                    logger.warning(f"Skipping candidate '{candidate_text}' because it appears to be a descriptive phrase.")
                    continue
                
                # Check if it contains at least one of the system variables
                if variables and not any(var in candidate_text for var in variables):
                    logger.warning(f"Skipping candidate '{candidate_text}' because it doesn't contain any system variables.")
                    continue
                
                cleaned_expr = clean_and_validate_expression(candidate_text, variables)
                if cleaned_expr:
                    logger.info(f"Extracted and validated B(x) using regex pattern {i+1}: {cleaned_expr}")
                    return cleaned_expr, False
        except re.error as re_err:
            logger.error(f"Regex error with pattern {i+1} ('{pattern_str}'): {re_err}")

    logger.warning(f"Could not reliably extract or validate a specific B(x) expression from LLM output via any method: {llm_text[:100]}...")
    return None, True


def clean_and_validate_expression(candidate_str: str, system_variables_str_list: List[str]) -> Optional[str]:
    """
    Clean and validate a potential barrier certificate expression string.
    Returns the cleaned string if valid and parsable by SymPy and contains system variables, otherwise None.
    """
    if not candidate_str:
        return None
    
    candidate_str = str(candidate_str).strip()
    
    # Handle specific patterns that might cause issues
    
    # 1. Remove B(x) = prefix if it exists (to avoid interpreting B and x as variables)
    b_prefix_match = re.match(r'B\s*\([^)]*\)\s*=\s*(.*)', candidate_str)
    if b_prefix_match:
        candidate_str = b_prefix_match.group(1).strip()
    
    # 2. Basic structure checks (parentheses, trailing operators)
    if candidate_str.count('(') != candidate_str.count(')'):
        logger.debug(f"CleanValidate: Invalid - Unbalanced parentheses in '{candidate_str}'")
        return None
    if candidate_str.endswith(('+', '-', '*', '/', '**', '^', '(')):
        logger.debug(f"CleanValidate: Invalid - Trailing operator/open paren in '{candidate_str}'")
        return None
    if re.search(r'B\(\s*\)\s*=', candidate_str, re.IGNORECASE):
        logger.debug(f"CleanValidate: Invalid - Empty B() function in '{candidate_str}'")
        return None

    # 3. Standard cleaning (LaTeX, descriptive text, trailing punctuation)
    cleaned_str = re.sub(r'\\[\(\)]', '', candidate_str)  
    cleaned_str = re.sub(r'\\[\{\}]', '', cleaned_str)
    cleaned_str = cleaned_str.replace('\\cdot', '*')
    cleaned_str = cleaned_str.replace('^', '**')
    
    descriptive_match = re.match(r"(.*?)(?:\s+(?:where|for|such that|on|ensuring|if|assuming|denotes|represents)\s+[a-zA-Z].*)", cleaned_str, re.DOTALL | re.IGNORECASE)
    if descriptive_match:
        cleaned_str = descriptive_match.group(1).strip()
    else:
        cleaned_str = cleaned_str.strip()
    cleaned_str = cleaned_str.rstrip('.,;')

    if not cleaned_str:
        logger.debug(f"CleanValidate: Candidate '{candidate_str}' became empty after cleaning.")
        return None

    # 4. Attempt to parse with SymPy
    try:
        local_sympy_dict = {var_name: sympy.symbols(var_name) for var_name in system_variables_str_list} if system_variables_str_list else {}
        parsed_expr = sympy.parse_expr(cleaned_str, local_dict=local_sympy_dict, transformations='all')
        
        if parsed_expr is None or parsed_expr is sympy.S.EmptySet: 
            logger.debug(f"CleanValidate: Candidate '{cleaned_str}' (from '{candidate_str}') parsed to SymPy EmptySet or None.")
            return None

    except (SyntaxError, TypeError, sympy.SympifyError, AttributeError, RecursionError) as e: 
        logger.warning(f"CleanValidate: Candidate '{cleaned_str}' (from '{candidate_str}') failed SymPy parsing: {e}")
        return None
    except Exception as e: 
        logger.warning(f"CleanValidate: Candidate '{cleaned_str}' (from '{candidate_str}') failed SymPy parsing with unexpected error: {e}")
        return None

    # 5. Check if the parsed expression contains any of the system variables (if specified)
    if system_variables_str_list:
        try:
            expr_free_symbols_names = {s.name for s in parsed_expr.free_symbols}
        except AttributeError:  # e.g. if parsed_expr is a number
            expr_free_symbols_names = set()

        # Fix: Check if parsed_expr is a tuple or has the is_number attribute before accessing it
        is_number = False
        if isinstance(parsed_expr, tuple):
            logger.warning(f"CleanValidate: Parsed expression is a tuple: {parsed_expr}")
            return None
        elif hasattr(parsed_expr, 'is_number'):
            is_number = parsed_expr.is_number
        else:
            logger.warning(f"CleanValidate: Parsed expression has unexpected type: {type(parsed_expr)}")
            return None

        if not any(var_name in expr_free_symbols_names for var_name in system_variables_str_list):
            if is_number:  # Allow constants if they parse correctly
                logger.debug(f"CleanValidate: Candidate '{cleaned_str}' (parsed: '{parsed_expr}') is a constant. Allowing as potentially valid.")
            else:
                logger.debug(f"CleanValidate: Candidate '{cleaned_str}' (parsed: '{parsed_expr}') does not contain any system variables: {system_variables_str_list}. Free symbols: {expr_free_symbols_names}")
                return None

    # 6. Return the cleaned string if all checks pass
    logger.debug(f"CleanValidate: Successfully validated candidate '{cleaned_str}' (from '{candidate_str}')")
    return cleaned_str


def clean_certificate_expression(expression: str) -> str:
    """
    Clean LaTeX artifacts and other formatting issues from certificate expressions.
    
    Parameters
    ----------
    expression : str
        Raw certificate expression string
        
    Returns
    -------
    str
        Cleaned expression string
    """
    if not expression:
        return expression
    
    # FIRST: Remove any text after the mathematical expression
    # Stop at common sentence starters or descriptive phrases
    sentence_patterns = [
        r'\s+is\s+.*$',                 # "is a valid...", "is appropriate..."
        r'\s+which\s+.*$',               # "which ensures..."
        r'\s+this\s+.*$',                # "this guarantees..."
        r'\s+that\s+.*$',                # "that proves..."
        r'\s+for\s+.*$',                 # "for the system..."
        r'\s+where\s+.*$',               # "where x and y..."
        r'\s+ensuring\s+.*$',            # "ensuring stability..."
        r'\s+guaranteeing\s+.*$',        # "guaranteeing safety..."
        r'\s+could\s+be\s+.*$',          # "could be appropriate"
        r'\s+would\s+be\s+.*$',          # "would be appropriate"
        r'\s+seems\s+.*$',               # "seems appropriate"
        r'\s+appears\s+.*$',             # "appears suitable"
        r'\s+might\s+be\s+.*$',          # "might be appropriate"
        r'\s+should\s+be\s+.*$',         # "should be suitable"
        r'\s+can\s+be\s+used\s*$',      # "can be used"
        r'\s+satisfies\s+.*$',           # "satisfies the conditions"
        r'\s+represents\s+.*$',          # "represents a barrier"
        r'\[/?INST\].*$',               # Remove [INST] or [/INST] and everything after
        r'\s+Great!.*$',                # "Great! That's correct..."
        r'\s+[Tt]hat\'s\s+correct.*$',  # "That's correct..."
    ]
    
    cleaned = expression
    for pattern in sentence_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove trailing [CUT] or similar bracketed text
    cleaned = re.sub(r'\s*\[.*?\]\s*$', '', cleaned).strip()
    
    # Remove LaTeX brackets and delimiters
    cleaned = re.sub(r'\\[\[\]()]', '', cleaned)  # Remove \[ \] \( \)
    cleaned = re.sub(r'\\[\{\}]', '', cleaned)    # Remove \{ \}
    
    # Remove standalone LaTeX brackets at the end
    cleaned = re.sub(r'\s*\\\]\s*$', '', cleaned)  # Remove trailing \]
    cleaned = re.sub(r'\s*\\\[\s*$', '', cleaned)  # Remove trailing \[
    cleaned = re.sub(r'\s*\\\)\s*$', '', cleaned)  # Remove trailing \)
    cleaned = re.sub(r'\s*\\\(\s*$', '', cleaned)  # Remove trailing \(
    
    # Convert LaTeX math operators to standard notation
    cleaned = cleaned.replace('\\cdot', '*')
    cleaned = cleaned.replace('\\times', '*')
    cleaned = cleaned.replace('\\div', '/')
    cleaned = cleaned.replace('^', '**')           # Convert exponentiation
    
    # Remove LaTeX commands that might appear
    cleaned = re.sub(r'\\[a-zA-Z]+\s*', '', cleaned)  # Remove LaTeX commands like \alpha, \beta
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Remove trailing punctuation that might cause parsing issues
    cleaned = cleaned.rstrip('.,;:')
    
    logger.debug(f"Cleaned certificate expression: '{expression}' -> '{cleaned}'")
    return cleaned


def has_placeholder_variables(expression: str) -> bool:
    """
    Check if expression contains placeholder variables (Greek letters, single letters, etc.).
    
    Parameters
    ----------
    expression : str
        Certificate expression string
        
    Returns
    -------
    bool
        True if expression contains placeholder variables
    """
    if not expression:
        return True
    
    # Greek letters and common placeholders
    placeholders = [
        'α', 'β', 'γ', 'δ', 'λ', 'μ', 'θ', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω',
        '\\alpha', '\\beta', '\\gamma', '\\delta', '\\lambda', '\\mu', '\\theta',
        '\\rho', '\\sigma', '\\tau', '\\phi', '\\chi', '\\psi', '\\omega'
    ]
    
    # Check for Greek letters and LaTeX placeholders
    for placeholder in placeholders:
        if placeholder in expression:
            return True
    
    # More specific patterns for template variables
    # Look for single letters that are multiplied by variables or used as coefficients
    template_patterns = [
        r'\b[a-h]\s*\*\s*x',      # a*x, b*x, etc.
        r'\b[a-h]\s*\*\s*y',      # a*y, b*y, etc.
        r'\b[a-h]\s*\*\s*z',      # a*z, b*z, etc.
        r'\b[a-h]\s*x\b',         # ax, bx (without *)
        r'\b[a-h]\s*y\b',         # ay, by (without *)
        r'\b[a-h]\s*z\b',         # az, bz (without *)
        r'\b[A-HJ-W]\s*x\b',      # Ax, Bx, etc.
        r'\b[A-HJ-W]\s*y\b',      # Ay, By, etc.
        r'\b[A-HJ-W]\s*z\b',      # Az, Bz, etc.
        r'\b[a-zA-Z]\s*_\d+',     # a_1, b_2, etc.
        r'\bc_\d+',               # c_1, c_2, etc.
        r'coeff',                 # coefficient
        r'param',                 # parameter
    ]
    
    for pattern in template_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            return True
    
    return False


def contains_state_variables(expression: str) -> bool:
    """
    Check if expression contains actual state variables (x, y, z).
    
    Parameters
    ----------
    expression : str
        Certificate expression string
        
    Returns
    -------
    bool
        True if expression contains state variables
    """
    if not expression:
        return False
    
    # Look for state variables x, y, z
    state_var_pattern = r'\b[xyz]\b'
    return bool(re.search(state_var_pattern, expression, re.IGNORECASE))


def is_template_expression(expression: str) -> bool:
    """
    Check if the expression is a template/placeholder rather than a concrete barrier certificate.
    
    Parameters
    ----------
    expression : str
        Certificate expression string
        
    Returns
    -------
    bool
        True if expression is a template
    """
    if not expression:
        return True
    
    # If it already has placeholder variables, it's a template
    if has_placeholder_variables(expression):
        return True
    
    # Common template patterns that should be rejected
    template_patterns = [
        # Template variables (single letters as coefficients) - more precise matching
        r'\b[a-h]\*[xy]',  # ax, bx, cy, etc. (single letters multiplying state variables)
        r'\b[a-h]\*\*2',   # a**2, b**2, etc. (single letters as base)
        r'\b[a-h][xy]\b',  # ax, by, etc. (single letter followed by state variable)
        
        # Common placeholder naming
        r'\bc[0-9]',  # c1, c2, c3, etc.
        r'\bcoeff',   # coeff, coefficient
        r'\bparam',   # param, parameter
        
        # Standalone single letters that aren't state variables
        r'^\s*[a-h]\s*$',  # Just 'a' or 'b' etc.
        r'^\s*[a-h]\s*\+',  # Starting with single letter like 'a +'
    ]
    
    # Check for any template patterns
    for pattern in template_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            logger.debug(f"Template pattern detected: '{pattern}' in '{expression}'")
            return True
    
    # Additional heuristic: if expression has more than 3 single-letter variables 
    # (excluding x, y, z which are likely state variables), it's probably a template
    single_letters = re.findall(r'\b[a-h]\b', expression.lower())
    
    if len(set(single_letters)) >= 3:  # 3 or more different single-letter template variables
        logger.debug(f"Too many template variables detected: {set(single_letters)} in '{expression}'")
        return True
    
    return False 