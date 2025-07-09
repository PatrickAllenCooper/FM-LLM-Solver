"""
Simplified condition parsing utilities.

This module extracts complex condition parsing logic and reduces nested conditionals
for better maintainability.
"""

import logging
import sympy
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def parse_single_condition(cond_str: str, local_dict: dict) -> Optional[sympy.core.relational.Relational]:
    """
    Parse a single condition string into a SymPy relational.
    
    Returns:
        Parsed relational or None if parsing fails
    """
    try:
        rel = sympy.sympify(cond_str, locals=local_dict)
        if not (
            isinstance(rel, sympy.logic.boolalg.BooleanAtom) or
            isinstance(rel, sympy.logic.boolalg.BooleanFunction) or
            isinstance(rel, sympy.core.relational.Relational)
        ):
            raise TypeError(f"Parsed condition '{cond_str}' is not a recognized Relational or SymPy Boolean construct.")
        return rel
    except (SyntaxError, TypeError, sympy.SympifyError) as e:
        logger.error(f"Failed to parse condition '{cond_str}'. Error: {str(e)}")
        return None


def parse_or_condition(cond_str: str, local_dict: dict) -> Optional[sympy.core.relational.Relational]:
    """
    Parse a condition containing logical OR operators.
    
    Returns:
        Combined relational or None if parsing fails
    """
    parts = cond_str.split(" or ")
    sub_relationals = []
    
    for part in parts:
        part = part.strip()
        rel_part = parse_single_condition(part, local_dict)
        if rel_part is None:
            return None
        sub_relationals.append(rel_part)
    
    if not sub_relationals:
        return None
    
    # Combine parts with logical OR
    combined_rel = sub_relationals[0]
    for i in range(1, len(sub_relationals)):
        combined_rel = sympy.logic.boolalg.Or(combined_rel, sub_relationals[i])
    
    return combined_rel


def parse_set_conditions_simplified(condition_strings: List[str], variables: List) -> Tuple[Optional[List], str]:
    """
    Simplified version of parse_set_conditions with reduced nested conditionals.
    
    Parameters
    ----------
    condition_strings : List[str]
        List of condition strings to parse
    variables : List
        List of SymPy symbols for variables
        
    Returns
    -------
    Tuple[Optional[List], str]
        (parsed_relationals, message)
    """
    if condition_strings is None:
        return [], "No conditions provided"
    
    if not isinstance(condition_strings, list):
        logging.error("Set conditions must be a list of strings.")
        return None, "Invalid condition format (not a list)"

    relationals = []
    local_dict = {var.name: var for var in variables}
    
    for cond_str in condition_strings:
        # Handle OR conditions first
        if " or " in cond_str:
            rel = parse_or_condition(cond_str, local_dict)
        else:
            rel = parse_single_condition(cond_str, local_dict)
        
        if rel is None:
            return None, f"Parsing error in condition: {cond_str}"
        
        relationals.append(rel)
    
    return relationals, "Conditions parsed successfully"


def validate_condition_structure(cond_str: str) -> bool:
    """
    Validate basic structure of a condition string.
    
    Returns:
        True if structure is valid, False otherwise
    """
    if not cond_str:
        return False
    
    # Check for balanced parentheses
    if cond_str.count('(') != cond_str.count(')'):
        return False
    
    # Check for trailing operators
    if cond_str.endswith(('+', '-', '*', '/', '**', '^', '(')):
        return False
    
    # Check for empty function calls
    if '()' in cond_str:
        return False
    
    return True


def clean_condition_string(cond_str: str) -> str:
    """
    Clean a condition string by removing LaTeX and descriptive text.
    
    Returns:
        Cleaned condition string
    """
    if not cond_str:
        return cond_str
    
    # Remove LaTeX commands
    import re
    cleaned_str = re.sub(r'\\[\(\)]', '', cond_str)
    cleaned_str = re.sub(r'\\[\{\}]', '', cleaned_str)
    cleaned_str = cleaned_str.replace('\\cdot', '*')
    cleaned_str = cleaned_str.replace('^', '**')
    
    # Remove descriptive text
    descriptive_match = re.match(
        r"(.*?)(?:\s+(?:where|for|such that|on|ensuring|if|assuming|denotes|represents)\s+[a-zA-Z].*)", 
        cleaned_str, 
        re.DOTALL | re.IGNORECASE
    )
    if descriptive_match:
        cleaned_str = descriptive_match.group(1).strip()
    else:
        cleaned_str = cleaned_str.strip()
    
    # Remove trailing punctuation
    cleaned_str = cleaned_str.rstrip('.,;')
    
    return cleaned_str 