"""
Shared utilities for data formatting in fine-tuning modules.

This module consolidates common data formatting functions used across multiple
fine-tuning modules to eliminate code duplication.
"""

import json
import logging
import sympy
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def format_instruction_example(system_description: str, barrier_certificate: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format a single example into the instruction-following JSON structure.
    
    Parameters
    ----------
    system_description : str
        The system dynamics description
    barrier_certificate : str
        The barrier certificate function
    metadata : Optional[Dict[str, Any]]
        Additional metadata to include
        
    Returns
    -------
    Dict[str, Any]
        Formatted instruction example
    """
    instruction = ("Given the autonomous system described by the following dynamics, "
                   "propose a suitable barrier certificate function B(x) and, if possible, "
                   "briefly outline the conditions it must satisfy.")
    
    example = {
        "instruction": instruction,
        "input": system_description,
        "output": barrier_certificate,
        "metadata": metadata or {"source": "manual"}
    }
    
    return example


def format_prompt_completion_example(system_description: str, barrier_certificate: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format a single example into a simpler prompt/completion structure.
    
    Parameters
    ----------
    system_description : str
        The system dynamics description
    barrier_certificate : str
        The barrier certificate function
    metadata : Optional[Dict[str, Any]]
        Additional metadata to include
        
    Returns
    -------
    Dict[str, Any]
        Formatted prompt/completion example
    """
    prompt = f"System Dynamics:\n{system_description}\n\nBarrier Certificate:"
    completion = f" {barrier_certificate}"
    
    example = {
        "prompt": prompt,
        "completion": completion,
        "metadata": metadata or {"source": "manual"}
    }
    
    return example


def format_synthetic_example(system_desc_text: str, certificate_str: str, 
                           format_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format synthetic training examples with enhanced variable extraction.
    
    Parameters
    ----------
    system_desc_text : str
        System description text
    certificate_str : str
        Certificate string
    format_type : str
        Output format type ("instruction" or "prompt_completion")
    metadata : Optional[Dict[str, Any]]
        Additional metadata to include
        
    Returns
    -------
    Dict[str, Any]
        Formatted synthetic example
    """
    # Add source metadata
    if metadata is None:
        metadata = {'source': 'synthetic_sos_placeholder'}
    
    # Extract variables from certificate string
    try:
        certificate_expr = sympy.sympify(certificate_str)
        variables = list(certificate_expr.free_symbols)
        var_names = sorted([str(var) for var in variables])
        if var_names:
            var_string = ', '.join(var_names)
        else:
            var_string = "x, y"  # Default fallback
    except Exception as e:
        logger.warning(f"Failed to parse certificate for variable extraction: {e}")
        var_string = "x, y"  # Fallback if parsing fails
    
    if format_type == "instruction":
        instruction = ("Given the autonomous system described by the following dynamics, "
                       "propose a suitable barrier certificate function B(x).")
        output = f"Barrier Certificate Candidate:\nB({var_string}) = {certificate_str}\nMetadata: {json.dumps(metadata)}"
        
        return {
            "instruction": instruction,
            "input": system_desc_text,
            "output": output
        }
    elif format_type == "prompt_completion":
        prompt = f"System Dynamics:\n{system_desc_text}\n\nBarrier Certificate:"
        completion = f" B({var_string}) = {certificate_str}\nMetadata: {json.dumps(metadata)}"
        
        return {
            "prompt": prompt,
            "completion": completion
        }
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def formatting_prompts_func(example: Dict[str, Any]) -> str:
    """
    Format dataset examples for instruction fine-tuning.
    
    Parameters
    ----------
    example : Dict[str, Any]
        Example dictionary with instruction, input, output fields
        
    Returns
    -------
    str
        Formatted text for training
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    # Basic concatenation, actual chat structure will be applied by the tokenizer's chat_template if available
    text = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n{output_text}"
    
    return text


def formatting_prompt_completion_func(example: Dict[str, Any]) -> str:
    """
    Format dataset examples for prompt/completion fine-tuning.
    
    Parameters
    ----------
    example : Dict[str, Any]
        Example dictionary with prompt and completion fields
        
    Returns
    -------
    str
        Formatted text for training
    """
    prompt = example.get('prompt', '')
    completion = example.get('completion', '')
    text = prompt + completion
    
    return text


def extract_metadata_from_example(data: Dict[str, Any], filepath: str = "") -> Optional[str]:
    """
    Extract source metadata from a training example.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Training example data
    filepath : str
        File path for fallback source inference
        
    Returns
    -------
    Optional[str]
        Extracted source metadata or None
    """
    # Attempt 1: Check for top-level 'metadata.source'
    if isinstance(data.get('metadata'), dict):
        return data['metadata'].get('source')
    
    # Attempt 2: Check if source is embedded in 'output' or 'completion'
    output_text = data.get('output', '') + data.get('completion', '')
    import re
    meta_match = re.search(r'Metadata:\s*({.*?})', output_text, re.IGNORECASE)
    if meta_match:
        try:
            embedded_meta = json.loads(meta_match.group(1))
            return embedded_meta.get('source')
        except json.JSONDecodeError:
            pass
    
    # Attempt 3: Infer source from filename if not found in data
    if filepath:
        filename = filepath.lower()
        if 'synthetic' in filename:
            return 'synthetic_inferred'
        elif 'extract' in filename:
            return 'llm_extracted_inferred'
        elif 'manual' in filename or 'finetuning_data.jsonl' in filename:
            return 'manual'
        else:
            return 'unknown'
    
    return None 