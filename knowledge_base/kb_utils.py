"""
Utility functions for knowledge base operations.

This module provides utility functions for working with different types of knowledge bases
(unified, discrete, continuous) in the FM-LLM-Solver system.
"""

import os
import logging
from typing import Tuple, Dict, Any

def get_kb_paths_by_type(cfg, kb_type: str) -> Tuple[str, str, str]:
    """
    Get knowledge base paths for a specific barrier certificate type.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
    kb_type : str
        Knowledge base type ('unified', 'discrete', 'continuous')
        
    Returns
    -------
    Tuple[str, str, str]
        (output_dir, vector_store_path, metadata_path)
    """
    if kb_type == 'discrete':
        output_dir = cfg.paths.kb_discrete_output_dir
        vector_file = cfg.paths.kb_discrete_vector_store_filename
        metadata_file = cfg.paths.kb_discrete_metadata_filename
    elif kb_type == 'continuous':
        output_dir = cfg.paths.kb_continuous_output_dir
        vector_file = cfg.paths.kb_continuous_vector_store_filename
        metadata_file = cfg.paths.kb_continuous_metadata_filename
    else:  # unified or default
        output_dir = cfg.paths.kb_output_dir
        vector_file = cfg.paths.kb_vector_store_filename
        metadata_file = cfg.paths.kb_metadata_filename
    
    vector_path = os.path.join(output_dir, vector_file)
    metadata_path = os.path.join(output_dir, metadata_file)
    
    return output_dir, vector_path, metadata_path

def determine_kb_type_from_config(cfg) -> str:
    """
    Determine the knowledge base type from configuration.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
        
    Returns
    -------
    str
        Knowledge base type ('unified', 'discrete', 'continuous')
    """
    # Check fine-tuning configuration first
    ft_type = cfg.fine_tuning.get('barrier_certificate_type', 'unified')
    
    # Check knowledge base configuration
    kb_type = cfg.knowledge_base.get('barrier_certificate_type', 'unified')
    
    # They should match, but fine-tuning takes precedence
    if ft_type != kb_type:
        logging.warning(f"Mismatch between fine-tuning type ({ft_type}) and KB type ({kb_type}). Using fine-tuning type.")
        return ft_type
    
    return kb_type

def get_active_kb_paths(cfg) -> Tuple[str, str, str]:
    """
    Get the paths for the currently active knowledge base based on configuration.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
        
    Returns
    -------
    Tuple[str, str, str]
        (output_dir, vector_store_path, metadata_path)
    """
    kb_type = determine_kb_type_from_config(cfg)
    return get_kb_paths_by_type(cfg, kb_type)

def get_ft_data_path_by_type(cfg, kb_type: str) -> str:
    """
    Get the fine-tuning data path for a specific barrier certificate type.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
    kb_type : str
        Knowledge base type ('unified', 'discrete', 'continuous')
        
    Returns
    -------
    str
        Path to the appropriate fine-tuning data file
    """
    if kb_type == 'discrete':
        return cfg.paths.ft_discrete_data_file
    elif kb_type == 'continuous':
        return cfg.paths.ft_continuous_data_file
    else:  # unified or default
        return cfg.paths.ft_combined_data_file

def kb_exists(cfg, kb_type: str) -> bool:
    """
    Check if a knowledge base exists for the given type.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
    kb_type : str
        Knowledge base type ('unified', 'discrete', 'continuous')
        
    Returns
    -------
    bool
        True if the knowledge base exists, False otherwise
    """
    _, vector_path, metadata_path = get_kb_paths_by_type(cfg, kb_type)
    return os.path.exists(vector_path) and os.path.exists(metadata_path)

def list_available_kbs(cfg) -> Dict[str, bool]:
    """
    List all available knowledge bases.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping KB type to availability
    """
    return {
        'unified': kb_exists(cfg, 'unified'),
        'discrete': kb_exists(cfg, 'discrete'),
        'continuous': kb_exists(cfg, 'continuous')
    }

def validate_kb_config(cfg) -> bool:
    """
    Validate the knowledge base configuration.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
        
    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    try:
        # Check that barrier certificate types match
        ft_type = cfg.fine_tuning.get('barrier_certificate_type', 'unified')
        kb_type = cfg.knowledge_base.get('barrier_certificate_type', 'unified')
        
        valid_types = ['unified', 'discrete', 'continuous']
        
        if ft_type not in valid_types:
            logging.error(f"Invalid fine-tuning barrier certificate type: {ft_type}")
            return False
            
        if kb_type not in valid_types:
            logging.error(f"Invalid knowledge base barrier certificate type: {kb_type}")
            return False
        
        # Check that required paths are configured
        required_paths = ['pdf_input_dir', 'kb_output_dir']
        
        if ft_type == 'discrete' or kb_type == 'discrete':
            required_paths.extend(['kb_discrete_output_dir', 'ft_discrete_data_file'])
            
        if ft_type == 'continuous' or kb_type == 'continuous':
            required_paths.extend(['kb_continuous_output_dir', 'ft_continuous_data_file'])
        
        for path_key in required_paths:
            if not hasattr(cfg.paths, path_key):
                logging.error(f"Missing required path configuration: {path_key}")
                return False
        
        # Check classification configuration if needed
        if kb_type in ['discrete', 'continuous']:
            if not hasattr(cfg.knowledge_base, 'classification'):
                logging.error("Classification configuration required for discrete/continuous KB types")
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating KB configuration: {e}")
        return False