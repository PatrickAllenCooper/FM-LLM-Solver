"""
Configuration Manager for FM-LLM-Solver
Handles configuration switching and environment-specific settings.
"""

import os
import yaml
import copy
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration settings with support for different barrier certificate types."""
    
    def __init__(self, base_config_path: str = "config.yaml"):
        """Initialize configuration manager with base config."""
        self.base_config_path = Path(base_config_path)
        self.base_config = self._load_yaml(self.base_config_path)
        self._current_config = copy.deepcopy(self.base_config)
        
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def set_barrier_type(self, barrier_type: str) -> None:
        """Set the barrier certificate type throughout the configuration.
        
        Args:
            barrier_type: One of "discrete", "continuous", or "unified"
        """
        valid_types = ["discrete", "continuous", "unified"]
        if barrier_type not in valid_types:
            raise ValueError(f"Invalid barrier type. Must be one of: {valid_types}")
        
        # Update barrier certificate type in all relevant sections
        if 'knowledge_base' in self._current_config:
            self._current_config['knowledge_base']['barrier_certificate_type'] = barrier_type
            
            # Update KB paths based on type
            if barrier_type == "discrete":
                kb_dir = "kb_data_discrete"
                vector_file = "paper_index_discrete.faiss"
                metadata_file = "paper_metadata_discrete.jsonl"
            elif barrier_type == "continuous":
                kb_dir = "kb_data_continuous"
                vector_file = "paper_index_continuous.faiss"
                metadata_file = "paper_metadata_continuous.jsonl"
            else:  # unified
                kb_dir = "kb_data"
                vector_file = "paper_index_mathpix.faiss"
                metadata_file = "paper_metadata_mathpix.jsonl"
            
            paths = self._current_config.get('paths', {})
            paths['kb_output_dir'] = kb_dir
            paths['kb_vector_store_filename'] = vector_file
            paths['kb_metadata_filename'] = metadata_file
            
        if 'fine_tuning' in self._current_config:
            self._current_config['fine_tuning']['barrier_certificate_type'] = barrier_type
            
            # Update fine-tuning output directory
            paths = self._current_config.get('paths', {})
            paths['ft_output_dir'] = f"${{paths.output_dir}}/finetuning_results_{barrier_type}"
            paths['eval_results_file'] = f"${{paths.output_dir}}/evaluation_results_{barrier_type}.csv"
    
    def set_environment(self, env: str) -> None:
        """Set environment-specific configurations.
        
        Args:
            env: Environment name (e.g., "development", "production", "testing")
        """
        # Adjust settings based on environment
        if env == "testing":
            # Reduce sample sizes for faster testing
            if 'evaluation' in self._current_config:
                verif = self._current_config['evaluation']['verification']
                verif['num_samples_lie'] = 1000
                verif['num_samples_boundary'] = 500
                verif['optimization_max_iter'] = 50
                verif['optimization_pop_size'] = 5
        elif env == "production":
            # Use full sample sizes for production
            if 'evaluation' in self._current_config:
                verif = self._current_config['evaluation']['verification']
                verif['num_samples_lie'] = 10000
                verif['num_samples_boundary'] = 5000
                verif['optimization_max_iter'] = 100
                verif['optimization_pop_size'] = 15
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return copy.deepcopy(self._current_config)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            path: Output path. If None, overwrites the base config.
        """
        output_path = Path(path) if path else self.base_config_path
        with open(output_path, 'w') as f:
            yaml.dump(self._current_config, f, default_flow_style=False, sort_keys=False)
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update a specific configuration section.
        
        Args:
            section: Section name (e.g., "fine_tuning", "evaluation")
            updates: Dictionary of updates to apply
        """
        if section not in self._current_config:
            self._current_config[section] = {}
        
        self._update_nested_dict(self._current_config[section], updates)
    
    def _update_nested_dict(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    @classmethod
    def create_minimal_config(cls, barrier_type: str = "discrete") -> 'ConfigManager':
        """Create a minimal configuration for testing.
        
        Args:
            barrier_type: Type of barrier certificates
            
        Returns:
            ConfigManager instance with minimal config
        """
        manager = cls.__new__(cls)
        manager.base_config_path = Path("config.yaml")
        manager.base_config = {}
        manager._current_config = {
            'paths': {
                'project_root': '.',
                'data_dir': 'data',
                'output_dir': 'output',
                'kb_output_dir': f'kb_data_{barrier_type}' if barrier_type != 'unified' else 'kb_data'
            },
            'knowledge_base': {
                'barrier_certificate_type': barrier_type,
                'embedding_model_name': 'all-mpnet-base-v2',
                'pipeline': 'open_source'
            },
            'fine_tuning': {
                'barrier_certificate_type': barrier_type,
                'base_model_name': 'Qwen/Qwen2.5-14B-Instruct',
                'use_adapter': True
            },
            'inference': {
                'rag_k': 3,
                'max_new_tokens': 512,
                'temperature': 0.6,
                'top_p': 0.9
            },
            'evaluation': {
                'verification': {
                    'num_samples_lie': 1000,
                    'num_samples_boundary': 500,
                    'numerical_tolerance': 1e-6,
                    'attempt_sos': True,
                    'attempt_optimization': True
                }
            }
        }
        return manager


# Convenience functions for backward compatibility
def load_config_for_barrier_type(barrier_type: str, base_config: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration for a specific barrier certificate type."""
    manager = ConfigManager(base_config)
    manager.set_barrier_type(barrier_type)
    return manager.get_config()


def save_unified_config(config: Dict[str, Any], path: str = "config.yaml") -> None:
    """Save a unified configuration file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False) 