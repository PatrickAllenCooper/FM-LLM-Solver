# paper_population/utils/config_loader.py
import os
import sys
from omegaconf import OmegaConf, MissingMandatoryValue

# Determine Project Root dynamically
UTILS_DIR = os.path.dirname(__file__)
# Go up one level from utils, as utils is directly in the project root
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Loads the YAML configuration file using OmegaConf.
    
    This function loads configuration settings from a YAML file and performs
    path resolution and variable interpolation. It handles both absolute and
    relative paths, resolving them correctly based on the project root.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file (default: config.yaml in project root)
        
    Returns
    -------
    OmegaConf.DictConfig
        The loaded and resolved configuration object, or None if loading fails
    
    Raises
    ------
    SystemExit
        If the configuration file is missing or cannot be loaded
    """
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
            print(f"Working directory: {os.getcwd()}", file=sys.stderr)
            print("Please ensure the config file exists or specify the correct path.", file=sys.stderr)
            sys.exit(1)
            
        # Load the config file
        conf = OmegaConf.load(config_path)
        print(f"Loaded configuration from {config_path}")

        # Resolve paths relative to project root
        if 'paths' in conf:
            # Ensure project_root itself is absolute first
            if 'project_root' in conf.paths and not os.path.isabs(conf.paths.project_root):
                # If project_root is relative, resolve it against config file directory
                conf.paths['project_root'] = os.path.abspath(os.path.join(os.path.dirname(config_path), conf.paths.project_root))
            else:
                # If project_root is not specified or already absolute, use calculated PROJECT_ROOT
                conf.paths['project_root'] = PROJECT_ROOT

            # Resolve other paths relative to project_root if they are not absolute
            for key, path_val in conf.paths.items():
                if isinstance(path_val, str) and not os.path.isabs(path_val) and key != 'project_root':
                    conf.paths[key] = os.path.join(conf.paths['project_root'], path_val)
            
            print(f"Project root resolved to: {conf.paths.project_root}")

        # Resolve variable interpolation in the config
        try:
            OmegaConf.resolve(conf)
        except Exception as e:
            print(f"Warning: Some variable interpolation failed: {e}", file=sys.stderr)
            print("Processing will continue with partially resolved configuration.", file=sys.stderr)
            
        return conf

    except MissingMandatoryValue as e:
        print(f"Error: Missing required configuration value: {e}", file=sys.stderr)
        print("Please check your configuration file and ensure all required values are present.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing configuration file {config_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_config(cfg, config_path=DEFAULT_CONFIG_PATH):
    """
    Saves the configuration object back to a YAML file.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The configuration object to save
    config_path : str
        Path to the configuration YAML file to save to
        
    Returns
    -------
    bool
        True if saved successfully, False otherwise
    """
    try:
        # Create a backup of the original config file
        if os.path.exists(config_path):
            backup_path = config_path + '.bak'
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        # Save the configuration
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        
        print(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    print(f"Project Root detected as: {PROJECT_ROOT}")
    print(f"Attempting to load config from: {DEFAULT_CONFIG_PATH}")
    cfg = load_config()
    if cfg:
        print("Config loaded successfully.")
        print(OmegaConf.to_yaml(cfg))
        # Example Access:
        # print("\nFine-tuning model:", cfg.fine_tuning.base_model_name)
        # print("KB Output Dir:", cfg.paths.kb_output_dir)
        # print("Evaluation RAG K:", cfg.evaluation.rag_k) # Should be resolved from inference section 