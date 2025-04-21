# paper_population/utils/config_loader.py
import os
import sys
from omegaconf import OmegaConf, MissingMandatoryValue

# Determine Project Root dynamically assuming this util is in paper_population/utils
UTILS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, "..", "..")) # Go up two levels from utils
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads the YAML configuration file using OmegaConf."""
    try:
        conf = OmegaConf.load(config_path)

        # Resolve paths relative to project root
        if 'paths' in conf:
            # Ensure project_root itself is absolute first
            if 'project_root' in conf.paths and not os.path.isabs(conf.paths.project_root):
                conf.paths['project_root'] = os.path.abspath(os.path.join(os.path.dirname(config_path), conf.paths.project_root))
            else:
                 # If project_root is not specified or absolute, assume config file's dir parent is root
                 # Or default to the calculated PROJECT_ROOT
                 conf.paths['project_root'] = PROJECT_ROOT

            for key, path_val in conf.paths.items():
                if isinstance(path_val, str) and not os.path.isabs(path_val) and key != 'project_root':
                    conf.paths[key] = os.path.join(conf.paths['project_root'], path_val)

        # Example: Check for required environment variables mentioned in comments
        # Note: OmegaConf doesn't directly parse env vars from comments,
        # this check needs to be done explicitly where the env var is used.
        # Example check (place in the script that uses the key):
        # mathpix_id = os.environ.get("MATHPIX_APP_ID")
        # if not mathpix_id:
        #     raise ValueError("Environment variable MATHPIX_APP_ID is not set.")

        # Merge with command line arguments if needed (can be done in the main script)
        # cli_conf = OmegaConf.from_cli()
        # conf = OmegaConf.merge(conf, cli_conf)

        OmegaConf.resolve(conf) # Resolve interpolations like ${section.key}
        return conf

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except MissingMandatoryValue as e:
        print(f"Error: Missing configuration value: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch other OmegaConf or general errors
        print(f"Error loading or processing configuration file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)

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