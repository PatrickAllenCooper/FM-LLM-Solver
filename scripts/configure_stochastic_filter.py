#!/usr/bin/env python3
"""
Configure Stochastic Barrier Certificate Filtering

This script provides an easy way to enable/disable and configure 
stochastic barrier certificate filtering for both knowledge base 
building and fine-tuning data creation.

Usage:
    python scripts/configure_stochastic_filter.py --help
    python scripts/configure_stochastic_filter.py --enable --mode exclude
    python scripts/configure_stochastic_filter.py --disable
    python scripts/configure_stochastic_filter.py --status
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config, save_config, DEFAULT_CONFIG_PATH
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def show_status(config_path):
    """Show current stochastic filtering configuration."""
    try:
        cfg = load_config(config_path)
        
        print("\n🔍 Current Stochastic Filter Configuration:")
        print("=" * 50)
        
        # Knowledge base filtering
        kb_enabled = cfg.knowledge_base.classification.stochastic_filter.get('enable', False)
        kb_mode = cfg.knowledge_base.classification.stochastic_filter.get('mode', 'exclude')
        kb_min_keywords = cfg.knowledge_base.classification.stochastic_filter.get('min_stochastic_keywords', 2)
        kb_threshold = cfg.knowledge_base.classification.stochastic_filter.get('stochastic_confidence_threshold', 0.4)
        
        print(f"📚 Knowledge Base Filtering:")
        print(f"   Status: {'✅ ENABLED' if kb_enabled else '❌ DISABLED'}")
        if kb_enabled:
            print(f"   Mode: {kb_mode.upper()}")
            print(f"   Min Keywords: {kb_min_keywords}")
            print(f"   Confidence Threshold: {kb_threshold}")
        
        # Fine-tuning filtering
        ft_enabled = cfg.fine_tuning.stochastic_filter.get('enable', False)
        ft_mode = cfg.fine_tuning.stochastic_filter.get('mode', 'exclude')
        ft_extracted = cfg.fine_tuning.stochastic_filter.get('apply_to_extracted_data', True)
        ft_manual = cfg.fine_tuning.stochastic_filter.get('apply_to_manual_data', False)
        ft_synthetic = cfg.fine_tuning.stochastic_filter.get('apply_to_synthetic_data', True)
        
        print(f"\n🎯 Fine-tuning Data Filtering:")
        print(f"   Status: {'✅ ENABLED' if ft_enabled else '❌ DISABLED'}")
        if ft_enabled:
            print(f"   Mode: {ft_mode.upper()}")
            print(f"   Apply to Extracted Data: {'✅' if ft_extracted else '❌'}")
            print(f"   Apply to Manual Data: {'✅' if ft_manual else '❌'}")
            print(f"   Apply to Synthetic Data: {'✅' if ft_synthetic else '❌'}")
        
        # Stochastic keywords
        stochastic_keywords = cfg.knowledge_base.classification.get('stochastic_keywords', [])
        print(f"\n🏷️ Stochastic Keywords ({len(stochastic_keywords)} total):")
        print("   " + ", ".join(stochastic_keywords[:10]) + ("..." if len(stochastic_keywords) > 10 else ""))
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False
    
    return True

def enable_filtering(config_path, mode, min_keywords, confidence_threshold, apply_to_extracted, apply_to_manual, apply_to_synthetic):
    """Enable stochastic filtering with specified configuration."""
    try:
        cfg = load_config(config_path)
        
        # Enable knowledge base filtering
        cfg.knowledge_base.classification.stochastic_filter.enable = True
        cfg.knowledge_base.classification.stochastic_filter.mode = mode
        cfg.knowledge_base.classification.stochastic_filter.min_stochastic_keywords = min_keywords
        cfg.knowledge_base.classification.stochastic_filter.stochastic_confidence_threshold = confidence_threshold
        
        # Enable fine-tuning filtering
        cfg.fine_tuning.stochastic_filter.enable = True
        cfg.fine_tuning.stochastic_filter.mode = mode
        cfg.fine_tuning.stochastic_filter.apply_to_extracted_data = apply_to_extracted
        cfg.fine_tuning.stochastic_filter.apply_to_manual_data = apply_to_manual
        cfg.fine_tuning.stochastic_filter.apply_to_synthetic_data = apply_to_synthetic
        
        # Save configuration
        save_config(cfg, config_path)
        
        print(f"✅ Stochastic filtering ENABLED with mode '{mode.upper()}'")
        print(f"   Configuration saved to: {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error enabling filtering: {e}")
        return False

def disable_filtering(config_path):
    """Disable stochastic filtering."""
    try:
        cfg = load_config(config_path)
        
        # Disable knowledge base filtering
        cfg.knowledge_base.classification.stochastic_filter.enable = False
        
        # Disable fine-tuning filtering
        cfg.fine_tuning.stochastic_filter.enable = False
        
        # Save configuration
        save_config(cfg, config_path)
        
        print("❌ Stochastic filtering DISABLED")
        print(f"   Configuration saved to: {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error disabling filtering: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Configure stochastic barrier certificate filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show current status
    python scripts/configure_stochastic_filter.py --status
    
    # Enable filtering to exclude stochastic papers
    python scripts/configure_stochastic_filter.py --enable --mode exclude
    
    # Enable filtering to include only stochastic papers
    python scripts/configure_stochastic_filter.py --enable --mode include
    
    # Disable filtering completely
    python scripts/configure_stochastic_filter.py --disable
    
    # Enable with custom settings
    python scripts/configure_stochastic_filter.py --enable --mode exclude \\
        --min-keywords 3 --confidence-threshold 0.5 \\
        --no-apply-to-manual
        """
    )
    
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                       help="Path to configuration file")
    
    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--enable", action="store_true",
                             help="Enable stochastic filtering")
    action_group.add_argument("--disable", action="store_true",
                             help="Disable stochastic filtering")
    action_group.add_argument("--status", action="store_true",
                             help="Show current filtering status")
    
    # Configuration options (only used with --enable)
    parser.add_argument("--mode", type=str, choices=["include", "exclude"], default="exclude",
                       help="Filter mode: 'include' only stochastic papers, 'exclude' stochastic papers")
    parser.add_argument("--min-keywords", type=int, default=2,
                       help="Minimum stochastic keywords required for classification")
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                       help="Confidence threshold for stochastic classification")
    
    # Fine-tuning specific options
    parser.add_argument("--apply-to-extracted", action="store_true", default=True,
                       help="Apply filtering to LLM-extracted data")
    parser.add_argument("--no-apply-to-extracted", dest="apply_to_extracted", action="store_false",
                       help="Don't apply filtering to LLM-extracted data")
    parser.add_argument("--apply-to-manual", action="store_true", default=False,
                       help="Apply filtering to manually created data")
    parser.add_argument("--no-apply-to-manual", dest="apply_to_manual", action="store_false",
                       help="Don't apply filtering to manually created data")
    parser.add_argument("--apply-to-synthetic", action="store_true", default=True,
                       help="Apply filtering to synthetically generated data")
    parser.add_argument("--no-apply-to-synthetic", dest="apply_to_synthetic", action="store_false",
                       help="Don't apply filtering to synthetically generated data")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    
    # Execute requested action
    if args.status:
        success = show_status(args.config)
    elif args.enable:
        success = enable_filtering(
            args.config, 
            args.mode, 
            args.min_keywords, 
            args.confidence_threshold,
            args.apply_to_extracted,
            args.apply_to_manual,
            args.apply_to_synthetic
        )
        if success:
            print("\n💡 Next steps:")
            print("   1. Rebuild knowledge base to apply filtering: scripts/knowledge_base/run_kb_builder.sh")
            print("   2. Regenerate fine-tuning data if needed")
    elif args.disable:
        success = disable_filtering(args.config)
        if success:
            print("\n💡 Filtering is now disabled. Rebuild knowledge base to include all papers.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 