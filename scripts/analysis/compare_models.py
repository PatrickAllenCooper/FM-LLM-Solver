#!/usr/bin/env python
# compare_models.py - Compare base model with fine-tuned model using the same benchmark

import os
import sys
import json
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
from evaluation.evaluate_pipeline import evaluate_pipeline
from omegaconf import OmegaConf

# Configure logging with file output
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_comparison_{timestamp}.log")
    
    # Set up file handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Detailed logs will be saved to: {log_file}")
    return log_file

def setup_base_model_config(cfg):
    """Create a copy of the config with base model settings (no fine-tuning, no RAG)."""
    # Create a deep copy of the config
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Modify config to use base model without fine-tuning
    base_cfg.paths.eval_results_file = os.path.join(base_cfg.paths.output_dir, "evaluation_results_base_model.csv")
    
    # Set RAG k to 0 to disable RAG
    base_cfg.evaluation.rag_k = 0
    base_cfg.inference.rag_k = 0
    
    # Remove adapter path so it uses only the base model
    base_cfg.fine_tuning.use_adapter = False
    
    # Return the modified config
    return base_cfg

def setup_finetuned_model_config(cfg):
    """Create a copy of the config for the fine-tuned model with RAG."""
    # Create a deep copy of the config
    ft_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Set results path 
    ft_cfg.paths.eval_results_file = os.path.join(ft_cfg.paths.output_dir, "evaluation_results_finetuned_model.csv")
    
    # Ensure adapter is used
    ft_cfg.fine_tuning.use_adapter = True
    
    # Return the modified config
    return ft_cfg

def generate_comparison_report(base_results_path, ft_results_path, output_path, model_name_prefix=""):
    """Generate a comparison report between base and fine-tuned model results."""
    try:
        # Load results from CSV files
        base_df = pd.read_csv(base_results_path)
        ft_df = pd.read_csv(ft_results_path)
        
        logging.info(f"Loaded base model results from {base_results_path} ({len(base_df)} systems)")
        logging.info(f"Loaded fine-tuned model results from {ft_results_path} ({len(ft_df)} systems)")
        
        # Add model column with optional prefix
        base_df['model'] = f'{model_name_prefix}Base Model'
        ft_df['model'] = f'{model_name_prefix}Fine-tuned Model with RAG'
        
        # Merge dataframes
        combined_df = pd.concat([base_df, ft_df], ignore_index=True)
        
        # Calculate summary statistics
        base_summary = {
            'total_systems': len(base_df),
            'generation_success': base_df['generation_successful'].sum(),
            'parsing_success': base_df['parsing_successful'].sum(),
            'numerical_pass': sum(base_df['final_verdict'] == 'Passed Numerical Checks'),
            'symbolic_pass': sum(base_df['final_verdict'] == 'Passed Symbolic Checks (Basic)') + 
                            sum(base_df['final_verdict'] == 'Passed SOS Checks'),
            'sos_pass': sum(base_df['final_verdict'] == 'Passed SOS Checks'),
            'overall_pass': sum((base_df['final_verdict'] == 'Passed Numerical Checks') | 
                               (base_df['final_verdict'] == 'Passed SOS Checks')),
            'avg_duration': base_df['duration_seconds'].mean()
        }
        
        ft_summary = {
            'total_systems': len(ft_df),
            'generation_success': ft_df['generation_successful'].sum(),
            'parsing_success': ft_df['parsing_successful'].sum(),
            'numerical_pass': sum(ft_df['final_verdict'] == 'Passed Numerical Checks'),
            'symbolic_pass': sum(ft_df['final_verdict'] == 'Passed Symbolic Checks (Basic)') + 
                           sum(ft_df['final_verdict'] == 'Passed SOS Checks'),
            'sos_pass': sum(ft_df['final_verdict'] == 'Passed SOS Checks'),
            'overall_pass': sum((ft_df['final_verdict'] == 'Passed Numerical Checks') | 
                              (ft_df['final_verdict'] == 'Passed SOS Checks')),
            'avg_duration': ft_df['duration_seconds'].mean()
        }
        
        # Log detailed summary statistics
        logging.info("===== DETAILED COMPARISON SUMMARY =====")
        logging.info(f"Total Systems: {base_summary['total_systems']}")
        
        metrics = [
            ('Generation Success', 'generation_success'),
            ('Parsing Success', 'parsing_success'),
            ('Numerical Checks Passed', 'numerical_pass'),
            ('Symbolic Checks Passed', 'symbolic_pass'),
            ('SOS Checks Passed', 'sos_pass'),
            ('Overall Pass Rate', 'overall_pass'),
            ('Average Processing Time (s)', 'avg_duration')
        ]
        
        for label, key in metrics:
            if key == 'avg_duration':
                base_val = base_summary[key]
                ft_val = ft_summary[key]
                diff = ft_val - base_val
                logging.info(f"{label}: Base={base_val:.2f}s, Fine-tuned={ft_val:.2f}s, Diff={diff:.2f}s")
            else:
                base_val = base_summary[key]
                base_pct = (base_val / base_summary['total_systems']) * 100
                ft_val = ft_summary[key]
                ft_pct = (ft_val / ft_summary['total_systems']) * 100
                diff = ft_val - base_val
                diff_pct = ft_pct - base_pct
                logging.info(f"{label}: Base={base_val}/{base_summary['total_systems']} ({base_pct:.1f}%), Fine-tuned={ft_val}/{ft_summary['total_systems']} ({ft_pct:.1f}%), Improvement={diff} ({diff_pct:.1f}%)")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Metric': [
                'Total Systems Evaluated',
                'Successful Generations',
                'Successfully Parsed Certificates',
                'Passed Numerical Checks',
                'Passed Symbolic Checks (Any)',
                'Passed SOS Checks',
                'Overall Success Rate',
                'Average Processing Time (s)'
            ],
            'Base Model': [
                base_summary['total_systems'],
                f"{base_summary['generation_success']} ({base_summary['generation_success']/base_summary['total_systems']*100:.1f}%)",
                f"{base_summary['parsing_success']} ({base_summary['parsing_success']/base_summary['total_systems']*100:.1f}%)",
                f"{base_summary['numerical_pass']} ({base_summary['numerical_pass']/base_summary['total_systems']*100:.1f}%)",
                f"{base_summary['symbolic_pass']} ({base_summary['symbolic_pass']/base_summary['total_systems']*100:.1f}%)",
                f"{base_summary['sos_pass']} ({base_summary['sos_pass']/base_summary['total_systems']*100:.1f}%)",
                f"{base_summary['overall_pass']} ({base_summary['overall_pass']/base_summary['total_systems']*100:.1f}%)",
                f"{base_summary['avg_duration']:.1f}"
            ],
            'Fine-tuned Model with RAG': [
                ft_summary['total_systems'],
                f"{ft_summary['generation_success']} ({ft_summary['generation_success']/ft_summary['total_systems']*100:.1f}%)",
                f"{ft_summary['parsing_success']} ({ft_summary['parsing_success']/ft_summary['total_systems']*100:.1f}%)",
                f"{ft_summary['numerical_pass']} ({ft_summary['numerical_pass']/ft_summary['total_systems']*100:.1f}%)",
                f"{ft_summary['symbolic_pass']} ({ft_summary['symbolic_pass']/ft_summary['total_systems']*100:.1f}%)",
                f"{ft_summary['sos_pass']} ({ft_summary['sos_pass']/ft_summary['total_systems']*100:.1f}%)",
                f"{ft_summary['overall_pass']} ({ft_summary['overall_pass']/ft_summary['total_systems']*100:.1f}%)",
                f"{ft_summary['avg_duration']:.1f}"
            ],
            'Improvement': [
                "N/A",
                f"{ft_summary['generation_success'] - base_summary['generation_success']} ({(ft_summary['generation_success']/ft_summary['total_systems'] - base_summary['generation_success']/base_summary['total_systems'])*100:.1f}%)",
                f"{ft_summary['parsing_success'] - base_summary['parsing_success']} ({(ft_summary['parsing_success']/ft_summary['total_systems'] - base_summary['parsing_success']/base_summary['total_systems'])*100:.1f}%)",
                f"{ft_summary['numerical_pass'] - base_summary['numerical_pass']} ({(ft_summary['numerical_pass']/ft_summary['total_systems'] - base_summary['numerical_pass']/base_summary['total_systems'])*100:.1f}%)",
                f"{ft_summary['symbolic_pass'] - base_summary['symbolic_pass']} ({(ft_summary['symbolic_pass']/ft_summary['total_systems'] - base_summary['symbolic_pass']/base_summary['total_systems'])*100:.1f}%)",
                f"{ft_summary['sos_pass'] - base_summary['sos_pass']} ({(ft_summary['sos_pass']/ft_summary['total_systems'] - base_summary['sos_pass']/base_summary['total_systems'])*100:.1f}%)",
                f"{ft_summary['overall_pass'] - base_summary['overall_pass']} ({(ft_summary['overall_pass']/ft_summary['total_systems'] - base_summary['overall_pass']/base_summary['total_systems'])*100:.1f}%)",
                f"{ft_summary['avg_duration'] - base_summary['avg_duration']:.1f}"
            ]
        })
        
        # Save comparison to CSV
        comparison_df.to_csv(output_path, index=False)
        
        # Create a detailed comparison of each system
        system_comparison = []
        
        logging.info("\n===== SYSTEM-BY-SYSTEM COMPARISON =====")
        
        for system_id in base_df['system_id'].unique():
            base_system = base_df[base_df['system_id'] == system_id].iloc[0]
            ft_system = ft_df[ft_df['system_id'] == system_id].iloc[0]
            
            # Check improvement
            base_verdict = base_system['final_verdict']
            ft_verdict = ft_system['final_verdict']
            if base_verdict == ft_verdict:
                verdict_change = "No Change"
            elif base_verdict in ["Passed Numerical Checks", "Passed SOS Checks"] and ft_verdict not in ["Passed Numerical Checks", "Passed SOS Checks"]:
                verdict_change = "Regression"
            elif base_verdict not in ["Passed Numerical Checks", "Passed SOS Checks"] and ft_verdict in ["Passed Numerical Checks", "Passed SOS Checks"]:
                verdict_change = "Improvement"
            else:
                verdict_change = "Different Failure Mode"
            
            # Log detailed system comparison
            logging.info(f"\nSystem: {system_id}")
            logging.info(f"  Base Model Verdict: {base_verdict}")
            logging.info(f"  Fine-tuned Model Verdict: {ft_verdict}")
            logging.info(f"  Outcome: {verdict_change}")
            logging.info(f"  Base Certificate: {base_system['parsed_certificate']}")
            logging.info(f"  Fine-tuned Certificate: {ft_system['parsed_certificate']}")
            
            system_comparison.append({
                'system_id': system_id,
                'base_certificate': base_system['parsed_certificate'],
                'ft_certificate': ft_system['parsed_certificate'],
                'base_verdict': base_verdict,
                'ft_verdict': ft_verdict,
                'verdict_change': verdict_change
            })
        
        system_df = pd.DataFrame(system_comparison)
        system_df.to_csv(os.path.join(os.path.dirname(output_path), "system_level_comparison.csv"), index=False)
        
        # Generate visualizations
        plt.figure(figsize=(12, 8))
        
        # Success rates bar chart
        metrics = ['generation_successful', 'parsing_successful', 'final_verdict']
        labels = ['Generation Success', 'Parsing Success', 'Verification Success']
        
        # Convert 'final_verdict' to boolean success
        combined_df['verification_success'] = combined_df['final_verdict'].apply(
            lambda x: 1 if x in ["Passed Numerical Checks", "Passed SOS Checks"] else 0
        )
        metrics[-1] = 'verification_success'
        
        success_data = []
        for metric, label in zip(metrics, labels):
            for model in combined_df['model'].unique():
                model_df = combined_df[combined_df['model'] == model]
                success_rate = model_df[metric].mean() * 100
                success_data.append({
                    'Metric': label,
                    'Model': model,
                    'Success Rate (%)': success_rate
                })
        
        success_df = pd.DataFrame(success_data)
        
        plt.subplot(2, 1, 1)
        sns.barplot(x='Metric', y='Success Rate (%)', hue='Model', data=success_df)
        plt.title('Model Performance Comparison')
        plt.ylim(0, 100)
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=0)
        plt.legend(loc='best')
        
        # System-level outcomes
        plt.subplot(2, 1, 2)
        outcome_counts = pd.DataFrame({
            'Status': ['Improved', 'No Change', 'Regression'],
            'Count': [
                sum(system_df['verdict_change'] == 'Improvement'),
                sum(system_df['verdict_change'] == 'No Change'),
                sum(system_df['verdict_change'] == 'Regression')
            ]
        })
        
        # Log outcome counts
        logging.info("\n===== OUTCOME SUMMARY =====")
        logging.info(f"Systems Improved: {sum(system_df['verdict_change'] == 'Improvement')}")
        logging.info(f"Systems with No Change: {sum(system_df['verdict_change'] == 'No Change')}")
        logging.info(f"Systems Regressed: {sum(system_df['verdict_change'] == 'Regression')}")
        logging.info(f"Systems with Different Failure Mode: {sum(system_df['verdict_change'] == 'Different Failure Mode')}")
        
        sns.barplot(x='Status', y='Count', data=outcome_counts, palette=['green', 'gray', 'red'])
        plt.title('System-Level Outcome Changes')
        plt.ylabel('Number of Systems')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(output_path), "model_comparison_charts.png"), dpi=300)
        
        logging.info(f"Comparison report generated at {output_path}")
        logging.info(f"System-level comparison saved to {os.path.join(os.path.dirname(output_path), 'system_level_comparison.csv')}")
        logging.info(f"Visualizations saved to {os.path.join(os.path.dirname(output_path), 'model_comparison_charts.png')}")
        
    except Exception as e:
        logging.error(f"Error generating comparison report: {e}", exc_info=True)
        raise
        
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare base model with fine-tuned model on barrier certificate generation.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration YAML file.")
    parser.add_argument("--base-only", action="store_true", help="Run only the base model evaluation.")
    parser.add_argument("--ft-only", action="store_true", help="Run only the fine-tuned model evaluation.")
    parser.add_argument("--report-only", action="store_true", help="Generate comparison report only (skip evaluations).")
    parser.add_argument("--log-dir", type=str, default="output/logs", help="Directory to save detailed log files.")
    parser.add_argument("--qwen15b", action="store_true", help="Run comparison with Qwen 15B model (separate output directory).")
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    
    # Log run start with timestamp
    start_time = datetime.now()
    logging.info(f"===== MODEL COMPARISON STARTED AT {start_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    logging.info(f"Command arguments: {args}")
    
    # Get the Python executable path
    python_path = sys.executable
    if not python_path:
        logging.error("Could not determine Python executable path.")
        return
    
    logging.info(f"Using Python executable: {python_path}")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Log configuration details
    logging.info("===== CONFIGURATION DETAILS =====")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Base model: {cfg.fine_tuning.base_model_name}")
    logging.info(f"Adapter path: {os.path.join(cfg.paths.ft_output_dir, 'final_adapter')}")
    logging.info(f"Knowledge base directory: {cfg.paths.kb_output_dir}")
    logging.info(f"Vector store: {cfg.paths.kb_vector_store_filename}")
    logging.info(f"Metadata file: {cfg.paths.kb_metadata_filename}")
    logging.info(f"Embedding model: {cfg.knowledge_base.embedding_model_name}")
    logging.info(f"RAG k: {cfg.inference.rag_k}")
    logging.info(f"Evaluation config: {OmegaConf.to_yaml(cfg.evaluation)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    # Set output directory based on model version
    if args.qwen15b:
        comparison_dir = os.path.join(cfg.paths.output_dir, "model_comparison_qwen15b")
        model_name_prefix = "Qwen 15B "
    else:
        comparison_dir = os.path.join(cfg.paths.output_dir, "model_comparison")
        model_name_prefix = ""
    
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_path = os.path.join(comparison_dir, "evaluation_results_base_model.csv")
    ft_results_path = os.path.join(comparison_dir, "evaluation_results_finetuned_model.csv")
    comparison_output_path = os.path.join(comparison_dir, f"model_comparison_report_{timestamp}.csv")
    
    # Run evaluations
    if not args.report_only:
        # Modify config for base model (no fine-tuning, no RAG)
        if not args.ft_only:
            logging.info(f"===== RUNNING {model_name_prefix}BASE MODEL EVALUATION =====")
            base_cfg = setup_base_model_config(cfg)
            # Override the default path with our custom output path
            base_cfg.paths.eval_results_file = base_results_path
            logging.debug(f"Base model config: {OmegaConf.to_yaml(base_cfg)}")
            evaluate_pipeline(base_cfg)
            logging.info(f"Base model evaluation complete. Results saved to {base_results_path}")
        
        # Run evaluation with fine-tuned model and RAG
        if not args.base_only:
            logging.info(f"===== RUNNING {model_name_prefix}FINE-TUNED MODEL EVALUATION =====")
            ft_cfg = setup_finetuned_model_config(cfg)
            # Override the default path with our custom output path
            ft_cfg.paths.eval_results_file = ft_results_path
            logging.debug(f"Fine-tuned model config: {OmegaConf.to_yaml(ft_cfg)}")
            evaluate_pipeline(ft_cfg)
            logging.info(f"Fine-tuned model evaluation complete. Results saved to {ft_results_path}")
    
    # Generate comparison report
    logging.info("===== GENERATING COMPARISON REPORT =====")
    generate_comparison_report(base_results_path, ft_results_path, comparison_output_path, model_name_prefix)
    
    # Calculate and log total run time
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"===== MODEL COMPARISON COMPLETED AT {end_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    logging.info(f"Total run time: {duration}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Summary report: {comparison_output_path}")
    logging.info(f"Charts: {os.path.join(comparison_dir, 'model_comparison_charts.png')}")
    logging.info(f"System-level comparison: {os.path.join(comparison_dir, 'system_level_comparison.csv')}")

if __name__ == "__main__":
    main() 