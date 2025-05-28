#!/usr/bin/env python3
"""
Parameterized Experiment Runner for FMLLMSolver.

This script automates running a comprehensive set of experiments across multiple dimensions:
1. Different LLM models
2. Varied knowledge base configurations
3. Various RAG retrieval settings
4. Different system types/domains
5. Different verification methods

Results are logged in a structured format for detailed analysis.
"""

import os
import sys
import argparse
import logging
import time
import json
import subprocess
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parameterized_experiments.log"),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
dotenv_path = Path(".env")
if dotenv_path.exists():
    logger.info(f"Loading environment variables from {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.info("No .env file found. Using system environment variables.")

def create_experiment_directory(base_dir="experiments"):
    """Create a timestamped directory for experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(base_dir) / f"experiment_batch_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir

def create_model_variants(base_config, model_list):
    """Create variants of the configuration with different LLM models."""
    variants = []
    for model_name in model_list:
        # Create a deep copy of the config
        config_copy = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        # Update the model name
        config_copy.fine_tuning.base_model_name = model_name
        variants.append({
            "name": f"model_{model_name.replace('/', '_')}",
            "config": config_copy,
            "description": f"LLM model variant using {model_name}",
            "dimension": "model_architecture"
        })
    return variants

def create_knowledge_base_variants(base_config, embedding_models, k_values):
    """Create variants of the configuration with different knowledge base settings."""
    variants = []
    
    # Vary embedding models
    for embedding_model in embedding_models:
        config_copy = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        config_copy.knowledge_base.embedding_model_name = embedding_model
        config_copy.embeddings.model_name = embedding_model
        variants.append({
            "name": f"kb_embedding_{embedding_model.replace('/', '_')}",
            "config": config_copy,
            "description": f"Knowledge base using {embedding_model} embeddings",
            "dimension": "knowledge_base_embedding"
        })
    
    # Vary RAG retrieval count (k)
    for k in k_values:
        config_copy = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        config_copy.inference.rag_k = k
        config_copy.evaluation.rag_k = k
        variants.append({
            "name": f"rag_k_{k}",
            "config": config_copy,
            "description": f"RAG retrieval with k={k} context chunks",
            "dimension": "rag_retrieval_count"
        })
    
    return variants

def create_system_variants(base_config, system_subsets):
    """Create variants targeting different system subsets."""
    variants = []
    
    # Get the full benchmark file path
    benchmark_path = Path(base_config.paths.eval_benchmark_file)
    
    for subset_name, filter_func in system_subsets.items():
        # Create the subset benchmark file
        with open(benchmark_path, 'r') as f:
            all_systems = json.load(f)
        
        subset_systems = [system for system in all_systems if filter_func(system)]
        if not subset_systems:
            logger.warning(f"No systems matched criteria for subset '{subset_name}'")
            continue
            
        subset_path = benchmark_path.parent / f"benchmark_{subset_name}.json"
        with open(subset_path, 'w') as f:
            json.dump(subset_systems, f, indent=2)
        
        # Create config variant using this subset
        config_copy = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        config_copy.paths.eval_benchmark_file = str(subset_path)
        
        variants.append({
            "name": f"systems_{subset_name}",
            "config": config_copy,
            "description": f"System subset: {subset_name} ({len(subset_systems)} systems)",
            "dimension": "system_type",
            "system_count": len(subset_systems)
        })
    
    return variants

def create_verification_variants(base_config):
    """Create variants with different verification settings."""
    variants = []
    
    # Numerical sampling only
    num_only_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    num_only_config.evaluation.verification.attempt_sos = False
    num_only_config.evaluation.verification.attempt_optimization = True
    variants.append({
        "name": "verify_numerical_only",
        "config": num_only_config,
        "description": "Verification using numerical sampling only",
        "dimension": "verification_method"
    })
    
    # Numerical + optimization
    num_opt_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    num_opt_config.evaluation.verification.attempt_sos = False
    num_opt_config.evaluation.verification.attempt_optimization = True
    num_opt_config.evaluation.verification.optimization_max_iter = 200  # Increase iterations
    variants.append({
        "name": "verify_numerical_optimization",
        "config": num_opt_config,
        "description": "Verification using numerical sampling + intensive optimization",
        "dimension": "verification_method"
    })
    
    # Full verification (SOS + numerical + optimization)
    full_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    full_config.evaluation.verification.attempt_sos = True
    full_config.evaluation.verification.attempt_optimization = True
    variants.append({
        "name": "verify_full",
        "config": full_config,
        "description": "Full verification (SOS when applicable + numerical + optimization)",
        "dimension": "verification_method"
    })
    
    return variants

def define_system_filters():
    """Define filter functions for system subsets."""
    return {
        "2d_systems": lambda sys: len(sys.get("state_variables", [])) == 2,
        "3d_plus_systems": lambda sys: len(sys.get("state_variables", [])) >= 3,
        "linear": lambda sys: all("**" not in dyn and "sin" not in dyn and "cos" not in dyn 
                               for dyn in sys.get("dynamics", [])),
        "nonlinear": lambda sys: any("**" in dyn or "sin" in dyn or "cos" in dyn 
                                  for dyn in sys.get("dynamics", [])),
        "polynomial": lambda sys: any("**" in dyn for dyn in sys.get("dynamics", [])) 
                               and all("sin" not in dyn and "cos" not in dyn for dyn in sys.get("dynamics", [])),
        "trigonometric": lambda sys: any("sin" in dyn or "cos" in dyn for dyn in sys.get("dynamics", [])),
        "mechanical": lambda sys: "mechanical" in sys.get("description", "").lower() 
                              or "pendulum" in sys.get("description", "").lower(),
        "biological": lambda sys: "biological" in sys.get("description", "").lower() 
                               or "predator" in sys.get("description", "").lower() 
                               or "prey" in sys.get("description", "").lower(),
        "electrical": lambda sys: "electrical" in sys.get("description", "").lower() 
                               or "circuit" in sys.get("description", "").lower(),
        "chemical": lambda sys: "chemical" in sys.get("description", "").lower() 
                              or "reaction" in sys.get("description", "").lower()
    }

def run_single_experiment(experiment_name, config, experiment_dir, skip_steps=None):
    """
    Run a single experiment with the given configuration.
    
    Args:
        experiment_name: Name for this experiment
        config: The configuration to use
        experiment_dir: Directory to save results
        skip_steps: List of steps to skip (e.g., ["data_fetching", "kb_building"])
    
    Returns:
        dict: Results summary
    """
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Create experiment subdirectory
    exp_subdir = experiment_dir / experiment_name
    exp_subdir.mkdir(exist_ok=True)
    
    # Save the configuration
    config_path = exp_subdir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    
    # Update results path to save in the experiment directory
    config.paths.eval_results_file = str(exp_subdir / "results.csv")
    
    # Save updated config
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    
    # Prepare command
    cmd = [sys.executable, "run_experiments.py", "--config", str(config_path)]
    
    # Add skip arguments
    if skip_steps:
        if "data_fetching" in skip_steps:
            cmd.append("--skip-data-fetching")
        if "kb_building" in skip_steps:
            cmd.append("--skip-kb-building")
        if "data_prep" in skip_steps:
            cmd.append("--skip-data-prep")
        if "finetuning" in skip_steps:
            cmd.append("--skip-finetuning")
    
    # Run the experiment
    logger.info(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr
        success = True
    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr
        success = False
        logger.error(f"Experiment failed with exit code {e.returncode}")
    
    duration = time.time() - start_time
    
    # Save stdout and stderr
    with open(exp_subdir / "stdout.log", "w") as f:
        f.write(stdout)
    with open(exp_subdir / "stderr.log", "w") as f:
        f.write(stderr)
    
    # Extract results summary from stdout if available
    summary = {
        "experiment_name": experiment_name,
        "success": success,
        "duration_seconds": duration
    }
    
    # Try to parse the results CSV for more details
    results_csv_path = exp_subdir / "results.csv"
    if results_csv_path.exists():
        try:
            results_df = pd.read_csv(results_csv_path)
            summary["total_systems"] = len(results_df)
            summary["generation_successful"] = results_df["generation_successful"].sum()
            summary["parsing_successful"] = results_df["parsing_successful"].sum()
            summary["final_verdict_passed"] = results_df[results_df["final_verdict"].str.contains("Passed")].shape[0]
            summary["numerical_passed"] = results_df[results_df["final_verdict"] == "Passed Numerical Checks"].shape[0]
            summary["symbolic_passed"] = results_df[results_df["final_verdict"] == "Passed Symbolic Checks (Basic)"].shape[0]
            summary["sos_passed"] = results_df[results_df["final_verdict"] == "Passed SOS Checks"].shape[0]
            summary["avg_duration"] = results_df["duration_seconds"].mean()
        except Exception as e:
            logger.error(f"Error parsing results CSV: {e}")
    
    # Save summary
    with open(exp_subdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment {experiment_name} completed in {duration:.2f} seconds")
    return summary

def generate_experiment_combinations(base_config, dimensions, experiment_dir, args):
    """
    Generate experiment configurations for all combinations of specified dimensions.
    
    Args:
        base_config: Base configuration
        dimensions: Dict of dimension names to lists of dimension variants
        experiment_dir: Directory to save results
        args: Command line arguments
    
    Returns:
        list: Experiment configurations to run
    """
    # Get the requested dimensions from args
    requested_dimensions = args.dimensions.split(",") if args.dimensions else []
    
    # If specific dimensions requested, filter to only those
    if requested_dimensions:
        dimensions = {k: v for k, v in dimensions.items() if k in requested_dimensions}
    
    # Generate all combinations of the dimensions
    dimension_keys = list(dimensions.keys())
    dimension_values = [dimensions[k] for k in dimension_keys]
    
    all_experiments = []
    
    if args.no_combinations:
        # Run each dimension independently (no cross-product)
        for dim_key, dim_variants in dimensions.items():
            for variant in dim_variants:
                variant_copy = variant.copy()
                variant_copy["name"] = f"{dim_key}_{variant['name']}"
                all_experiments.append(variant_copy)
    else:
        # Generate all combinations
        for combination in itertools.product(*dimension_values):
            # Start with the base config
            combined_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
            
            # Apply each variant's configuration changes
            experiment_name_parts = []
            experiment_dimensions = []
            
            for variant in combination:
                # Apply this variant's config changes
                combined_config = OmegaConf.merge(combined_config, variant["config"])
                experiment_name_parts.append(variant["name"])
                experiment_dimensions.append(variant["dimension"])
            
            # Create a name for this experiment combination
            experiment_name = "__".join(experiment_name_parts)
            
            # Ensure no name collisions if the experiment name is too long
            if len(experiment_name) > 100:
                import hashlib
                name_hash = hashlib.md5(experiment_name.encode()).hexdigest()[:8]
                experiment_name = f"combined_experiment_{name_hash}"
            
            all_experiments.append({
                "name": experiment_name,
                "config": combined_config,
                "description": f"Combined experiment across dimensions: {', '.join(experiment_dimensions)}",
                "dimensions": experiment_dimensions
            })
    
    # If a limit is specified, sample that many experiments
    if args.limit and len(all_experiments) > args.limit:
        if args.random_sample:
            import random
            all_experiments = random.sample(all_experiments, args.limit)
        else:
            all_experiments = all_experiments[:args.limit]
    
    logger.info(f"Generated {len(all_experiments)} experiment configurations")
    return all_experiments

def analyze_experiment_results(experiment_dir):
    """
    Analyze all experiment results and generate summary reports.
    
    Args:
        experiment_dir: Directory containing experiment results
    """
    logger.info("Analyzing experiment results...")
    
    # Find all summary.json files
    summary_files = list(experiment_dir.glob("*/summary.json"))
    if not summary_files:
        logger.warning("No experiment summaries found to analyze")
        return
    
    # Load all summaries
    all_summaries = []
    for summary_file in summary_files:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            # Add the experiment directory name
            summary["experiment_dir"] = summary_file.parent.name
            all_summaries.append(summary)
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save the full summary
    summary_df.to_csv(experiment_dir / "all_experiments_summary.csv", index=False)
    
    # Generate visualizations
    generate_result_visualizations(summary_df, experiment_dir)
    
    # Generate text report
    generate_text_report(summary_df, experiment_dir)

def generate_result_visualizations(summary_df, experiment_dir):
    """Generate visualizations from experiment results."""
    plt.figure(figsize=(12, 8))
    
    # Success Rates by Experiment
    if "total_systems" in summary_df.columns and "final_verdict_passed" in summary_df.columns:
        summary_df["success_rate"] = summary_df["final_verdict_passed"] / summary_df["total_systems"] * 100
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x="experiment_name", y="success_rate", data=summary_df)
        plt.title("Success Rate by Experiment")
        plt.xlabel("Experiment")
        plt.ylabel("Success Rate (%)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(experiment_dir / "success_rate_by_experiment.png")
        
    # Distribution of Verification Methods
    if all(col in summary_df.columns for col in ["numerical_passed", "symbolic_passed", "sos_passed"]):
        plt.figure(figsize=(10, 6))
        verification_data = summary_df[["numerical_passed", "symbolic_passed", "sos_passed"]].sum()
        verification_data.plot(kind="bar")
        plt.title("Distribution of Successful Verification Methods")
        plt.xlabel("Verification Method")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(experiment_dir / "verification_methods_distribution.png")
    
    # Average Duration by Experiment
    if "avg_duration" in summary_df.columns:
        plt.figure(figsize=(12, 8))
        sns.barplot(x="experiment_name", y="avg_duration", data=summary_df)
        plt.title("Average Processing Duration by Experiment")
        plt.xlabel("Experiment")
        plt.ylabel("Average Duration (seconds)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(experiment_dir / "avg_duration_by_experiment.png")
    
    # Try to extract dimension information for grouped analysis
    if "experiment_name" in summary_df.columns:
        # Extract dimension from experiment name if possible
        try:
            # This assumes experiment names follow dimension_value format
            summary_df["dimension"] = summary_df["experiment_name"].apply(
                lambda x: x.split("__")[0].split("_")[0] if "__" in x else x.split("_")[0]
            )
            
            # Success Rate by Dimension
            if "success_rate" in summary_df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x="dimension", y="success_rate", data=summary_df)
                plt.title("Success Rate Distribution by Dimension")
                plt.xlabel("Dimension")
                plt.ylabel("Success Rate (%)")
                plt.tight_layout()
                plt.savefig(experiment_dir / "success_rate_by_dimension.png")
        except Exception as e:
            logger.warning(f"Could not analyze by dimension: {e}")

def generate_text_report(summary_df, experiment_dir):
    """Generate a text summary report of experiment results."""
    report_lines = []
    report_lines.append("# Experiment Results Summary")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Experiments: {len(summary_df)}")
    report_lines.append("")
    
    # Overall statistics
    if "success" in summary_df.columns:
        successful_exps = summary_df["success"].sum()
        report_lines.append(f"Successful Experiments: {successful_exps}/{len(summary_df)} ({successful_exps/len(summary_df)*100:.1f}%)")
    
    if "total_systems" in summary_df.columns and "final_verdict_passed" in summary_df.columns:
        total_systems = summary_df["total_systems"].sum()
        passed_systems = summary_df["final_verdict_passed"].sum()
        if total_systems > 0:
            report_lines.append(f"Overall Success Rate: {passed_systems}/{total_systems} ({passed_systems/total_systems*100:.1f}%)")
    
    report_lines.append("")
    report_lines.append("## Top Performing Experiments")
    
    # Top experiments by success rate
    if "success_rate" in summary_df.columns:
        top_experiments = summary_df.sort_values("success_rate", ascending=False).head(5)
        report_lines.append("\n### By Success Rate")
        for _, row in top_experiments.iterrows():
            report_lines.append(f"- {row['experiment_name']}: {row['success_rate']:.1f}% ({row['final_verdict_passed']}/{row['total_systems']})")
    
    # Breakdown by verification method
    report_lines.append("\n## Verification Method Breakdown")
    if all(col in summary_df.columns for col in ["numerical_passed", "symbolic_passed", "sos_passed"]):
        total_passed = summary_df["final_verdict_passed"].sum()
        numerical = summary_df["numerical_passed"].sum()
        symbolic = summary_df["symbolic_passed"].sum()
        sos = summary_df["sos_passed"].sum()
        
        if total_passed > 0:
            report_lines.append(f"- Numerical Verification: {numerical}/{total_passed} ({numerical/total_passed*100:.1f}%)")
            report_lines.append(f"- Symbolic Verification: {symbolic}/{total_passed} ({symbolic/total_passed*100:.1f}%)")
            report_lines.append(f"- SOS Verification: {sos}/{total_passed} ({sos/total_passed*100:.1f}%)")
    
    # Write the report
    with open(experiment_dir / "experiment_summary_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Text report generated: {experiment_dir / 'experiment_summary_report.md'}")

def main():
    parser = argparse.ArgumentParser(description="Parameterized experiment runner for FMLLMSolver")
    
    # Main options
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, 
                        help="Path to base configuration file")
    parser.add_argument("--dimensions", type=str, default="",
                        help="Comma-separated list of dimensions to vary (e.g., 'model,rag,system')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of experiments to run")
    parser.add_argument("--random-sample", action="store_true",
                        help="Randomly sample experiments if using --limit")
    parser.add_argument("--no-combinations", action="store_true",
                        help="Run each dimension independently (no cross-product)")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Base directory for experiment results")
    
    # Skip options
    parser.add_argument("--skip-data-fetching", action="store_true",
                        help="Skip data fetching for all experiments")
    parser.add_argument("--skip-kb-building", action="store_true", 
                        help="Skip knowledge base building for all experiments")
    parser.add_argument("--skip-finetuning", action="store_true",
                        help="Skip model fine-tuning for all experiments")
    
    # Model options
    parser.add_argument("--models", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Comma-separated list of models to test")
    
    # Knowledge base options
    parser.add_argument("--embedding-models", type=str, default="all-mpnet-base-v2",
                        help="Comma-separated list of embedding models to test")
    parser.add_argument("--rag-k-values", type=str, default="3,5,7",
                        help="Comma-separated list of RAG k values to test")
    
    args = parser.parse_args()
    
    # Load base configuration
    logger.info(f"Loading base configuration from: {args.config}")
    base_config = load_config(args.config)
    if not base_config:
        logger.error("Failed to load base configuration")
        return
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(args.output_dir)
    
    # Define experiment dimensions
    model_list = args.models.split(",")
    embedding_models = args.embedding_models.split(",")
    k_values = [int(k) for k in args.rag_k_values.split(",")]
    
    # Create dimension variants
    dimensions = {
        "model": create_model_variants(base_config, model_list),
        "knowledge_base": create_knowledge_base_variants(base_config, embedding_models, k_values),
        "system": create_system_variants(base_config, define_system_filters()),
        "verification": create_verification_variants(base_config)
    }
    
    # Generate experiment combinations
    experiments = generate_experiment_combinations(base_config, dimensions, experiment_dir, args)
    
    # Determine steps to skip
    skip_steps = []
    if args.skip_data_fetching:
        skip_steps.append("data_fetching")
    if args.skip_kb_building:
        skip_steps.append("kb_building")
    if args.skip_finetuning:
        skip_steps.append("finetuning")
    
    # Save experiment plan
    experiment_plan = {
        "timestamp": datetime.now().isoformat(),
        "base_config": args.config,
        "dimensions": {k: [v["name"] for v in variants] for k, variants in dimensions.items()},
        "experiments": [{"name": exp["name"], "dimensions": exp.get("dimensions", [])} for exp in experiments],
        "skip_steps": skip_steps
    }
    with open(experiment_dir / "experiment_plan.json", "w") as f:
        json.dump(experiment_plan, f, indent=2)
    
    # Run experiments
    results = []
    total_experiments = len(experiments)
    for i, experiment in enumerate(experiments):
        logger.info(f"Running experiment {i+1}/{total_experiments}: {experiment['name']}")
        result = run_single_experiment(
            experiment["name"], 
            experiment["config"], 
            experiment_dir,
            skip_steps
        )
        results.append(result)
    
    # Analyze results
    analyze_experiment_results(experiment_dir)
    
    logger.info(f"All experiments completed. Results available in: {experiment_dir}")
    logger.info(f"Summary report: {experiment_dir / 'experiment_summary_report.md'}")

if __name__ == "__main__":
    main() 