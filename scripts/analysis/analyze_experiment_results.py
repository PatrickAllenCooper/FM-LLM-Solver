#!/usr/bin/env python3
"""
Advanced Experiment Results Analyzer for FMLLMSolver

This script provides in-depth analysis of parameterized experiment results:
1. Detailed success rates across multiple dimensions
2. Certificate complexity analysis
3. Cross-dimensional correlations 
4. Verification method effectiveness
5. LLM performance patterns
6. System type performance analysis

Usage:
    python analyze_experiment_results.py --experiment-dir /path/to/experiment_batch_dir
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
from sklearn.metrics import confusion_matrix
import networkx as nx
from scipy.stats import pearsonr, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def load_experiment_data(experiment_dir):
    """Load and combine all experiment result data."""
    experiment_dir = Path(experiment_dir)
    
    # Load experiment plan
    plan_path = experiment_dir / "experiment_plan.json"
    if not plan_path.exists():
        logger.warning(f"Experiment plan not found at {plan_path}")
        experiment_plan = None
    else:
        with open(plan_path, 'r') as f:
            experiment_plan = json.load(f)
    
    # Load experiment summaries
    summary_path = experiment_dir / "all_experiments_summary.csv"
    if summary_path.exists():
        summaries_df = pd.read_csv(summary_path)
    else:
        # If summary CSV doesn't exist, generate it from individual summaries
        summary_files = list(experiment_dir.glob("*/summary.json"))
        if not summary_files:
            logger.error(f"No experiment summaries found in {experiment_dir}")
            return None, None, None
        
        summaries = []
        for summary_file in summary_files:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                summary["experiment_dir"] = summary_file.parent.name
                summaries.append(summary)
        
        summaries_df = pd.DataFrame(summaries)
    
    # Load individual result CSVs and combine
    result_files = list(experiment_dir.glob("*/results.csv"))
    result_dfs = []
    
    for result_file in result_files:
        try:
            exp_name = result_file.parent.name
            df = pd.read_csv(result_file)
            df["experiment_name"] = exp_name
            result_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {result_file}: {e}")
    
    if not result_dfs:
        logger.error("No result CSV files found")
        return summaries_df, None, experiment_plan
    
    # Combine all result dataframes
    combined_results_df = pd.concat(result_dfs, ignore_index=True)
    
    return summaries_df, combined_results_df, experiment_plan

def analyze_certificate_complexity(results_df):
    """Analyze complexity of generated barrier certificates."""
    if "parsed_certificate" not in results_df.columns:
        logger.warning("No certificate data found for complexity analysis")
        return None
    
    # Add complexity metrics
    results_df = results_df.copy()
    # Filter out failed parses or empty certificates
    valid_certs = results_df[results_df["parsed_certificate"].notna() & 
                           (results_df["parsed_certificate"] != "")]
    
    if len(valid_certs) == 0:
        logger.warning("No valid certificates found for complexity analysis")
        return None
    
    # Compute complexity metrics
    complexity_data = []
    for _, row in valid_certs.iterrows():
        cert = row["parsed_certificate"]
        
        # Basic length
        length = len(cert)
        
        # Count operators
        operators = len(re.findall(r'[+\-*/]', cert))
        
        # Count exponents
        exponents = len(re.findall(r'\*\*', cert))
        
        # Highest degree (approximately)
        highest_degree = 1
        exp_matches = re.findall(r'\*\*(\d+)', cert)
        if exp_matches:
            highest_degree = max([int(d) for d in exp_matches])
        
        # Check if certificate is quadratic (has x**2, y**2 terms)
        is_quadratic = bool(re.search(r'x\*\*2|y\*\*2|z\*\*2', cert))
        
        # Check if certificate is linear
        is_linear = not is_quadratic and not exponents and operators > 0
        
        # Success (verification passed)
        is_successful = "Passed" in row["final_verdict"]
        
        complexity_data.append({
            "certificate": cert,
            "length": length,
            "operators": operators,
            "exponents": exponents,
            "highest_degree": highest_degree,
            "is_quadratic": is_quadratic,
            "is_linear": is_linear,
            "is_successful": is_successful,
            "system_id": row["system_id"],
            "experiment_name": row["experiment_name"]
        })
    
    return pd.DataFrame(complexity_data)

def extract_experiment_dimensions(experiment_name, experiment_plan):
    """Extract dimension information from experiment name and plan."""
    if not experiment_plan or "experiments" not in experiment_plan:
        return {}
    
    # Try to find this experiment in the plan
    for exp in experiment_plan["experiments"]:
        if exp["name"] == experiment_name and "dimensions" in exp:
            return {dim: True for dim in exp["dimensions"]}
    
    # Fallback: try to parse from name
    dimensions = {}
    
    # Extract model information
    if "model_" in experiment_name:
        model_match = re.search(r'model_([^_]+)', experiment_name)
        if model_match:
            dimensions["model"] = model_match.group(1)
    
    # Extract RAG k value
    k_match = re.search(r'rag_k_(\d+)', experiment_name)
    if k_match:
        dimensions["rag_k"] = int(k_match.group(1))
    
    # Extract embedding model
    emb_match = re.search(r'kb_embedding_([^_]+)', experiment_name)
    if emb_match:
        dimensions["embedding"] = emb_match.group(1)
    
    # Extract system type
    sys_match = re.search(r'systems_([^_]+)', experiment_name)
    if sys_match:
        dimensions["system_type"] = sys_match.group(1)
    
    # Extract verification method
    verify_match = re.search(r'verify_([^_]+)', experiment_name)
    if verify_match:
        dimensions["verification"] = verify_match.group(1)
    
    return dimensions

def enrich_data_with_dimensions(df, experiment_plan):
    """Add dimension columns to the dataframe based on experiment names."""
    if df is None or len(df) == 0:
        return df
    
    # Initialize dimension columns
    dimension_columns = {
        "model": None,
        "rag_k": None,
        "embedding": None,
        "system_type": None,
        "verification": None
    }
    
    enriched_df = df.copy()
    for col in dimension_columns:
        enriched_df[col] = None
    
    # Extract dimensions for each experiment
    if "experiment_name" in enriched_df.columns:
        for exp_name in enriched_df["experiment_name"].unique():
            dimensions = extract_experiment_dimensions(exp_name, experiment_plan)
            for dim_name, dim_value in dimensions.items():
                if dim_name in dimension_columns:
                    mask = enriched_df["experiment_name"] == exp_name
                    enriched_df.loc[mask, dim_name] = dim_value
    
    return enriched_df

def generate_success_rate_analysis(enriched_df, output_dir):
    """Generate analysis of success rates across different dimensions."""
    if enriched_df is None or "final_verdict" not in enriched_df.columns:
        logger.warning("Cannot generate success rate analysis - missing data")
        return
    
    logger.info("Generating success rate analysis")
    output_dir = Path(output_dir)
    
    # Calculate success flag
    enriched_df["success"] = enriched_df["final_verdict"].str.contains("Passed")
    
    # Create plots directory
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Analyze by dimension
    dimension_columns = ["model", "rag_k", "embedding", "system_type", "verification"]
    valid_dimensions = [dim for dim in dimension_columns if enriched_df[dim].notna().any()]
    
    plt.figure(figsize=(14, 10))
    
    for i, dimension in enumerate(valid_dimensions):
        if len(enriched_df[dimension].unique()) <= 1:
            continue  # Skip dimensions with only one value
            
        # Group by dimension and calculate success rate
        success_by_dim = enriched_df.groupby(dimension)["success"].mean().reset_index()
        success_by_dim["success_rate"] = success_by_dim["success"] * 100
        
        # Create subplot
        plt.subplot(len(valid_dimensions), 1, i+1)
        
        # Plot
        sns.barplot(x=dimension, y="success_rate", data=success_by_dim)
        plt.title(f"Success Rate by {dimension}")
        plt.xlabel(dimension)
        plt.ylabel("Success Rate (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate_by_dimension.png")
    plt.close()
    
    # If sufficient data, create cross-dimensional heatmaps
    for i, dim1 in enumerate(valid_dimensions):
        for dim2 in valid_dimensions[i+1:]:
            # Check if we have sufficient data for both dimensions
            if len(enriched_df[dim1].unique()) <= 1 or len(enriched_df[dim2].unique()) <= 1:
                continue
                
            # Create cross-tabulation
            try:
                heatmap_data = pd.pivot_table(
                    enriched_df, 
                    values="success",
                    index=dim1,
                    columns=dim2,
                    aggfunc=lambda x: np.mean(x) * 100  # Convert to percentage
                )
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f")
                plt.title(f"Success Rate (%) by {dim1} and {dim2}")
                plt.tight_layout()
                plt.savefig(plots_dir / f"success_heatmap_{dim1}_{dim2}.png")
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create heatmap for {dim1} and {dim2}: {e}")
    
    # System-level analysis
    if "system_id" in enriched_df.columns:
        try:
            # Success rate by system
            system_success = enriched_df.groupby("system_id")["success"].mean().reset_index()
            system_success["success_rate"] = system_success["success"] * 100
            system_success = system_success.sort_values("success_rate", ascending=False)
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x="system_id", y="success_rate", data=system_success)
            plt.title("Success Rate by System")
            plt.xlabel("System ID")
            plt.ylabel("Success Rate (%)")
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(plots_dir / "success_rate_by_system.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create system success plot: {e}")

def analyze_verification_methods(enriched_df, output_dir):
    """Analyze the effectiveness of different verification methods."""
    if enriched_df is None:
        return
    
    verification_cols = ["num_lie_passed", "num_bound_passed", "sym_lie_passed", 
                         "sym_bound_passed", "sos_attempted", "sos_passed"]
    
    # Check if we have verification data
    if not all(col in enriched_df.columns for col in verification_cols):
        logger.warning("Missing verification data columns")
        return
    
    logger.info("Analyzing verification methods")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Calculate verification method success rates
    verification_success = {
        "Numerical": enriched_df["num_lie_passed"] & enriched_df["num_bound_passed"],
        "Symbolic": enriched_df["sym_lie_passed"] & enriched_df["sym_bound_passed"],
        "SOS": enriched_df["sos_passed"]
    }
    
    # Convert to DataFrame
    verification_df = pd.DataFrame(verification_success)
    
    # Calculate success rates
    success_rates = verification_df.mean() * 100
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    success_rates.plot(kind="bar")
    plt.title("Success Rate by Verification Method")
    plt.xlabel("Verification Method")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "verification_method_success.png")
    plt.close()
    
    # Create confusion matrix between methods
    for method1, method2 in [("Numerical", "Symbolic"), ("Numerical", "SOS"), ("Symbolic", "SOS")]:
        if verification_df[method1].sum() > 0 and verification_df[method2].sum() > 0:
            # Create confusion matrix
            cm = confusion_matrix(verification_df[method1], verification_df[method2])
            
            # Convert to DataFrame for better display
            cm_df = pd.DataFrame(
                cm, 
                index=[f"{method1} {i}" for i in ["Fail", "Pass"]], 
                columns=[f"{method2} {i}" for i in ["Fail", "Pass"]]
            )
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Verification Method Comparison: {method1} vs {method2}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"verification_comparison_{method1}_{method2}.png")
            plt.close()
    
    # Analyze combined methods
    enriched_df["any_method_passed"] = (
        verification_df["Numerical"] | 
        verification_df["Symbolic"] | 
        verification_df["SOS"]
    )
    enriched_df["all_methods_passed"] = (
        verification_df["Numerical"] & 
        verification_df["Symbolic"] & 
        verification_df["SOS"]
    )
    
    # Create summary statistics
    summary = {
        "any_method_passed": enriched_df["any_method_passed"].mean() * 100,
        "all_methods_passed": enriched_df["all_methods_passed"].mean() * 100,
        "numerical_only": (verification_df["Numerical"] & ~verification_df["Symbolic"] & ~verification_df["SOS"]).mean() * 100,
        "symbolic_only": (~verification_df["Numerical"] & verification_df["Symbolic"] & ~verification_df["SOS"]).mean() * 100,
        "sos_only": (~verification_df["Numerical"] & ~verification_df["Symbolic"] & verification_df["SOS"]).mean() * 100
    }
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    pd.Series(summary).plot(kind="bar")
    plt.title("Verification Method Combinations")
    plt.xlabel("Method Combination")
    plt.ylabel("Percentage of Certificates (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "verification_method_combinations.png")
    plt.close()

def analyze_certificate_complexity_results(complexity_df, output_dir):
    """Analyze the relationship between certificate complexity and success rate."""
    if complexity_df is None:
        return
    
    logger.info("Analyzing certificate complexity")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Histograms of complexity metrics
    plt.figure(figsize=(16, 12))
    
    # Length histogram
    plt.subplot(2, 2, 1)
    sns.histplot(data=complexity_df, x="length", hue="is_successful", multiple="stack")
    plt.title("Certificate Length Distribution")
    plt.xlabel("Length (characters)")
    plt.ylabel("Count")
    
    # Operators histogram
    plt.subplot(2, 2, 2)
    sns.histplot(data=complexity_df, x="operators", hue="is_successful", multiple="stack")
    plt.title("Number of Operators Distribution")
    plt.xlabel("Number of Operators")
    plt.ylabel("Count")
    
    # Exponents histogram
    plt.subplot(2, 2, 3)
    sns.histplot(data=complexity_df, x="exponents", hue="is_successful", multiple="stack")
    plt.title("Number of Exponents Distribution")
    plt.xlabel("Number of Exponents")
    plt.ylabel("Count")
    
    # Highest degree histogram
    plt.subplot(2, 2, 4)
    sns.histplot(data=complexity_df, x="highest_degree", hue="is_successful", multiple="stack")
    plt.title("Highest Degree Distribution")
    plt.xlabel("Highest Degree")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "certificate_complexity_histograms.png")
    plt.close()
    
    # Success rate by complexity metrics
    metrics = ["length", "operators", "exponents", "highest_degree"]
    
    # Success rate by complexity metric (binned)
    plt.figure(figsize=(16, 12))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Create bins for the metric
        # Use fewer bins for metrics with small range
        if complexity_df[metric].nunique() < 10:
            bins = complexity_df[metric].nunique()
        else:
            bins = 10
            
        # Create binned metric
        complexity_df[f"{metric}_binned"] = pd.cut(complexity_df[metric], bins=bins)
        
        # Calculate success rate by bin
        success_by_bin = complexity_df.groupby(f"{metric}_binned")["is_successful"].mean().reset_index()
        success_by_bin["success_rate"] = success_by_bin["is_successful"] * 100
        
        # Plot
        if len(success_by_bin) > 1:  # Only plot if we have multiple bins
            sns.barplot(x=f"{metric}_binned", y="success_rate", data=success_by_bin)
            plt.title(f"Success Rate by {metric}")
            plt.xlabel(metric)
            plt.ylabel("Success Rate (%)")
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate_by_complexity.png")
    plt.close()
    
    # Certificate form analysis
    form_data = {
        "form": ["Quadratic", "Linear", "Other"],
        "count": [
            complexity_df["is_quadratic"].sum(),
            complexity_df["is_linear"].sum(),
            len(complexity_df) - complexity_df["is_quadratic"].sum() - complexity_df["is_linear"].sum()
        ],
        "success_rate": [
            complexity_df[complexity_df["is_quadratic"]]["is_successful"].mean() * 100,
            complexity_df[complexity_df["is_linear"]]["is_successful"].mean() * 100,
            complexity_df[~(complexity_df["is_quadratic"] | complexity_df["is_linear"])]["is_successful"].mean() * 100
        ]
    }
    
    form_df = pd.DataFrame(form_data)
    
    # Plot certificate form distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.pie(form_df["count"], labels=form_df["form"], autopct='%1.1f%%')
    plt.title("Certificate Form Distribution")
    
    plt.subplot(1, 2, 2)
    sns.barplot(x="form", y="success_rate", data=form_df)
    plt.title("Success Rate by Certificate Form")
    plt.xlabel("Certificate Form")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "certificate_form_analysis.png")
    plt.close()

def generate_comprehensive_report(summaries_df, results_df, output_dir):
    """Generate a comprehensive report of findings."""
    output_dir = Path(output_dir)
    report_path = output_dir / "comprehensive_analysis_report.md"
    
    report_lines = []
    report_lines.append("# Barrier Certificate Generation: Comprehensive Analysis Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. Overall Performance
    report_lines.append("## 1. Overall Performance")
    if "success" in results_df.columns:
        overall_success_rate = results_df["success"].mean() * 100
        report_lines.append(f"Overall Success Rate: **{overall_success_rate:.2f}%**")
    if "parsing_successful" in results_df.columns:
        parsing_rate = results_df["parsing_successful"].mean() * 100
        report_lines.append(f"Certificate Parsing Success Rate: **{parsing_rate:.2f}%**")
    
    # System Type Analysis
    if "system_type" in results_df.columns and results_df["system_type"].notna().any():
        report_lines.append("\n### Performance by System Type")
        system_performance = results_df.groupby("system_type")["success"].agg(
            ["count", "mean"]
        ).reset_index()
        system_performance["success_rate"] = system_performance["mean"] * 100
        system_performance = system_performance.sort_values("success_rate", ascending=False)
        
        # Create markdown table
        report_lines.append("\n| System Type | Count | Success Rate |")
        report_lines.append("| --- | --- | --- |")
        for _, row in system_performance.iterrows():
            report_lines.append(f"| {row['system_type']} | {int(row['count'])} | {row['success_rate']:.2f}% |")
    
    # 2. Model Comparison
    if "model" in results_df.columns and results_df["model"].notna().any():
        report_lines.append("\n## 2. Model Performance Comparison")
        model_performance = results_df.groupby("model")["success"].agg(
            ["count", "mean"]
        ).reset_index()
        model_performance["success_rate"] = model_performance["mean"] * 100
        model_performance = model_performance.sort_values("success_rate", ascending=False)
        
        # Create markdown table
        report_lines.append("\n| Model | Count | Success Rate |")
        report_lines.append("| --- | --- | --- |")
        for _, row in model_performance.iterrows():
            report_lines.append(f"| {row['model']} | {int(row['count'])} | {row['success_rate']:.2f}% |")
    
    # 3. Knowledge Base Impact
    if "rag_k" in results_df.columns and results_df["rag_k"].notna().any():
        report_lines.append("\n## 3. Knowledge Base Impact")
        
        # RAG k analysis
        report_lines.append("\n### Impact of RAG k value")
        rag_performance = results_df.groupby("rag_k")["success"].agg(
            ["count", "mean"]
        ).reset_index()
        rag_performance["success_rate"] = rag_performance["mean"] * 100
        rag_performance = rag_performance.sort_values("rag_k")
        
        # Create markdown table
        report_lines.append("\n| RAG k | Count | Success Rate |")
        report_lines.append("| --- | --- | --- |")
        for _, row in rag_performance.iterrows():
            report_lines.append(f"| {int(row['rag_k'])} | {int(row['count'])} | {row['success_rate']:.2f}% |")
    
    # 4. Verification Methods
    report_lines.append("\n## 4. Verification Method Analysis")
    
    # Add verification method breakdown if available
    verification_cols = ["num_lie_passed", "num_bound_passed", "sym_lie_passed", 
                         "sym_bound_passed", "sos_passed"]
    
    if all(col in results_df.columns for col in verification_cols):
        verification_success = {
            "Numerical": (results_df["num_lie_passed"] & results_df["num_bound_passed"]).mean() * 100,
            "Symbolic": (results_df["sym_lie_passed"] & results_df["sym_bound_passed"]).mean() * 100,
            "SOS": results_df["sos_passed"].mean() * 100
        }
        
        report_lines.append("\n### Verification Success by Method")
        report_lines.append("\n| Verification Method | Success Rate |")
        report_lines.append("| --- | --- |")
        for method, rate in verification_success.items():
            report_lines.append(f"| {method} | {rate:.2f}% |")
    
    # 5. Certificate Complexity
    if "certificate" in results_df.columns and "parsed_certificate" in results_df.columns:
        report_lines.append("\n## 5. Certificate Complexity Analysis")
        
        # Calculate average length of successful vs. unsuccessful certificates
        if "parsed_certificate" in results_df.columns and "success" in results_df.columns:
            # Filter valid certificates
            valid_certs = results_df[results_df["parsed_certificate"].notna() & 
                                   (results_df["parsed_certificate"] != "")]
            
            if len(valid_certs) > 0:
                valid_certs["cert_length"] = valid_certs["parsed_certificate"].str.len()
                
                avg_length_success = valid_certs[valid_certs["success"]]["cert_length"].mean()
                avg_length_failure = valid_certs[~valid_certs["success"]]["cert_length"].mean()
                
                report_lines.append("\n### Certificate Length Analysis")
                report_lines.append(f"Average length of successful certificates: **{avg_length_success:.1f}** characters")
                report_lines.append(f"Average length of unsuccessful certificates: **{avg_length_failure:.1f}** characters")
    
    # 6. System-Level Performance
    if "system_id" in results_df.columns:
        report_lines.append("\n## 6. System-Level Analysis")
        
        # Most successful systems
        system_success = results_df.groupby("system_id")["success"].mean().reset_index()
        system_success["success_rate"] = system_success["success"] * 100
        
        # Top 5 most successful systems
        top_systems = system_success.sort_values("success_rate", ascending=False).head(5)
        
        report_lines.append("\n### Top Performing Systems")
        report_lines.append("\n| System ID | Success Rate |")
        report_lines.append("| --- | --- |")
        for _, row in top_systems.iterrows():
            report_lines.append(f"| {row['system_id']} | {row['success_rate']:.2f}% |")
        
        # Bottom 5 least successful systems
        bottom_systems = system_success.sort_values("success_rate").head(5)
        
        report_lines.append("\n### Most Challenging Systems")
        report_lines.append("\n| System ID | Success Rate |")
        report_lines.append("| --- | --- |")
        for _, row in bottom_systems.iterrows():
            report_lines.append(f"| {row['system_id']} | {row['success_rate']:.2f}% |")
    
    # 7. Key Findings and Recommendations
    report_lines.append("\n## 7. Key Findings and Recommendations")
    
    # Generate findings based on data analysis
    findings = []
    
    # Find best model
    if "model" in results_df.columns and results_df["model"].notna().any():
        model_perfs = results_df.groupby("model")["success"].mean()
        if len(model_perfs) > 0:
            best_model = model_perfs.idxmax()
            best_rate = model_perfs.max() * 100
            findings.append(f"The most effective model was **{best_model}** with a success rate of **{best_rate:.2f}%**.")
    
    # Find best RAG k
    if "rag_k" in results_df.columns and results_df["rag_k"].notna().any():
        rag_perfs = results_df.groupby("rag_k")["success"].mean()
        if len(rag_perfs) > 0:
            best_k = rag_perfs.idxmax()
            best_k_rate = rag_perfs.max() * 100
            findings.append(f"The optimal RAG retrieval count (k) was **{best_k}** with a success rate of **{best_k_rate:.2f}%**.")
    
    # Best system type
    if "system_type" in results_df.columns and results_df["system_type"].notna().any():
        system_perfs = results_df.groupby("system_type")["success"].mean()
        if len(system_perfs) > 1:  # Only if we have multiple system types
            best_system = system_perfs.idxmax()
            worst_system = system_perfs.idxmin()
            findings.append(f"LLMs performed best on **{best_system}** systems and struggled most with **{worst_system}** systems.")
    
    # Add findings to report
    if findings:
        report_lines.append("\n### Key Findings")
        for finding in findings:
            report_lines.append(f"- {finding}")
    
    # Add recommendations
    report_lines.append("\n### Recommendations")
    report_lines.append("Based on the analysis, we recommend:")
    
    # Generate recommendations based on findings
    recommendations = []
    
    # Model recommendation
    if "model" in results_df.columns and results_df["model"].notna().any():
        model_perfs = results_df.groupby("model")["success"].mean()
        if len(model_perfs) > 0:
            best_model = model_perfs.idxmax()
            recommendations.append(f"Use **{best_model}** as the primary LLM for barrier certificate generation.")
    
    # RAG recommendation
    if "rag_k" in results_df.columns and results_df["rag_k"].notna().any():
        rag_perfs = results_df.groupby("rag_k")["success"].mean()
        if len(rag_perfs) > 0:
            best_k = rag_perfs.idxmax()
            recommendations.append(f"Set the RAG retrieval count to **k={best_k}** for optimal performance.")
    
    # Verification recommendation
    verification_cols = ["num_lie_passed", "sym_lie_passed", "sos_passed"]
    if all(col in results_df.columns for col in verification_cols):
        try:
            numerical = (results_df["num_lie_passed"] & results_df["num_bound_passed"]).mean()
            symbolic = (results_df["sym_lie_passed"] & results_df["sym_bound_passed"]).mean()
            sos = results_df["sos_passed"].mean()
            
            methods = ["numerical", "symbolic", "SOS"]
            rates = [numerical, symbolic, sos]
            
            best_method = methods[np.argmax(rates)]
            recommendations.append(f"Prioritize **{best_method}** verification for the most reliable results.")
        except Exception:
            pass
    
    # Add system-specific recommendations
    if "system_type" in results_df.columns and "success" in results_df.columns:
        try:
            system_success = results_df.groupby(["system_type", "model"])["success"].mean().reset_index()
            
            if len(system_success) > 0:
                # For each system type, find the best model
                best_models = system_success.loc[system_success.groupby("system_type")["success"].idxmax()]
                
                if len(best_models) > 1:
                    recommendations.append("Consider using different models for different system types:")
                    for _, row in best_models.iterrows():
                        recommendations.append(f"  - For **{row['system_type']}** systems, use **{row['model']}** (success rate: {row['success']*100:.2f}%)")
        except Exception:
            pass
    
    # Add recommendations to report
    for rec in recommendations:
        report_lines.append(f"- {rec}")
    
    # Write report to file
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Comprehensive analysis report written to {report_path}")
    return report_path

def generate_correlation_matrix(enriched_df, output_dir):
    """Generate correlation matrix between different experiment dimensions and success metrics."""
    if enriched_df is None or len(enriched_df) == 0:
        logger.warning("Cannot generate correlation matrix - insufficient data")
        return
    
    logger.info("Generating correlation matrix")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Make sure success column exists
    if "success" not in enriched_df.columns:
        enriched_df["success"] = enriched_df["final_verdict"].str.contains("Passed")
    
    # Extract numeric columns we can use for correlation
    numeric_cols = []
    
    # Convert categorical dimensions to numeric when possible
    if "rag_k" in enriched_df.columns:
        if enriched_df["rag_k"].dtype == object:
            try:
                enriched_df["rag_k_numeric"] = pd.to_numeric(enriched_df["rag_k"], errors="coerce")
                numeric_cols.append("rag_k_numeric")
            except:
                pass
        else:
            numeric_cols.append("rag_k")
    
    # Add other numeric metrics we care about
    metric_cols = [
        "success", 
        "parsing_successful", 
        "generation_successful", 
        "duration_seconds",
    ]
    
    for col in metric_cols:
        if col in enriched_df.columns:
            if enriched_df[col].dtype in (np.int64, np.float64, bool):
                numeric_cols.append(col)
    
    # Include numeric metrics from verification
    verification_cols = [
        "num_lie_passed", "num_bound_passed", 
        "sym_lie_passed", "sym_bound_passed", 
        "sos_passed"
    ]
    
    for col in verification_cols:
        if col in enriched_df.columns:
            if enriched_df[col].dtype in (np.int64, np.float64, bool):
                numeric_cols.append(col)
    
    # Skip if we don't have enough numeric columns
    if len(numeric_cols) < 2:
        logger.warning("Insufficient numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_df = enriched_df[numeric_cols].corr(method='pearson')
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_df, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True, 
        fmt=".2f"
    )
    
    plt.title("Correlation Matrix of Experiment Metrics")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_matrix.png")
    plt.close()
    
    # Also generate a correlation matrix focused on success drivers
    if "success" in numeric_cols:
        success_correlations = corr_df["success"].sort_values(ascending=False)
        
        # Plot the factors most correlated with success
        plt.figure(figsize=(10, 6))
        success_correlations.drop("success").plot(kind="bar")
        plt.title("Factors Correlating with Success Rate")
        plt.xlabel("Factor")
        plt.ylabel("Correlation Coefficient")
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "success_correlation_factors.png")
        plt.close()

def generate_system_difficulty_analysis(enriched_df, output_dir):
    """Generate network graph and analysis of system difficulty patterns."""
    if enriched_df is None or "system_id" not in enriched_df.columns:
        logger.warning("Cannot generate system difficulty analysis - missing data")
        return
    
    logger.info("Analyzing system difficulty patterns")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Ensure success column exists
    if "success" not in enriched_df.columns:
        enriched_df["success"] = enriched_df["final_verdict"].str.contains("Passed")
    
    # Calculate system success rates
    system_success = enriched_df.groupby("system_id")["success"].agg(
        ["count", "mean"]
    ).reset_index()
    system_success["success_rate"] = system_success["mean"] * 100
    system_success["difficulty"] = 100 - system_success["success_rate"]
    system_success = system_success.sort_values("difficulty", ascending=False)
    
    # Create system difficulty bar plot
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("RdYlGn_r", len(system_success))
    
    bars = sns.barplot(
        x="system_id", 
        y="difficulty", 
        data=system_success,
        palette=palette,
        hue="difficulty",
        dodge=False
    )
    
    # Add count annotations
    for idx, row in enumerate(system_success.itertuples()):
        bars.text(
            idx, 
            row.difficulty + 1, 
            f"n={row.count}", 
            ha='center',
            size=8,
            color='black'
        )
    
    plt.title("System Difficulty Ranking")
    plt.xlabel("System ID")
    plt.ylabel("Difficulty (100% - Success Rate)")
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "system_difficulty_ranking.png")
    plt.close()
    
    # If system_type column exists, analyze difficulty by system type
    if "system_type" in enriched_df.columns and enriched_df["system_type"].notna().any():
        system_type_difficulty = enriched_df.groupby("system_type")["success"].agg(
            ["count", "mean"]
        ).reset_index()
        system_type_difficulty["success_rate"] = system_type_difficulty["mean"] * 100
        system_type_difficulty["difficulty"] = 100 - system_type_difficulty["success_rate"]
        system_type_difficulty = system_type_difficulty.sort_values("difficulty", ascending=False)
        
        # Create system type difficulty bar plot
        plt.figure(figsize=(12, 6))
        palette = sns.color_palette("RdYlGn_r", len(system_type_difficulty))
        
        sns.barplot(
            x="system_type", 
            y="difficulty", 
            data=system_type_difficulty,
            palette=palette,
            hue="difficulty",
            dodge=False
        )
        
        plt.title("System Type Difficulty Ranking")
        plt.xlabel("System Type")
        plt.ylabel("Difficulty (100% - Success Rate)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend([],[], frameon=False)
        plt.tight_layout()
        plt.savefig(plots_dir / "system_type_difficulty.png")
        plt.close()
    
    # Create network graph of system relationships based on success patterns
    # Only do this if we have multiple models or dimensions to compare
    if (("model" in enriched_df.columns and enriched_df["model"].nunique() > 1) or
        ("verification" in enriched_df.columns and enriched_df["verification"].nunique() > 1)):
        
        try:
            # Create network of systems
            G = nx.Graph()
            
            # Add nodes for each system
            for idx, row in system_success.iterrows():
                G.add_node(
                    row["system_id"], 
                    difficulty=row["difficulty"],
                    success_rate=row["success_rate"],
                    count=row["count"]
                )
            
            # Calculate similarity between systems based on which models/settings succeed on them
            similarity_matrix = {}
            
            for sys1 in system_success["system_id"]:
                for sys2 in system_success["system_id"]:
                    if sys1 != sys2 and (sys1, sys2) not in similarity_matrix and (sys2, sys1) not in similarity_matrix:
                        # Get success patterns for sys1 and sys2
                        sys1_exp = enriched_df[enriched_df["system_id"] == sys1]
                        sys2_exp = enriched_df[enriched_df["system_id"] == sys2]
                        
                        # Only proceed if we have enough data
                        if len(sys1_exp) >= 3 and len(sys2_exp) >= 3:
                            # Calculate Jaccard similarity based on successful experiments
                            model_success_sys1 = set(sys1_exp[sys1_exp["success"]]["experiment_name"])
                            model_success_sys2 = set(sys2_exp[sys2_exp["success"]]["experiment_name"])
                            
                            union_size = len(model_success_sys1.union(model_success_sys2))
                            if union_size > 0:
                                intersection_size = len(model_success_sys1.intersection(model_success_sys2))
                                similarity = intersection_size / union_size
                                
                                # Only add edges between highly similar systems
                                if similarity > 0.5:
                                    G.add_edge(sys1, sys2, weight=similarity)
                                similarity_matrix[(sys1, sys2)] = similarity
            
            if len(G.edges()) > 0:
                plt.figure(figsize=(12, 12))
                
                # Calculate node sizes based on count
                node_sizes = [G.nodes[node]["count"] * 50 for node in G.nodes()]
                
                # Calculate node colors based on difficulty
                difficulties = [G.nodes[node]["difficulty"] for node in G.nodes()]
                norm = plt.Normalize(min(difficulties), max(difficulties))
                node_colors = [plt.cm.RdYlGn_r(norm(diff)) for diff in difficulties]
                
                # Calculate edge widths based on similarity
                edge_widths = [G[u][v]["weight"] * 3 for u, v in G.edges()]
                
                # Calculate positions
                pos = nx.spring_layout(G, seed=42)
                
                # Draw network
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="gray")
                nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
                
                plt.title("System Similarity Network\n(Systems with similar success patterns)")
                plt.axis("off")
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_label("Difficulty")
                
                plt.tight_layout()
                plt.savefig(plots_dir / "system_similarity_network.png")
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to create system network graph: {e}")

def generate_model_system_interaction_analysis(enriched_df, output_dir):
    """Generate analysis of how different models perform across system types."""
    if (enriched_df is None or "model" not in enriched_df.columns or 
        enriched_df["model"].notna().sum() == 0 or "system_id" not in enriched_df.columns):
        logger.warning("Cannot generate model-system interaction analysis - missing data")
        return
    
    logger.info("Analyzing model performance across systems")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Ensure success column exists
    if "success" not in enriched_df.columns:
        enriched_df["success"] = enriched_df["final_verdict"].str.contains("Passed")
    
    # Create model x system matrix
    if enriched_df["model"].nunique() >= 2:
        try:
            pivot_df = pd.pivot_table(
                enriched_df,
                values="success",
                index="model",
                columns="system_id",
                aggfunc=lambda x: np.mean(x) * 100  # Convert to percentage
            )
            
            # Fill any remaining NaN values with 0
            pivot_df = pivot_df.fillna(0)
            
            # Plot the heatmap
            plt.figure(figsize=(max(12, len(pivot_df.columns) * 0.7), max(8, len(pivot_df.index) * 0.7)))
            
            # Calculate optimal column order - group similar systems
            col_linkage = hierarchy.linkage(pdist(pivot_df.T), method='ward')
            col_order = hierarchy.dendrogram(col_linkage, no_plot=True)['leaves']
            pivot_reordered = pivot_df.iloc[:, col_order]
            
            sns.heatmap(
                pivot_reordered,
                annot=True,
                cmap="YlGnBu",
                fmt=".1f",
                linewidths=0.5
            )
            
            plt.title("Model Performance by System (Success Rate %)")
            plt.ylabel("Model")
            plt.xlabel("System ID")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(plots_dir / "model_system_performance_heatmap.png")
            plt.close()
            
            # Generate a variance analysis - which systems have the most varied performance across models
            system_variances = pivot_df.var().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 6))
            system_variances.plot(kind="bar")
            plt.title("Performance Variance Across Models by System")
            plt.xlabel("System ID")
            plt.ylabel("Success Rate Variance")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(plots_dir / "system_performance_variance.png")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create model-system performance heatmap: {e}")
    
    # If we have system types, analyze model performance by system type
    if "system_type" in enriched_df.columns and enriched_df["system_type"].notna().any():
        if enriched_df["model"].nunique() >= 2 and enriched_df["system_type"].nunique() >= 2:
            try:
                type_pivot = pd.pivot_table(
                    enriched_df,
                    values="success",
                    index="model",
                    columns="system_type",
                    aggfunc=lambda x: np.mean(x) * 100  # Convert to percentage
                )
                
                # Fill any remaining NaN values with 0
                type_pivot = type_pivot.fillna(0)
                
                # Plot the heatmap
                plt.figure(figsize=(max(10, len(type_pivot.columns) * 1.2), max(8, len(type_pivot.index) * 0.7)))
                
                sns.heatmap(
                    type_pivot,
                    annot=True,
                    cmap="YlGnBu",
                    fmt=".1f",
                    linewidths=0.5
                )
                
                plt.title("Model Performance by System Type (Success Rate %)")
                plt.ylabel("Model")
                plt.xlabel("System Type")
                plt.tight_layout()
                plt.savefig(plots_dir / "model_system_type_heatmap.png")
                plt.close()
                
                # Create a radar chart for model comparison across system types
                if len(type_pivot.columns) >= 3:
                    # Prepare data for radar chart
                    categories = type_pivot.columns.tolist()
                    n_models = len(type_pivot.index)
                    
                    # Create figure and polar axis
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, polar=True)
                    
                    # Calculate angle for each category
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]  # Close the loop
                    
                    # Add axis lines
                    plt.xticks(angles[:-1], categories)
                    
                    # Set y-limits
                    ax.set_ylim(0, 100)
                    
                    # Draw the data
                    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
                    
                    for i, model in enumerate(type_pivot.index):
                        values = type_pivot.loc[model].values.flatten().tolist()
                        values += values[:1]  # Close the loop
                        
                        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
                        ax.fill(angles, values, alpha=0.1, color=colors[i])
                    
                    # Add legend
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    plt.title("Model Performance Across System Types", y=1.1)
                    plt.tight_layout()
                    plt.savefig(plots_dir / "model_system_type_radar.png")
                    plt.close()
                    
            except Exception as e:
                logger.warning(f"Failed to create model-system type performance analysis: {e}")

def analyze_failure_modes(enriched_df, output_dir):
    """Analyze and visualize common failure modes in the experiments."""
    if enriched_df is None or "final_verdict" not in enriched_df.columns:
        logger.warning("Cannot generate failure mode analysis - missing data")
        return
    
    logger.info("Analyzing failure modes")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract failure reasons
    failed_exps = enriched_df[~enriched_df["final_verdict"].str.contains("Passed")]
    
    if len(failed_exps) == 0:
        logger.info("No failures found for failure mode analysis")
        return
    
    # Analyze verification failures
    if "verification_reason" in failed_exps.columns:
        try:
            # Extract key phrases from verification reasons
            failure_patterns = [
                (r"numerical.*lie|lie.*numerical", "Lie Derivative Numerical Violation"),
                (r"numerical.*boundary|boundary.*numerical", "Boundary Numerical Violation"),
                (r"optimization.*counterexample", "Optimization Found Counterexample"),
                (r"symbolic.*check", "Symbolic Check Failed"),
                (r"SOS.*failed", "SOS Verification Failed"),
                (r"parsing.*failed", "Certificate Parsing Failed"),
                (r"verification.*crashed|error", "Verification Crashed/Error"),
            ]
            
            failure_counts = defaultdict(int)
            
            for _, row in failed_exps.iterrows():
                if pd.isna(row["verification_reason"]):
                    failure_counts["No Verification Reason"] += 1
                    continue
                    
                reason = row["verification_reason"].lower()
                matched = False
                
                for pattern, category in failure_patterns:
                    if re.search(pattern, reason):
                        failure_counts[category] += 1
                        matched = True
                        break
                
                if not matched:
                    failure_counts["Other Reasons"] += 1
            
            # Create DataFrame from counts
            failure_df = pd.DataFrame({
                "Failure Mode": list(failure_counts.keys()),
                "Count": list(failure_counts.values())
            }).sort_values("Count", ascending=False)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = sns.barplot(x="Failure Mode", y="Count", data=failure_df)
            plt.title("Distribution of Failure Modes")
            plt.xlabel("Failure Mode")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Add count labels
            for i, v in enumerate(failure_df["Count"]):
                bars.text(i, v + 0.1, str(v), ha='center')
                
            plt.tight_layout()
            plt.savefig(plots_dir / "failure_mode_distribution.png")
            plt.close()
            
            # Analyze failure modes by key dimensions
            for dim in ["model", "system_type", "verification"]:
                if dim in failed_exps.columns and failed_exps[dim].notna().any() and failed_exps[dim].nunique() > 1:
                    # Create failure mode matrix by dimension
                    failure_modes = []
                    
                    for _, row in failed_exps.iterrows():
                        if pd.isna(row["verification_reason"]) or pd.isna(row[dim]):
                            continue
                            
                        reason = row["verification_reason"].lower()
                        mode = "Other"
                        
                        for pattern, category in failure_patterns:
                            if re.search(pattern, reason):
                                mode = category
                                break
                        
                        failure_modes.append({
                            "Dimension": row[dim],
                            "Failure Mode": mode,
                            "Count": 1
                        })
                    
                    if failure_modes:
                        # Convert to DataFrame and aggregate
                        failure_dim_df = pd.DataFrame(failure_modes)
                        pivot_table = pd.pivot_table(
                            failure_dim_df,
                            values="Count",
                            index="Dimension",
                            columns="Failure Mode",
                            aggfunc="sum",
                            fill_value=0
                        )
                        
                        # Create heatmap
                        plt.figure(figsize=(max(12, len(pivot_table.columns) * 0.8), max(8, len(pivot_table.index) * 0.6)))
                        sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlOrRd")
                        plt.title(f"Failure Modes by {dim}")
                        plt.tight_layout()
                        plt.savefig(plots_dir / f"failure_modes_by_{dim}.png")
                        plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create failure mode analysis: {e}")
    
    # Analyze parsing failures
    parsing_failures = enriched_df[
        (enriched_df["generation_successful"] == True) & 
        (enriched_df["parsing_successful"] == False)
    ]
    
    if len(parsing_failures) > 0:
        logger.info(f"Found {len(parsing_failures)} parsing failures")
        
        # Analyze parsing failures by dimension
        for dim in ["model", "system_type"]:
            if dim in parsing_failures.columns and parsing_failures[dim].notna().any() and parsing_failures[dim].nunique() > 1:
                parsing_by_dim = parsing_failures.groupby(dim).size().reset_index(name="count")
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=dim, y="count", data=parsing_by_dim)
                plt.title(f"Certificate Parsing Failures by {dim}")
                plt.xlabel(dim)
                plt.ylabel("Number of Parsing Failures")
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(plots_dir / f"parsing_failures_by_{dim}.png")
                plt.close()

def analyze_certificate_characteristics(enriched_df, output_dir):
    """Analyze characteristics of generated certificates in relation to verification success."""
    if enriched_df is None or "parsed_certificate" not in enriched_df.columns:
        logger.warning("Cannot analyze certificate characteristics - missing data")
        return
    
    # First calculate certificate complexity metrics
    complexity_df = analyze_certificate_complexity(enriched_df)
    if complexity_df is None:
        return
    
    logger.info("Analyzing certificate characteristics")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create multi-dimensional scatter plot of certificate properties
    plt.figure(figsize=(12, 10))
    
    # Use length and operators as x, y and exponents for size
    scatter = plt.scatter(
        complexity_df["length"], 
        complexity_df["operators"],
        s=complexity_df["exponents"] * 20 + 20,  # Scale exponents for visibility
        c=complexity_df["is_successful"].astype(int),
        cmap="viridis",
        alpha=0.7
    )
    
    plt.title("Certificate Properties and Success")
    plt.xlabel("Certificate Length")
    plt.ylabel("Number of Operators")
    plt.grid(alpha=0.3)
    
    # Create a custom legend for the success/failure color
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(1), label='Success', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(0), label='Failure', markersize=10)
    ]
    
    # Add the legend for success/failure
    plt.legend(handles=legend_elements, title="Verification Result", loc="upper right")
    
    # Add a colorbar legend for the sizes
    sizes = sorted(set(complexity_df["exponents"]))
    if len(sizes) > 1:
        for size in sizes:
            plt.scatter([], [], s=size*20+20, c='gray', alpha=0.7, label=f'{size} Exponents')
        plt.legend(title="Exponents", loc="upper left")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "certificate_property_scatter.png")
    plt.close()
    
    # Analyze success rate by certificate degree
    success_by_degree = complexity_df.groupby("highest_degree")["is_successful"].agg(
        ["count", "mean"]
    ).reset_index()
    success_by_degree["success_rate"] = success_by_degree["mean"] * 100
    success_by_degree = success_by_degree.sort_values("highest_degree")
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x="highest_degree", y="success_rate", data=success_by_degree)
    plt.title("Success Rate by Certificate Degree")
    plt.xlabel("Highest Degree in Certificate")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add count annotations
    for idx, row in enumerate(success_by_degree.itertuples()):
        bars.text(
            idx, 
            row.success_rate + 1, 
            f"n={row.count}", 
            ha='center',
            size=10
        )
    
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate_by_degree.png")
    plt.close()
    
    # Analyze certificate form distribution by system type
    if "system_type" in enriched_df.columns and enriched_df["system_type"].notna().any():
        # Merge system type info into complexity df
        system_type_map = {}
        for _, row in enriched_df.iterrows():
            if not pd.isna(row["system_type"]) and not pd.isna(row["system_id"]):
                system_type_map[row["system_id"]] = row["system_type"]
        
        if system_type_map:
            complexity_df["system_type"] = complexity_df["system_id"].map(system_type_map)
            
            # Calculate form distribution by system type
            form_data = []
            
            for sys_type in complexity_df["system_type"].unique():
                if pd.isna(sys_type):
                    continue
                    
                type_certs = complexity_df[complexity_df["system_type"] == sys_type]
                
                form_data.append({
                    "system_type": sys_type,
                    "quadratic": type_certs["is_quadratic"].mean() * 100,
                    "linear": type_certs["is_linear"].mean() * 100,
                    "other": (1 - type_certs["is_quadratic"].mean() - type_certs["is_linear"].mean()) * 100,
                    "count": len(type_certs)
                })
            
            if form_data:
                form_df = pd.DataFrame(form_data)
                
                # Create stacked bar chart
                form_plot_df = form_df.melt(
                    id_vars=["system_type", "count"],
                    value_vars=["quadratic", "linear", "other"],
                    var_name="Form",
                    value_name="Percentage"
                )
                
                plt.figure(figsize=(max(10, len(form_df) * 1.2), 8))
                chart = sns.barplot(
                    x="system_type",
                    y="Percentage",
                    hue="Form",
                    data=form_plot_df
                )
                
                plt.title("Certificate Form Distribution by System Type")
                plt.xlabel("System Type")
                plt.ylabel("Percentage")
                plt.legend(title="Certificate Form")
                plt.xticks(rotation=45, ha="right")
                
                # Add count annotations
                for idx, row in enumerate(form_df.itertuples()):
                    chart.text(
                        idx, 
                        105, 
                        f"n={row.count}", 
                        ha='center',
                        size=9
                    )
                
                plt.tight_layout()
                plt.savefig(plots_dir / "certificate_form_by_system_type.png")
                plt.close()

def generate_multivariate_analysis(enriched_df, output_dir):
    """Generate advanced multivariate analysis of success factors."""
    if enriched_df is None:
        logger.warning("Cannot generate multivariate analysis - missing data")
        return
    
    logger.info("Generating multivariate analysis of success factors")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Ensure success column exists
    if "success" not in enriched_df.columns and "final_verdict" in enriched_df.columns:
        enriched_df["success"] = enriched_df["final_verdict"].str.contains("Passed")
        
    # Add interaction terms between RAG k and model
    if ("rag_k" in enriched_df.columns and "model" in enriched_df.columns and
        enriched_df["rag_k"].notna().any() and enriched_df["model"].notna().any()):
        
        # Create crosstab of success rates
        try:
            interaction_pivot = pd.pivot_table(
                enriched_df,
                values="success",
                index="model",
                columns="rag_k",
                aggfunc=lambda x: np.mean(x) * 100  # Convert to percentage
            )
            
            # Create interaction heatmap
            plt.figure(figsize=(max(10, len(interaction_pivot.columns) * 1.0), max(8, len(interaction_pivot.index) * 0.6)))
            sns.heatmap(
                interaction_pivot,
                annot=True,
                cmap="YlGnBu",
                fmt=".1f"
            )
            
            plt.title("Success Rate (%) by Model and RAG k")
            plt.tight_layout()
            plt.savefig(plots_dir / "model_rag_k_interaction.png")
            plt.close()
            
            # Calculate optimal RAG k for each model
            optimal_k = interaction_pivot.idxmax(axis=1).reset_index()
            optimal_k.columns = ["Model", "Optimal RAG k"]
            
            # Add the max success rate
            optimal_k["Max Success Rate"] = interaction_pivot.max(axis=1).values
            
            # Create a table visualization
            plt.figure(figsize=(10, len(optimal_k) * 0.5 + 2))
            plt.axis('off')
            plt.title("Optimal RAG k by Model")
            
            # Create table data
            table_data = []
            for _, row in optimal_k.iterrows():
                table_data.append([row["Model"], row["Optimal RAG k"], f"{row['Max Success Rate']:.1f}%"])
            
            # Create the table
            table = plt.table(
                cellText=table_data,
                colLabels=["Model", "Optimal RAG k", "Success Rate"],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.3, 0.3]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "optimal_rag_k_by_model.png")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create model-RAG interaction analysis: {e}")
    
    # System dimensionality analysis
    # Try to extract dimensionality from system_id using common patterns
    try:
        if "system_id" in enriched_df.columns:
            # Try to identify 2D, 3D, 4D systems based on system ID patterns
            enriched_df["dimensionality"] = None
            
            # Look for xD in system_id
            dim_pattern = r'(\d+)[dD]'
            
            for idx, row in enriched_df.iterrows():
                sys_id = row["system_id"]
                # Skip if nan
                if pd.isna(sys_id):
                    continue
                
                # Try to extract dimensionality from system_id
                match = re.search(dim_pattern, sys_id)
                if match:
                    enriched_df.at[idx, "dimensionality"] = int(match.group(1))
                # Try to identify dimensionality from common names
                elif "2d" in sys_id.lower():
                    enriched_df.at[idx, "dimensionality"] = 2
                elif "3d" in sys_id.lower():
                    enriched_df.at[idx, "dimensionality"] = 3
                elif "4d" in sys_id.lower():
                    enriched_df.at[idx, "dimensionality"] = 4
            
            # If we identified dimensions, analyze success by dimensionality
            if enriched_df["dimensionality"].notna().any():
                dim_success = enriched_df.groupby("dimensionality")["success"].agg(
                    ["count", "mean"]
                ).reset_index()
                dim_success["success_rate"] = dim_success["mean"] * 100
                dim_success = dim_success.sort_values("dimensionality")
                
                plt.figure(figsize=(10, 6))
                bars = sns.barplot(x="dimensionality", y="success_rate", data=dim_success)
                
                # Add count annotations
                for idx, row in enumerate(dim_success.itertuples()):
                    bars.text(
                        idx, 
                        row.success_rate + 1, 
                        f"n={row.count}", 
                        ha='center',
                        size=10
                    )
                
                plt.title("Success Rate by System Dimensionality")
                plt.xlabel("Number of Dimensions")
                plt.ylabel("Success Rate (%)")
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(plots_dir / "success_by_dimensionality.png")
                plt.close()
                
                # Check interactions with other dimensions like model or RAG k
                if "model" in enriched_df.columns and enriched_df["model"].notna().any():
                    try:
                        dim_model_pivot = pd.pivot_table(
                            enriched_df,
                            values="success",
                            index="model",
                            columns="dimensionality",
                            aggfunc=lambda x: np.mean(x) * 100  # Convert to percentage
                        )
                        
                        # Plot the interaction
                        plt.figure(figsize=(max(10, len(dim_model_pivot.columns) * 1.2), max(8, len(dim_model_pivot.index) * 0.6)))
                        sns.heatmap(
                            dim_model_pivot,
                            annot=True,
                            cmap="YlGnBu",
                            fmt=".1f"
                        )
                        
                        plt.title("Success Rate (%) by Model and System Dimensionality")
                        plt.xlabel("System Dimensionality")
                        plt.ylabel("Model")
                        plt.tight_layout()
                        plt.savefig(plots_dir / "model_dimensionality_interaction.png")
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Failed to create model-dimensionality interaction analysis: {e}")
    except Exception as e:
        logger.warning(f"Failed to analyze system dimensionality: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze FMLLMSolver experiment results")
    parser.add_argument("--experiment-dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save analysis results (defaults to experiment-dir/analysis)")
    parser.add_argument("--interactive", action="store_true",
                        help="Generate interactive visualizations with Plotly")
    args = parser.parse_args()
    
    # Set output directory
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory does not exist: {experiment_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Load experiment data
    logger.info(f"Loading experiment data from {experiment_dir}")
    summaries_df, results_df, experiment_plan = load_experiment_data(experiment_dir)
    
    if summaries_df is None:
        logger.error("No experiment summaries found")
        return
    
    # Enrich data with dimension information
    enriched_results_df = None
    if results_df is not None:
        logger.info("Enriching results data with dimension information")
        enriched_results_df = enrich_data_with_dimensions(results_df, experiment_plan)
    
    # Generate analysis
    if enriched_results_df is not None:
        # Basic visualizations (existing)
        generate_success_rate_analysis(enriched_results_df, output_dir)
        
        # NEW: Advanced correlation analysis
        generate_correlation_matrix(enriched_results_df, output_dir)
        
        # NEW: System difficulty analysis
        generate_system_difficulty_analysis(enriched_results_df, output_dir)
        
        # NEW: Model-system interaction analysis
        generate_model_system_interaction_analysis(enriched_results_df, output_dir)
        
        # Certificate complexity (existing)
        complexity_df = analyze_certificate_complexity(enriched_results_df)
        if complexity_df is not None:
            analyze_certificate_complexity_results(complexity_df, output_dir)
            
            # NEW: Detailed certificate characteristics analysis
            analyze_certificate_characteristics(enriched_results_df, output_dir)
        
        # Verification method analysis (existing)
        analyze_verification_methods(enriched_results_df, output_dir)
        
        # NEW: Failure mode analysis
        analyze_failure_modes(enriched_results_df, output_dir)
        
        # NEW: Multivariate analysis
        generate_multivariate_analysis(enriched_results_df, output_dir)
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(summaries_df, enriched_results_df, output_dir)
        logger.info(f"Analysis complete. Report available at: {report_path}")
    else:
        logger.warning("Could not analyze detailed results - only summary information available")
    
    logger.info(f"Analysis results saved to {output_dir}")

if __name__ == "__main__":
    main() 