"""
Shared utilities for experiment analysis and result processing.

This module consolidates common analysis functions used across multiple
experiment analysis scripts to eliminate code duplication.
"""

import os
import json
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Any

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


def generate_correlation_matrix(enriched_df, output_dir):
    """Generate correlation matrix of experiment metrics."""
    if enriched_df is None:
        logger.warning("Cannot generate correlation matrix - missing data")
        return
    
    logger.info("Generating correlation matrix")
    output_dir = Path(output_dir)
    plots_dir = output_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Identify numeric columns for correlation analysis
    numeric_cols = []
    for col in enriched_df.columns:
        if enriched_df[col].dtype in ['int64', 'float64']:
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