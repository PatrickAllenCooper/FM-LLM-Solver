#!/usr/bin/env python
"""
Compare results between Qwen 7B and Qwen 15B model runs.
This script analyzes the performance differences between model sizes.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_comparison_data(qwen7b_dir="output/model_comparison", qwen15b_dir="output/model_comparison_qwen15b"):
    """
    Load results from both model size comparisons.
    
    Parameters
    ----------
    qwen7b_dir : str
        Directory with Qwen 7B comparison results
    qwen15b_dir : str
        Directory with Qwen 15B comparison results
        
    Returns
    -------
    tuple
        DataFrames with system-level and summary data for both models
    """
    # Find the most recent comparison report in each directory
    qwen7b_summary = max(glob(os.path.join(qwen7b_dir, "model_comparison_report_*.csv")), 
                         key=os.path.getctime, default=None)
    qwen15b_summary = max(glob(os.path.join(qwen15b_dir, "model_comparison_report_*.csv")),
                         key=os.path.getctime, default=None)
    
    # Find system-level comparison files
    qwen7b_system = os.path.join(qwen7b_dir, "system_level_comparison.csv")
    qwen15b_system = os.path.join(qwen15b_dir, "system_level_comparison.csv")
    
    if not all([qwen7b_summary, qwen15b_summary, os.path.exists(qwen7b_system), os.path.exists(qwen15b_system)]):
        logging.error("Couldn't find all required comparison files. Make sure both model comparisons have been run.")
        return None, None, None, None
    
    try:
        # Load the data
        qwen7b_summary_df = pd.read_csv(qwen7b_summary)
        qwen15b_summary_df = pd.read_csv(qwen15b_summary)
        qwen7b_system_df = pd.read_csv(qwen7b_system)
        qwen15b_system_df = pd.read_csv(qwen15b_system)
        
        # Add model size identifier
        qwen7b_summary_df['model_size'] = '7B'
        qwen15b_summary_df['model_size'] = '15B'
        qwen7b_system_df['model_size'] = '7B'
        qwen15b_system_df['model_size'] = '15B'
        
        logging.info(f"Loaded Qwen 7B summary from {qwen7b_summary}")
        logging.info(f"Loaded Qwen 15B summary from {qwen15b_summary}")
        
        return qwen7b_summary_df, qwen15b_summary_df, qwen7b_system_df, qwen15b_system_df
        
    except Exception as e:
        logging.error(f"Error loading comparison data: {e}")
        return None, None, None, None

def extract_metric_values(summary_df):
    """Extract numeric values from the summary dataframe metric strings."""
    metrics_dict = {}
    
    # Process each model (base and fine-tuned) separately
    for model_col in ['Base Model', 'Fine-tuned Model with RAG']:
        metrics = {}
        
        for idx, row in summary_df.iterrows():
            metric_name = row['Metric']
            value_str = row[model_col]
            
            # Skip non-numeric metrics
            if metric_name == 'Total Systems Evaluated':
                if isinstance(value_str, (int, float)):
                    metrics[metric_name] = value_str
                else:
                    try:
                        metrics[metric_name] = int(value_str)
                    except:
                        metrics[metric_name] = np.nan
                continue
                
            # Extract numeric portion from strings like "10 (50.0%)"
            if isinstance(value_str, str) and '(' in value_str:
                count_str = value_str.split('(')[0].strip()
                pct_str = value_str.split('(')[1].replace(')', '').replace('%', '').strip()
                
                try:
                    metrics[f"{metric_name}_count"] = int(count_str)
                    metrics[f"{metric_name}_pct"] = float(pct_str)
                except:
                    metrics[f"{metric_name}_count"] = np.nan
                    metrics[f"{metric_name}_pct"] = np.nan
            else:
                # Try to convert to float (for processing time)
                try:
                    metrics[metric_name] = float(value_str)
                except:
                    metrics[metric_name] = np.nan
        
        # Add model column name to metrics keys
        model_key = 'base' if 'Base' in model_col else 'ft'
        metrics_dict[model_key] = metrics
    
    return metrics_dict

def generate_model_size_comparison(qwen7b_summary, qwen15b_summary, qwen7b_system, qwen15b_system, output_dir):
    """
    Generate comparison charts and analysis between model sizes.
    
    Parameters
    ----------
    qwen7b_summary, qwen15b_summary : DataFrame
        Summary metrics for each model size
    qwen7b_system, qwen15b_system : DataFrame
        System-level comparisons for each model size
    output_dir : str
        Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract numeric metrics from summary dataframes
    qwen7b_metrics = extract_metric_values(qwen7b_summary)
    qwen15b_metrics = extract_metric_values(qwen15b_summary)
    
    # === CHART 1: Comparison of success rates ===
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    metric_keys = [
        ('Successful Generations_pct', 'Generation Success'),
        ('Successfully Parsed Certificates_pct', 'Certificate Parsing'),
        ('Passed Numerical Checks_pct', 'Numerical Verification'),
        ('Passed SOS Checks_pct', 'SOS Verification'),
        ('Overall Success Rate_pct', 'Overall Success')
    ]
    
    model_configs = [
        ('base', '7B', 'Qwen 7B Base'),
        ('ft', '7B', 'Qwen 7B Fine-tuned'),
        ('base', '15B', 'Qwen 15B Base'),
        ('ft', '15B', 'Qwen 15B Fine-tuned')
    ]
    
    # Collect data for bar chart
    chart_data = []
    for metric_key, metric_label in metric_keys:
        for model_type, model_size, model_label in model_configs:
            metrics = qwen7b_metrics if model_size == '7B' else qwen15b_metrics
            if model_type in metrics and metric_key in metrics[model_type]:
                value = metrics[model_type][metric_key]
                chart_data.append({
                    'Metric': metric_label,
                    'Model': model_label,
                    'Success Rate (%)': value,
                    'Model Size': model_size,
                    'Model Type': 'Base' if model_type == 'base' else 'Fine-tuned'
                })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create bar chart
    ax = plt.subplot(111)
    sns.barplot(x='Metric', y='Success Rate (%)', hue='Model', data=chart_df, ax=ax)
    plt.title('Success Rates by Model Size and Type', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xticks(rotation=15)
    plt.legend(title='Model', fontsize=12)
    plt.tight_layout()
    
    # Save chart
    success_chart_path = os.path.join(output_dir, 'model_size_success_comparison.png')
    plt.savefig(success_chart_path, dpi=300)
    plt.close()
    
    # === CHART 2: Improvement from Base to Fine-tuned by Size ===
    plt.figure(figsize=(12, 6))
    
    improvement_data = []
    for metric_key, metric_label in metric_keys:
        for model_size, metrics in [('7B', qwen7b_metrics), ('15B', qwen15b_metrics)]:
            if 'base' in metrics and 'ft' in metrics and metric_key in metrics['base'] and metric_key in metrics['ft']:
                base_value = metrics['base'][metric_key]
                ft_value = metrics['ft'][metric_key]
                improvement = ft_value - base_value
                improvement_data.append({
                    'Metric': metric_label,
                    'Model Size': model_size,
                    'Improvement (%)': improvement
                })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Create improvement chart
    ax = plt.subplot(111)
    sns.barplot(x='Metric', y='Improvement (%)', hue='Model Size', data=improvement_df, ax=ax)
    plt.title('Improvement from Fine-tuning by Model Size', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Improvement (%)', fontsize=14)
    plt.xticks(rotation=15)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend(title='Model Size', fontsize=12)
    plt.tight_layout()
    
    # Save chart
    improvement_chart_path = os.path.join(output_dir, 'model_size_improvement_comparison.png')
    plt.savefig(improvement_chart_path, dpi=300)
    plt.close()
    
    # === CHART 3: System-level verdict changes comparison ===
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    system_outcomes = []
    
    for model_size, system_df in [('7B', qwen7b_system), ('15B', qwen15b_system)]:
        outcomes = system_df['verdict_change'].value_counts()
        system_outcomes.append({
            'Model Size': model_size,
            'Improvements': outcomes.get('Improvement', 0),
            'No Change': outcomes.get('No Change', 0),
            'Regressions': outcomes.get('Regression', 0),
            'Different Failure': outcomes.get('Different Failure Mode', 0)
        })
    
    system_outcome_df = pd.DataFrame(system_outcomes)
    system_outcome_df = system_outcome_df.set_index('Model Size')
    
    # Calculate percentages
    total_systems = {
        '7B': len(qwen7b_system),
        '15B': len(qwen15b_system)
    }
    
    system_outcome_pct = system_outcome_df.copy()
    for size in system_outcome_pct.index:
        system_outcome_pct.loc[size] = system_outcome_pct.loc[size] / total_systems[size] * 100
    
    # Create stacked bar chart
    ax = plt.subplot(111)
    system_outcome_pct.plot(kind='bar', stacked=True, ax=ax, 
                          color=['green', 'gray', 'red', 'orange'])
    
    plt.title('System-Level Outcomes by Model Size', fontsize=16)
    plt.xlabel('Model Size', fontsize=14)
    plt.ylabel('Percentage of Systems (%)', fontsize=14)
    plt.legend(title='Outcome')
    
    # Add count labels on bars
    for i, size in enumerate(system_outcome_df.index):
        total = 0
        for col in system_outcome_df.columns:
            count = system_outcome_df.loc[size, col]
            if count > 0:
                pct = system_outcome_pct.loc[size, col]
                plt.text(i, total + (pct/2), f"{count}", 
                       ha='center', va='center', fontweight='bold')
            total += system_outcome_pct.loc[size, col]
    
    plt.tight_layout()
    
    # Save chart
    outcomes_chart_path = os.path.join(output_dir, 'model_size_system_outcomes.png')
    plt.savefig(outcomes_chart_path, dpi=300)
    plt.close()
    
    # === Generate Summary Report ===
    summary_md = f"""# Qwen Model Size Comparison Report

## Overview
This report compares the performance of Qwen 7B and Qwen 15B models on barrier certificate generation.

## Success Rates
![Success Rates by Model](model_size_success_comparison.png)

## Improvement from Fine-tuning
![Improvement from Fine-tuning](model_size_improvement_comparison.png)

## System-Level Outcomes
![System Outcomes](model_size_system_outcomes.png)

## Key Findings

### Base Model Performance
- Qwen 7B base model success rate: {qwen7b_metrics['base'].get('Overall Success Rate_pct', 'N/A')}%
- Qwen 15B base model success rate: {qwen15b_metrics['base'].get('Overall Success Rate_pct', 'N/A')}%
- Size advantage: {qwen15b_metrics['base'].get('Overall Success Rate_pct', 0) - qwen7b_metrics['base'].get('Overall Success Rate_pct', 0):.1f}%

### Fine-tuned Model Performance
- Qwen 7B fine-tuned model success rate: {qwen7b_metrics['ft'].get('Overall Success Rate_pct', 'N/A')}%
- Qwen 15B fine-tuned model success rate: {qwen15b_metrics['ft'].get('Overall Success Rate_pct', 'N/A')}%
- Size advantage: {qwen15b_metrics['ft'].get('Overall Success Rate_pct', 0) - qwen7b_metrics['ft'].get('Overall Success Rate_pct', 0):.1f}%

### Improvement from Fine-tuning
- Qwen 7B improvement: {qwen7b_metrics['ft'].get('Overall Success Rate_pct', 0) - qwen7b_metrics['base'].get('Overall Success Rate_pct', 0):.1f}%
- Qwen 15B improvement: {qwen15b_metrics['ft'].get('Overall Success Rate_pct', 0) - qwen15b_metrics['base'].get('Overall Success Rate_pct', 0):.1f}%

## System-Level Changes
- Qwen 7B systems improved: {system_outcome_df.loc['7B', 'Improvements']} ({system_outcome_pct.loc['7B', 'Improvements']:.1f}%)
- Qwen 15B systems improved: {system_outcome_df.loc['15B', 'Improvements']} ({system_outcome_pct.loc['15B', 'Improvements']:.1f}%)

## Conclusion
- Overall performance comparison between model sizes
- How size impacts fine-tuning benefits
- Which model size provides the best balance of performance and efficiency

Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'model_size_comparison_report.md')
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    
    return {
        'success_chart': success_chart_path,
        'improvement_chart': improvement_chart_path,
        'outcomes_chart': outcomes_chart_path,
        'summary_report': summary_path
    }

def main():
    parser = argparse.ArgumentParser(description="Compare performance between Qwen 7B and Qwen 15B models")
    parser.add_argument("--qwen7b-dir", type=str, default="output/model_comparison",
                        help="Directory containing Qwen 7B comparison results")
    parser.add_argument("--qwen15b-dir", type=str, default="output/model_comparison_qwen15b",
                        help="Directory containing Qwen 15B comparison results")
    parser.add_argument("--output-dir", type=str, default="output/model_size_comparison",
                        help="Directory to save comparison results")
    args = parser.parse_args()
    
    # Load comparison data for both model sizes
    qwen7b_summary, qwen15b_summary, qwen7b_system, qwen15b_system = load_comparison_data(
        args.qwen7b_dir, args.qwen15b_dir
    )
    
    if qwen7b_summary is None or qwen15b_summary is None:
        logging.error("Failed to load comparison data. Make sure both model comparisons have been run.")
        sys.exit(1)
    
    # Generate comparison of model sizes
    output_files = generate_model_size_comparison(
        qwen7b_summary, qwen15b_summary, qwen7b_system, qwen15b_system, args.output_dir
    )
    
    logging.info(f"Model size comparison completed successfully.")
    logging.info(f"Summary report: {output_files['summary_report']}")
    print(f"\nModel size comparison complete. Report saved to: {output_files['summary_report']}")
    
if __name__ == "__main__":
    main() 