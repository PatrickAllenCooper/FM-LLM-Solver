#!/usr/bin/env python
"""
Advanced analysis and visualization of model comparison results.
This script processes the outputs from compare_models.py and generates
additional visualizations and insights.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter, defaultdict
import json
from glob import glob
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"output/logs/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

def load_comparison_data(comparison_dir=None, system_comparison_file=None, summary_file=None):
    """
    Load comparison data from output files.
    
    Parameters
    ----------
    comparison_dir : str, optional
        Directory containing comparison results
    system_comparison_file : str, optional
        Path to system comparison CSV file
    summary_file : str, optional
        Path to summary CSV file
        
    Returns
    -------
    tuple
        (system_df, summary_df) - DataFrames with comparison data
    """
    # If no specific files provided, find most recent in output dir
    if comparison_dir is None:
        comparison_dir = "output/model_comparison"
    
    if system_comparison_file is None:
        # Find system comparison file
        system_comparison_file = os.path.join(comparison_dir, "system_level_comparison.csv")
    
    if summary_file is None:
        # Find most recent summary file
        summary_files = glob(os.path.join(comparison_dir, "model_comparison_report_*.csv"))
        if summary_files:
            summary_file = max(summary_files, key=os.path.getctime)
        else:
            logging.error(f"No summary files found in {comparison_dir}")
            return None, None
    
    logging.info(f"Loading system comparison from: {system_comparison_file}")
    logging.info(f"Loading summary from: {summary_file}")
    
    try:
        system_df = pd.read_csv(system_comparison_file)
        summary_df = pd.read_csv(summary_file)
        return system_df, summary_df
    except Exception as e:
        logging.error(f"Error loading comparison data: {e}")
        return None, None

def analyze_certificate_patterns(system_df):
    """
    Analyze patterns in the generated certificates.
    
    Parameters
    ----------
    system_df : pandas.DataFrame
        DataFrame with system-level comparison data
        
    Returns
    -------
    dict
        Dictionary with certificate pattern analysis
    """
    logging.info("Analyzing certificate patterns")
    
    # Extract certificate forms
    base_certificates = system_df["base_certificate"].dropna().tolist()
    ft_certificates = system_df["ft_certificate"].dropna().tolist()
    
    # Pattern analysis function
    def extract_patterns(certificates):
        patterns = {
            "quadratic": 0,  # Contains x**2, y**2, etc.
            "linear": 0,     # Contains only linear terms
            "higher_order": 0,  # Contains terms with power > 2
            "mixed": 0,      # Contains products of different variables
            "constant_term": 0,  # Contains constant term
            "empty": 0       # Empty or None
        }
        
        term_usage = Counter()  # Count specific term patterns
        variable_usage = Counter()  # Count which variables are used
        
        for cert in certificates:
            if pd.isna(cert) or not cert or cert == "None":
                patterns["empty"] += 1
                continue
                
            # Count variable occurrences
            vars_found = re.findall(r'([a-zA-Z]+)(?:\*\*|\^)?', cert)
            variable_usage.update(vars_found)
            
            # Check for patterns
            has_quadratic = bool(re.search(r'[a-zA-Z]+\*\*2', cert))
            has_higher = bool(re.search(r'[a-zA-Z]+\*\*[3-9]', cert))
            has_mixed = bool(re.search(r'[a-zA-Z]+\s*\*\s*[a-zA-Z]+', cert))
            has_constant = bool(re.search(r'(^|[+\-\s]\s*)[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?($|[+\-\s])', cert))
            
            # Linear only = no powers and no mixed terms
            linear_only = not (has_quadratic or has_higher or has_mixed)
            
            # Update pattern counts
            if has_higher:
                patterns["higher_order"] += 1
            elif has_quadratic and not has_higher:
                patterns["quadratic"] += 1
            elif linear_only and not has_higher and not has_quadratic:
                patterns["linear"] += 1
                
            if has_mixed:
                patterns["mixed"] += 1
                
            if has_constant:
                patterns["constant_term"] += 1
                
            # Analyze specific terms (x**2, y**2, etc.)
            terms = re.findall(r'([a-zA-Z]+(?:\*\*[0-9]+)?)', cert)
            term_usage.update(terms)
        
        return {
            "patterns": patterns,
            "term_usage": dict(term_usage.most_common(10)),
            "variable_usage": dict(variable_usage.most_common(10))
        }
    
    # Analyze both sets of certificates
    base_analysis = extract_patterns(base_certificates)
    ft_analysis = extract_patterns(ft_certificates)
    
    # Compare complexity
    base_avg_length = np.mean([len(c) for c in base_certificates if not pd.isna(c) and c])
    ft_avg_length = np.mean([len(c) for c in ft_certificates if not pd.isna(c) and c])
    
    return {
        "base_model": base_analysis,
        "fine_tuned_model": ft_analysis,
        "complexity_comparison": {
            "base_avg_length": base_avg_length,
            "ft_avg_length": ft_avg_length,
            "length_difference": ft_avg_length - base_avg_length,
            "length_ratio": ft_avg_length / base_avg_length if base_avg_length > 0 else float('inf')
        }
    }

def analyze_system_effectiveness(system_df):
    """
    Analyze which types of systems benefit most from fine-tuning.
    
    Parameters
    ----------
    system_df : pandas.DataFrame
        DataFrame with system-level comparison data
        
    Returns
    -------
    dict
        Dictionary with system effectiveness analysis
    """
    logging.info("Analyzing system effectiveness")
    
    # Extract system IDs and categorize them
    system_categories = {
        "simple": ["simple_linear_1d", "simple_quadratic_2d", "half_plane_barrier", 
                  "linear_double_integrator", "simple_parabolic_barrier", "example_1"],
        "linear": ["linear_stable_1", "linear_coupled", "3d_simple", "4d_linear", "circuit_rlc"],
        "nonlinear": ["example_2", "nonpoly_trig", "van_der_pol", "polynomial_high_degree", 
                     "pendulum", "lotka_volterra"],
        "complex": ["hybrid_switching", "complex_regions", "chemical_reaction"]
    }
    
    # Create reverse mapping from system ID to category
    system_to_category = {}
    for category, systems in system_categories.items():
        for system in systems:
            system_to_category[system] = category
    
    # Add category column to DataFrame
    system_df["category"] = system_df["system_id"].map(
        lambda x: system_to_category.get(x, "other")
    )
    
    # Analyze by category
    category_outcomes = {}
    for category in system_categories.keys():
        category_systems = system_df[system_df["category"] == category]
        if len(category_systems) == 0:
            continue
            
        outcomes = category_systems["verdict_change"].value_counts()
        
        base_success = sum(category_systems["base_verdict"].str.contains("Passed").fillna(False))
        ft_success = sum(category_systems["ft_verdict"].str.contains("Passed").fillna(False))
        
        category_outcomes[category] = {
            "total_systems": len(category_systems),
            "improvements": outcomes.get("Improvement", 0),
            "regressions": outcomes.get("Regression", 0),
            "no_change": outcomes.get("No Change", 0),
            "base_success_rate": base_success / len(category_systems) if len(category_systems) > 0 else 0,
            "ft_success_rate": ft_success / len(category_systems) if len(category_systems) > 0 else 0,
        }
    
    return {
        "category_outcomes": category_outcomes,
        "system_categories": system_df.groupby("category")["verdict_change"].value_counts().unstack().fillna(0).to_dict()
    }

def create_advanced_visualizations(system_df, summary_df, certificate_analysis, system_analysis, output_dir):
    """
    Create advanced visualizations of comparison results.
    
    Parameters
    ----------
    system_df : pandas.DataFrame
        DataFrame with system-level comparison data
    summary_df : pandas.DataFrame
        DataFrame with summary comparison data
    certificate_analysis : dict
        Certificate pattern analysis results
    system_analysis : dict
        System effectiveness analysis results
    output_dir : str
        Directory to save visualizations
        
    Returns
    -------
    list
        Paths to created visualization files
    """
    logging.info(f"Creating advanced visualizations in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_files = []
    
    # ===== 1. Certificate Pattern Comparison =====
    plt.figure(figsize=(14, 8))
    
    # Extract pattern data
    base_patterns = certificate_analysis["base_model"]["patterns"]
    ft_patterns = certificate_analysis["fine_tuned_model"]["patterns"]
    
    # Prepare data for plotting
    pattern_types = list(base_patterns.keys())
    base_counts = [base_patterns[p] for p in pattern_types]
    ft_counts = [ft_patterns[p] for p in pattern_types]
    
    # Plot
    plt.subplot(121)
    x = np.arange(len(pattern_types))
    width = 0.35
    plt.bar(x - width/2, base_counts, width, label='Base Model')
    plt.bar(x + width/2, ft_counts, width, label='Fine-tuned Model')
    plt.xlabel('Certificate Pattern')
    plt.ylabel('Count')
    plt.title('Certificate Pattern Comparison')
    plt.xticks(x, pattern_types, rotation=45)
    plt.legend()
    
    # Term usage comparison (top 5)
    plt.subplot(122)
    base_term_usage = certificate_analysis["base_model"]["term_usage"]
    ft_term_usage = certificate_analysis["fine_tuned_model"]["term_usage"]
    
    # Get common terms between both models
    common_terms = set(list(base_term_usage.keys())[:5]) | set(list(ft_term_usage.keys())[:5])
    common_terms = sorted(list(common_terms))[:7]  # Limit to top 7 for readability
    
    # Prepare data
    base_term_counts = [base_term_usage.get(term, 0) for term in common_terms]
    ft_term_counts = [ft_term_usage.get(term, 0) for term in common_terms]
    
    # Plot
    x = np.arange(len(common_terms))
    plt.bar(x - width/2, base_term_counts, width, label='Base Model')
    plt.bar(x + width/2, ft_term_counts, width, label='Fine-tuned Model')
    plt.xlabel('Term')
    plt.ylabel('Count')
    plt.title('Most Common Certificate Terms')
    plt.xticks(x, common_terms, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    pattern_viz_file = os.path.join(output_dir, "certificate_pattern_comparison.png")
    plt.savefig(pattern_viz_file, dpi=300)
    visualization_files.append(pattern_viz_file)
    plt.close()
    
    # ===== 2. System Category Effectiveness =====
    plt.figure(figsize=(16, 8))
    
    # Extract category data
    categories = list(system_analysis["category_outcomes"].keys())
    
    # Success rate comparison
    plt.subplot(121)
    base_success_rates = [system_analysis["category_outcomes"][cat]["base_success_rate"] * 100 for cat in categories]
    ft_success_rates = [system_analysis["category_outcomes"][cat]["ft_success_rate"] * 100 for cat in categories]
    
    x = np.arange(len(categories))
    plt.bar(x - width/2, base_success_rates, width, label='Base Model')
    plt.bar(x + width/2, ft_success_rates, width, label='Fine-tuned Model')
    plt.xlabel('System Category')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by System Category')
    plt.xticks(x, categories)
    plt.ylim(0, 100)
    plt.legend()
    
    # Outcome by category
    plt.subplot(122)
    
    # Prepare data
    improvement_counts = [system_analysis["category_outcomes"][cat].get("improvements", 0) for cat in categories]
    regression_counts = [system_analysis["category_outcomes"][cat].get("regressions", 0) for cat in categories]
    no_change_counts = [system_analysis["category_outcomes"][cat].get("no_change", 0) for cat in categories]
    
    # Create stacked bar chart
    plt.bar(categories, improvement_counts, label='Improvements', color='green')
    plt.bar(categories, no_change_counts, bottom=improvement_counts, label='No Change', color='gray')
    plt.bar(categories, regression_counts, 
            bottom=[i+n for i, n in zip(improvement_counts, no_change_counts)], 
            label='Regressions', color='red')
    
    plt.xlabel('System Category')
    plt.ylabel('Count')
    plt.title('Outcomes by System Category')
    plt.legend()
    
    plt.tight_layout()
    category_viz_file = os.path.join(output_dir, "system_category_effectiveness.png")
    plt.savefig(category_viz_file, dpi=300)
    visualization_files.append(category_viz_file)
    plt.close()
    
    # ===== 3. Certificate Complexity Comparison =====
    plt.figure(figsize=(10, 6))
    
    # Add complexity indicator to the dataframe
    system_df['base_cert_length'] = system_df['base_certificate'].fillna('').apply(len)
    system_df['ft_cert_length'] = system_df['ft_certificate'].fillna('').apply(len)
    system_df['length_difference'] = system_df['ft_cert_length'] - system_df['base_cert_length']
    
    # Sort by length difference
    sorted_systems = system_df.sort_values('length_difference', ascending=False)
    
    # Plot top 10 systems with biggest difference
    top_systems = sorted_systems.head(10)
    plt.bar(top_systems['system_id'], top_systems['length_difference'], 
            color=[('green' if x >= 0 else 'red') for x in top_systems['length_difference']])
    plt.xlabel('System ID')
    plt.ylabel('Certificate Length Difference (Fine-tuned - Base)')
    plt.title('Top 10 Systems by Certificate Complexity Difference')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    complexity_viz_file = os.path.join(output_dir, "certificate_complexity_comparison.png")
    plt.savefig(complexity_viz_file, dpi=300)
    visualization_files.append(complexity_viz_file)
    plt.close()
    
    # ===== 4. Detailed Comparison by System Type =====
    if 'category' in system_df.columns:
        # Create a scatter plot showing both models' performance by system category
        plt.figure(figsize=(12, 8))
        
        # Create a categorical success level for each verdict
        def verdict_to_success_level(verdict):
            if pd.isna(verdict):
                return 0
            elif "Failed" in verdict:
                return 1
            elif "Basic" in verdict:
                return 2
            elif "Numerical" in verdict:
                return 3
            elif "SOS" in verdict:
                return 4
            else:
                return 0
        
        system_df['base_success_level'] = system_df['base_verdict'].apply(verdict_to_success_level)
        system_df['ft_success_level'] = system_df['ft_verdict'].apply(verdict_to_success_level)
        
        # Create scatter plot
        categories = system_df['category'].unique()
        colors = {'simple': 'blue', 'linear': 'green', 'nonlinear': 'red', 'complex': 'purple', 'other': 'gray'}
        
        for category in categories:
            category_data = system_df[system_df['category'] == category]
            plt.scatter(category_data['base_success_level'], category_data['ft_success_level'], 
                        label=category, color=colors.get(category, 'gray'), alpha=0.7, s=100)
        
        # Add reference line (y=x)
        max_level = max(system_df['base_success_level'].max(), system_df['ft_success_level'].max())
        plt.plot([0, max_level+1], [0, max_level+1], 'k--', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Base Model Success Level')
        plt.ylabel('Fine-tuned Model Success Level')
        plt.title('Base vs. Fine-tuned Model Success Level by System Category')
        plt.xticks([0, 1, 2, 3, 4], ['None', 'Failed', 'Basic', 'Numerical', 'SOS'])
        plt.yticks([0, 1, 2, 3, 4], ['None', 'Failed', 'Basic', 'Numerical', 'SOS'])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        category_scatter_file = os.path.join(output_dir, "system_category_scatter.png")
        plt.savefig(category_scatter_file, dpi=300)
        visualization_files.append(category_scatter_file)
        plt.close()
    
    return visualization_files

def generate_comprehensive_report(system_df, summary_df, certificate_analysis, system_analysis, visualization_files, output_dir):
    """
    Generate a comprehensive HTML report of the analysis.
    
    Parameters
    ----------
    system_df : pandas.DataFrame
        DataFrame with system-level comparison data
    summary_df : pandas.DataFrame
        DataFrame with summary comparison data
    certificate_analysis : dict
        Certificate pattern analysis results
    system_analysis : dict
        System effectiveness analysis results
    visualization_files : list
        Paths to visualization files
    output_dir : str
        Directory to save report
        
    Returns
    -------
    str
        Path to generated report
    """
    logging.info("Generating comprehensive report")
    
    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Analysis - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .visualization {{ margin: 20px 0; text-align: center; }}
            .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            .neutral {{ color: #7f8c8d; }}
            .summary-box {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin-bottom: 20px; 
                background-color: #f8f9fa; 
                border-radius: 4px; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Comparison Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary-box">
                <h2>Summary Results</h2>
                <p>Base Model: <strong>{summary_df.iloc[0,1] if not summary_df.empty else "N/A"}</strong></p>
                <p>Fine-tuned Model with RAG: <strong>{summary_df.iloc[0,2] if not summary_df.empty else "N/A"}</strong></p>
                <p>Total Systems Evaluated: <strong>{len(system_df)}</strong></p>
                
                <h3>Outcome Distribution</h3>
                <ul>
                    <li><span class="success">Improvements: {sum(system_df['verdict_change'] == 'Improvement')}</span></li>
                    <li><span class="neutral">No Change: {sum(system_df['verdict_change'] == 'No Change')}</span></li>
                    <li><span class="failure">Regressions: {sum(system_df['verdict_change'] == 'Regression')}</span></li>
                </ul>
            </div>
            
            <h2>Visualizations</h2>
    """
    
    # Add visualizations
    for viz_file in visualization_files:
        viz_name = os.path.basename(viz_file).replace('.png', '').replace('_', ' ').title()
        html_content += f"""
            <div class="visualization">
                <h3>{viz_name}</h3>
                <img src="{os.path.basename(viz_file)}" alt="{viz_name}">
            </div>
        """
    
    # Certificate Pattern Analysis
    html_content += """
            <h2>Certificate Pattern Analysis</h2>
            <h3>Certificate Complexity</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Base Model</th>
                    <th>Fine-tuned Model</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Average Certificate Length</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f} ({:.1f}%)</td>
                </tr>
            </table>
    """.format(
        certificate_analysis["complexity_comparison"]["base_avg_length"],
        certificate_analysis["complexity_comparison"]["ft_avg_length"],
        certificate_analysis["complexity_comparison"]["length_difference"],
        (certificate_analysis["complexity_comparison"]["length_ratio"] - 1) * 100
    )
    
    # Pattern Summary
    html_content += """
            <h3>Certificate Pattern Distribution</h3>
            <table>
                <tr>
                    <th>Pattern</th>
                    <th>Base Model</th>
                    <th>Fine-tuned Model</th>
                    <th>Difference</th>
                </tr>
    """
    
    for pattern in certificate_analysis["base_model"]["patterns"].keys():
        base_count = certificate_analysis["base_model"]["patterns"][pattern]
        ft_count = certificate_analysis["fine_tuned_model"]["patterns"][pattern]
        diff = ft_count - base_count
        html_content += f"""
                <tr>
                    <td>{pattern}</td>
                    <td>{base_count}</td>
                    <td>{ft_count}</td>
                    <td>{'<span class="success">+' + str(diff) + '</span>' if diff > 0 else '<span class="failure">' + str(diff) + '</span>' if diff < 0 else '0'}</td>
                </tr>
        """
    
    html_content += """
            </table>
    """
    
    # System Category Analysis
    html_content += """
            <h2>System Category Analysis</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Total Systems</th>
                    <th>Base Success Rate</th>
                    <th>Fine-tuned Success Rate</th>
                    <th>Improvements</th>
                    <th>No Change</th>
                    <th>Regressions</th>
                </tr>
    """
    
    for category, data in system_analysis["category_outcomes"].items():
        html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{data["total_systems"]}</td>
                    <td>{data["base_success_rate"]*100:.1f}%</td>
                    <td>{data["ft_success_rate"]*100:.1f}%</td>
                    <td>{data.get("improvements", 0)}</td>
                    <td>{data.get("no_change", 0)}</td>
                    <td>{data.get("regressions", 0)}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>System-Level Comparison</h2>
            <table>
                <tr>
                    <th>System ID</th>
                    <th>Category</th>
                    <th>Base Verdict</th>
                    <th>Fine-tuned Verdict</th>
                    <th>Outcome</th>
                    <th>Base Certificate</th>
                    <th>Fine-tuned Certificate</th>
                </tr>
    """
    
    # Add each system's details
    for _, row in system_df.iterrows():
        outcome_class = "success" if row["verdict_change"] == "Improvement" else "failure" if row["verdict_change"] == "Regression" else "neutral"
        category = row.get("category", "N/A")
        
        html_content += f"""
                <tr>
                    <td>{row["system_id"]}</td>
                    <td>{category}</td>
                    <td>{row["base_verdict"]}</td>
                    <td>{row["ft_verdict"]}</td>
                    <td class="{outcome_class}">{row["verdict_change"]}</td>
                    <td><code>{row["base_certificate"]}</code></td>
                    <td><code>{row["ft_certificate"]}</code></td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_file = os.path.join(output_dir, f"comparison_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logging.info(f"Comprehensive report saved to: {report_file}")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description="Advanced analysis and visualization of model comparison results")
    parser.add_argument("--comparison-dir", type=str, default="output/model_comparison",
                        help="Directory containing comparison results")
    parser.add_argument("--system-file", type=str, default=None,
                        help="Path to system comparison CSV file (default: auto-detect)")
    parser.add_argument("--summary-file", type=str, default=None,
                        help="Path to summary CSV file (default: most recent)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations and report (default: comparison_dir/analysis_TIMESTAMP)")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.comparison_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    logging.info(f"Starting analysis with output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load comparison data
    system_df, summary_df = load_comparison_data(
        args.comparison_dir, args.system_file, args.summary_file
    )
    
    if system_df is None or summary_df is None:
        logging.error("Failed to load comparison data. Exiting.")
        return 1
    
    # Analyze certificate patterns
    certificate_analysis = analyze_certificate_patterns(system_df)
    
    # Analyze system effectiveness
    system_analysis = analyze_system_effectiveness(system_df)
    
    # Create visualizations
    visualization_files = create_advanced_visualizations(
        system_df, summary_df, certificate_analysis, system_analysis, args.output_dir
    )
    
    # Generate comprehensive report
    report_file = generate_comprehensive_report(
        system_df, summary_df, certificate_analysis, system_analysis, 
        visualization_files, args.output_dir
    )
    
    logging.info(f"Analysis complete. Report saved to: {report_file}")
    print(f"\nAnalysis complete. Open the report at:\n{report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 