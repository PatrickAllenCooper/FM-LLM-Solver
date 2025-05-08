#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_verification_data(json_path):
    """Load verification data from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded verification data for {len(data)} systems")
        return data
    except Exception as e:
        logging.error(f"Failed to load verification data: {e}")
        return None

def parse_barrier_certificate(certificate_str):
    """Parse a barrier certificate string into a SymPy expression."""
    try:
        # Define common variables
        x, y, z = sp.symbols('x y z')
        expr = parse_expr(certificate_str, local_dict={'x': x, 'y': y, 'z': z})
        return expr, [x, y, z]
    except Exception as e:
        logging.error(f"Failed to parse barrier certificate '{certificate_str}': {e}")
        return None, None

def plot_barrier_certificate(expr, variables, system_id, title=None, xlim=(-2, 2), ylim=(-2, 2), resolution=100, 
                             init_points=None, unsafe_points=None, lie_points=None, show_violations=True):
    """Plot a 3D surface of the barrier certificate and optionally violation points."""
    if expr is None:
        return None
        
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create mesh grid
    x_var, y_var = variables[0], variables[1]
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Convert SymPy expression to NumPy function
    f_np = sp.lambdify([x_var, y_var], expr, 'numpy')
    Z = f_np(X, Y)
    
    # Plot surface
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Plot zero level (barrier)
    ax1.contour(X, Y, Z, [0], colors='g', linestyles='solid', linewidths=2)
    
    # Add violation points if available
    if show_violations:
        if init_points and len(init_points) > 0:
            x_init = [p['point']['x'] for p in init_points if isinstance(p, dict) and 'point' in p]
            y_init = [p['point']['y'] for p in init_points if isinstance(p, dict) and 'point' in p]
            z_init = [p['B_value'] for p in init_points if isinstance(p, dict) and 'B_value' in p]
            if x_init and y_init and z_init:
                ax1.scatter(x_init, y_init, z_init, color='red', s=50, label='Initial Set Violations')
            
        if unsafe_points and len(unsafe_points) > 0:
            x_unsafe = [p['point']['x'] for p in unsafe_points if isinstance(p, dict) and 'point' in p]
            y_unsafe = [p['point']['y'] for p in unsafe_points if isinstance(p, dict) and 'point' in p]
            z_unsafe = [p['B_value'] for p in unsafe_points if isinstance(p, dict) and 'B_value' in p]
            if x_unsafe and y_unsafe and z_unsafe:
                ax1.scatter(x_unsafe, y_unsafe, z_unsafe, color='orange', s=50, label='Unsafe Set Violations')
            
        if lie_points and len(lie_points) > 0:
            x_lie = [p['point']['x'] for p in lie_points if isinstance(p, dict) and 'point' in p]
            y_lie = [p['point']['y'] for p in lie_points if isinstance(p, dict) and 'point' in p]
            z_lie = [f_np(p['point']['x'], p['point']['y']) for p in lie_points if isinstance(p, dict) and 'point' in p]
            if x_lie and y_lie and z_lie:
                ax1.scatter(x_lie, y_lie, z_lie, color='purple', s=50, label='Lie Derivative Violations')
    
    # Set labels and title
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('B(x,y)')
    ax1.set_title(f"3D View of Barrier Certificate: {expr}")
    
    # 2D Contour Plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.coolwarm)
    barrier = ax2.contour(X, Y, Z, [0], colors='green', linewidths=2)
    ax2.clabel(barrier, inline=True, fontsize=10, fmt='B=0')
    
    # Add violation points to contour plot
    if show_violations:
        if init_points and len(init_points) > 0:
            x_init = [p['point']['x'] for p in init_points if isinstance(p, dict) and 'point' in p]
            y_init = [p['point']['y'] for p in init_points if isinstance(p, dict) and 'point' in p]
            if x_init and y_init:
                ax2.scatter(x_init, y_init, color='red', s=50, label='Initial Set Violations')
            
        if unsafe_points and len(unsafe_points) > 0:
            x_unsafe = [p['point']['x'] for p in unsafe_points if isinstance(p, dict) and 'point' in p]
            y_unsafe = [p['point']['y'] for p in unsafe_points if isinstance(p, dict) and 'point' in p]
            if x_unsafe and y_unsafe:
                ax2.scatter(x_unsafe, y_unsafe, color='orange', s=50, label='Unsafe Set Violations')
            
        if lie_points and len(lie_points) > 0:
            x_lie = [p['point']['x'] for p in lie_points if isinstance(p, dict) and 'point' in p]
            y_lie = [p['point']['y'] for p in lie_points if isinstance(p, dict) and 'point' in p]
            if x_lie and y_lie:
                ax2.scatter(x_lie, y_lie, color='purple', s=50, label='Lie Derivative Violations')
    
    # Add color bar
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('B(x,y) value')
    
    # Set labels and title
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f"2D View with B(x,y)=0 Barrier (System {system_id})")
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.9)
    
    return fig

def generate_verification_report(verification_data):
    """Generate a summary report of the verification results."""
    total_systems = len(verification_data)
    passed_systems = 0
    failed_systems = 0
    error_systems = 0
    
    # Summary statistics
    for system_data in verification_data:
        verdict = system_data.get('verification_details', {}).get('final_verdict', 'Unknown')
        if "Passed" in verdict:
            passed_systems += 1
        elif "Failed" in verdict:
            failed_systems += 1
        else:
            error_systems += 1
    
    print("\n=== Verification Report ===")
    print(f"Total Systems: {total_systems}")
    print(f"Passed: {passed_systems} ({passed_systems/total_systems*100:.1f}%)")
    print(f"Failed: {failed_systems} ({failed_systems/total_systems*100:.1f}%)")
    print(f"Errors: {error_systems} ({error_systems/total_systems*100:.1f}%)")
    print("=========================")
    
    # Create a DataFrame for detailed system-by-system report
    report_data = []
    for system_data in verification_data:
        system_id = system_data.get('system_id', 'Unknown')
        candidate_b = system_data.get('candidate_B', 'Unknown')
        details = system_data.get('verification_details', {})
        
        # Extract key verification information
        system_report = {
            'system_id': system_id,
            'barrier_certificate': candidate_b,
            'final_verdict': details.get('final_verdict', 'Unknown'),
            'numerical_lie_check': details.get('numerical_sampling_lie_passed', None),
            'numerical_boundary_check': details.get('numerical_sampling_boundary_passed', None),
            'symbolic_checks': details.get('symbolic_lie_check_passed', None),
            'sos_attempted': details.get('sos_attempted', False),
            'sos_passed': details.get('sos_passed', None),
            'verification_time': details.get('verification_time_seconds', 0)
        }
        report_data.append(system_report)
    
    # Convert to DataFrame and return
    report_df = pd.DataFrame(report_data)
    return report_df

def main():
    parser = argparse.ArgumentParser(description="Visualize barrier certificate verification results")
    parser.add_argument("--data", type=str, required=True, help="Path to verification data JSON file")
    parser.add_argument("--system", type=str, default=None, help="Specific system ID to visualize (default: all)")
    parser.add_argument("--save", action="store_true", help="Save plots to files instead of displaying")
    parser.add_argument("--output", type=str, default="verification_plots", help="Directory to save plots (default: verification_plots)")
    parser.add_argument("--report", action="store_true", help="Generate verification report")
    
    args = parser.parse_args()
    
    # Load verification data
    verification_data = load_verification_data(args.data)
    if not verification_data:
        return
    
    # Generate report if requested
    if args.report:
        report_df = generate_verification_report(verification_data)
        print("\nDetailed System Report:")
        print(report_df.to_string())
        
        # Save report to CSV
        report_path = os.path.join(os.path.dirname(args.data), "verification_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\nReport saved to {report_path}")
    
    # Create output directory if saving plots
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    # Filter systems if specific system ID is provided
    if args.system:
        verification_data = [data for data in verification_data if data.get('system_id') == args.system]
        if not verification_data:
            logging.error(f"No verification data found for system ID: {args.system}")
            return
    
    # Process each system
    for system_data in verification_data:
        system_id = system_data.get('system_id', 'Unknown')
        candidate_b = system_data.get('candidate_B', 'Unknown')
        details = system_data.get('verification_details', {})
        
        logging.info(f"Visualizing system {system_id} with certificate: {candidate_b}")
        
        # Parse barrier certificate
        expr, variables = parse_barrier_certificate(candidate_b)
        if expr is None:
            continue
        
        # Get violation points
        num_sampling_data = details.get('numerical_sampling_details', {})
        # Try to extract these from overall verification details as well to handle older format
        init_points = num_sampling_data.get('init_violation_points', [])
        unsafe_points = num_sampling_data.get('unsafe_violation_points', [])
        lie_points = num_sampling_data.get('lie_violation_points', [])
        
        # Generate title based on verification result
        verdict = details.get('final_verdict', 'Unknown')
        title = f"System {system_id}: {verdict}"
        
        # Plot barrier certificate
        fig = plot_barrier_certificate(
            expr, variables, system_id, title=title,
            init_points=init_points, unsafe_points=unsafe_points, lie_points=lie_points
        )
        
        if fig is not None:
            if args.save:
                output_path = os.path.join(args.output, f"system_{system_id}_barrier.png")
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved plot to {output_path}")
            else:
                plt.show()
            plt.close(fig)
    
    if args.save and verification_data:
        print(f"\nAll plots saved to {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main() 