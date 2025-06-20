#!/usr/bin/env python3
"""
Generate discrete-time barrier certificate training data.

This script creates training examples specifically for discrete-time systems
using proper x[k+1] notation and discrete barrier conditions.
"""

import json
import numpy as np
from pathlib import Path

def create_discrete_time_training_data():
    """Generate training examples for discrete-time barrier certificates."""
    
    examples = []
    
    # Example 1: Simple stable linear system
    examples.append({
        "instruction": "Generate a barrier certificate for the discrete-time system",
        "input": """System Dynamics:
x[k+1] = 0.8*x[k] + 0.1*y[k]  
y[k+1] = -0.1*x[k] + 0.9*y[k]

Initial Set: x[0]**2 + y[0]**2 <= 0.1
Unsafe Set: x[k]**2 + y[k]**2 >= 2.0
State Variables: x, y""",
        "output": """For this discrete-time linear system, I need to find B(x,y) such that:
1. B(x[0], y[0]) <= 0 for initial states (safe inside initial set)
2. B(x[k+1], y[k+1]) - B(x[k], y[k]) <= 0 (non-increasing along trajectories)  
3. B(x,y) >= 0 at unsafe boundary (positive in unsafe region)

Let me try B(x,y) = x**2 + y**2 - 2.0

Verification:
- In initial set: B <= 0.1 - 2.0 = -1.9 < 0 ✓
- At unsafe boundary: B = 2.0 - 2.0 = 0 ✓  
- For difference: ΔB = B(x[k+1], y[k+1]) - B(x[k], y[k])
  = (0.8x + 0.1y)² + (-0.1x + 0.9y)² - (x² + y²)
  = 0.64x² + 0.16xy + 0.01y² + 0.01x² - 0.18xy + 0.81y² - x² - y²
  = -0.34x² - 0.02xy - 0.18y²

The matrix [[-0.34, -0.01], [-0.01, -0.18]] is negative definite, so ΔB < 0 ✓

Barrier Certificate: B(x,y) = x**2 + y**2 - 2.0""",
        "metadata": {"source": "discrete_time_manual", "system_type": "discrete"}
    })
    
    # Example 2: Nonlinear discrete system  
    examples.append({
        "instruction": "Generate a barrier certificate for the discrete-time system",
        "input": """System Dynamics:
x[k+1] = 0.9*x[k] - 0.1*y[k]**2
y[k+1] = 0.1*x[k] + 0.8*y[k]

Initial Set: x[0]**2 + y[0]**2 <= 0.25
Unsafe Set: x[k] >= 1.5
State Variables: x, y""",
        "output": """For this discrete-time nonlinear system with unsafe region x >= 1.5, I need a barrier that separates the regions.

Since the initial set has radius 0.5 and unsafe region starts at x = 1.5, I'll try a simple linear barrier:
B(x,y) = 1.5 - x

Verification:
- In initial set: max |x[0]| ≤ 0.5, so B ≥ 1.5 - 0.5 = 1.0 > 0 (should be ≤ 0)

This doesn't work. Let me try B(x,y) = x - 1.5 (flipped sign):
- In initial set: B ≤ 0.5 - 1.5 = -1.0 < 0 ✓
- At unsafe boundary x = 1.5: B = 0 ✓
- For difference: ΔB = (0.9*x[k] - 0.1*y[k]²) - 1.5 - (x[k] - 1.5)
  = 0.9*x[k] - 0.1*y[k]² - x[k] = -0.1*x[k] - 0.1*y[k]²

For points where |x|, |y| are small, ΔB ≈ -0.1*x[k] - 0.1*y[k]² ≤ 0 when x[k] ≥ 0.
This barrier works for trajectories starting near origin.

Barrier Certificate: B(x,y) = x - 1.5""",
        "metadata": {"source": "discrete_time_manual", "system_type": "discrete"}
    })
    
    # Example 3: Stable rotation matrix
    examples.append({
        "instruction": "Generate a barrier certificate for the discrete-time system", 
        "input": """System Dynamics:
x[k+1] = 0.95*x[k] - 0.05*y[k]
y[k+1] = 0.05*x[k] + 0.95*y[k]

Initial Set: x[0]**2 + y[0]**2 <= 0.01
Unsafe Set: x[k]**2 + y[k]**2 >= 1.0
State Variables: x, y""",
        "output": """This is a stable discrete-time linear system (rotation + contraction).
The matrix A = [[0.95, -0.05], [0.05, 0.95]] has spectral radius ρ(A) = 0.95 < 1.

For such systems, B(x,y) = x² + y² - c works well where c is between the initial and unsafe boundaries.

Let me try B(x,y) = x**2 + y**2 - 1.0:

Verification:
- Initial set: B ≤ 0.01 - 1.0 = -0.99 < 0 ✓
- Unsafe boundary: B = 1.0 - 1.0 = 0 ✓  
- For the system ||x[k+1]||² = ||A*x[k]||² ≤ ||A||²*||x[k]||²

Since ||A||² = λ_max(A^T*A) where A^T*A = [[0.9025, 0], [0, 0.9025]]:
||x[k+1]||² = 0.9025*||x[k]||²

Therefore: ΔB = ||x[k+1]||² - ||x[k]||² = (0.9025 - 1)*||x[k]||² = -0.0975*||x[k]||² ≤ 0 ✓

Barrier Certificate: B(x,y) = x**2 + y**2 - 1.0""",
        "metadata": {"source": "discrete_time_manual", "system_type": "discrete"}
    })
    
    # Example 4: More complex nonlinear case
    examples.append({
        "instruction": "Generate a barrier certificate for the discrete-time system",
        "input": """System Dynamics:
x[k+1] = 0.7*x[k] + 0.2*sin(y[k])
y[k+1] = 0.8*y[k] - 0.1*x[k]**2

Initial Set: x[0]**2 + y[0]**2 <= 0.04
Unsafe Set: abs(x[k]) >= 1.2 or abs(y[k]) >= 1.2  
State Variables: x, y""",
        "output": """This nonlinear discrete-time system has trigonometric and polynomial terms.
The unsafe set defines a "box" boundary: |x| ≥ 1.2 or |y| ≥ 1.2.

For such cases, I'll try a barrier based on the maximum norm:
B(x,y) = max(abs(x), abs(y)) - 1.2

However, max() is not smooth. Let me use a smooth approximation:
B(x,y) = x**2 + y**2 - 1.44

Verification:
- Initial set: B ≤ 0.04 - 1.44 = -1.4 < 0 ✓
- Unsafe boundary (e.g., x = ±1.2, y = 0): B = 1.44 - 1.44 = 0 ✓

For ΔB analysis, this is complex due to nonlinearity. The general approach is:
1. |sin(y)| ≤ |y| for small |y|, so system is approximately linear near origin
2. Contraction factors (0.7, 0.8) suggest stability
3. Cross-coupling terms (sin(y), x²) are small near origin

For small initial region, the barrier should work due to dominant linear contraction.

Barrier Certificate: B(x,y) = x**2 + y**2 - 1.44""",
        "metadata": {"source": "discrete_time_manual", "system_type": "discrete"}
    })
    
    return examples

def main():
    """Generate and save discrete-time training data."""
    examples = create_discrete_time_training_data()
    
    # Save to JSONL file
    output_file = Path("data/ft_data_discrete_time.jsonl")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(examples)} discrete-time training examples")
    print(f"Saved to: {output_file}")
    
    # Also append to existing discrete data
    existing_file = Path("data/ft_data_discrete.jsonl")
    if existing_file.exists():
        with open(existing_file, 'a') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        print(f"Also appended to: {existing_file}")

if __name__ == "__main__":
    main() 