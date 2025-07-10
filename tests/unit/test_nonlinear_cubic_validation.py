#!/usr/bin/env python3
"""Test nonlinear cubic system validation"""

import sys
import os
import numpy as np
import sympy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester

# Create tester
tester = CertificateValidationTester()

# Define the nonlinear cubic system
system = {
    "name": "nonlinear_cubic_2d",
    "dynamics": ["dx/dt = -x**3 - y", "dy/dt = x - y**3"],
    "initial_set": ["x**2 + y**2 <= 0.1"],
    "unsafe_set": ["x**2 + y**2 >= 2.0"]
}

print("Nonlinear Cubic System Analysis")
print("=" * 60)
print(f"Dynamics: {system['dynamics']}")
print(f"Initial set: {system['initial_set'][0]} (radius ≈ {np.sqrt(0.1):.3f})")
print(f"Unsafe set: {system['unsafe_set'][0]} (radius ≈ {np.sqrt(2.0):.3f})")
print()

# Test certificates
certificates = [
    "x**2 + y**2 - 0.5",
    "x**2 + y**2 - 0.75",
    "x**2 + y**2 - 1.0",
    "x**2 + y**2 - 1.5"
]

# Analyze Lie derivative for this system
x, y = sympy.symbols('x y')
B = x**2 + y**2 - 0.5  # Example certificate

# Calculate Lie derivative
dB_dx = 2*x
dB_dy = 2*y
f1 = -x**3 - y
f2 = x - y**3

lie_derivative = dB_dx * f1 + dB_dy * f2
print("Lie derivative for B = x² + y² - c:")
print(f"dB/dt = 2x(-x³ - y) + 2y(x - y³)")
print(f"     = -2x⁴ - 2xy + 2xy - 2y⁴")
print(f"     = -2x⁴ - 2y⁴")
print()

# Test points to check if Lie derivative can be positive
print("Testing Lie derivative at various points:")
test_points = [
    (0.1, 0.1),
    (0.5, 0.5),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.7, 0.7),
]

for px, py in test_points:
    lie_val = -2*px**4 - 2*py**4
    print(f"  At ({px}, {py}): dB/dt = {lie_val:.6f} ≤ 0 ✓")

print("\nConclusion: The Lie derivative is ALWAYS negative!")
print("This means these certificates SHOULD be valid for the nonlinear system.")
print()

# Now test each certificate
for cert in certificates:
    print(f"\nTesting certificate: {cert}")
    print("-" * 40)
    
    result = tester.validate_certificate_mathematically(cert, system, n_samples=30)
    
    print(f"Valid: {result['valid']}")
    print(f"Number of violations: {result['num_violations']}")
    
    if result['violations']:
        print("Sample violations:")
        for v in result['violations'][:3]:
            print(f"  - {v}")

print("\n" + "="*60)
print("IMPORTANT FINDING:")
print("The test expectations for the nonlinear cubic system are WRONG!")
print("Simple Lyapunov functions CAN work for this specific nonlinear system")
print("because the Lie derivative -2x⁴ - 2y⁴ is always negative.") 