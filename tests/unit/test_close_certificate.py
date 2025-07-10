#!/usr/bin/env python3
"""Test x² + y² - 0.3 certificate"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester

# Create tester
tester = CertificateValidationTester()

# Define the system
system = {
    "name": "linear_stable_2d",
    "dynamics": ["dx/dt = -x", "dy/dt = -y"],
    "initial_set": ["x**2 + y**2 <= 0.25"],  # radius 0.5
    "unsafe_set": ["x**2 + y**2 >= 4.0"]     # radius 2.0
}

# Test the specific certificate
certificate = "x**2 + y**2 - 0.3"

print(f"Testing certificate: {certificate}")
print(f"Initial set: {system['initial_set'][0]} (radius 0.5)")
print(f"Unsafe set: {system['unsafe_set'][0]} (radius 2.0)")
print()

# Analysis
print("Mathematical analysis:")
print(f"Certificate zero level: x² + y² = 0.3 (radius {np.sqrt(0.3):.3f})")
print(f"Initial set boundary: x² + y² = 0.25 (radius 0.5)")
print(f"Gap: {np.sqrt(0.3) - 0.5:.3f} ({(np.sqrt(0.3) - 0.5)/0.5*100:.1f}% of initial radius)")
print()

# At initial boundary
print("At initial set boundary (x² + y² = 0.25):")
print(f"B = 0.25 - 0.3 = -0.05")
print()

# Validate with high precision
result = tester.validate_certificate_mathematically(certificate, system, n_samples=50)

print(f"Validation result: {'VALID' if result['valid'] else 'INVALID'}")
print(f"Number of violations: {result['num_violations']}")

if result['violations']:
    print("\nViolations found:")
    for i, violation in enumerate(result['violations'][:10]):
        print(f"  {i+1}. {violation}")
    if len(result['violations']) > 10:
        print(f"  ... and {len(result['violations']) - 10} more")

print("\nConclusion:")
print("This certificate SHOULD be invalid because the gap (9.5%) is too small.")
print("The zero level set is too close to the initial set boundary.")
print("This could lead to numerical issues in practice.") 