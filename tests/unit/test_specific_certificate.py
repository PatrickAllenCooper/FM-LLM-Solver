#!/usr/bin/env python3
"""Test specific certificate validation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester

# Create tester
tester = CertificateValidationTester()

# Define the system
system = {
    "name": "linear_stable_2d",
    "dynamics": ["dx/dt = -x", "dy/dt = -y"],
    "initial_set": ["x**2 + y**2 <= 0.25"],
    "unsafe_set": ["x**2 + y**2 >= 4.0"]
}

# Test the specific certificate
certificate = "x**2 + y**2 - 0.5"

print(f"Testing certificate: {certificate}")
print(f"System: {system['name']}")
print(f"Initial set: {system['initial_set'][0]}")
print(f"Unsafe set: {system['unsafe_set'][0]}")
print()

# Validate
result = tester.validate_certificate_mathematically(certificate, system, n_samples=20)

print(f"Valid: {result['valid']}")
print(f"Lie derivative: {result['lie_derivative']}")
print(f"Number of violations: {result['num_violations']}")

if result['violations']:
    print("\nViolations found:")
    for i, violation in enumerate(result['violations'][:5]):  # Show first 5
        print(f"  {i+1}. {violation}")
    if len(result['violations']) > 5:
        print(f"  ... and {len(result['violations']) - 5} more")
else:
    print("\nNo violations found - certificate is VALID!") 