#!/usr/bin/env python3
"""Debug validation logic for certificates"""

import sympy
import numpy as np

# Test case: linear stable 2D with certificate that should fail
x, y = sympy.symbols('x y')

# System
print("System: dx/dt = -x, dy/dt = -y")
print("Initial set: x² + y² ≤ 0.25 (radius 0.5)")
print("Unsafe set: x² + y² ≥ 4.0 (radius 2.0)")
print()

# Test certificates that are passing but shouldn't
test_certificates = [
    ("x**2 + y**2 - 0.5", "Too close to initial set"),
    ("x**2 + y**2 - 1.5", "Should work but marked as invalid"),
    ("x**2 + y**2 - 2.0", "Should work but marked as invalid"),
]

for cert_str, reason in test_certificates:
    print(f"\nTesting: {cert_str} ({reason})")
    print("-" * 50)
    
    B = sympy.parse_expr(cert_str)
    
    # Check initial set: B should be ≤ 0
    # At boundary of initial set (r=0.5), x²+y² = 0.25
    init_boundary_val = 0.25 - float(cert_str.split('-')[1])
    print(f"At initial boundary: B = {init_boundary_val:.3f}", "✓" if init_boundary_val <= 0 else "✗")
    
    # Check unsafe set: B should be > 0  
    # At boundary of unsafe set (r=2.0), x²+y² = 4.0
    unsafe_boundary_val = 4.0 - float(cert_str.split('-')[1])
    print(f"At unsafe boundary: B = {unsafe_boundary_val:.3f}", "✓" if unsafe_boundary_val > 0 else "✗")
    
    # Lie derivative
    dynamics = [-x, -y]
    dB_dx = sympy.diff(B, x)
    dB_dy = sympy.diff(B, y)
    lie_derivative = dB_dx * dynamics[0] + dB_dy * dynamics[1]
    print(f"Lie derivative: {lie_derivative} = -2(x²+y²) ≤ 0 ✓")
    
    # Why might this be considered invalid?
    if cert_str == "x**2 + y**2 - 0.5":
        print("\nIssue: Certificate level set (0.5) is too close to initial set (0.25)")
        print("This could allow trajectories to reach the barrier from initial set")
    elif cert_str in ["x**2 + y**2 - 1.5", "x**2 + y**2 - 2.0"]:
        print("\nThis should be VALID! The barrier properly separates initial from unsafe.")
        print("Test expectation is WRONG - these should be marked as expected_valid=True")

# For nonlinear system
print("\n" + "="*60)
print("Nonlinear system: dx/dt = -x³ - y, dy/dt = x - y³")
print("Initial set: x² + y² ≤ 0.1 (radius ~0.316)")
print("Unsafe set: x² + y² ≥ 2.0 (radius ~1.414)")

# Check Lie derivative for nonlinear system
B = x**2 + y**2 - 0.5
dynamics_nl = [-x**3 - y, x - y**3]
dB_dx = sympy.diff(B, x)
dB_dy = sympy.diff(B, y)
lie_derivative_nl = dB_dx * dynamics_nl[0] + dB_dy * dynamics_nl[1]

print(f"\nFor B = x² + y² - 0.5:")
print(f"Lie derivative: {lie_derivative_nl}")
print(f"Simplified: {lie_derivative_nl.simplify()}")

# Test at some points
test_points = [
    (0.5, 0.5),
    (0.3, 0.3),
    (0.7, 0.0),
]

print("\nLie derivative at test points:")
for px, py in test_points:
    lie_val = float(lie_derivative_nl.subs({x: px, y: py}))
    print(f"At ({px}, {py}): dB/dt = {lie_val:.3f}", "✓" if lie_val <= 0 else "✗ POSITIVE!")

print("\nConclusion: For nonlinear systems, simple Lyapunov functions may not work!")
print("The Lie derivative can be positive in some regions, making the certificate invalid.") 