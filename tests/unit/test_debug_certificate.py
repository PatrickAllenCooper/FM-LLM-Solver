#!/usr/bin/env python3
"""Debug certificate validation issues"""

import sympy

# Test case: linear stable 2D system
x, y = sympy.symbols("x y")
B = x**2 + y**2 - 1.0  # This should be valid

# System dynamics
dynamics = [-x, -y]  # dx/dt = -x, dy/dt = -y

# Calculate Lie derivative
dB_dx = sympy.diff(B, x)
dB_dy = sympy.diff(B, y)
lie_derivative = dB_dx * dynamics[0] + dB_dy * dynamics[1]

print(f"Certificate: {B}")
print(f"Lie derivative: {lie_derivative}")
print(f"Simplified: {lie_derivative.simplify()}")

# Test at some points
test_points = [
    {"x": 0.3, "y": 0.3},  # Inside initial set boundary
    {"x": 0.7, "y": 0.7},  # Between initial and barrier
    {"x": 1.5, "y": 1.5},  # Between barrier and unsafe
]

print("\nBarrier values and Lie derivatives at test points:")
for point in test_points:
    B_val = float(B.subs(point))
    lie_val = float(lie_derivative.subs(point))
    print(f"At {point}: B = {B_val:.3f}, dB/dt = {lie_val:.3f}")

# Check conditions
print("\nChecking barrier conditions:")
print("1. Initial set (x²+y² ≤ 0.25): B should be ≤ 0")
print(
    "   At boundary (r=0.5): B =",
    float(B.subs({"x": 0.5, "y": 0})),
    "✓" if float(B.subs({"x": 0.5, "y": 0})) <= 0 else "✗",
)

print("2. Unsafe set (x²+y² ≥ 4.0): B should be > 0")
print(
    "   At boundary (r=2.0): B =",
    float(B.subs({"x": 2.0, "y": 0})),
    "✓" if float(B.subs({"x": 2.0, "y": 0})) > 0 else "✗",
)

print("3. Lie derivative should be ≤ 0 everywhere")
print("   This is:", lie_derivative, "= -2(x²+y²) ≤ 0 ✓")

print("\nSo why is this certificate being rejected?")
print("Let's check the actual validation logic...")

# The issue might be in how we're checking the sets
# Initial set: x²+y² ≤ 0.25, so at the boundary x²+y² = 0.25
# For B = x²+y² - 1.0, at the initial set boundary: B = 0.25 - 1.0 = -0.75 ✓
# Unsafe set: x²+y² ≥ 4.0, so at the boundary x²+y² = 4.0
# For B = x²+y² - 1.0, at the unsafe set boundary: B = 4.0 - 1.0 = 3.0 ✓

# But what about points INSIDE the unsafe set?
print("\nChecking points inside unsafe set (should have B > 0):")
inside_unsafe = [
    {"x": 0.5, "y": 0.5},  # x²+y² = 0.5 < 4.0, so NOT in unsafe set
    {"x": 1.5, "y": 1.5},  # x²+y² = 4.5 > 4.0, so IN unsafe set
    {"x": 3.0, "y": 0},  # x²+y² = 9.0 > 4.0, so IN unsafe set
]

for point in inside_unsafe:
    r_squared = point["x"] ** 2 + point["y"] ** 2
    in_unsafe = r_squared >= 4.0
    B_val = float(B.subs(point))
    print(f"At {point}: r² = {r_squared:.1f}, in unsafe = {in_unsafe}, B = {B_val:.3f}")
