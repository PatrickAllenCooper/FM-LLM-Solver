#!/usr/bin/env python3
"""Analyze certificate validity for x² + y² - 0.5"""

import numpy as np

print("System Analysis:")
print("Initial set: x² + y² ≤ 0.25 (inside circle of radius 0.5)")
print("Unsafe set: x² + y² ≥ 4.0 (outside circle of radius 2.0)")
print("Certificate: B(x,y) = x² + y² - 0.5")
print()

# Theory: A valid barrier certificate must:
# 1. B ≤ 0 on the initial set
# 2. B > 0 on the unsafe set  
# 3. dB/dt ≤ 0 along trajectories (Lie derivative condition)

print("Checking conditions:")
print()

# Condition 1: B ≤ 0 on initial set
print("1. B ≤ 0 on initial set (x² + y² ≤ 0.25):")
print("   At boundary (x² + y² = 0.25): B = 0.25 - 0.5 = -0.25 ✓")
print("   At center (0,0): B = 0 - 0.5 = -0.5 ✓")
print("   Maximum B in initial set: -0.25 ≤ 0 ✓")

# Condition 2: B > 0 on unsafe set
print("\n2. B > 0 on unsafe set (x² + y² ≥ 4.0):")
print("   At boundary (x² + y² = 4.0): B = 4.0 - 0.5 = 3.5 > 0 ✓")
print("   All points in unsafe set have B > 0 ✓")

# Condition 3: Lie derivative
print("\n3. Lie derivative dB/dt ≤ 0:")
print("   B = x² + y² - 0.5")
print("   dB/dx = 2x, dB/dy = 2y")
print("   Dynamics: dx/dt = -x, dy/dt = -y")
print("   dB/dt = 2x(-x) + 2y(-y) = -2x² - 2y² ≤ 0 ✓")

print("\nConclusion: x² + y² - 0.5 SHOULD BE VALID!")
print()

# BUT wait - let's check the level set B = 0
print("Critical observation:")
print("The level set B = 0 occurs at x² + y² = 0.5")
print("This is a circle of radius sqrt(0.5) ≈ 0.707")
print()
print("Initial set boundary is at radius 0.5")
print("Certificate zero level is at radius 0.707")
print()
print("The issue might be numerical precision in validation...")

# Let's check what happens at the exact boundary
r_init = 0.5
print(f"\nAt initial set boundary (r = {r_init}):")
print(f"x² + y² = {r_init**2} = 0.25")
print(f"B = 0.25 - 0.5 = -0.25")
print()

# Check if there's a buffer zone issue
print("Possible issue: Some validators require a buffer zone")
print("between the initial set and the zero level set of B.")
print("This ensures numerical robustness.")

# Calculate the gap
gap = np.sqrt(0.5) - 0.5
print(f"\nGap between initial boundary and B=0: {gap:.3f}")
print("This might be considered too small for numerical stability.") 