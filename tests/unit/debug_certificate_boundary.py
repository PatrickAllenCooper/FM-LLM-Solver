#!/usr/bin/env python3
"""Debug why x² + y² - 0.5 certificate is failing"""

import numpy as np

# System: initial set x² + y² ≤ 0.25 (radius 0.5)
# Certificate: x² + y² - 0.5

print("Initial set: x² + y² ≤ 0.25 (radius 0.5)")
print("Certificate: B(x,y) = x² + y² - 0.5")
print()

# Check boundary points
print("At initial set boundary (radius 0.5):")
r = 0.5
for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    B_val = x**2 + y**2 - 0.5
    print(f"  At ({x:.3f}, {y:.3f}): B = {B_val:.6f}")

print("\nProblem: B = 0 exactly at initial set boundary!")
print("This means trajectories starting at the boundary are ON the barrier.")
print("For a valid barrier, we need B < 0 strictly inside and at the boundary of initial set.")

print("\nFor comparison, with B = x² + y² - 0.75:")
for angle in [0, np.pi/4, np.pi/2]:
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    B_val = x**2 + y**2 - 0.75
    print(f"  At ({x:.3f}, {y:.3f}): B = {B_val:.3f} < 0 ✓")

# Check a point slightly outside initial set
print("\nJust outside initial set (r=0.51):")
r = 0.51
x = r
y = 0
B_val = x**2 + y**2 - 0.5
print(f"  At ({x:.3f}, {y:.3f}): B = {B_val:.6f} > 0")
print("\nThis could allow trajectories to escape the initial set and reach the barrier!") 