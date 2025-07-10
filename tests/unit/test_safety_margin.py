#!/usr/bin/env python3
"""Test safety margin for barrier certificates"""

import numpy as np

# System: initial set x² + y² ≤ 0.25 (radius 0.5)
# Certificate: x² + y² - c

print("Safety Margin Analysis")
print("=" * 50)
print("Initial set: x² + y² ≤ 0.25 (radius 0.5)")
print()

certificates = [
    ("x² + y² - 0.25", 0.25, "AT initial boundary - definitely invalid"),
    ("x² + y² - 0.3", 0.3, "Very close to initial - likely invalid"),
    ("x² + y² - 0.5", 0.5, "Gap = 0.207 - borderline"),
    ("x² + y² - 0.75", 0.75, "Gap = 0.366 - reasonable"),
    ("x² + y² - 1.0", 1.0, "Gap = 0.5 - good separation"),
]

print("Certificate Analysis:")
for cert_name, c_val, desc in certificates:
    # Zero level set is at radius sqrt(c)
    r_zero = np.sqrt(c_val)
    r_init = 0.5
    gap = r_zero - r_init
    
    print(f"\n{cert_name} ({desc})")
    print(f"  Zero level at radius: {r_zero:.3f}")
    print(f"  Gap from initial boundary: {gap:.3f}")
    print(f"  Gap as % of initial radius: {gap/r_init*100:.1f}%")
    
    # Suggested criterion: gap should be at least 20% of initial radius
    if gap < 0.1 * r_init:
        print("  ❌ Too close (< 10% gap)")
    elif gap < 0.2 * r_init:
        print("  ⚠️ Borderline (10-20% gap)")
    else:
        print("  ✅ Good separation (> 20% gap)")

print("\n" + "="*50)
print("Recommendation:")
print("For numerical robustness, require gap ≥ 20% of initial radius")
print("This would make x² + y² - 0.5 borderline/invalid for practical use") 