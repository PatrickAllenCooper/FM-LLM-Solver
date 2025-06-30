# Verification System

FM-LLM Solver includes multiple verification methods to validate generated barrier certificates.

## Overview

The verification pipeline checks if a proposed barrier certificate B(x) satisfies:
1. B(x) > 0 for x ∈ Initial Set
2. B(x) ≤ 0 for x ∈ Unsafe Set  
3. dB/dt ≤ 0 along trajectories (Lie derivative condition)

## Verification Methods

### 1. Numerical Verification
- Samples points from initial and unsafe sets
- Evaluates barrier certificate at sample points
- Checks Lie derivative at grid points
- Fast but not complete

### 2. Symbolic Verification
- Computes Lie derivative symbolically using SymPy
- Checks mathematical properties analytically
- More rigorous than numerical sampling
- Limited by symbolic computation complexity

### 3. SOS Verification
- Uses Sum-of-Squares programming
- Provides formal guarantees for polynomial systems
- Requires MOSEK or SCS solver
- Most rigorous but computationally intensive

## Usage

### Command Line
```bash
python evaluation/verify_certificate.py \
  --dynamics "dx/dt = -x^3 - y, dy/dt = x - y^3" \
  --certificate "x^4 + y^4" \
  --initial_set "x^2 + y^2 <= 0.1" \
  --unsafe_set "x >= 1.5"
```

### Python API
```python
from evaluation.verify_certificate import verify_barrier_certificate

result = verify_barrier_certificate(
    dynamics={"x": "-x**3 - y", "y": "x - y**3"},
    certificate="x**4 + y**4",
    initial_set="x**2 + y**2 <= 0.1",
    unsafe_set="x >= 1.5",
    domain_bounds={"x": [-2, 2], "y": [-2, 2]}
)

print(f"Valid: {result['valid']}")
print(f"Details: {result['details']}")
```

## Configuration

In `config.yaml`:
```yaml
verification:
  numerical:
    num_samples: 1000
    grid_density: 20
  
  symbolic:
    simplify: true
    timeout: 30
  
  sos:
    solver: "mosek"  # or "scs"
    degree: 4
```

## Special Cases

### Discrete-Time Systems
For x[k+1] = f(x[k]):
- Checks B(f(x)) - B(x) ≤ 0
- Uses difference instead of derivative

### Stochastic Systems
For dx = f(x)dt + g(x)dW:
- Computes infinitesimal generator
- Checks LB(x) ≤ 0 where L is the generator

### Domain-Bounded Certificates
- Restricts verification to specified domain
- Useful for local barrier certificates
- Add domain_bounds parameter

## Interpreting Results

```json
{
  "valid": true,
  "numerical_check": true,
  "symbolic_check": true,
  "sos_check": false,
  "details": {
    "initial_set_satisfied": true,
    "unsafe_set_satisfied": true,
    "lie_derivative_satisfied": true,
    "sample_violations": 0,
    "computation_time": 1.23
  }
}
```

## Limitations

1. **Numerical**: May miss edge cases
2. **Symbolic**: Struggles with complex expressions
3. **SOS**: Only for polynomial systems
4. **All methods**: Assume exact system model

## Best Practices

1. Start with numerical verification (fast)
2. Use symbolic for polynomial systems
3. Apply SOS for formal guarantees
4. Always verify in target deployment domain
5. Consider multiple certificates if one fails 