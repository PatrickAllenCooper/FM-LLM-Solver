# FM-LLM-Solver Verification System: Technical Overview

## System Architecture

The FM-LLM-Solver implements automated barrier certificate generation for dynamic systems using fine-tuned large language models coupled with multi-method verification. The system addresses the computational challenge of verifying safety properties for autonomous systems by automating the traditionally manual process of barrier certificate derivation.

## Problem Formulation

Given a dynamic system with state variables x ∈ ℝⁿ, we consider:
- **Continuous-time systems**: ẋ = f(x) where f: ℝⁿ → ℝⁿ
- **Discrete-time systems**: x[k+1] = f(x[k]) where f: ℝⁿ → ℝⁿ

Safety verification requires finding a barrier function B: ℝⁿ → ℝ satisfying:
1. B(x) ≤ 0 for all x ∈ X₀ (initial set)
2. B(x) ≥ 0 for all x ∈ Xᵤ (unsafe set)
3. For continuous systems: ∇B(x)·f(x) ≤ 0 for all x ∈ Xₛ (safe set)
4. For discrete systems: B(f(x)) - B(x) ≤ 0 for all x ∈ Xₛ (safe set)

## Three-Method Verification Architecture

### Method 1: Numerical Verification via Monte Carlo Sampling

**Implementation**: The numerical verification method uses pseudorandom sampling to empirically validate barrier certificate conditions through statistical testing.

**Algorithm**:
```
For i = 1 to N_samples:
    x_i ~ Uniform(sampling_bounds)
    if x_i ∈ safe_set:
        if continuous_system:
            compute dB/dt|_{x_i} = ∇B(x_i) · f(x_i)
            if dB/dt|_{x_i} > tolerance: record_violation()
        else: // discrete_system
            compute ΔB|_{x_i} = B(f(x_i)) - B(x_i)
            if ΔB|_{x_i} > tolerance: record_violation()
```

**User Parameters**:
- `num_samples_lie`: Sample count for Lie derivative/difference condition verification
  - Default: 9,100 (continuous), 15,000 (discrete)
  - Range: 100-50,000
  - Impact: Higher values reduce Type II error probability at linear computational cost
- `num_samples_boundary`: Sample count for initial/unsafe set boundary verification  
  - Default: 4,600 (continuous), 8,000 (discrete)
  - Range: 100-25,000
  - Impact: Controls boundary condition verification confidence
- `numerical_tolerance`: Epsilon threshold for numerical inequality evaluation
  - Default: 1e-6
  - Range: 1e-12 to 1e-3
  - Impact: Smaller values require more precise barrier functions

**Discrete-Time Optimizations**:
The system applies automatic parameter adjustments for discrete-time systems:
- Sample bounds compression: `bounds_new = bounds_old * 0.7` to focus on reachable regions
- Increased sample density: 1.5× multiplier for both Lie and boundary sample counts
- Stricter tolerance enforcement: Fixed at 1e-6 regardless of user setting

**Lambdification Process**:
Symbolic expressions are converted to numerical functions using SymPy's `lambdify()` with NumPy backend for vectorized evaluation. Failed lambdification results in verification method skip.

### Method 2: Symbolic Verification via Computer Algebra

**Implementation**: Uses SymPy computer algebra system to perform exact symbolic manipulation and analytical verification of barrier certificate conditions.

**Algorithm Components**:
1. **Expression Parsing**: Convert string representations to SymPy symbolic expressions
2. **Derivative Computation**: 
   - Continuous: `dB_dt = sum(diff(B, var) * dynamics[var] for var in variables)`
   - Discrete: `delta_B = B.subs(var_substitutions) - B`
3. **Symbolic Simplification**: Apply SymPy simplification algorithms
4. **Constraint Analysis**: Attempt symbolic proof of inequality conditions

**Limitations**:
- Requires rational polynomial expressions for reliable operation
- Complex transcendental functions may cause symbolic verification failure
- Set membership verification limited to polynomial inequality constraints
- No user-configurable parameters (deterministic symbolic computation)

**Implementation Details**:
```python
# Continuous system Lie derivative
lie_derivative = sum(sympy.diff(B, var) * dynamics_dict[var] 
                    for var in state_variables)

# Discrete system difference  
var_substitutions = {var: dynamics_dict[var] for var in state_variables}
delta_B = B.subs(var_substitutions) - B
```

### Method 3: Sum-of-Squares (SOS) Verification via Semidefinite Programming

**Implementation**: Reformulates barrier certificate verification as semidefinite programming (SDP) problems using sum-of-squares decomposition.

**Mathematical Formulation**:
For polynomial systems, the SOS method verifies:

1. **Lie Derivative Condition**: Find SOS polynomials s₀(x), s₁(x), ..., sₘ(x) such that:
   ```
   -∇B(x)·f(x) = s₀(x) + Σᵢ sᵢ(x)·gᵢ(x)
   ```
   where gᵢ(x) ≥ 0 define the safe set constraints.

2. **Initial Set Condition**: Find SOS polynomials such that:
   ```
   -B(x) = s₀(x) + Σⱼ sⱼ(x)·hⱼ(x)  
   ```
   where hⱼ(x) ≥ 0 define the initial set.

3. **Unsafe Set Condition**: Find SOS polynomials such that:
   ```
   B(x) = s₀(x) + Σₖ sₖ(x)·kₖ(x)
   ```
   where kₖ(x) ≥ 0 define the unsafe set.

**SDP Formulation**:
Each SOS polynomial sᵢ(x) is represented as:
```
sᵢ(x) = z(x)ᵀ Qᵢ z(x)
```
where z(x) is a vector of monomials and Qᵢ ⪰ 0 is a positive semidefinite matrix variable.

**User Parameters**:
- `attempt_sos`: Boolean flag to enable SOS verification
  - Default: True
  - Automatically disabled for non-polynomial systems
- `sos_default_degree`: Degree bound for SOS multiplier polynomials
  - Default: 2
  - Range: 1-6 (practical computational limits)
  - Impact: Higher degrees enable verification of more complex certificates but increase SDP problem size exponentially

**Solver Configuration**:
- Primary solver: MOSEK (requires license)
- Fallback solver: SCS (open-source)
- Solver selection: Automatic based on availability
- Convergence tolerance: Uses solver defaults (typically 1e-6 to 1e-8)

**Computational Complexity**:
SDP problem size scales as O((n+d choose d)³) where n is the number of variables and d is the polynomial degree. Practical limits typically constrain problems to n ≤ 10, d ≤ 4.

**Applicability Conditions**:
- System dynamics must be polynomial
- Barrier function must be polynomial  
- Set constraints must be polynomial inequalities
- All coefficients must be rational numbers

### Method 4: Optimization-Based Falsification (Optional)

**Implementation**: Uses differential evolution metaheuristic to search for counterexamples that violate barrier certificate conditions.

**Objective Functions**:
1. **Lie Derivative Maximization**: `maximize f₁(x) = ∇B(x)·f(x)` subject to x ∈ safe_set
2. **Initial Set Barrier Maximization**: `maximize f₂(x) = B(x)` subject to x ∈ initial_set  
3. **Unsafe Set Barrier Minimization**: `minimize f₃(x) = B(x)` subject to x ∉ unsafe_set

**Algorithm Parameters**:
- `optimization_max_iter`: Maximum iterations for differential evolution
  - Default: 100
  - Range: 10-1000
  - Impact: Higher values increase search thoroughness at linear computational cost
- `optimization_pop_size`: Population size for evolutionary algorithm
  - Default: 15  
  - Range: 5-100
  - Impact: Larger populations improve global search capability

**Differential Evolution Configuration**:
- Mutation factor: (0.5, 1.0) adaptive range
- Crossover probability: 0.7
- Selection strategy: 'immediate' updating
- Convergence tolerance: 0.01

## Web Interface Parameter Control

The web interface provides parameter control through an expandable "Advanced Verification Settings" panel:

**Parameter Validation**:
- Input field constraints prevent invalid parameter ranges
- Automatic type conversion from string inputs to appropriate numeric types
- Discrete system detection triggers automatic parameter optimization overrides

**Default Parameter Selection**:
- Continuous systems: Use base configuration values from `config.yaml`
- Discrete systems: Apply 1.5× sample multipliers and compressed bounds
- Parameter inheritance: User overrides take precedence over automatic optimizations

**Form Processing**:
```javascript
// Parameter parsing in web interface
data.num_samples_lie = parseInt(data.num_samples_lie);
data.numerical_tolerance = parseFloat(data.numerical_tolerance);  
data.attempt_sos = document.getElementById('attempt_sos').checked;
```

## System Detection and Auto-Configuration

**Discrete vs. Continuous Detection**:
The system uses regex pattern matching to classify system types:

```python
discrete_patterns = [
    r'([a-zA-Z_]\w*)\s*\{\s*k\s*\+\s*1\s*\}\s*=',  # x{k+1} = 
    r'([a-zA-Z_]\w*)\[\s*k\s*\+\s*1\s*\]\s*=',      # x[k+1] = 
]

continuous_patterns = [
    r'd([a-zA-Z_]\w*)/dt\s*=',  # dx/dt = 
    r'([a-zA-Z_]\w*)\'\s*=',    # x' = 
]
```

**Auto-Configuration Logic**:
Upon discrete system detection:
1. Sampling bounds compression by 30% to focus on discrete reachable sets
2. Sample count increases: 1.5× for Lie derivative checks, 1.5× for boundary checks
3. Numerical tolerance override to 1e-6 for improved discrete transition precision

This implementation provides rigorous mathematical verification through multiple complementary methods while maintaining computational efficiency and user configurability. 