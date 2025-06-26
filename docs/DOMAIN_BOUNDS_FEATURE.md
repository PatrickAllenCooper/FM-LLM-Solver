# Domain Bounds for Barrier Certificates

## Overview

The Domain Bounds feature allows users to explicitly define the validity region for barrier certificates. This is a crucial enhancement because barrier certificates are typically only valid within specific domains, and verification outside these bounds may not be meaningful.

## Motivation

Barrier certificates prove safety properties, but they are not necessarily valid across the entire state space. Key reasons for domain bounds:

1. **Mathematical Validity**: The certificate may only satisfy barrier conditions within a bounded region
2. **Numerical Stability**: Verification becomes unreliable at extreme values
3. **Physical Constraints**: Real systems operate within physical limits
4. **Computational Efficiency**: Focus verification on relevant regions

## Implementation

### Database Models

Domain bounds are stored in multiple models:

- **QueryLog**: `certificate_domain_bounds` (JSON), `domain_bounds_conditions` (JSON)
- **Conversation**: `domain_bounds` (JSON), `domain_conditions` (JSON)
- **VerificationResult**: `domain_bounds_check_passed` (Boolean), `domain_bounds_violations` (Integer)
- **SystemBenchmark**: `certificate_domain_bounds` (JSON), `domain_bounds_description` (Text)

### Data Format

Domain bounds are stored as JSON dictionaries mapping variable names to [min, max] ranges:

```json
{
  "x": [-2.0, 2.0],
  "y": [-1.5, 1.5],
  "z": [0.0, 3.0]
}
```

### Certificate Generation

When generating certificates, domain bounds are:

1. **Included in Prompts**: LLM receives explicit domain constraints
2. **Used for Guidance**: Helps select appropriate functional forms
3. **Documented**: Domain bounds appear in generated explanations

Example prompt enhancement:
```
🎯 DOMAIN BOUNDS CONSTRAINT: This barrier certificate must be valid within the specified domain:
   Domain: x ∈ [-2, 2], y ∈ [-1.5, 1.5]
   - The certificate MUST satisfy all barrier conditions within this domain
   - Outside this domain, the certificate validity is not guaranteed
```

### Verification Process

Domain bounds verification includes:

1. **Sampling within Domain**: Generate test points within specified bounds
2. **Barrier Condition Checks**: Verify all three barrier conditions within domain
3. **Violation Tracking**: Count and categorize any violations found
4. **Detailed Reporting**: Provide specific violation information

#### Verification Steps

1. **Initial Set Check**: B(x) ≤ 0 for x in initial set ∩ domain
2. **Unsafe Set Check**: B(x) ≥ 0 for x in (safe set \ unsafe set) ∩ domain  
3. **Primary Condition**: dB/dt ≤ 0 (continuous) or ΔB ≤ 0 (discrete) in safe set ∩ domain

### Web Interface

#### Direct Generation Mode

Users can specify domain bounds via an expandable "Domain Bounds" section:

- **Variable Input**: Text field for variable name (e.g., "x")
- **Range Input**: Numeric fields for minimum and maximum values
- **Dynamic Addition**: Add/remove domain bound rows as needed
- **Validation**: Ensures proper format and reasonable values

#### Conversational Mode

Domain bounds are extracted from conversation text using pattern matching:

- `"domain: x ∈ [-2, 2], y ∈ [-1, 1]"`
- `"x in [-2, 2]"`, `"y in [-1, 1]"`
- `"bounds: x [-2, 2], y [-1, 1]"`
- `"valid for x between -2 and 2"`

## Usage Examples

### Example 1: Simple 2D System

```json
{
  "domain_bounds": {
    "x": [-2.0, 2.0],
    "y": [-2.0, 2.0]
  },
  "description": "Certificate valid in bounded region where dynamics are well-behaved"
}
```

### Example 2: Asymmetric Bounds

```json
{
  "domain_bounds": {
    "x": [-1.5, 3.0],
    "y": [0.0, 2.5]
  },
  "description": "Asymmetric domain reflecting system constraints"
}
```

### Example 3: Physical System

```json
{
  "domain_bounds": {
    "position": [-10.0, 10.0],
    "velocity": [-5.0, 5.0]
  },
  "description": "Physical limits for position and velocity"
}
```

## API Integration

### Certificate Generation

```python
# Generate certificate with domain bounds
result = certificate_generator.generate_certificate(
    system_description=system_desc,
    model_key="qwen_7b_continuous",
    rag_k=3,
    domain_bounds={"x": [-2, 2], "y": [-1, 1]}
)
```

### Verification

```python
# Verify certificate within domain bounds
verification_result = verification_service.verify_certificate(
    certificate_str="x**2 + y**2 - 1.0",
    system_description=system_desc,
    param_overrides=None,
    domain_bounds={"x": [-2, 2], "y": [-1, 1]}
)
```

### Conversation Integration

```python
# Set domain bounds for conversation
conversation.set_domain_bounds_dict({"x": [-2, 2], "y": [-1, 1]})

# Extract bounds from conversation text
bounds = conversation_service._extract_domain_bounds_from_conversation(conversation)
```

## Benchmark Integration

Benchmark systems now include domain bounds:

```json
{
  "id": "example_system",
  "certificate_domain_bounds": {
    "x": [-2.0, 2.0],
    "y": [-2.0, 2.0]
  },
  "domain_bounds_description": "Certificate valid in bounded region where dynamics are well-behaved"
}
```

## Verification Results

Domain bounds verification results include:

```json
{
  "domain_bounds_check_passed": true,
  "domain_bounds_violations": 0,
  "domain_bounds_reason": "Domain bounds check: 0 violations found. Checked 150 initial points, 200 non-unsafe points, 180 safe points within domain bounds.",
  "domain_bounds_details": {
    "violations": 0,
    "initial_violations": 0,
    "unsafe_violations": 0,
    "condition_violations": 0,
    "samples_checked": {
      "initial": 150,
      "outside_unsafe": 200,
      "safe": 180,
      "total": 1000
    }
  }
}
```

## Best Practices

### Choosing Domain Bounds

1. **Physical Constraints**: Use actual system limits when available
2. **Initial/Unsafe Sets**: Ensure bounds encompass relevant sets
3. **Numerical Stability**: Avoid extremely large or small values
4. **Safety Margins**: Include some buffer around critical regions

### Verification Considerations

1. **Sample Density**: More samples needed for larger domains
2. **Boundary Effects**: Pay attention to domain boundaries
3. **Set Intersections**: Verify proper intersection with system sets
4. **Performance**: Larger domains require more computation

### Common Patterns

1. **Symmetric Bounds**: `x ∈ [-a, a]` for centered systems
2. **Asymmetric Bounds**: Different limits for different directions
3. **Hierarchical Bounds**: Tighter bounds for critical variables
4. **Dynamic Bounds**: Bounds that depend on other variables

## Troubleshooting

### Common Issues

1. **Empty Domain**: Bounds too restrictive, no valid samples
2. **Boundary Violations**: Certificate fails near domain edges
3. **Set Mismatch**: Domain doesn't properly contain system sets
4. **Numerical Issues**: Extreme bounds cause computation problems

### Debugging Tips

1. **Check Sample Distribution**: Ensure adequate coverage
2. **Visualize Bounds**: Plot domain and system sets
3. **Test Boundary Conditions**: Manually verify edge cases
4. **Gradual Expansion**: Start with tight bounds, expand as needed

## Future Enhancements

Potential improvements to the domain bounds feature:

1. **Adaptive Bounds**: Automatically adjust based on system analysis
2. **Non-Rectangular Domains**: Support for more complex shapes
3. **Interactive Visualization**: Graphical domain specification
4. **Optimization Integration**: Use bounds to guide certificate search
5. **Uncertainty Quantification**: Handle uncertain domain boundaries

## Configuration

Domain bounds behavior can be configured:

- **Default Bounds**: Fallback bounds when none specified
- **Validation Rules**: Constraints on bound values
- **Sample Density**: Number of verification samples per domain
- **Tolerance Settings**: Numerical tolerances for boundary checks

This feature significantly enhances the reliability and applicability of barrier certificate verification by ensuring certificates are only validated within their intended domains of operation. 