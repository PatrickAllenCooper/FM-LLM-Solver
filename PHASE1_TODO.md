# Phase 1: Foundation Fix - Comprehensive TODO List

## 🎯 Overview
**Goal**: Fix critical theory violations and establish correct mathematical foundation
**Timeline**: 2 weeks (10 working days)
**Success Metric**: >90% accuracy on ground truth test suite

---

## 📅 Week 1: Core Theory Fixes

### Day 1-2: Fix Barrier Certificate Theory ✅

#### TODO 1.1: Update Core Validation Logic
- [ ] **File**: `tests/unit/test_certificate_validation_accuracy.py`
  - [ ] Replace line ~160: Change initial set check from `B > 0` to `B ≤ c₁`
  - [ ] Add level set computation before validation
  - [ ] Update violation messages to include level set info
  
```python
# Current (WRONG):
if B_val > 0:
    violations.append(f"Initial set violation at {point}: B={B_val}")

# Fixed:
if B_val > levels.initial_max:
    violations.append(f"Initial set violation at {point}: B={B_val} > c1={levels.initial_max}")
```

#### TODO 1.2: Implement Level Set Tracker
- [ ] **File**: Create `utils/level_set_tracker.py`
  - [ ] Implement `compute_level_sets()` function
  - [ ] Add `validate_separation()` method
  - [ ] Create visualization for level sets
  
```python
class LevelSetTracker:
    def compute_level_sets(self, B, initial_set, unsafe_set):
        # Sample sets densely
        # Find max(B) on initial, min(B) on unsafe
        # Return c1, c2, separation
        pass
```

#### TODO 1.3: Update Verification Service
- [ ] **File**: `web_interface/verification_service.py`
  - [ ] Import new `BarrierCertificateValidator`
  - [ ] Replace numerical verification logic
  - [ ] Add level set info to response
  
#### TODO 1.4: Create Theory Compliance Tests
- [ ] **File**: Create `tests/test_theory_compliance.py`
  - [ ] Test known-good certificates pass
  - [ ] Test known-bad certificates fail
  - [ ] Test edge cases (B=0 on boundaries)

```python
test_cases = [
    # Should PASS
    {"B": "x**2 + y**2 - 1.0", "system": stable_2d, "expected": True},
    {"B": "x**2 + y**2 - 2.0", "system": stable_2d, "expected": True},
    
    # Should FAIL
    {"B": "x**2 + y**2 - 0.1", "system": stable_2d, "expected": False},
    {"B": "x**2 + y**2", "system": stable_2d, "expected": False},  # No separation
]
```

---

### Day 3-4: Fix Validation Logic 🔧

#### TODO 2.1: Fix Set Membership Testing
- [ ] **File**: `utils/set_membership.py` (create new)
  - [ ] Implement robust point-in-set testing
  - [ ] Handle equality constraints properly
  - [ ] Add epsilon-ball testing for boundaries

```python
class SetMembershipTester:
    def is_in_set(self, point, constraints, epsilon=1e-6):
        """
        Test if point satisfies all constraints
        Handles <=, >=, =, and composite constraints
        """
        for constraint in constraints:
            if not self._evaluate_constraint(point, constraint, epsilon):
                return False
        return True
```

#### TODO 2.2: Fix Boundary Detection
- [ ] **File**: Update `tests/unit/test_certificate_validation_accuracy.py`
  - [ ] Line ~200-250: Fix unsafe set checking logic
  - [ ] Current: Checks points OUTSIDE unsafe set (wrong)
  - [ ] Fixed: Check points INSIDE unsafe set

```python
# Current (WRONG):
if not unsafe_func(xi, yi):  # Point is NOT in unsafe set
    if B_val <= 0:
        violations.append(...)

# Fixed:
if unsafe_func(xi, yi):  # Point IS in unsafe set
    if B_val < 0:  # Should be >= 0 in unsafe set
        violations.append(...)
```

#### TODO 2.3: Implement Adaptive Tolerance
- [ ] **File**: `utils/adaptive_tolerance.py` (create new)
  - [ ] Compute set-dependent tolerances
  - [ ] Scale tolerance with problem size
  - [ ] Add confidence intervals

```python
class AdaptiveTolerance:
    def compute_tolerance(self, set_bounds, problem_scale):
        # Base tolerance
        base_tol = 1e-6
        
        # Scale with problem
        scale_factor = np.mean(np.abs(set_bounds))
        
        # Adaptive tolerance
        return base_tol * max(1.0, scale_factor)
```

#### TODO 2.4: Integration Tests for Validation
- [ ] **File**: `tests/integration/test_validation_pipeline.py`
  - [ ] Test complete validation pipeline
  - [ ] Test with various system types
  - [ ] Benchmark performance

---

### Day 5: Fix Extraction Pipeline 🔍

#### TODO 3.1: Fix Decimal Number Extraction
- [ ] **File**: `utils/certificate_extraction.py`
  - [ ] Line ~50: Update regex pattern for decimals
  - [ ] Handle scientific notation (1e-10, 1.5e3)
  - [ ] Preserve full precision

```python
# Current regex (loses decimals):
r'B\s*\([^)]*\)\s*=\s*([^;\n]+)'

# Fixed regex:
r'B\s*\([^)]*\)\s*=\s*([^;\n]+?)(?=\s*(?:$|\n|;))'
# With proper decimal capture in clean_and_validate_expression()
```

#### TODO 3.2: Enhance Template Detection
- [ ] **File**: `utils/certificate_extraction.py`
  - [ ] Line ~440: Strengthen `is_template_expression()`
  - [ ] Add more template patterns
  - [ ] Implement confidence scoring

```python
template_patterns = [
    # Current patterns...
    
    # Add new patterns:
    r'\b[a-zA-Z]_\d+',      # a_1, b_2, etc.
    r'\bcoeff_[a-zA-Z]+',   # coeff_x, coeff_y
    r'\b[α-ωΑ-Ω]',          # Greek letters
    r'<[^>]+>',             # Placeholder brackets
]
```

#### TODO 3.3: Add Format Support
- [ ] **File**: `utils/certificate_extraction.py`
  - [ ] Support LaTeX format (already started)
  - [ ] Support MathML
  - [ ] Support ASCII math

#### TODO 3.4: Edge Case Tests
- [ ] **File**: `tests/unit/test_extraction_edge_cases.py`
  - [ ] Test very large/small numbers
  - [ ] Test complex expressions
  - [ ] Test malformed input handling

```python
edge_cases = [
    "B(x,y) = 1.234567890123456789e-100",
    "B(x,y) = x**2 + y**2 - 1.0000000000000001",
    "B(x,y) = (x**2 + y**2 - 1) / (x**2 + y**2 + 1)",  # Rational
    "B(x,y) = |x|**2 + |y|**2 - 1",  # Absolute values
]
```

---

## 📅 Week 2: Testing & Baseline

### Day 6-7: Comprehensive Test Suite 🧪

#### TODO 4.1: Create Ground Truth Dataset
- [ ] **File**: `tests/ground_truth/barrier_certificates.json`
  - [ ] 20+ verified barrier certificates
  - [ ] Various system types
  - [ ] Include edge cases

```json
{
  "linear_stable_2d": {
    "system": {
      "dynamics": ["dx/dt = -x", "dy/dt = -y"],
      "initial_set": ["x**2 + y**2 <= 0.25"],
      "unsafe_set": ["x**2 + y**2 >= 4.0"]
    },
    "valid_certificates": [
      {"expr": "x**2 + y**2 - 1.0", "valid": true},
      {"expr": "x**2 + y**2 - 2.0", "valid": true},
      {"expr": "0.5*x**2 + 0.5*y**2 - 0.5", "valid": true}
    ],
    "invalid_certificates": [
      {"expr": "x**2 + y**2 - 0.1", "valid": false, "reason": "No separation"},
      {"expr": "x**2 + y**2", "valid": false, "reason": "B > 0 everywhere"}
    ]
  }
}
```

#### TODO 4.2: Implement Test Harness
- [ ] **File**: `tests/test_harness.py`
  - [ ] Load ground truth data
  - [ ] Run extraction + validation
  - [ ] Compare with expected results
  - [ ] Generate detailed reports

```python
class TestHarness:
    def run_test_suite(self, ground_truth_file):
        results = {
            'extraction': {'passed': 0, 'failed': 0},
            'validation': {'passed': 0, 'failed': 0},
            'end_to_end': {'passed': 0, 'failed': 0}
        }
        
        for system_name, test_data in ground_truth.items():
            self.test_system(system_name, test_data, results)
        
        return self.generate_report(results)
```

#### TODO 4.3: Automated Test Runner
- [ ] **File**: `tests/run_phase1_tests.py`
  - [ ] Run all unit tests
  - [ ] Run integration tests
  - [ ] Run ground truth validation
  - [ ] CI/CD integration ready

#### TODO 4.4: Report Generator
- [ ] **File**: `tests/report_generator.py`
  - [ ] HTML report with visualizations
  - [ ] Accuracy metrics per component
  - [ ] Failure analysis
  - [ ] Performance metrics

---

### Day 8-9: Performance Benchmarking 📊

#### TODO 5.1: Performance Profiler
- [ ] **File**: `tests/benchmarks/profiler.py`
  - [ ] Time each component
  - [ ] Memory usage tracking
  - [ ] Identify bottlenecks

```python
class PerformanceProfiler:
    def profile_component(self, component, test_data):
        import cProfile
        import memory_profiler
        
        # Time profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run component
        result = component.process(test_data)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        
        return {
            'time': stats.total_tt,
            'memory': memory_profiler.memory_usage(),
            'bottlenecks': self.identify_bottlenecks(stats)
        }
```

#### TODO 5.2: Accuracy Metrics Implementation
- [ ] **File**: `tests/metrics.py`
  - [ ] Precision/Recall/F1 for extraction
  - [ ] True/False Positive/Negative rates
  - [ ] Confusion matrices
  - [ ] Per-system-type breakdown

#### TODO 5.3: Optimization Targets
- [ ] **File**: `tests/benchmarks/optimization_targets.py`
  - [ ] Identify slow functions
  - [ ] Suggest optimizations
  - [ ] Track improvement over time

#### TODO 5.4: Benchmark Suite
- [ ] **File**: `tests/benchmarks/run_benchmarks.py`
  - [ ] Standard benchmark systems
  - [ ] Scaling tests (2D to 10D)
  - [ ] Stress tests
  - [ ] Compare with baseline

---

### Day 10: Documentation & Release 📚

#### TODO 6.1: Theory Documentation
- [ ] **File**: `docs/BARRIER_CERTIFICATE_THEORY.md`
  - [ ] Correct mathematical formulation
  - [ ] Examples with worked solutions
  - [ ] Common misconceptions
  - [ ] References to literature

#### TODO 6.2: API Documentation
- [ ] **File**: `docs/API_REFERENCE.md`
  - [ ] Document all public functions
  - [ ] Parameter descriptions
  - [ ] Return value specs
  - [ ] Usage examples

#### TODO 6.3: Migration Guide
- [ ] **File**: `docs/MIGRATION_GUIDE.md`
  - [ ] Breaking changes from old version
  - [ ] How to update existing code
  - [ ] New features available

#### TODO 6.4: Contribution Guidelines
- [ ] **File**: `CONTRIBUTING.md`
  - [ ] Code style guide
  - [ ] Testing requirements
  - [ ] PR process
  - [ ] Issue templates

---

## 🎯 Validation Criteria

### Exit Criteria for Phase 1:
1. - [ ] All theory compliance tests pass (100%)
2. - [ ] Extraction accuracy ≥ 90%
3. - [ ] Validation accuracy ≥ 85%
4. - [ ] End-to-end accuracy ≥ 85%
5. - [ ] No false positives on ground truth
6. - [ ] Performance: <100ms per certificate
7. - [ ] All documentation complete
8. - [ ] CI/CD tests passing

### Deliverables:
1. - [ ] Fixed validation system
2. - [ ] Comprehensive test suite
3. - [ ] Performance benchmarks
4. - [ ] Complete documentation
5. - [ ] Migration guide

---

## 🚀 Quick Start Commands

```bash
# Run theory fix tests
python tests/unit/barrier_theory_fix.py

# Run extraction tests
python tests/unit/test_extraction_edge_cases.py

# Run full test suite
python tests/run_phase1_tests.py

# Generate performance report
python tests/benchmarks/run_benchmarks.py

# Build documentation
cd docs && make html
```

---

## 📝 Daily Checklist Template

### Day X Checklist:
- [ ] Morning: Review yesterday's work
- [ ] Code: Implement assigned TODOs
- [ ] Test: Write/run unit tests
- [ ] Document: Update relevant docs
- [ ] Commit: Clean commits with good messages
- [ ] Review: Self-review or peer review
- [ ] Plan: Update tomorrow's tasks

---

## 🔄 Progress Tracking

| Day | Planned Tasks | Completed | Blockers | Notes |
|-----|--------------|-----------|----------|-------|
| 1   | TODO 1.1-1.2 |    ❌     |   None   |       |
| 2   | TODO 1.3-1.4 |    ❌     |   None   |       |
| 3   | TODO 2.1-2.2 |    ❌     |   None   |       |
| 4   | TODO 2.3-2.4 |    ❌     |   None   |       |
| 5   | TODO 3.1-3.4 |    ❌     |   None   |       |
| 6   | TODO 4.1-4.2 |    ❌     |   None   |       |
| 7   | TODO 4.3-4.4 |    ❌     |   None   |       |
| 8   | TODO 5.1-5.2 |    ❌     |   None   |       |
| 9   | TODO 5.3-5.4 |    ❌     |   None   |       |
| 10  | TODO 6.1-6.4 |    ❌     |   None   |       | 