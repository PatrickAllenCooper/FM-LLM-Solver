#!/usr/bin/env python3
"""
Adaptive Tolerance Module
Provides problem-dependent tolerance computation for robust validation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveTolerance:
    """
    Computes adaptive tolerances based on:
    - Problem scale (size of sets)
    - Numerical precision requirements
    - System dynamics characteristics
    - Confidence intervals
    """
    
    def __init__(self, base_tolerance: float = 1e-6):
        self.base_tolerance = base_tolerance
        
    def compute_tolerance(self, 
                         set_bounds: Dict[str, Tuple[float, float]],
                         problem_scale: Optional[float] = None) -> float:
        """
        Compute adaptive tolerance based on problem characteristics.
        
        Args:
            set_bounds: Bounds for each variable
            problem_scale: Optional explicit problem scale
            
        Returns:
            Adaptive tolerance value
        """
        # If explicit scale provided, use it
        if problem_scale is not None:
            scale_factor = abs(problem_scale)
        else:
            # Estimate scale from bounds
            scales = []
            for var, (low, high) in set_bounds.items():
                if np.isfinite(low) and np.isfinite(high):
                    scales.append(high - low)
                    
            scale_factor = np.mean(scales) if scales else 1.0
            
        # Adaptive tolerance scales with problem size
        # but has minimum and maximum bounds
        adaptive_tol = self.base_tolerance * max(1.0, scale_factor / 10.0)
        
        # Clamp to reasonable range
        min_tol = 1e-10
        max_tol = 1e-3
        
        return np.clip(adaptive_tol, min_tol, max_tol)
    
    def compute_set_tolerance(self,
                            initial_set: List[str],
                            unsafe_set: List[str],
                            variables: List[str]) -> Dict[str, float]:
        """
        Compute tolerances specific to initial and unsafe sets.
        
        Returns:
            Dictionary with tolerances for different validation steps
        """
        # Extract bounds from constraints
        initial_bounds = self._extract_bounds(initial_set, variables)
        unsafe_bounds = self._extract_bounds(unsafe_set, variables)
        
        # Compute scales
        initial_scale = self._compute_set_scale(initial_bounds)
        unsafe_scale = self._compute_set_scale(unsafe_bounds)
        
        # Different tolerances for different checks
        return {
            'initial_set': self.compute_tolerance(initial_bounds, initial_scale),
            'unsafe_set': self.compute_tolerance(unsafe_bounds, unsafe_scale),
            'separation': self.base_tolerance * min(initial_scale, unsafe_scale) / 100.0,
            'lie_derivative': self.base_tolerance,
            'boundary': self.base_tolerance * 10  # More lenient for boundaries
        }
    
    def compute_confidence_interval(self,
                                  value: float,
                                  scale: float,
                                  n_samples: int) -> Tuple[float, float]:
        """
        Compute confidence interval for a computed value.
        
        Args:
            value: The computed value
            scale: Problem scale
            n_samples: Number of samples used
            
        Returns:
            (lower_bound, upper_bound) for 95% confidence
        """
        # Standard error decreases with more samples
        std_error = scale / np.sqrt(n_samples)
        
        # 95% confidence interval (approximately 2 standard errors)
        margin = 2 * std_error * self.base_tolerance
        
        return (value - margin, value + margin)
    
    def validate_with_tolerance(self,
                              computed_value: float,
                              expected_value: float,
                              tolerance: Optional[float] = None) -> bool:
        """
        Check if computed value matches expected within tolerance.
        
        Args:
            computed_value: Value computed numerically
            expected_value: Expected theoretical value
            tolerance: Optional specific tolerance
            
        Returns:
            True if values match within tolerance
        """
        if tolerance is None:
            tolerance = self.base_tolerance
            
        # Handle special cases
        if np.isnan(computed_value) or np.isnan(expected_value):
            return False
            
        if np.isinf(expected_value):
            # For infinite expected values, check if computed is large enough
            return np.isinf(computed_value) or abs(computed_value) > 1/tolerance
            
        # Relative tolerance for large values, absolute for small
        if abs(expected_value) > 1.0:
            # Relative tolerance
            rel_error = abs(computed_value - expected_value) / abs(expected_value)
            return rel_error <= tolerance
        else:
            # Absolute tolerance
            return abs(computed_value - expected_value) <= tolerance
    
    def get_numerical_precision(self, system_type: str) -> Dict[str, float]:
        """
        Get recommended numerical precision settings for different system types.
        
        Args:
            system_type: Type of system ('linear', 'nonlinear', 'hybrid')
            
        Returns:
            Dictionary with precision settings
        """
        if system_type == 'linear':
            return {
                'integration_tol': 1e-8,
                'optimization_tol': 1e-6,
                'sampling_density': 100,
                'max_iterations': 1000
            }
        elif system_type == 'nonlinear':
            return {
                'integration_tol': 1e-10,
                'optimization_tol': 1e-8,
                'sampling_density': 500,
                'max_iterations': 5000
            }
        elif system_type == 'hybrid':
            return {
                'integration_tol': 1e-9,
                'optimization_tol': 1e-7,
                'sampling_density': 200,
                'max_iterations': 2000
            }
        else:
            # Default conservative settings
            return {
                'integration_tol': 1e-10,
                'optimization_tol': 1e-8,
                'sampling_density': 200,
                'max_iterations': 2000
            }
    
    def _extract_bounds(self, constraints: List[str], variables: List[str]) -> Dict[str, Tuple[float, float]]:
        """Extract variable bounds from constraints"""
        bounds = {var: [-np.inf, np.inf] for var in variables}
        
        for constraint in constraints:
            # Look for simple bounds
            for var in variables:
                if f"{var} <=" in constraint:
                    try:
                        parts = constraint.split('<=')
                        if parts[0].strip() == var:
                            val = float(parts[1].strip())
                            bounds[var] = (bounds[var][0], min(bounds[var][1], val))
                    except:
                        pass
                        
                if f"{var} >=" in constraint:
                    try:
                        parts = constraint.split('>=')
                        if parts[0].strip() == var:
                            val = float(parts[1].strip())
                            bounds[var] = (max(bounds[var][0], val), bounds[var][1])
                    except:
                        pass
            
            # Look for quadratic constraints (circles/spheres)
            if '**2' in constraint:
                try:
                    if '<=' in constraint:
                        rhs = float(constraint.split('<=')[1].strip())
                        r = np.sqrt(rhs)
                        for var in variables:
                            if f"{var}**2" in constraint:
                                bounds[var] = (max(bounds[var][0], -r), 
                                             min(bounds[var][1], r))
                    elif '>=' in constraint:
                        rhs = float(constraint.split('>=')[1].strip())
                        r = np.sqrt(rhs)
                        # For >= constraints, the bounds extend outward
                        # We'll use a reasonable limit
                        for var in variables:
                            if f"{var}**2" in constraint:
                                current_bound = max(abs(bounds[var][0]), abs(bounds[var][1]))
                                new_bound = max(current_bound, r * 2)
                                bounds[var] = (-new_bound, new_bound)
                except:
                    pass
                    
        # Replace infinities with reasonable values
        for var in bounds:
            low, high = bounds[var]
            if np.isinf(low):
                low = -100.0
            if np.isinf(high):
                high = 100.0
            bounds[var] = (low, high)
            
        return bounds
    
    def _compute_set_scale(self, bounds: Dict[str, Tuple[float, float]]) -> float:
        """Compute characteristic scale of a set from its bounds"""
        scales = []
        
        for var, (low, high) in bounds.items():
            if np.isfinite(low) and np.isfinite(high):
                scales.append(high - low)
                
        if not scales:
            return 1.0
            
        # Use RMS scale as characteristic scale
        return np.sqrt(np.mean(np.square(scales)))


class ToleranceManager:
    """
    Manages tolerances across different components of validation.
    """
    
    def __init__(self, base_tolerance: float = 1e-6):
        self.adaptive = AdaptiveTolerance(base_tolerance)
        self.tolerances = {}
        
    def setup_problem(self,
                     initial_set: List[str],
                     unsafe_set: List[str],
                     variables: List[str],
                     system_type: str = 'nonlinear'):
        """
        Setup tolerances for a specific problem.
        
        Args:
            initial_set: Initial set constraints
            unsafe_set: Unsafe set constraints
            variables: System variables
            system_type: Type of system
        """
        # Compute set-specific tolerances
        self.tolerances = self.adaptive.compute_set_tolerance(
            initial_set, unsafe_set, variables
        )
        
        # Add system-specific settings
        self.precision = self.adaptive.get_numerical_precision(system_type)
        
        # Log tolerance settings
        logger.info(f"Tolerance settings for {system_type} system:")
        for key, value in self.tolerances.items():
            logger.info(f"  {key}: {value:.2e}")
            
    def get_tolerance(self, check_type: str) -> float:
        """
        Get tolerance for specific check type.
        
        Args:
            check_type: Type of check ('initial_set', 'unsafe_set', etc.)
            
        Returns:
            Appropriate tolerance value
        """
        return self.tolerances.get(check_type, self.adaptive.base_tolerance)
    
    def validate(self, computed: float, expected: float, check_type: str) -> bool:
        """
        Validate a computed value against expected with appropriate tolerance.
        
        Args:
            computed: Computed value
            expected: Expected value
            check_type: Type of check being performed
            
        Returns:
            True if validation passes
        """
        tolerance = self.get_tolerance(check_type)
        return self.adaptive.validate_with_tolerance(computed, expected, tolerance)


# Test the module
if __name__ == "__main__":
    # Test adaptive tolerance
    print("Testing Adaptive Tolerance Module")
    print("=" * 50)
    
    # Test 1: Basic tolerance computation
    adaptive = AdaptiveTolerance()
    
    bounds1 = {'x': (-1, 1), 'y': (-1, 1)}
    tol1 = adaptive.compute_tolerance(bounds1)
    print(f"Tolerance for unit box: {tol1:.2e}")
    
    bounds2 = {'x': (-100, 100), 'y': (-100, 100)}
    tol2 = adaptive.compute_tolerance(bounds2)
    print(f"Tolerance for large box: {tol2:.2e}")
    
    # Test 2: Set-specific tolerances
    print("\nSet-specific tolerances:")
    initial_set = ["x**2 + y**2 <= 0.25"]
    unsafe_set = ["x**2 + y**2 >= 4.0"]
    
    set_tols = adaptive.compute_set_tolerance(initial_set, unsafe_set, ['x', 'y'])
    for key, value in set_tols.items():
        print(f"  {key}: {value:.2e}")
    
    # Test 3: Confidence intervals
    print("\nConfidence intervals:")
    value = 1.0
    scale = 2.0
    for n_samples in [10, 100, 1000]:
        lower, upper = adaptive.compute_confidence_interval(value, scale, n_samples)
        print(f"  {n_samples} samples: [{lower:.6f}, {upper:.6f}]")
    
    # Test 4: Tolerance manager
    print("\nTolerance Manager:")
    manager = ToleranceManager()
    manager.setup_problem(initial_set, unsafe_set, ['x', 'y'], 'nonlinear')
    
    # Test validation
    print("\nValidation tests:")
    test_cases = [
        (1.0, 1.0001, 'initial_set', "Close values"),
        (1.0, 1.1, 'initial_set', "Far values"),
        (0.0, 1e-7, 'separation', "Near zero"),
    ]
    
    for computed, expected, check_type, desc in test_cases:
        result = manager.validate(computed, expected, check_type)
        print(f"  {desc}: {result}") 