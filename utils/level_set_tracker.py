#!/usr/bin/env python3
"""
Level Set Tracker for Barrier Certificates
Computes and tracks level sets c1 and c2 for proper barrier certificate validation
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LevelSetInfo:
    """Information about barrier certificate level sets"""
    initial_max: float  # c1: max(B(x)) for x in Initial Set
    unsafe_min: float   # c2: min(B(x)) for x in Unsafe Set
    separation: float   # c2 - c1 (must be > 0)
    initial_samples: int
    unsafe_samples: int
    
    @property
    def is_valid(self) -> bool:
        """Check if level sets properly separate initial and unsafe sets"""
        return self.separation > 0
    
    @property
    def separation_margin(self) -> float:
        """Get relative separation margin"""
        if abs(self.initial_max) < 1e-10:
            return float('inf') if self.separation > 0 else 0
        return self.separation / abs(self.initial_max)


class LevelSetTracker:
    """
    Tracks and validates level sets for barrier certificates.
    
    Correct barrier certificate theory:
    1. B(x) ≤ c1 for all x in Initial Set
    2. B(x) ≥ c2 for all x in Unsafe Set  
    3. c1 < c2 (separation condition)
    4. dB/dt ≤ 0 when B(x) = c for c ∈ [c1, c2]
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def compute_level_sets(self, 
                          barrier_expr: str,
                          initial_set: List[str], 
                          unsafe_set: List[str],
                          variables: List[str],
                          n_samples: int = 1000) -> LevelSetInfo:
        """
        Compute level set values c1 and c2 for the barrier certificate.
        
        Args:
            barrier_expr: Barrier certificate expression as string
            initial_set: List of constraints defining initial set
            unsafe_set: List of constraints defining unsafe set
            variables: List of variable names
            n_samples: Number of samples to use
            
        Returns:
            LevelSetInfo with computed values
        """
        # Parse barrier certificate
        var_symbols = sp.symbols(variables)
        B = sp.parse_expr(barrier_expr)
        B_func = sp.lambdify(var_symbols, B, 'numpy')
        
        # Sample initial set and find max(B)
        initial_samples = self._sample_constrained_set(initial_set, variables, n_samples)
        if len(initial_samples) > 0:
            initial_values = np.array([B_func(*sample) for sample in initial_samples])
            c1 = np.max(initial_values)
            logger.debug(f"Initial set: {len(initial_samples)} samples, max(B) = {c1:.6f}")
        else:
            c1 = -np.inf
            logger.warning("No valid samples found in initial set")
            
        # Sample unsafe set and find min(B)
        unsafe_samples = self._sample_constrained_set(unsafe_set, variables, n_samples)
        if len(unsafe_samples) > 0:
            unsafe_values = np.array([B_func(*sample) for sample in unsafe_samples])
            c2 = np.min(unsafe_values)
            logger.debug(f"Unsafe set: {len(unsafe_samples)} samples, min(B) = {c2:.6f}")
        else:
            c2 = np.inf
            logger.warning("No valid samples found in unsafe set")
            
        return LevelSetInfo(
            initial_max=c1,
            unsafe_min=c2,
            separation=c2 - c1,
            initial_samples=len(initial_samples),
            unsafe_samples=len(unsafe_samples)
        )
    
    def validate_separation(self, level_info: LevelSetInfo, min_margin: float = 0.01) -> Dict:
        """
        Validate that level sets properly separate initial and unsafe sets.
        
        Args:
            level_info: Computed level set information
            min_margin: Minimum required separation margin (relative)
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Check basic separation
        if not level_info.is_valid:
            results['valid'] = False
            results['errors'].append(
                f"No separation: c1={level_info.initial_max:.6f} >= c2={level_info.unsafe_min:.6f}"
            )
        else:
            results['checks']['separation'] = True
            
        # Check separation margin
        if level_info.separation_margin < min_margin:
            results['warnings'].append(
                f"Small separation margin: {level_info.separation_margin:.3f} < {min_margin}"
            )
            
        # Check sample coverage
        if level_info.initial_samples < 10:
            results['warnings'].append(
                f"Low initial set coverage: only {level_info.initial_samples} samples"
            )
        if level_info.unsafe_samples < 10:
            results['warnings'].append(
                f"Low unsafe set coverage: only {level_info.unsafe_samples} samples"
            )
            
        results['level_info'] = {
            'c1': level_info.initial_max,
            'c2': level_info.unsafe_min,
            'separation': level_info.separation,
            'margin': level_info.separation_margin
        }
        
        return results
    
    def visualize_level_sets(self, 
                            barrier_expr: str,
                            initial_set: List[str],
                            unsafe_set: List[str], 
                            variables: List[str],
                            level_info: LevelSetInfo,
                            bounds: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create visualization data for level sets (2D only).
        
        Returns:
            Dictionary with plot data or None if not 2D
        """
        if len(variables) != 2:
            logger.warning("Visualization only supported for 2D systems")
            return None
            
        if bounds is None:
            bounds = {variables[0]: [-3, 3], variables[1]: [-3, 3]}
            
        # Create grid
        x_range = np.linspace(bounds[variables[0]][0], bounds[variables[0]][1], 100)
        y_range = np.linspace(bounds[variables[1]][0], bounds[variables[1]][1], 100)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Evaluate barrier on grid
        var_symbols = sp.symbols(variables)
        B = sp.parse_expr(barrier_expr)
        B_func = sp.lambdify(var_symbols, B, 'numpy')
        Z = B_func(X, Y)
        
        return {
            'X': X,
            'Y': Y,
            'Z': Z,
            'c1': level_info.initial_max,
            'c2': level_info.unsafe_min,
            'levels': [level_info.initial_max, 0, level_info.unsafe_min],
            'x_label': variables[0],
            'y_label': variables[1]
        }
    
    def _sample_constrained_set(self,
                               constraints: List[str],
                               variables: List[str], 
                               n_samples: int) -> List[Tuple[float, ...]]:
        """
        Sample points from a set defined by constraints.
        
        Uses a combination of:
        1. Boundary sampling (for sets like x^2 + y^2 <= r^2)
        2. Grid sampling (for box constraints)
        3. Random rejection sampling
        """
        samples = []
        var_symbols = sp.symbols(variables)
        
        # Parse constraints
        parsed_constraints = []
        for constraint in constraints:
            if '<=' in constraint:
                lhs, rhs = constraint.split('<=')
                parsed_constraints.append(
                    (sp.parse_expr(lhs) - sp.parse_expr(rhs), '<=')
                )
            elif '>=' in constraint:
                lhs, rhs = constraint.split('>=')
                parsed_constraints.append(
                    (sp.parse_expr(lhs) - sp.parse_expr(rhs), '>=')
                )
            elif '=' in constraint and not any(op in constraint for op in ['<=', '>=', '!=']):
                lhs, rhs = constraint.split('=')
                parsed_constraints.append(
                    (sp.parse_expr(lhs) - sp.parse_expr(rhs), '=')
                )
                
        # Try to detect special structures
        if len(constraints) == 1 and len(variables) == 2:
            # Check for circular constraint
            constraint_str = constraints[0]
            if 'x**2' in constraint_str and 'y**2' in constraint_str:
                samples.extend(self._sample_circular_set(constraint_str, variables, n_samples))
                
        # Estimate bounds
        bounds = self._estimate_bounds(constraints, variables)
        
        # Grid sampling for better coverage
        if len(variables) <= 3:
            grid_samples = self._grid_sample(bounds, variables, parsed_constraints, 
                                           min(n_samples // 2, 100))
            samples.extend(grid_samples)
            
        # Random rejection sampling for remaining samples
        remaining = n_samples - len(samples)
        if remaining > 0:
            random_samples = self._rejection_sample(bounds, variables, parsed_constraints, 
                                                   remaining)
            samples.extend(random_samples)
            
        return samples[:n_samples]  # Limit to requested number
    
    def _sample_circular_set(self, 
                            constraint: str,
                            variables: List[str],
                            n_samples: int) -> List[Tuple[float, float]]:
        """Special sampling for circular/elliptical sets"""
        samples = []
        
        # Parse to extract radius
        if '<=' in constraint:
            lhs, rhs = constraint.split('<=')
            try:
                r_squared = float(rhs.strip())
                r = np.sqrt(r_squared)
                
                # Sample on boundary
                angles = np.linspace(0, 2*np.pi, n_samples // 2)
                for angle in angles:
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    samples.append((x, y))
                    
                # Sample interior
                for _ in range(n_samples // 2):
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(0, r)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    samples.append((x, y))
                    
            except:
                pass  # Fall back to general sampling
                
        elif '>=' in constraint:
            # For unsafe sets (exterior of circle)
            lhs, rhs = constraint.split('>=')
            try:
                r_squared = float(rhs.strip())
                r = np.sqrt(r_squared)
                
                # Sample on boundary
                angles = np.linspace(0, 2*np.pi, n_samples)
                for angle in angles:
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    samples.append((x, y))
                    
            except:
                pass
                
        return samples
    
    def _grid_sample(self,
                    bounds: Dict,
                    variables: List[str],
                    constraints: List[Tuple],
                    n_points: int) -> List[Tuple[float, ...]]:
        """Grid sampling for systematic coverage"""
        samples = []
        
        # Determine grid size per dimension
        grid_size = int(np.power(n_points, 1/len(variables)))
        
        # Create grid
        grids = []
        for var in variables:
            grids.append(np.linspace(bounds[var][0], bounds[var][1], grid_size))
            
        # Check all grid points
        if len(variables) == 2:
            for x in grids[0]:
                for y in grids[1]:
                    point = (x, y)
                    if self._check_constraints(point, variables, constraints):
                        samples.append(point)
        elif len(variables) == 3:
            for x in grids[0]:
                for y in grids[1]:
                    for z in grids[2]:
                        point = (x, y, z)
                        if self._check_constraints(point, variables, constraints):
                            samples.append(point)
                            
        return samples
    
    def _rejection_sample(self,
                         bounds: Dict,
                         variables: List[str],
                         constraints: List[Tuple],
                         n_samples: int) -> List[Tuple[float, ...]]:
        """Rejection sampling for random coverage"""
        samples = []
        max_attempts = n_samples * 100
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # Generate random point
            point = tuple(np.random.uniform(bounds[var][0], bounds[var][1]) 
                         for var in variables)
            
            # Check constraints
            if self._check_constraints(point, variables, constraints):
                samples.append(point)
                
            attempts += 1
            
        if attempts == max_attempts:
            logger.warning(f"Rejection sampling: only found {len(samples)}/{n_samples} samples")
            
        return samples
    
    def _check_constraints(self,
                          point: Tuple[float, ...],
                          variables: List[str],
                          constraints: List[Tuple]) -> bool:
        """Check if a point satisfies all constraints"""
        point_dict = dict(zip(variables, point))
        
        for expr, op in constraints:
            value = float(expr.subs(point_dict))
            
            if op == '<=' and value > self.tolerance:
                return False
            elif op == '>=' and value < -self.tolerance:
                return False
            elif op == '=' and abs(value) > self.tolerance:
                return False
                
        return True
    
    def _estimate_bounds(self, constraints: List[str], variables: List[str]) -> Dict:
        """Estimate reasonable bounds for variables from constraints"""
        bounds = {var: [-5.0, 5.0] for var in variables}
        
        # Try to extract bounds from constraints
        for constraint in constraints:
            # Look for circular constraints
            if '**2' in constraint and any(op in constraint for op in ['<=', '>=']):
                try:
                    if '<=' in constraint:
                        rhs = float(constraint.split('<=')[1].strip())
                    else:
                        rhs = float(constraint.split('>=')[1].strip())
                    r = np.sqrt(abs(rhs)) * 1.5  # Add margin
                    
                    for var in variables:
                        if var in constraint:
                            bounds[var] = [-r, r]
                except:
                    pass
                    
            # Look for box constraints
            for var in variables:
                # Pattern: var <= value
                if f"{var} <=" in constraint:
                    try:
                        value = float(constraint.split('<=')[1].strip())
                        bounds[var][1] = min(bounds[var][1], value)
                    except:
                        pass
                        
                # Pattern: var >= value
                if f"{var} >=" in constraint:
                    try:
                        value = float(constraint.split('>=')[1].strip())
                        bounds[var][0] = max(bounds[var][0], value)
                    except:
                        pass
                        
        return bounds


# Standalone test
if __name__ == "__main__":
    # Test the level set tracker
    tracker = LevelSetTracker()
    
    # Test system
    barrier = "x**2 + y**2 - 1.0"
    initial_set = ["x**2 + y**2 <= 0.25"]
    unsafe_set = ["x**2 + y**2 >= 4.0"]
    variables = ["x", "y"]
    
    # Compute level sets
    level_info = tracker.compute_level_sets(barrier, initial_set, unsafe_set, variables)
    print(f"Level sets computed:")
    print(f"  c1 (initial max): {level_info.initial_max:.6f}")
    print(f"  c2 (unsafe min): {level_info.unsafe_min:.6f}")
    print(f"  Separation: {level_info.separation:.6f}")
    print(f"  Valid: {level_info.is_valid}")
    
    # Validate separation
    validation = tracker.validate_separation(level_info)
    print(f"\nValidation results:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Warnings: {validation['warnings']}")
    print(f"  Errors: {validation['errors']}") 