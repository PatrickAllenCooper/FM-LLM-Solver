#!/usr/bin/env python3
"""
Set Membership Testing Module
Provides robust methods for testing if points belong to constrained sets
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


class SetMembershipTester:
    """
    Robust set membership testing with support for:
    - Inequality constraints (<=, >=, <, >)
    - Equality constraints (=)
    - Composite constraints (AND/OR)
    - Epsilon-ball testing for boundaries
    """
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        
    def is_in_set(self, 
                  point: Union[Tuple[float, ...], np.ndarray],
                  constraints: List[str],
                  variables: List[str],
                  strict: bool = False) -> bool:
        """
        Test if a point satisfies all constraints.
        
        Args:
            point: The point to test
            constraints: List of constraint strings
            variables: List of variable names
            strict: If True, use strict inequalities
            
        Returns:
            True if point satisfies all constraints
        """
        point_dict = dict(zip(variables, point))
        
        for constraint in constraints:
            if not self._evaluate_constraint(constraint, point_dict, strict):
                return False
                
        return True
    
    def is_on_boundary(self,
                      point: Union[Tuple[float, ...], np.ndarray],
                      constraints: List[str],
                      variables: List[str],
                      tolerance: Optional[float] = None) -> bool:
        """
        Test if a point is on the boundary of a set.
        
        A point is on the boundary if at least one constraint is satisfied
        with equality (within tolerance).
        """
        if tolerance is None:
            tolerance = self.epsilon
            
        point_dict = dict(zip(variables, point))
        
        for constraint in constraints:
            if self._is_constraint_boundary(constraint, point_dict, tolerance):
                return True
                
        return False
    
    def distance_to_set(self,
                       point: Union[Tuple[float, ...], np.ndarray],
                       constraints: List[str],
                       variables: List[str]) -> float:
        """
        Compute approximate distance from point to set.
        
        Returns:
            Negative if inside set, positive if outside
        """
        point_dict = dict(zip(variables, point))
        distances = []
        
        for constraint in constraints:
            dist = self._constraint_distance(constraint, point_dict)
            distances.append(dist)
            
        # For AND constraints, take the maximum (worst violation)
        return max(distances) if distances else 0.0
    
    def sample_boundary(self,
                       constraints: List[str],
                       variables: List[str],
                       n_samples: int = 100) -> List[Tuple[float, ...]]:
        """
        Sample points on the boundary of a set.
        
        Specialized for common cases like circles and boxes.
        """
        samples = []
        
        # Try to detect special structures
        if len(constraints) == 1 and len(variables) == 2:
            constraint = constraints[0]
            
            # Circular boundary
            if 'x**2' in constraint and 'y**2' in constraint:
                if '<=' in constraint or '>=' in constraint:
                    samples.extend(self._sample_circular_boundary(constraint, n_samples))
                    
        # Add more special cases as needed
        
        # Fallback: use numerical methods
        if not samples:
            samples = self._sample_boundary_numerical(constraints, variables, n_samples)
            
        return samples[:n_samples]
    
    def _evaluate_constraint(self,
                           constraint: str,
                           point_dict: Dict[str, float],
                           strict: bool) -> bool:
        """Evaluate a single constraint at a point"""
        try:
            # Parse constraint
            if '<=' in constraint:
                lhs, rhs = constraint.split('<=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return lhs_val <= rhs_val + (0 if strict else self.epsilon)
                
            elif '>=' in constraint:
                lhs, rhs = constraint.split('>=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return lhs_val >= rhs_val - (0 if strict else self.epsilon)
                
            elif '<' in constraint and '=' not in constraint:
                lhs, rhs = constraint.split('<')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return lhs_val < rhs_val - self.epsilon
                
            elif '>' in constraint and '=' not in constraint:
                lhs, rhs = constraint.split('>')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return lhs_val > rhs_val + self.epsilon
                
            elif '=' in constraint and not any(op in constraint for op in ['<=', '>=', '!=']):
                lhs, rhs = constraint.split('=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return abs(lhs_val - rhs_val) <= self.epsilon
                
            else:
                logger.warning(f"Unknown constraint format: {constraint}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating constraint {constraint}: {e}")
            return False
    
    def _evaluate_expression(self, expr: str, point_dict: Dict[str, float]) -> float:
        """Evaluate a mathematical expression at a point"""
        # Parse with sympy
        parsed = sp.parse_expr(expr)
        # Substitute values
        return float(parsed.subs(point_dict))
    
    def _is_constraint_boundary(self,
                              constraint: str,
                              point_dict: Dict[str, float],
                              tolerance: float) -> bool:
        """Check if constraint is satisfied with equality"""
        try:
            if '<=' in constraint:
                lhs, rhs = constraint.split('<=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return abs(lhs_val - rhs_val) <= tolerance
                
            elif '>=' in constraint:
                lhs, rhs = constraint.split('>=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return abs(lhs_val - rhs_val) <= tolerance
                
            elif '=' in constraint and not any(op in constraint for op in ['<=', '>=', '!=']):
                # Already an equality constraint
                return self._evaluate_constraint(constraint, point_dict, False)
                
        except:
            pass
            
        return False
    
    def _constraint_distance(self,
                           constraint: str,
                           point_dict: Dict[str, float]) -> float:
        """
        Compute signed distance to constraint boundary.
        Negative if constraint is satisfied, positive if violated.
        """
        try:
            if '<=' in constraint:
                lhs, rhs = constraint.split('<=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return lhs_val - rhs_val  # Negative if satisfied
                
            elif '>=' in constraint:
                lhs, rhs = constraint.split('>=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return rhs_val - lhs_val  # Negative if satisfied
                
            elif '=' in constraint and not any(op in constraint for op in ['<=', '>=', '!=']):
                lhs, rhs = constraint.split('=')
                lhs_val = self._evaluate_expression(lhs.strip(), point_dict)
                rhs_val = self._evaluate_expression(rhs.strip(), point_dict)
                return abs(lhs_val - rhs_val)  # Always positive
                
        except:
            pass
            
        return float('inf')  # Unknown constraint
    
    def _sample_circular_boundary(self, constraint: str, n_samples: int) -> List[Tuple[float, float]]:
        """Sample points on a circular boundary"""
        samples = []
        
        # Parse radius
        if '<=' in constraint:
            lhs, rhs = constraint.split('<=')
            r_squared = float(rhs.strip())
        elif '>=' in constraint:
            lhs, rhs = constraint.split('>=')
            r_squared = float(rhs.strip())
        else:
            return samples
            
        r = np.sqrt(r_squared)
        
        # Sample uniformly on circle
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        for angle in angles:
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            samples.append((x, y))
            
        return samples
    
    def _sample_boundary_numerical(self,
                                 constraints: List[str],
                                 variables: List[str],
                                 n_samples: int) -> List[Tuple[float, ...]]:
        """
        Numerical boundary sampling using gradient projection.
        This is a fallback for complex constraint sets.
        """
        samples = []
        
        # Start with random points and project to boundary
        bounds = self._estimate_bounds(constraints, variables)
        
        for _ in range(n_samples * 10):  # Oversample
            # Random starting point
            point = np.array([np.random.uniform(bounds[v][0], bounds[v][1]) 
                            for v in variables])
            
            # Try to project to boundary
            projected = self._project_to_boundary(point, constraints, variables)
            
            if projected is not None:
                samples.append(tuple(projected))
                
            if len(samples) >= n_samples:
                break
                
        return samples
    
    def _project_to_boundary(self,
                           point: np.ndarray,
                           constraints: List[str],
                           variables: List[str],
                           max_iter: int = 50) -> Optional[np.ndarray]:
        """Project a point onto the constraint boundary"""
        current = point.copy()
        
        for _ in range(max_iter):
            # Check which constraints are active
            point_dict = dict(zip(variables, current))
            
            # Find the most violated or nearest constraint
            min_dist = float('inf')
            closest_constraint = None
            
            for constraint in constraints:
                dist = abs(self._constraint_distance(constraint, point_dict))
                if dist < min_dist:
                    min_dist = dist
                    closest_constraint = constraint
                    
            if min_dist < self.epsilon:
                # Already on boundary
                return current
                
            if closest_constraint is None:
                return None
                
            # Move towards constraint boundary
            # This is a simplified version - could use proper gradient projection
            step_size = min(0.1, min_dist)
            gradient = self._constraint_gradient(closest_constraint, current, variables)
            
            if gradient is not None:
                current = current - step_size * gradient
                
        return None if min_dist > self.epsilon * 10 else current
    
    def _constraint_gradient(self,
                           constraint: str,
                           point: np.ndarray,
                           variables: List[str]) -> Optional[np.ndarray]:
        """Compute gradient of constraint function"""
        try:
            # Parse constraint to get the function
            if '<=' in constraint:
                lhs, rhs = constraint.split('<=')
                func_str = f"({lhs}) - ({rhs})"
            elif '>=' in constraint:
                lhs, rhs = constraint.split('>=')
                func_str = f"({rhs}) - ({lhs})"
            else:
                return None
                
            # Compute gradient using sympy
            var_symbols = sp.symbols(variables)
            func = sp.parse_expr(func_str)
            
            gradient = []
            for var in var_symbols:
                grad_component = sp.diff(func, var)
                grad_val = float(grad_component.subs(dict(zip(variables, point))))
                gradient.append(grad_val)
                
            return np.array(gradient)
            
        except:
            return None
    
    def _estimate_bounds(self, constraints: List[str], variables: List[str]) -> Dict[str, Tuple[float, float]]:
        """Estimate reasonable bounds for variables"""
        bounds = {var: [-10.0, 10.0] for var in variables}
        
        # Try to extract from constraints
        for constraint in constraints:
            # Look for patterns like x <= value or x >= value
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
                        
            # Look for circular constraints
            if '**2' in constraint:
                try:
                    if '<=' in constraint:
                        rhs = float(constraint.split('<=')[1].strip())
                        r = np.sqrt(rhs) * 1.5
                        for var in variables:
                            if var in constraint:
                                bounds[var] = (-r, r)
                except:
                    pass
                    
        return bounds


# Convenience functions
def is_in_initial_set(point: Union[Tuple[float, ...], np.ndarray],
                     initial_constraints: List[str],
                     variables: List[str]) -> bool:
    """Check if point is in initial set"""
    tester = SetMembershipTester()
    return tester.is_in_set(point, initial_constraints, variables)


def is_in_unsafe_set(point: Union[Tuple[float, ...], np.ndarray],
                    unsafe_constraints: List[str],
                    variables: List[str]) -> bool:
    """Check if point is in unsafe set"""
    tester = SetMembershipTester()
    return tester.is_in_set(point, unsafe_constraints, variables)


# Test the module
if __name__ == "__main__":
    tester = SetMembershipTester()
    
    # Test 1: Circle membership
    print("Test 1: Circle membership")
    constraints = ["x**2 + y**2 <= 1.0"]
    print(f"  (0, 0) in unit circle: {tester.is_in_set((0, 0), constraints, ['x', 'y'])}")
    print(f"  (1, 0) in unit circle: {tester.is_in_set((1, 0), constraints, ['x', 'y'])}")
    print(f"  (2, 0) in unit circle: {tester.is_in_set((2, 0), constraints, ['x', 'y'])}")
    
    # Test 2: Boundary detection
    print("\nTest 2: Boundary detection")
    print(f"  (1, 0) on boundary: {tester.is_on_boundary((1, 0), constraints, ['x', 'y'])}")
    print(f"  (0.5, 0) on boundary: {tester.is_on_boundary((0.5, 0), constraints, ['x', 'y'])}")
    
    # Test 3: Distance to set
    print("\nTest 3: Distance to set")
    print(f"  Distance from (0, 0): {tester.distance_to_set((0, 0), constraints, ['x', 'y']):.3f}")
    print(f"  Distance from (2, 0): {tester.distance_to_set((2, 0), constraints, ['x', 'y']):.3f}")
    
    # Test 4: Boundary sampling
    print("\nTest 4: Boundary sampling")
    boundary_points = tester.sample_boundary(constraints, ['x', 'y'], n_samples=4)
    for i, point in enumerate(boundary_points):
        print(f"  Boundary point {i+1}: ({point[0]:.3f}, {point[1]:.3f})") 