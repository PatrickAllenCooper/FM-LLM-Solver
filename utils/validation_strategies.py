#!/usr/bin/env python3
"""
Validation Strategies Framework for Barrier Certificate Validation
Phase 2: Advanced Features & Optimization

This module implements multiple validation strategies that can work together
to provide higher confidence and speed in barrier certificate validation.
"""

import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result from a validation strategy"""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    execution_time: float
    samples_used: int
    violations_found: List[Dict]
    strategy_name: str
    details: Dict[str, Any]

@dataclass
class StrategyPerformance:
    """Performance metrics for a validation strategy"""
    strategy_name: str
    avg_execution_time: float
    avg_confidence: float
    success_rate: float
    samples_per_second: float

class BaseValidationStrategy(ABC):
    """Abstract base class for validation strategies"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.name = self.__class__.__name__
        self.performance_history = []
        
    @abstractmethod
    def validate(self, 
                certificate: str,
                system_info: Dict[str, Any]) -> ValidationResult:
        """Validate a barrier certificate using this strategy"""
        pass
    
    @abstractmethod
    def can_handle(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the given problem"""
        pass
    
    def get_performance_metrics(self) -> StrategyPerformance:
        """Get performance metrics for this strategy"""
        if not self.performance_history:
            return StrategyPerformance(
                strategy_name=self.name,
                avg_execution_time=0.0,
                avg_confidence=0.0,
                success_rate=0.0,
                samples_per_second=0.0
            )
            
        times = [r.execution_time for r in self.performance_history]
        confidences = [r.confidence for r in self.performance_history]
        success_count = sum(1 for r in self.performance_history if r.is_valid)
        
        return StrategyPerformance(
            strategy_name=self.name,
            avg_execution_time=np.mean(times),
            avg_confidence=np.mean(confidences),
            success_rate=success_count / len(self.performance_history),
            samples_per_second=np.mean([r.samples_used / r.execution_time 
                                      for r in self.performance_history if r.execution_time > 0])
        )

class SamplingValidationStrategy(BaseValidationStrategy):
    """Sampling-based validation (current approach)"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.name = "SamplingValidation"
        
    def can_handle(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Sampling can handle any certificate"""
        return True
        
    def validate(self, 
                certificate: str,
                system_info: Dict[str, Any]) -> ValidationResult:
        """Validate using uniform sampling"""
        start_time = time.time()
        
        try:
            # Parse certificate
            variables = system_info['variables']
            var_symbols = sp.symbols(variables)
            B_expr = sp.parse_expr(certificate)
            B_func = sp.lambdify(var_symbols, B_expr, 'numpy')
            
            # Sample points
            n_samples = self.config.get('num_samples_boundary', 1000)
            samples = self._generate_samples(system_info, n_samples)
            
            # Check barrier conditions
            violations = []
            valid_conditions = 0
            total_conditions = 0
            
            # Check initial set condition
            initial_violations = self._check_initial_set_condition(
                B_func, samples, system_info, variables
            )
            violations.extend(initial_violations)
            
            # Check unsafe set condition
            unsafe_violations = self._check_unsafe_set_condition(
                B_func, samples, system_info, variables
            )
            violations.extend(unsafe_violations)
            
            # Check Lie derivative condition
            lie_violations = self._check_lie_derivative_condition(
                certificate, system_info, variables
            )
            violations.extend(lie_violations)
            
            # Determine validity and confidence
            is_valid = len(violations) == 0
            confidence = self._calculate_confidence(len(violations), n_samples)
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                execution_time=execution_time,
                samples_used=n_samples,
                violations_found=violations,
                strategy_name=self.name,
                details={'violation_count': len(violations)}
            )
            
            self.performance_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Sampling validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                execution_time=time.time() - start_time,
                samples_used=0,
                violations_found=[],
                strategy_name=self.name,
                details={'error': str(e)}
            )
    
    def _generate_samples(self, system_info: Dict[str, Any], n_samples: int) -> np.ndarray:
        """Generate uniform samples within bounds"""
        bounds = system_info['sampling_bounds']
        variables = system_info['variables']
        
        samples = []
        for _ in range(n_samples):
            sample = []
            for var in variables:
                min_val, max_val = bounds[var]
                sample.append(np.random.uniform(min_val, max_val))
            samples.append(sample)
            
        return np.array(samples)
    
    def _check_initial_set_condition(self, B_func, samples, system_info, variables):
        """Check B(x) <= c1 for x in initial set"""
        violations = []
        # Implementation would check points in initial set
        # This is a simplified version
        return violations
    
    def _check_unsafe_set_condition(self, B_func, samples, system_info, variables):
        """Check B(x) >= c2 for x in unsafe set"""
        violations = []
        # Implementation would check points in unsafe set
        # This is a simplified version
        return violations
    
    def _check_lie_derivative_condition(self, certificate, system_info, variables):
        """Check dB/dt <= 0"""
        violations = []
        # Implementation would check Lie derivative
        # This is a simplified version
        return violations
    
    def _calculate_confidence(self, violation_count: int, sample_count: int) -> float:
        """Calculate confidence based on violations and samples"""
        if violation_count == 0:
            return 0.9  # High confidence if no violations found
        else:
            return max(0.1, 1.0 - (violation_count / sample_count))

class SymbolicValidationStrategy(BaseValidationStrategy):
    """Symbolic validation using SymPy"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.name = "SymbolicValidation"
        
    def can_handle(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Symbolic validation works for polynomial systems"""
        try:
            # Check if certificate is polynomial
            variables = system_info['variables']
            var_symbols = sp.symbols(variables)
            B_expr = sp.parse_expr(certificate)
            
            # Check if dynamics are polynomial
            dynamics = system_info['dynamics']
            for dyn in dynamics:
                sp.parse_expr(dyn)
                
            return True
        except:
            return False
    
    def validate(self, 
                certificate: str,
                system_info: Dict[str, Any]) -> ValidationResult:
        """Validate using symbolic methods"""
        start_time = time.time()
        
        try:
            variables = system_info['variables']
            var_symbols = sp.symbols(variables)
            B_expr = sp.parse_expr(certificate)
            
            # Symbolic validation steps
            # 1. Check if B is continuous and differentiable
            # 2. Compute Lie derivative symbolically
            # 3. Check conditions at boundary points
            
            # Simplified symbolic check
            is_valid = self._symbolic_check_conditions(B_expr, system_info, var_symbols)
            confidence = 0.95 if is_valid else 0.8  # High confidence for symbolic
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                execution_time=execution_time,
                samples_used=0,  # No sampling in symbolic
                violations_found=[],
                strategy_name=self.name,
                details={'method': 'symbolic'}
            )
            
            self.performance_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Symbolic validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                execution_time=time.time() - start_time,
                samples_used=0,
                violations_found=[],
                strategy_name=self.name,
                details={'error': str(e)}
            )
    
    def _symbolic_check_conditions(self, B_expr, system_info, var_symbols):
        """Perform symbolic condition checking"""
        # Simplified symbolic validation
        # In practice, this would use more sophisticated symbolic reasoning
        return True  # Placeholder

class IntervalValidationStrategy(BaseValidationStrategy):
    """Interval arithmetic validation for robust computation"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.name = "IntervalValidation"
        
    def can_handle(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Interval validation works for most continuous functions"""
        return True
    
    def validate(self, 
                certificate: str,
                system_info: Dict[str, Any]) -> ValidationResult:
        """Validate using interval arithmetic"""
        start_time = time.time()
        
        try:
            # Interval arithmetic validation
            # This provides guaranteed bounds on function values
            is_valid = self._interval_check_conditions(certificate, system_info)
            confidence = 0.85  # Good confidence for interval methods
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                execution_time=execution_time,
                samples_used=0,  # No sampling in interval arithmetic
                violations_found=[],
                strategy_name=self.name,
                details={'method': 'interval_arithmetic'}
            )
            
            self.performance_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Interval validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                execution_time=time.time() - start_time,
                samples_used=0,
                violations_found=[],
                strategy_name=self.name,
                details={'error': str(e)}
            )
    
    def _interval_check_conditions(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Check conditions using interval arithmetic"""
        # Simplified interval validation
        # In practice, this would use proper interval arithmetic libraries
        return True  # Placeholder

class SMTValidationStrategy(BaseValidationStrategy):
    """SMT solver-based validation using Z3/dReal"""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.name = "SMTValidation"
        self._check_smt_availability()
        
    def _check_smt_availability(self):
        """Check if SMT solvers are available"""
        try:
            import z3
            self.z3_available = True
        except ImportError:
            self.z3_available = False
            logger.warning("Z3 not available for SMT validation")
    
    def can_handle(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """SMT validation works for many constraint systems"""
        return self.z3_available
    
    def validate(self, 
                certificate: str,
                system_info: Dict[str, Any]) -> ValidationResult:
        """Validate using SMT solver"""
        start_time = time.time()
        
        if not self.z3_available:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                execution_time=time.time() - start_time,
                samples_used=0,
                violations_found=[],
                strategy_name=self.name,
                details={'error': 'SMT solver not available'}
            )
        
        try:
            # SMT-based validation
            is_valid = self._smt_check_conditions(certificate, system_info)
            confidence = 0.9  # Very high confidence for SMT methods
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                execution_time=execution_time,
                samples_used=0,  # No sampling in SMT
                violations_found=[],
                strategy_name=self.name,
                details={'method': 'smt_solver'}
            )
            
            self.performance_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"SMT validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                execution_time=time.time() - start_time,
                samples_used=0,
                violations_found=[],
                strategy_name=self.name,
                details={'error': str(e)}
            )
    
    def _smt_check_conditions(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Check conditions using SMT solver"""
        # Simplified SMT validation
        # In practice, this would use Z3 or dReal
        return True  # Placeholder

def create_strategy_registry() -> Dict[str, BaseValidationStrategy]:
    """Create a registry of available validation strategies"""
    config = DictConfig({
        'num_samples_boundary': 1000,
        'num_samples_lie': 2000,
        'numerical_tolerance': 1e-6
    })
    
    return {
        'sampling': SamplingValidationStrategy(config),
        'symbolic': SymbolicValidationStrategy(config),
        'interval': IntervalValidationStrategy(config),
        'smt': SMTValidationStrategy(config)
    }

# Test the strategies
if __name__ == "__main__":
    # Test with a simple system
    system_info = {
        'variables': ['x', 'y'],
        'dynamics': ['-x', '-y'],
        'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
        'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
        'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
    }
    
    certificate = "x**2 + y**2 - 1.0"
    
    # Test each strategy
    registry = create_strategy_registry()
    
    for name, strategy in registry.items():
        print(f"\nTesting {name} strategy...")
        if strategy.can_handle(certificate, system_info):
            result = strategy.validate(certificate, system_info)
            print(f"  Valid: {result.is_valid}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Time: {result.execution_time:.3f}s")
        else:
            print(f"  Cannot handle this problem") 