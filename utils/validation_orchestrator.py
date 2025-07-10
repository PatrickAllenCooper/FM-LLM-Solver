#!/usr/bin/env python3
"""
Validation Orchestrator for Barrier Certificate Validation
Phase 2: Advanced Features & Optimization

This module orchestrates multiple validation strategies to provide
optimal performance and confidence in barrier certificate validation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig

from .validation_strategies import (
    BaseValidationStrategy, 
    ValidationResult, 
    StrategyPerformance,
    create_strategy_registry
)

logger = logging.getLogger(__name__)

@dataclass
class OrchestratedResult:
    """Result from orchestrated validation"""
    is_valid: bool
    confidence: float
    execution_time: float
    strategies_used: List[str]
    individual_results: Dict[str, ValidationResult]
    consensus_achieved: bool
    details: Dict[str, Any]

class ValidationOrchestrator:
    """
    Orchestrates multiple validation strategies for optimal performance
    and confidence in barrier certificate validation.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.strategies = create_strategy_registry()
        self.performance_history = []
        
        # Strategy selection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.max_strategies = config.get('max_strategies', 3)
        self.parallel_execution = config.get('parallel_execution', True)
        
    def validate(self, 
                certificate: str,
                system_info: Dict[str, Any]) -> OrchestratedResult:
        """
        Validate a barrier certificate using orchestrated strategies
        """
        start_time = time.time()
        
        # 1. Select appropriate strategies
        selected_strategies = self._select_strategies(certificate, system_info)
        
        # 2. Execute strategies (parallel or sequential)
        if self.parallel_execution and len(selected_strategies) > 1:
            individual_results = self._execute_parallel(selected_strategies, certificate, system_info)
        else:
            individual_results = self._execute_sequential(selected_strategies, certificate, system_info)
        
        # 3. Combine results
        final_result = self._combine_results(individual_results)
        
        # 4. Add execution time
        final_result.execution_time = time.time() - start_time
        
        # 5. Store performance data
        self.performance_history.append(final_result)
        
        return final_result
    
    def _select_strategies(self, 
                          certificate: str, 
                          system_info: Dict[str, Any]) -> List[BaseValidationStrategy]:
        """
        Intelligently select validation strategies based on problem characteristics
        """
        available_strategies = []
        
        # Check which strategies can handle this problem
        for name, strategy in self.strategies.items():
            if strategy.can_handle(certificate, system_info):
                available_strategies.append((name, strategy))
        
        if not available_strategies:
            logger.warning("No strategies can handle this problem")
            return []
        
        # Score strategies based on problem characteristics
        scored_strategies = []
        for name, strategy in available_strategies:
            score = self._score_strategy(strategy, certificate, system_info)
            scored_strategies.append((score, name, strategy))
        
        # Sort by score and select top strategies
        scored_strategies.sort(key=lambda x: x[0], reverse=True)
        selected = [strategy for _, _, strategy in scored_strategies[:self.max_strategies]]
        
        logger.info(f"Selected strategies: {[s.name for s in selected]}")
        return selected
    
    def _score_strategy(self, 
                       strategy: BaseValidationStrategy,
                       certificate: str,
                       system_info: Dict[str, Any]) -> float:
        """
        Score a strategy based on problem characteristics and historical performance
        """
        score = 0.0
        
        # 1. Historical performance (40% weight)
        performance = strategy.get_performance_metrics()
        if performance.success_rate > 0:
            score += 0.4 * performance.success_rate
        
        # 2. Problem-specific scoring (60% weight)
        problem_score = self._calculate_problem_score(strategy, certificate, system_info)
        score += 0.6 * problem_score
        
        return score
    
    def _calculate_problem_score(self,
                               strategy: BaseValidationStrategy,
                               certificate: str,
                               system_info: Dict[str, Any]) -> float:
        """
        Calculate problem-specific score for a strategy
        """
        score = 0.5  # Base score
        
        # Check certificate complexity
        if '**' in certificate or '*' in certificate:
            if strategy.name == "SymbolicValidation":
                score += 0.2  # Symbolic good for polynomials
            elif strategy.name == "SamplingValidation":
                score += 0.1  # Sampling works for any function
        
        # Check system dimension
        n_vars = len(system_info['variables'])
        if n_vars <= 2:
            if strategy.name == "SMTValidation":
                score += 0.2  # SMT good for low dimensions
        elif n_vars > 4:
            if strategy.name == "SamplingValidation":
                score += 0.2  # Sampling scales well
        
        # Check if system is polynomial
        try:
            import sympy as sp
            variables = system_info['variables']
            var_symbols = sp.symbols(variables)
            sp.parse_expr(certificate)
            
            # Check dynamics
            dynamics = system_info['dynamics']
            polynomial_system = True
            for dyn in dynamics:
                try:
                    sp.parse_expr(dyn)
                except:
                    polynomial_system = False
                    break
            
            if polynomial_system:
                if strategy.name == "SymbolicValidation":
                    score += 0.3  # Symbolic excellent for polynomials
                elif strategy.name == "SMTValidation":
                    score += 0.2  # SMT good for polynomials
        except:
            pass
        
        return min(1.0, score)
    
    def _execute_parallel(self,
                         strategies: List[BaseValidationStrategy],
                         certificate: str,
                         system_info: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Execute strategies in parallel
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            # Submit all strategies
            future_to_strategy = {
                executor.submit(strategy.validate, certificate, system_info): strategy
                for strategy in strategies
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy.name] = result
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed: {e}")
                    # Create error result
                    results[strategy.name] = ValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        execution_time=0.0,
                        samples_used=0,
                        violations_found=[],
                        strategy_name=strategy.name,
                        details={'error': str(e)}
                    )
        
        return results
    
    def _execute_sequential(self,
                          strategies: List[BaseValidationStrategy],
                          certificate: str,
                          system_info: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Execute strategies sequentially
        """
        results = {}
        
        for strategy in strategies:
            try:
                result = strategy.validate(certificate, system_info)
                results[strategy.name] = result
                
                # Early termination if high confidence achieved
                if result.confidence >= self.confidence_threshold:
                    logger.info(f"Early termination: {strategy.name} achieved high confidence")
                    break
                    
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                results[strategy.name] = ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    execution_time=0.0,
                    samples_used=0,
                    violations_found=[],
                    strategy_name=strategy.name,
                    details={'error': str(e)}
                )
        
        return results
    
    def _combine_results(self, 
                        individual_results: Dict[str, ValidationResult]) -> OrchestratedResult:
        """
        Combine results from multiple strategies
        """
        if not individual_results:
            return OrchestratedResult(
                is_valid=False,
                confidence=0.0,
                execution_time=0.0,
                strategies_used=[],
                individual_results={},
                consensus_achieved=False,
                details={'error': 'No strategies executed'}
            )
        
        # Calculate consensus
        valid_results = [r.is_valid for r in individual_results.values()]
        confidence_scores = [r.confidence for r in individual_results.values()]
        
        # Consensus logic
        consensus_achieved = len(set(valid_results)) == 1  # All agree
        avg_confidence = np.mean(confidence_scores)
        
        # Final decision based on consensus and confidence
        if consensus_achieved:
            final_valid = valid_results[0]
            final_confidence = min(0.95, avg_confidence + 0.1)  # Boost confidence for consensus
        else:
            # No consensus - use weighted average
            weights = np.array(confidence_scores)
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted voting
            weighted_valid = np.sum(weights * np.array(valid_results))
            final_valid = weighted_valid > 0.5
            
            final_confidence = avg_confidence * 0.8  # Reduce confidence for disagreement
        
        return OrchestratedResult(
            is_valid=final_valid,
            confidence=final_confidence,
            execution_time=0.0,  # Will be set by caller
            strategies_used=list(individual_results.keys()),
            individual_results=individual_results,
            consensus_achieved=consensus_achieved,
            details={
                'consensus': consensus_achieved,
                'avg_confidence': avg_confidence,
                'strategy_count': len(individual_results)
            }
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of orchestrated validation
        """
        if not self.performance_history:
            return {
                'total_validations': 0,
                'avg_execution_time': 0.0,
                'avg_confidence': 0.0,
                'consensus_rate': 0.0,
                'strategy_usage': {}
            }
        
        total_validations = len(self.performance_history)
        avg_execution_time = np.mean([r.execution_time for r in self.performance_history])
        avg_confidence = np.mean([r.confidence for r in self.performance_history])
        consensus_rate = np.mean([r.consensus_achieved for r in self.performance_history])
        
        # Strategy usage statistics
        strategy_usage = {}
        for result in self.performance_history:
            for strategy_name in result.strategies_used:
                strategy_usage[strategy_name] = strategy_usage.get(strategy_name, 0) + 1
        
        return {
            'total_validations': total_validations,
            'avg_execution_time': avg_execution_time,
            'avg_confidence': avg_confidence,
            'consensus_rate': consensus_rate,
            'strategy_usage': strategy_usage
        }
    
    def get_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """
        Get performance metrics for all strategies
        """
        return {
            name: strategy.get_performance_metrics()
            for name, strategy in self.strategies.items()
        }

# Test the orchestrator
if __name__ == "__main__":
    # Test configuration
    config = DictConfig({
        'confidence_threshold': 0.8,
        'max_strategies': 3,
        'parallel_execution': True,
        'num_samples_boundary': 1000,
        'num_samples_lie': 2000,
        'numerical_tolerance': 1e-6
    })
    
    # Test system
    system_info = {
        'variables': ['x', 'y'],
        'dynamics': ['-x', '-y'],
        'initial_set_conditions': ['x**2 + y**2 <= 0.25'],
        'unsafe_set_conditions': ['x**2 + y**2 >= 4.0'],
        'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
    }
    
    certificate = "x**2 + y**2 - 1.0"
    
    # Create orchestrator
    orchestrator = ValidationOrchestrator(config)
    
    # Run validation
    print("Running orchestrated validation...")
    result = orchestrator.validate(certificate, system_info)
    
    print(f"\nOrchestrated Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Strategies Used: {result.strategies_used}")
    print(f"  Consensus Achieved: {result.consensus_achieved}")
    
    # Show individual results
    print(f"\nIndividual Results:")
    for strategy_name, individual_result in result.individual_results.items():
        print(f"  {strategy_name}:")
        print(f"    Valid: {individual_result.is_valid}")
        print(f"    Confidence: {individual_result.confidence:.3f}")
        print(f"    Time: {individual_result.execution_time:.3f}s")
    
    # Show performance summary
    print(f"\nPerformance Summary:")
    summary = orchestrator.get_performance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}") 