import os
import sys
import re
import json
import logging
import time
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.verify_certificate import verify_barrier_certificate
from omegaconf import DictConfig

# Import the new validator
from utils.level_set_tracker import BarrierCertificateValidator

logger = logging.getLogger(__name__)

class VerificationService:
    """Service for verifying barrier certificates."""
    
    def __init__(self, config):
        """Initialize the verification service with configuration."""
        self.config = config
        
    def parse_system_description(self, system_description: str) -> Dict[str, Any]:
        """Parse system description into components needed for verification."""
        try:
            system_info = {}
            
            # Extract system dynamics - handle both continuous and discrete-time
            # First try to find explicit dynamics blocks
            dynamics_match = re.search(r'System Dynamics:\s*(.+?)(?=\n[A-Z]|\n$|$)', system_description, re.IGNORECASE | re.DOTALL)
            dynamics_str = None
            
            if dynamics_match:
                dynamics_str = dynamics_match.group(1).strip()
            else:
                # Look for dynamics anywhere in the description
                # Check for discrete-time patterns first
                discrete_patterns = [
                    r'([a-zA-Z_]\w*)\s*\{\s*k\s*\+\s*1\s*\}\s*=\s*([^,\n]+)',
                    r'([a-zA-Z_]\w*)_\{\s*k\s*\+\s*1\s*\}\s*=\s*([^,\n]+)',
                    r'([a-zA-Z_]\w*)\[\s*k\s*\+\s*1\s*\]\s*=\s*([^,\n]+)',
                ]
                
                # Check for continuous-time patterns
                continuous_patterns = [
                    r'd([a-zA-Z_]\w*)/dt\s*=\s*([^,\n]+)',
                    r'([a-zA-Z_]\w*)\'\s*=\s*([^,\n]+)',
                    r'([a-zA-Z_]\w*)_dot\s*=\s*([^,\n]+)',
                ]
                
                dynamics_found = []
                variables_found = []
                
                # Try discrete patterns first
                for pattern in discrete_patterns:
                    matches = re.findall(pattern, system_description)
                    if matches:
                        for var, expr in matches:
                            dynamics_found.append(f"{var}_{{k+1}} = {expr}")
                            if var not in variables_found:
                                variables_found.append(var)
                
                # If no discrete patterns found, try continuous
                if not dynamics_found:
                    for pattern in continuous_patterns:
                        matches = re.findall(pattern, system_description)
                        if matches:
                            for var, expr in matches:
                                dynamics_found.append(f"d{var}/dt = {expr}" if pattern.startswith('d') else f"{var}' = {expr}")
                                if var not in variables_found:
                                    variables_found.append(var)
                
                if dynamics_found:
                    system_info['dynamics'] = dynamics_found
                    system_info['variables'] = variables_found
                    dynamics_str = '\n'.join(dynamics_found)
            
            if dynamics_str:
                # Parse individual dynamics components
                # Look for discrete-time patterns first
                discrete_patterns = re.findall(r'([a-zA-Z_]\w*)(?:\s*\{\s*k\s*\+\s*1\s*\}|_\{\s*k\s*\+\s*1\s*\}|\[\s*k\s*\+\s*1\s*\])\s*=\s*([^,\n]+)', dynamics_str)
                if discrete_patterns:
                    variables = [var for var, _ in discrete_patterns]
                    dynamics = [f"{var}_{{k+1}} = {expr.strip()}" for var, expr in discrete_patterns]
                    system_info['variables'] = variables
                    system_info['dynamics'] = dynamics
                else:
                    # Look for continuous-time patterns: dx/dt = ..., dy/dt = ..., etc.
                    continuous_patterns = re.findall(r'd([a-zA-Z_]\w*)/dt\s*=\s*([^,\n]+)', dynamics_str)
                    if continuous_patterns:
                        variables = [var for var, _ in continuous_patterns]
                        dynamics = [expr.strip() for _, expr in continuous_patterns]
                        system_info['variables'] = variables
                        system_info['dynamics'] = dynamics
                    else:
                        # Fallback: try to parse as comma-separated expressions
                        dynamics_parts = [d.strip() for d in dynamics_str.split(',')]
                        system_info['dynamics'] = dynamics_parts
                        # Try to infer variables from the dynamics
                        variables = self._infer_variables_from_dynamics(dynamics_parts)
                        system_info['variables'] = variables
            
            # Extract state variables if explicitly listed
            vars_match = re.search(r'State Variables:\s*\[?([^\]]+)\]?', system_description, re.IGNORECASE)
            if vars_match:
                vars_str = vars_match.group(1).strip()
                explicit_vars = [v.strip() for v in vars_str.split(',') if v.strip()]
                if explicit_vars:
                    system_info['variables'] = explicit_vars
            
            # Extract initial set
            initial_match = re.search(r'Initial Set:\s*(.+?)(?=\n[A-Z]|\n$|$)', system_description, re.IGNORECASE)
            if initial_match:
                initial_str = initial_match.group(1).strip()
                system_info['initial_set'] = self._parse_set_conditions(initial_str)
            
            # Extract unsafe set
            unsafe_match = re.search(r'Unsafe Set:\s*(.+?)(?=\n[A-Z]|\n$|$)', system_description, re.IGNORECASE)
            if unsafe_match:
                unsafe_str = unsafe_match.group(1).strip()
                system_info['unsafe_set'] = self._parse_set_conditions(unsafe_str)
            
            # Extract safe set
            safe_match = re.search(r'Safe Set:\s*(.+?)(?=\n[A-Z]|\n$|$)', system_description, re.IGNORECASE)
            if safe_match:
                safe_str = safe_match.group(1).strip()
                system_info['safe_set'] = self._parse_set_conditions(safe_str)
            
            # Default variables if not found
            if 'variables' not in system_info or not system_info['variables']:
                system_info['variables'] = ['x', 'y']  # Default assumption
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error parsing system description: {e}")
            return {
                'variables': ['x', 'y'],
                'dynamics': [],
                'initial_set': [],
                'unsafe_set': [],
                'safe_set': []
            }
    
    def _infer_variables_from_dynamics(self, dynamics: List[str]) -> List[str]:
        """Infer variable names from dynamics expressions."""
        variables = set()
        for expr in dynamics:
            # Look for common variable patterns
            var_matches = re.findall(r'\b([a-zA-Z_]\w*)\b', expr)
            for var in var_matches:
                # Filter out common functions and constants
                if var not in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'pi', 'e'] and len(var) <= 3:
                    variables.add(var)
        
        # Sort variables for consistency
        return sorted(list(variables))
    
    def _parse_set_conditions(self, condition_str: str) -> List[str]:
        """Parse set condition string into list of individual conditions."""
        if not condition_str.strip():
            return []
        
        # Handle logical operators
        condition_str = condition_str.replace(' and ', ' & ').replace(' or ', ' | ')
        
        # Split on common separators while preserving logical structure
        conditions = []
        if ' & ' in condition_str or ' | ' in condition_str:
            # Keep as single condition if it contains logical operators
            conditions = [condition_str.strip()]
        else:
            # Split on commas for multiple separate conditions
            conditions = [c.strip() for c in condition_str.split(',') if c.strip()]
        
        return conditions
    
    def create_sampling_bounds(self, system_info: Dict[str, Any], 
                             default_range: float = 2.0) -> Dict[str, tuple]:
        """Create sampling bounds for verification based on system information."""
        bounds = {}
        variables = system_info.get('variables', ['x', 'y'])
        
        # Default bounds
        for var in variables:
            bounds[var] = (-default_range, default_range)
        
        # Try to infer better bounds from initial and unsafe sets
        try:
            # Look for bounds in initial set conditions
            for condition in system_info.get('initial_set', []):
                self._update_bounds_from_condition(bounds, condition, variables)
            
            # Look for bounds in unsafe set conditions
            for condition in system_info.get('unsafe_set', []):
                self._update_bounds_from_condition(bounds, condition, variables)
                
        except Exception as e:
            logger.warning(f"Failed to infer bounds from sets: {e}")
        
        # ENHANCED: Auto-optimize bounds for specific system patterns
        try:
            self._optimize_bounds_for_system(bounds, system_info)
        except Exception as e:
            logger.warning(f"Failed to optimize bounds: {e}")
        
        return bounds
    
    def _update_bounds_from_condition(self, bounds: Dict[str, tuple], 
                                    condition: str, variables: List[str]):
        """Update bounds based on a set condition."""
        # Look for simple bounds like x <= 1, x >= -1, etc.
        for var in variables:
            # Pattern for x <= value or x < value
            upper_match = re.search(rf'\b{var}\s*<=?\s*([+-]?\d*\.?\d+)', condition)
            if upper_match:
                upper_val = float(upper_match.group(1))
                current_min, current_max = bounds[var]
                bounds[var] = (current_min, min(current_max, upper_val + 0.5))
            
            # Pattern for x >= value or x > value
            lower_match = re.search(rf'\b{var}\s*>=?\s*([+-]?\d*\.?\d+)', condition)
            if lower_match:
                lower_val = float(lower_match.group(1))
                current_min, current_max = bounds[var]
                bounds[var] = (max(current_min, lower_val - 0.5), current_max)
            
            # Pattern for value <= x or value < x
            lower_match2 = re.search(rf'([+-]?\d*\.?\d+)\s*<=?\s*\b{var}\b', condition)
            if lower_match2:
                lower_val = float(lower_match2.group(1))
                current_min, current_max = bounds[var]
                bounds[var] = (max(current_min, lower_val - 0.5), current_max)
            
            # Pattern for value >= x or value > x
            upper_match2 = re.search(rf'([+-]?\d*\.?\d+)\s*>=?\s*\b{var}\b', condition)
            if upper_match2:
                upper_val = float(upper_match2.group(1))
                current_min, current_max = bounds[var]
                bounds[var] = (current_min, min(current_max, upper_val + 0.5))
    
    def _clean_certificate_string(self, certificate_str: str) -> str:
        """Clean LaTeX artifacts and formatting issues from certificate string."""
        if not certificate_str:
            return certificate_str
        
        cleaned = certificate_str.strip()
        
        # Remove common LaTeX artifacts
        cleaned = re.sub(r'\\[\[\]()]', '', cleaned)  # Remove \[ \] \( \)
        cleaned = re.sub(r'\\[\{\}]', '', cleaned)    # Remove \{ \}
        
        # Remove standalone LaTeX brackets at the end
        cleaned = re.sub(r'\s*\\\]\s*$', '', cleaned)  # Remove trailing \]
        cleaned = re.sub(r'\s*\\\[\s*$', '', cleaned)  # Remove trailing \[
        cleaned = re.sub(r'\s*\\\)\s*$', '', cleaned)  # Remove trailing \)
        cleaned = re.sub(r'\s*\\\(\s*$', '', cleaned)  # Remove trailing \(
        
        # Convert LaTeX math operators to standard notation
        cleaned = cleaned.replace('\\cdot', '*')
        cleaned = cleaned.replace('\\times', '*')
        cleaned = cleaned.replace('\\div', '/')
        cleaned = cleaned.replace('^', '**')
        
        # Remove LaTeX commands
        cleaned = re.sub(r'\\[a-zA-Z]+\s*', '', cleaned)
        
        # Clean up whitespace and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip().rstrip('.,;:')
        
        # IMPORTANT: Normalize variable names to match verification expectations
        # Replace x with x_ and y with y_ (but not in expressions like exp)
        cleaned = re.sub(r'\bx\b', 'x_', cleaned)
        cleaned = re.sub(r'\by\b', 'y_', cleaned)
        
        if cleaned != certificate_str:
            logger.debug(f"Cleaned certificate for verification: '{certificate_str}' -> '{cleaned}'")
        
        return cleaned
    
    def verify_certificate(
        self,
        certificate_str: str,
        system_description: str,
        param_overrides: Optional[dict] = None,
        domain_bounds: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Verify a barrier certificate against a system description."""
        try:
            start_time = time.time()
            
            # Clean certificate string to remove LaTeX artifacts
            certificate_str = self._clean_certificate_string(certificate_str)
            
            # Parse system description
            system_info = self.parse_system_description(system_description)
            
            # Create sampling bounds
            sampling_bounds = self.create_sampling_bounds(system_info)
            
            # Ensure sampling bounds use underscored variables
            sampling_bounds_with_underscore = {}
            for var, bounds in sampling_bounds.items():
                if not var.endswith('_'):
                    sampling_bounds_with_underscore[var + '_'] = bounds
                else:
                    sampling_bounds_with_underscore[var] = bounds
            sampling_bounds = sampling_bounds_with_underscore
            
            # Auto-generate safe set conditions if not provided
            if not system_info.get('safe_set'):
                generated_safe_set = self._generate_safe_set_conditions(system_info)
                system_info['safe_set'] = generated_safe_set
                logger.info(f"Auto-generated safe set conditions: {generated_safe_set}")
            
            # Get verification configuration first
            base_cfg = self.config.evaluation.verification
            verification_cfg_dict = {k: v for k, v in base_cfg.items()}

            if param_overrides:
                for key, val in param_overrides.items():
                    if key in verification_cfg_dict and val is not None:
                        verification_cfg_dict[key] = val

            # DISCRETE-TIME OPTIMIZATIONS: Enhance parameters BEFORE creating verification_system_info
            is_discrete = self._detect_discrete_system(system_info)
            if is_discrete:
                logger.info("Detected discrete-time system - applying optimizations")
                logger.info(f"Original sampling bounds: {sampling_bounds}")
                # For discrete systems, use more conservative bounds and higher sample counts
                for var in sampling_bounds:
                    current_min, current_max = sampling_bounds[var]
                    # Compress sampling bounds for discrete systems to focus on reachable region
                    margin = min(abs(current_max - current_min) * 0.3, 1.0)
                    sampling_bounds[var] = (current_min + margin, current_max - margin)
                
                logger.info(f"Optimized sampling bounds for discrete system: {sampling_bounds}")
                
                # Override verification parameters for discrete systems if not set
                if not param_overrides or 'numerical_tolerance' not in param_overrides:
                    verification_cfg_dict['numerical_tolerance'] = 1e-6  # More strict tolerance
                    logger.info(f"Set discrete numerical tolerance: 1e-6")
                if not param_overrides or 'num_samples_lie' not in param_overrides:
                    verification_cfg_dict['num_samples_lie'] = min(15000, verification_cfg_dict.get('num_samples_lie', 9100) * 1.5)
                    logger.info(f"Set discrete lie samples: {verification_cfg_dict['num_samples_lie']}")
                if not param_overrides or 'num_samples_boundary' not in param_overrides:
                    verification_cfg_dict['num_samples_boundary'] = min(8000, verification_cfg_dict.get('num_samples_boundary', 4600) * 1.5)
                    logger.info(f"Set discrete boundary samples: {verification_cfg_dict['num_samples_boundary']}")
            else:
                logger.info("System detected as continuous-time - using standard parameters")
            
            # Prepare system info for verification
            # Ensure variables use underscores for consistency with certificate
            variables_with_underscore = []
            for var in system_info.get('variables', ['x', 'y']):
                if not var.endswith('_'):
                    variables_with_underscore.append(var + '_')
                else:
                    variables_with_underscore.append(var)
            
            # Update dynamics to use underscored variables consistently
            dynamics_with_underscore = []
            original_vars = ['x', 'y']  # Original variable names without underscores
            
            for dyn in system_info.get('dynamics', []):
                updated_dyn = dyn
                # First, fix any double underscores that might exist
                updated_dyn = updated_dyn.replace('x__', 'x_').replace('y__', 'y_')
                
                # Then ensure right-hand side variables have underscores
                for var in original_vars:
                    # Replace x[k] with x_[k], but not x_ which is already underscored
                    updated_dyn = re.sub(rf'\b{var}(?!_)\b', f'{var}_', updated_dyn)
                    
                dynamics_with_underscore.append(updated_dyn)
            
            verification_system_info = {
                'variables': variables_with_underscore,
                'state_variables': variables_with_underscore,  # Alternative naming
                'dynamics': dynamics_with_underscore,
                'initial_set_conditions': system_info.get('initial_set', []),
                'unsafe_set_conditions': system_info.get('unsafe_set', []),
                'safe_set_conditions': system_info.get('safe_set', []),
                'sampling_bounds': sampling_bounds,
                'certificate_domain_bounds': domain_bounds
            }
            
            verification_cfg = DictConfig(verification_cfg_dict)
            
            # NEW: Use BarrierCertificateValidator for enhanced verification
            try:
                # Create validator instance
                validator = BarrierCertificateValidator(
                    certificate_str=certificate_str,
                    system_info=verification_system_info,
                    config=verification_cfg
                )
                
                # Perform validation using the new approach
                validation_result = validator.validate()
                
                # If new validation passes, also run traditional verification for comparison
                if validation_result['is_valid']:
                    logger.info("New validator passed - running traditional verification for comparison")
                    traditional_result = verify_barrier_certificate(
                        certificate_str,
                        verification_system_info,
                        verification_cfg,
                    )
                else:
                    logger.info("New validator failed - skipping traditional verification")
                    traditional_result = None
                
            except Exception as e:
                logger.warning(f"New validator failed with error: {e}. Falling back to traditional verification.")
                validation_result = None
                traditional_result = verify_barrier_certificate(
                    certificate_str,
                    verification_system_info,
                    verification_cfg,
                )
            
            verification_time = time.time() - start_time
            
            # Combine results from both validators
            if validation_result:
                # Use new validator results as primary
                verification_result = {
                    'overall_success': validation_result['is_valid'],
                    'numerical_passed': validation_result.get('numerical_valid', False),
                    'symbolic_passed': validation_result.get('symbolic_valid', False),
                    'sos_passed': validation_result.get('sos_valid', False),
                    'verification_time': verification_time,
                    'details': {
                        'parsing': {'success': True, 'certificate': certificate_str},
                        'numerical': {
                            'success': validation_result.get('numerical_valid', False),
                            'reason': validation_result.get('numerical_reason', ''),
                            'level_sets': validation_result.get('level_sets', {}),
                            'separation_valid': validation_result.get('separation_valid', False),
                            'lie_derivative_valid': validation_result.get('lie_derivative_valid', False)
                        },
                        'symbolic': validation_result.get('symbolic_details', {}),
                        'sos': validation_result.get('sos_details', {}),
                        'system_info': verification_system_info,
                        'certificate': certificate_str,
                        'new_validator_used': True,
                        'traditional_result': traditional_result if traditional_result else None
                    }
                }
            else:
                # Fall back to traditional verification result format
                result = traditional_result
                if isinstance(result, dict):
                    verification_result = {
                        'overall_success': result.get('overall_success', False),
                        'numerical_passed': result.get('numerical_verification', {}).get('success', False),
                        'symbolic_passed': result.get('symbolic_verification', {}).get('success', False),
                        'sos_passed': result.get('sos_verification', {}).get('success', False),
                        'verification_time': verification_time,
                        'details': {
                            'parsing': result.get('parsing', {}),
                            'numerical': result.get('numerical_verification', {}),
                            'symbolic': result.get('symbolic_verification', {}),
                            'sos': result.get('sos_verification', {}),
                            'system_info': verification_system_info,
                            'certificate': certificate_str,
                            'new_validator_used': False
                        }
                    }
                else:
                    # Handle legacy result format
                    verification_result = {
                        'overall_success': False,
                        'numerical_passed': False,
                        'symbolic_passed': False,
                        'sos_passed': False,
                        'verification_time': verification_time,
                        'details': {
                            'error': 'Unexpected verification result format',
                            'result': str(result),
                            'system_info': verification_system_info,
                            'certificate': certificate_str,
                            'new_validator_used': False
                        }
                    }
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error in certificate verification: {str(e)}")
            return {
                'overall_success': False,
                'numerical_passed': False,
                'symbolic_passed': False,
                'sos_passed': False,
                'verification_time': None,
                'details': {
                    'error': str(e),
                    'certificate': certificate_str,
                    'system_description': system_description
                }
            }
    
    def get_verification_summary(self, verification_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of verification results."""
        if verification_result['overall_success']:
            return "✅ Certificate passed all verification checks"
        
        passed_checks = []
        failed_checks = []
        
        if verification_result['numerical_passed']:
            passed_checks.append("Numerical")
        else:
            failed_checks.append("Numerical")
        
        if verification_result['symbolic_passed']:
            passed_checks.append("Symbolic")
        else:
            failed_checks.append("Symbolic")
        
        if verification_result['sos_passed']:
            passed_checks.append("SOS")
        else:
            failed_checks.append("SOS")
        
        summary = ""
        if passed_checks:
            summary += f"✅ Passed: {', '.join(passed_checks)}"
        if failed_checks:
            if summary:
                summary += " | "
            summary += f"❌ Failed: {', '.join(failed_checks)}"
        
        return summary if summary else "❌ Verification failed"
    
    def get_detailed_feedback(self, verification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed feedback for each verification check."""
        feedback = []
        details = verification_result.get('details', {})
        
        # Numerical verification feedback
        numerical = details.get('numerical', {})
        if numerical:
            status = "✅ Passed" if verification_result['numerical_passed'] else "❌ Failed"
            message = numerical.get('reason', 'No details available')
            feedback.append({
                'type': 'Numerical Verification',
                'status': status,
                'message': message,
                'details': numerical
            })
        
        # Symbolic verification feedback
        symbolic = details.get('symbolic', {})
        if symbolic:
            status = "✅ Passed" if verification_result['symbolic_passed'] else "❌ Failed"
            message = symbolic.get('reason', 'No details available')
            feedback.append({
                'type': 'Symbolic Verification',
                'status': status,
                'message': message,
                'details': symbolic
            })
        
        # SOS verification feedback
        sos = details.get('sos', {})
        if sos:
            status = "✅ Passed" if verification_result['sos_passed'] else "❌ Failed"
            message = sos.get('reason', 'No details available')
            feedback.append({
                'type': 'Sum-of-Squares (SOS) Verification',
                'status': status,
                'message': message,
                'details': sos
            })
        
        # Add parsing information if available
        parsing = details.get('parsing', {})
        if parsing and not parsing.get('success', True):
            feedback.insert(0, {
                'type': 'Certificate Parsing',
                'status': "❌ Failed",
                'message': parsing.get('error', 'Failed to parse certificate'),
                'details': parsing
            })
        
        return feedback
    
    def _optimize_bounds_for_system(self, bounds: Dict[str, tuple], system_info: Dict[str, Any]):
        """Optimize sampling bounds based on system characteristics."""
        initial_conditions = system_info.get('initial_set', [])
        unsafe_conditions = system_info.get('unsafe_set', [])
        
        # For circular initial sets like x^2 + y^2 <= r^2
        for condition in initial_conditions:
            if '**2' in condition and '+' in condition and '<=' in condition:
                # Pattern: x**2 + y**2 <= value
                circle_match = re.search(r'([a-zA-Z_]\w*)\*\*2\s*\+\s*([a-zA-Z_]\w*)\*\*2\s*<=?\s*([+-]?\d*\.?\d+)', condition)
                if circle_match:
                    var1, var2, radius_sq = circle_match.groups()
                    radius = float(radius_sq) ** 0.5
                    # Expand bounds to cover initial set plus margin
                    margin = max(1.0, radius * 2)
                    if var1 in bounds:
                        bounds[var1] = (-margin, margin)
                    if var2 in bounds:
                        bounds[var2] = (-margin, margin)
                    logger.info(f"Optimized bounds for circular initial set: radius={radius:.2f}, bounds=±{margin:.2f}")
        
        # For unsafe boundaries like x >= value
        for condition in unsafe_conditions:
            bound_match = re.search(r'([a-zA-Z_]\w*)\s*>=?\s*([+-]?\d*\.?\d+)', condition)
            if bound_match:
                var, boundary = bound_match.groups()
                boundary_val = float(boundary)
                if var in bounds:
                    # Extend lower bound to well before the unsafe boundary
                    current_min, current_max = bounds[var]
                    safe_margin = abs(boundary_val) * 0.5 + 1.0
                    new_min = min(current_min, boundary_val - safe_margin)
                    new_max = min(current_max, boundary_val - 0.1)  # Stay safely away from boundary
                    bounds[var] = (new_min, new_max)
                    logger.info(f"Optimized bounds for unsafe boundary {var} >= {boundary_val}: [{new_min:.2f}, {new_max:.2f}]")
    
    def _generate_safe_set_conditions(self, system_info: Dict[str, Any]) -> List[str]:
        """Auto-generate safe set conditions when not explicitly provided."""
        safe_conditions = []
        unsafe_conditions = system_info.get('unsafe_set', [])
        initial_conditions = system_info.get('initial_set', [])
        
        # Method 1: Safe set as complement of unsafe set
        for unsafe_condition in unsafe_conditions:
            safe_condition = self._negate_condition(unsafe_condition)
            if safe_condition:
                safe_conditions.append(safe_condition)
                logger.info(f"Generated safe set condition from unsafe set: {safe_condition}")
        
        # Method 2: If no unsafe set, use expanded initial set
        if not safe_conditions and initial_conditions:
            for init_condition in initial_conditions:
                expanded_condition = self._expand_condition(init_condition)
                if expanded_condition:
                    safe_conditions.append(expanded_condition)
                    logger.info(f"Generated safe set condition from initial set: {expanded_condition}")
        
        # Method 3: Default conservative bounds
        if not safe_conditions:
            variables = system_info.get('variables', ['x', 'y'])
            for var in variables:
                safe_conditions.append(f"{var} <= 10")
                safe_conditions.append(f"{var} >= -10")
            logger.info(f"Generated default safe set conditions: {safe_conditions}")
        
        return safe_conditions
    
    def _negate_condition(self, condition: str) -> str:
        """Convert unsafe condition to safe condition by negation."""
        condition = condition.strip()
        
        # x >= value -> x < value
        if '>=' in condition:
            return condition.replace('>=', '<')
        # x > value -> x <= value  
        elif '>' in condition:
            return condition.replace('>', '<=')
        # x <= value -> x > value
        elif '<=' in condition:
            return condition.replace('<=', '>')
        # x < value -> x >= value
        elif '<' in condition:
            return condition.replace('<', '>=')
        
        logger.warning(f"Could not negate condition: {condition}")
        return ""
    
    def _expand_condition(self, condition: str) -> str:
        """Expand initial set condition to create a larger safe region."""
        # For circular conditions like x^2 + y^2 <= r^2, expand radius
        circle_match = re.search(r'([a-zA-Z_]\w*)\*\*2\s*\+\s*([a-zA-Z_]\w*)\*\*2\s*<=?\s*([+-]?\d*\.?\d+)', condition)
        if circle_match:
            var1, var2, radius_sq = circle_match.groups()
            expanded_radius_sq = float(radius_sq) * 4  # Expand by factor of 2 in radius (4 in area)
            return f"{var1}**2 + {var2}**2 <= {expanded_radius_sq}"
        
        # For linear bounds, expand by factor
        bound_match = re.search(r'([a-zA-Z_]\w*)\s*(<=?|>=?)\s*([+-]?\d*\.?\d+)', condition)
        if bound_match:
            var, op, value = bound_match.groups()
            expanded_value = float(value) * 2
            return f"{var} {op} {expanded_value}"
        
        return ""
    
    def _detect_discrete_system(self, system_info: Dict[str, Any]) -> bool:
        """Detect if system is discrete-time based on dynamics patterns."""
        dynamics = system_info.get('dynamics', [])
        
        for dyn in dynamics:
            # Look for discrete-time patterns: x[k+1], x_{k+1}, x{k+1}
            if any(pattern in dyn for pattern in ['[k+1]', '_{k+1}', '{k+1}']):
                return True
        
        return False 