#!/usr/bin/env python3
"""
Real LLM GPU Testing Suite for FM-LLM Solver

CRITICAL: Tests the complete pipeline with actual GPU-accelerated LLM inference
to validate filtering, parsing, and numerical checking on real model outputs.

Hardware Requirements: NVIDIA GPU with 6+ GB VRAM
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fm_llm_solver.services.model_provider import QwenProvider
from fm_llm_solver.core.types import ModelConfig, ModelProvider
from utils.certificate_extraction import extract_certificate_from_llm_output, is_template_expression
from utils.certificate_extraction import clean_and_validate_expression
from fm_llm_solver.services.verifier import CertificateVerifier
from utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RealLLMTestResult:
    """Result from real LLM testing."""
    test_name: str
    system_description: str
    prompt_tokens: int
    raw_llm_output: str
    generation_time: float
    extracted_certificate: Optional[str]
    extraction_success: bool
    is_template: bool
    cleaned_certificate: Optional[str]
    verification_result: Optional[Dict]
    numerical_checks_passed: bool
    overall_success: bool
    error_message: Optional[str] = None
    gpu_memory_used: Optional[float] = None

class RealLLMTestSuite:
    """Test suite for real GPU-accelerated LLM inference."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize with real model provider."""
        self.model_name = model_name
        self.provider = None
        self.verifier = None
        self.config = None
        self.results = []
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! This test requires GPU.")
        
        logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        logger.info(f"🔋 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    def setup(self):
        """Setup real model and services."""
        logger.info("🔄 Setting up real LLM testing environment...")
        
        # Load config
        self.config = load_config()
        
        # Initialize verifier
        self.verifier = CertificateVerifier(self.config)
        
        # Setup model config for GPU inference with quantization
        model_config = ModelConfig(
            provider=ModelProvider.QWEN,
            name=self.model_name,
            device="cuda",
            quantization="4bit",  # Use 4-bit quantization for 7B model on 8GB GPU
            temperature=0.1,
            max_tokens=512
        )
        
        # Initialize model provider
        logger.info(f"🤖 Loading real model: {self.model_name}")
        logger.info("⏳ This may take 2-3 minutes for first load...")
        
        start_time = time.time()
        self.provider = QwenProvider()
        self.provider.load_model(model_config)
        load_time = time.time() - start_time
        
        logger.info(f"✅ Model loaded in {load_time:.1f} seconds")
        logger.info(f"🔥 GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
    def get_test_systems(self) -> List[Dict]:
        """Get test systems for real LLM testing."""
        return [
            {
                "name": "stable_linear_2d",
                "description": """
System: dx/dt = -x, dy/dt = -y
Initial set: x² + y² ≤ 0.25
Unsafe set: x² + y² ≥ 4.0
Domain: x ∈ [-3, 3], y ∈ [-3, 3]

Generate a barrier certificate B(x,y) that separates the initial set from the unsafe set.
                """.strip(),
                "variables": ["x", "y"],
                "expected_form": "quadratic"
            },
            {
                "name": "nonlinear_2d",
                "description": """
System: dx/dt = -x + x*y, dy/dt = -y - x²
Initial set: x² + y² ≤ 0.1
Unsafe set: x² + y² ≥ 3.0
Domain: x ∈ [-2, 2], y ∈ [-2, 2]

Generate a barrier certificate B(x,y) for this nonlinear system.
                """.strip(),
                "variables": ["x", "y"],
                "expected_form": "polynomial"
            },
            {
                "name": "complex_polynomial",
                "description": """
System: dx/dt = -x - y + 0.1*x³, dy/dt = x - y + 0.1*y³
Initial set: x² + y² ≤ 0.5
Unsafe set: x² + y² ≥ 5.0
Domain: x ∈ [-3, 3], y ∈ [-3, 3]

Generate a polynomial barrier certificate that handles the cubic nonlinearity.
                """.strip(),
                "variables": ["x", "y"],
                "expected_form": "high_order_polynomial"
            },
            {
                "name": "discrete_time_system",
                "description": """
Discrete-time system: x[k+1] = 0.8*x[k] + 0.1*y[k], y[k+1] = -0.1*x[k] + 0.9*y[k]
Initial set: x² + y² ≤ 0.2
Unsafe set: x² + y² ≥ 2.0
Domain: x ∈ [-2, 2], y ∈ [-2, 2]

Generate a barrier certificate for this discrete-time linear system.
                """.strip(),
                "variables": ["x", "y"],
                "expected_form": "quadratic"
            },
            {
                "name": "challenging_coefficients",
                "description": """
System: dx/dt = -0.7*x + 0.3*y, dy/dt = -0.2*x - 1.1*y
Initial set: 2*x² + 3*y² ≤ 1.0
Unsafe set: x² + y² ≥ 8.0
Domain: x ∈ [-4, 4], y ∈ [-4, 4]

Generate a barrier certificate with specific numerical coefficients.
                """.strip(),
                "variables": ["x", "y"],
                "expected_form": "specific_coefficients"
            }
        ]
    
    def build_prompt(self, system_description: str) -> str:
        """Build prompt for barrier certificate generation."""
        return f"""You are an expert in control theory and barrier certificates. Your task is to generate a valid barrier certificate for the given dynamical system.

Instructions:
1. Analyze the system dynamics, initial set, and unsafe set
2. Generate a polynomial barrier certificate B(x,y) that satisfies:
   - B(x,y) ≤ 0 for all points in the initial set
   - B(x,y) > 0 for all points in the unsafe set  
   - The Lie derivative ∇B·f(x,y) ≤ 0 along system trajectories
3. Use specific numerical coefficients (not templates like 'a', 'b', 'c')
4. Format your answer as: BARRIER_CERTIFICATE_START
B(x,y) = [your certificate]
BARRIER_CERTIFICATE_END

System Description:
{system_description}

Generate the barrier certificate:"""

    def test_single_system(self, system: Dict) -> RealLLMTestResult:
        """Test certificate generation for a single system."""
        logger.info(f"🧪 Testing system: {system['name']}")
        
        # Build prompt
        prompt = self.build_prompt(system['description'])
        prompt_tokens = len(prompt.split())
        
        # Record GPU memory before generation
        initial_memory = torch.cuda.memory_allocated()
        
        # Generate with real LLM
        try:
            if self.provider is None:
                raise RuntimeError("Model provider not initialized")
                
            start_time = time.time()
            raw_output = self.provider.generate_text(
                prompt=prompt,
                max_tokens=256,
                temperature=0.1
            )
            generation_time = time.time() - start_time
            
            # Record GPU memory after generation
            final_memory = torch.cuda.memory_allocated()
            gpu_memory_used = (final_memory - initial_memory) / 1e6  # MB
            
            logger.info(f"📝 Raw LLM output ({generation_time:.1f}s):")
            logger.info(f"'{raw_output[:200]}...'")
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return RealLLMTestResult(
                test_name=system['name'],
                system_description=system['description'],
                prompt_tokens=prompt_tokens,
                raw_llm_output="",
                generation_time=0.0,
                extracted_certificate=None,
                extraction_success=False,
                is_template=True,
                cleaned_certificate=None,
                verification_result=None,
                numerical_checks_passed=False,
                overall_success=False,
                error_message=str(e)
            )
        
        # Test certificate extraction from real output
        try:
            extracted_result = extract_certificate_from_llm_output(
                raw_output, system['variables']
            )
            extracted_cert = extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
            extraction_success = extracted_cert is not None
            
            logger.info(f"🔍 Extracted certificate: '{extracted_cert}'")
            
        except Exception as e:
            logger.error(f"❌ Certificate extraction failed: {e}")
            extracted_cert = None
            extraction_success = False
        
        # Test template detection
        is_template = is_template_expression(extracted_cert) if extracted_cert else True
        logger.info(f"📋 Is template: {is_template}")
        
        # Test certificate cleaning
        cleaned_cert = None
        if extracted_cert and not is_template:
            try:
                cleaned_cert = clean_and_validate_expression(extracted_cert, system['variables'])
                logger.info(f"🧹 Cleaned certificate: '{cleaned_cert}'")
            except Exception as e:
                logger.warning(f"⚠️ Certificate cleaning failed: {e}")
        
        # Test numerical verification
        verification_result = None
        numerical_checks_passed = False
        
        if cleaned_cert:
            try:
                # Create mock system for verification
                mock_system = {
                    "dynamics": system['description'].split('\n')[0].split(': ')[1].split(', '),
                    "initial_set": [system['description'].split('Initial set: ')[1].split('\n')[0]],
                    "unsafe_set": [system['description'].split('Unsafe set: ')[1].split('\n')[0]]
                }
                
                # Perform basic numerical checks
                verification_result = self._perform_numerical_checks(
                    cleaned_cert, mock_system, system['variables']
                )
                numerical_checks_passed = verification_result.get('passed', False)
                
                logger.info(f"✓ Numerical checks: {'PASSED' if numerical_checks_passed else 'FAILED'}")
                
            except Exception as e:
                logger.warning(f"⚠️ Numerical verification failed: {e}")
                verification_result = {"error": str(e), "passed": False}
        
        # Overall success criteria
        overall_success = (
            extraction_success and 
            not is_template and 
            cleaned_cert is not None and
            numerical_checks_passed
        )
        
        result = RealLLMTestResult(
            test_name=system['name'],
            system_description=system['description'],
            prompt_tokens=prompt_tokens,
            raw_llm_output=raw_output,
            generation_time=generation_time,
            extracted_certificate=extracted_cert,
            extraction_success=extraction_success,
            is_template=is_template,
            cleaned_certificate=cleaned_cert,
            verification_result=verification_result,
            numerical_checks_passed=numerical_checks_passed,
            overall_success=overall_success,
            gpu_memory_used=gpu_memory_used
        )
        
        logger.info(f"🎯 Overall success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        return result
    
    def _perform_numerical_checks(
        self, 
        certificate: str, 
        system: Dict, 
        variables: List[str]
    ) -> Dict:
        """Perform numerical checks on the certificate."""
        try:
            import sympy as sp
            
            # Parse certificate into symbolic expression
            var_symbols = [sp.Symbol(var) for var in variables]
            cert_expr = sp.sympify(certificate)
            
            # Basic checks
            checks = {
                "is_polynomial": cert_expr.is_polynomial(*var_symbols),
                "has_specific_coefficients": not any(
                    str(coeff) in ['a', 'b', 'c', 'd', 'e', 'f'] 
                    for coeff in cert_expr.free_symbols
                ),
                "degree": sp.degree(cert_expr),
                "passed": True
            }
            
            # Check if all coefficients are numeric
            free_symbols = cert_expr.free_symbols
            non_var_symbols = free_symbols - set(var_symbols)
            checks["has_only_numeric_coeffs"] = len(non_var_symbols) == 0
            
            # Overall pass/fail
            checks["passed"] = (
                checks["is_polynomial"] and 
                checks["has_specific_coefficients"] and
                checks["has_only_numeric_coeffs"]
            )
            
            return checks
            
        except ImportError:
            # Fallback without sympy
            basic_checks = {
                "template_variables": any(
                    var in certificate.lower() 
                    for var in ['a*', 'b*', 'c*', 'alpha', 'beta', 'gamma']
                ),
                "has_numbers": any(char.isdigit() for char in certificate),
                "passed": True
            }
            
            basic_checks["passed"] = (
                not basic_checks["template_variables"] and 
                basic_checks["has_numbers"]
            )
            
            return basic_checks
        
        except Exception as e:
            return {"error": str(e), "passed": False}
    
    def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive real LLM tests."""
        logger.info("🎯 Starting comprehensive real LLM testing...")
        
        test_systems = self.get_test_systems()
        results = []
        
        total_start_time = time.time()
        
        for system in test_systems:
            result = self.test_single_system(system)
            results.append(result)
            self.results.append(result)
            
            # Clear GPU cache between tests
            torch.cuda.empty_cache()
            time.sleep(1)  # Brief pause
        
        total_time = time.time() - total_start_time
        
        # Analyze results
        analysis = self._analyze_results(results)
        analysis['total_test_time'] = total_time
        
        logger.info("📊 Test Summary:")
        logger.info(f"   Total tests: {analysis['total_tests']}")
        logger.info(f"   Successful generations: {analysis['successful_generations']}")
        logger.info(f"   Successful extractions: {analysis['successful_extractions']}")
        logger.info(f"   Non-template certificates: {analysis['non_template_certs']}")
        logger.info(f"   Numerical checks passed: {analysis['numerical_checks_passed']}")
        logger.info(f"   Overall success rate: {analysis['overall_success_rate']:.1%}")
        logger.info(f"   Average generation time: {analysis['avg_generation_time']:.2f}s")
        logger.info(f"   Total GPU time: {total_time:.1f}s")
        
        return analysis
    
    def _analyze_results(self, results: List[RealLLMTestResult]) -> Dict:
        """Analyze test results."""
        total_tests = len(results)
        successful_generations = sum(1 for r in results if r.raw_llm_output)
        successful_extractions = sum(1 for r in results if r.extraction_success)
        non_template_certs = sum(1 for r in results if not r.is_template)
        numerical_checks_passed = sum(1 for r in results if r.numerical_checks_passed)
        overall_successes = sum(1 for r in results if r.overall_success)
        
        avg_generation_time = np.mean([r.generation_time for r in results if r.generation_time > 0])
        total_gpu_memory = sum(r.gpu_memory_used or 0 for r in results)
        
        return {
            'total_tests': total_tests,
            'successful_generations': successful_generations,
            'successful_extractions': successful_extractions,
            'non_template_certs': non_template_certs,
            'numerical_checks_passed': numerical_checks_passed,
            'overall_successes': overall_successes,
            'generation_success_rate': successful_generations / total_tests if total_tests > 0 else 0,
            'extraction_success_rate': successful_extractions / total_tests if total_tests > 0 else 0,
            'template_rejection_rate': (total_tests - non_template_certs) / total_tests if total_tests > 0 else 0,
            'numerical_success_rate': numerical_checks_passed / total_tests if total_tests > 0 else 0,
            'overall_success_rate': overall_successes / total_tests if total_tests > 0 else 0,
            'avg_generation_time': avg_generation_time,
            'total_gpu_memory_used': total_gpu_memory,
            'results': [asdict(r) for r in results]
        }
    
    def save_results(self, filename: str = "real_llm_test_results.json"):
        """Save test results to file."""
        analysis = self._analyze_results(self.results)
        
        output_path = Path("test_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to: {output_path}")
        return output_path
    
    def cleanup(self):
        """Cleanup resources."""
        if self.provider:
            self.provider.unload_model()
        torch.cuda.empty_cache()
        logger.info("🧹 Cleanup completed")

def main():
    """Main test runner."""
    test_suite = RealLLMTestSuite()
    
    try:
        # Setup
        test_suite.setup()
        
        # Run tests
        analysis = test_suite.run_comprehensive_tests()
        
        # Save results
        results_file = test_suite.save_results()
        
        # Print final summary
        print("\n" + "="*60)
        print("REAL LLM GPU TESTING SUMMARY")
        print("="*60)
        print(f"Overall Success Rate: {analysis['overall_success_rate']:.1%}")
        print(f"Generation Success Rate: {analysis['generation_success_rate']:.1%}")
        print(f"Extraction Success Rate: {analysis['extraction_success_rate']:.1%}")
        print(f"Template Rejection Rate: {analysis['template_rejection_rate']:.1%}")
        print(f"Numerical Verification Rate: {analysis['numerical_success_rate']:.1%}")
        print(f"Average Generation Time: {analysis['avg_generation_time']:.2f}s")
        print(f"Results saved to: {results_file}")
        
        # Determine overall status
        if analysis['overall_success_rate'] >= 0.8:
            print("\n🎉 STATUS: EXCELLENT - Real LLM pipeline working correctly!")
            return 0
        elif analysis['overall_success_rate'] >= 0.6:
            print("\n⚠️  STATUS: GOOD - Minor issues need attention")
            return 1
        else:
            print("\n❌ STATUS: NEEDS WORK - Major issues with real LLM pipeline")
            return 2
            
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        return 3
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    exit(main()) 