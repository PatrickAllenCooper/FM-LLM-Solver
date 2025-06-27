#!/usr/bin/env python3
"""
Test script for web interface LLM generation.
Focuses on consistent barrier certificate generation and verification.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web_interface.certificate_generator import CertificateGenerator
from web_interface.verification_service import VerificationService
from utils.config_loader import load_config


class WebLLMGenerationTester:
    """Test LLM generation for web interface."""
    
    def __init__(self):
        """Initialize tester."""
        print("Loading configuration...")
        self.config = load_config("config.yaml")
        
        print("Initializing services...")
        self.generator = CertificateGenerator(self.config)
        self.verifier = VerificationService(self.config)
        
        self.results = []
        
    def test_generation(self, system_desc, domain_bounds, name="Test"):
        """Test generation for a single system."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"System: {system_desc}")
        print(f"Domain: {domain_bounds}")
        print("-"*60)
        
        # Generate certificate
        start_time = time.time()
        gen_result = self.generator.generate_certificate(
            system_description=system_desc,
            model_key="base",
            rag_k=0,  # No RAG for now
            domain_bounds=domain_bounds
        )
        gen_time = time.time() - start_time
        
        result = {
            'name': name,
            'system': system_desc,
            'domain': domain_bounds,
            'generation_time': gen_time,
            'generation_success': gen_result['success'],
            'certificate': gen_result.get('certificate'),
            'llm_output': gen_result.get('llm_output', ''),
            'error': gen_result.get('error'),
            'verification': None
        }
        
        print(f"Generation: {'✅ SUCCESS' if gen_result['success'] else '❌ FAILED'}")
        
        if gen_result['success'] and gen_result.get('certificate'):
            print(f"Certificate: {gen_result['certificate']}")
            
            # Verify certificate
            print("\nVerifying certificate...")
            verif_start = time.time()
            
            verif_result = self.verifier.verify_certificate(
                gen_result['certificate'],
                system_desc,
                {'num_samples_lie': 1000, 'num_samples_boundary': 500},
                domain_bounds
            )
            verif_time = time.time() - verif_start
            
            result['verification'] = {
                'time': verif_time,
                'overall_success': verif_result.get('overall_success', False),
                'numerical': verif_result.get('numerical_passed', False),
                'symbolic': verif_result.get('symbolic_passed', False),
                'sos': verif_result.get('sos_passed', False)
            }
            
            print(f"Verification: {'✅ PASSED' if verif_result['overall_success'] else '❌ FAILED'}")
            print(f"  Numerical: {'✅' if verif_result['numerical_passed'] else '❌'}")
            print(f"  Symbolic: {'✅' if verif_result['symbolic_passed'] else '❌'}")
            print(f"  SOS: {'✅' if verif_result['sos_passed'] else '❌'}")
        else:
            if gen_result.get('error'):
                print(f"Error: {gen_result['error']}")
            print(f"LLM Output Preview: {gen_result.get('llm_output', '')[:200]}...")
            
        self.results.append(result)
        return result
        
    def run_test_suite(self):
        """Run comprehensive test suite."""
        print("\n🚀 Starting Web LLM Generation Test Suite")
        print("="*60)
        
        test_cases = [
            {
                'name': 'Simple Linear Stable',
                'system': 'Discrete-time system: x[k+1] = 0.9*x[k] + 0.1*y[k], y[k+1] = -0.1*x[k] + 0.8*y[k]',
                'domain': {'x': [-5, 5], 'y': [-5, 5]}
            },
            {
                'name': 'Decoupled System',
                'system': 'Discrete-time system: x[k+1] = 0.95*x[k], y[k+1] = 0.9*y[k]',
                'domain': {'x': [-2, 2], 'y': [-2, 2]}
            },
            {
                'name': 'Coupled with Bounds',
                'system': 'Discrete-time system: x[k+1] = 0.8*x[k] - 0.2*y[k], y[k+1] = 0.1*x[k] + 0.85*y[k]',
                'domain': {'x': [-3, 3], 'y': [-3, 3]}
            },
            {
                'name': 'Near-Identity System',
                'system': 'Discrete-time system: x[k+1] = 0.99*x[k] - 0.01*y[k], y[k+1] = 0.01*x[k] + 0.99*y[k]',
                'domain': {'x': [-10, 10], 'y': [-10, 10]}
            },
            {
                'name': 'Strong Coupling',
                'system': 'Discrete-time system: x[k+1] = 0.7*x[k] + 0.3*y[k], y[k+1] = -0.3*x[k] + 0.7*y[k]',
                'domain': {'x': [-4, 4], 'y': [-4, 4]}
            }
        ]
        
        # Run tests
        for test_case in test_cases:
            self.test_generation(
                test_case['system'],
                test_case['domain'],
                test_case['name']
            )
            
        # Generate summary
        self.print_summary()
        
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total = len(self.results)
        gen_success = sum(1 for r in self.results if r['generation_success'])
        verif_success = sum(1 for r in self.results if r['verification'] and r['verification']['overall_success'])
        
        print(f"Total Tests: {total}")
        print(f"Generation Success: {gen_success}/{total} ({gen_success/total*100:.1f}%)")
        print(f"Verification Success: {verif_success}/{gen_success if gen_success > 0 else 1} ({verif_success/(gen_success if gen_success > 0 else 1)*100:.1f}%)")
        
        print("\nDetailed Results:")
        for r in self.results:
            status = "✅" if r['generation_success'] and r['verification'] and r['verification']['overall_success'] else "❌"
            print(f"\n{status} {r['name']}:")
            if r['certificate']:
                print(f"   Certificate: {r['certificate']}")
                if r['verification']:
                    v = r['verification']
                    print(f"   Verification: Num={v['numerical']}, Sym={v['symbolic']}, SOS={v['sos']}")
            else:
                print(f"   Failed: {r.get('error', 'No certificate generated')}")
                
    def analyze_failures(self):
        """Analyze failure patterns."""
        print("\n" + "="*60)
        print("🔍 FAILURE ANALYSIS")
        print("="*60)
        
        failures = [r for r in self.results if not r['generation_success'] or (r['verification'] and not r['verification']['overall_success'])]
        
        if not failures:
            print("No failures to analyze! 🎉")
            return
            
        # Categorize failures
        gen_failures = [f for f in failures if not f['generation_success']]
        verif_failures = [f for f in failures if f['generation_success'] and f['verification'] and not f['verification']['overall_success']]
        
        if gen_failures:
            print(f"\nGeneration Failures ({len(gen_failures)}):")
            for f in gen_failures:
                print(f"  - {f['name']}: {f.get('error', 'Unknown error')}")
                if f['llm_output']:
                    print(f"    LLM Output: {f['llm_output'][:100]}...")
                    
        if verif_failures:
            print(f"\nVerification Failures ({len(verif_failures)}):")
            for f in verif_failures:
                v = f['verification']
                print(f"  - {f['name']}: Num={v['numerical']}, Sym={v['symbolic']}, SOS={v['sos']}")
                print(f"    Certificate: {f['certificate']}")
                
    def save_results(self, filename="web_llm_test_results.json"):
        """Save results to file."""
        output_path = Path("results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total_tests': len(self.results),
                    'generation_success': sum(1 for r in self.results if r['generation_success']),
                    'verification_success': sum(1 for r in self.results if r['verification'] and r['verification']['overall_success'])
                },
                'results': self.results
            }, f, indent=2)
            
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main entry point."""
    tester = WebLLMGenerationTester()
    
    # Run test suite
    tester.run_test_suite()
    
    # Analyze failures
    tester.analyze_failures()
    
    # Save results
    tester.save_results()
    
    # Return exit code based on success
    total = len(tester.results)
    success = sum(1 for r in tester.results if r['generation_success'] and r['verification'] and r['verification']['overall_success'])
    
    if success == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - success} tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 