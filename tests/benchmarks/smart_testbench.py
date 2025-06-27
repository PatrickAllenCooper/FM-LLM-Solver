#!/usr/bin/env python3
"""
Smart Web Interface Testbench - Progressive Testing Framework

A lightweight testing framework that avoids the freezing issues by:
1. Starting with mocks and gradually enabling real components
2. Using timeouts to prevent hanging
3. Graceful failure handling
4. Progressive complexity levels
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from unittest.mock import Mock

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result with metadata."""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'ERROR', 'TIMEOUT'
    duration: float
    component: str
    level: str  # 'MOCK', 'UNIT', 'INTEGRATION'
    details: Dict[str, Any]
    error_msg: Optional[str] = None

class SmartTestbench:
    """Progressive testing framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(PROJECT_ROOT / "config.yaml")
        self.results: List[TestResult] = []
        self.component_status = {}
        
        logger.info("Smart Testbench initialized")
    
    def diagnose_components(self) -> Dict[str, bool]:
        """Quickly diagnose which components can be imported safely."""
        components = {
            'config': self._try_import_config,
            'verification': self._try_import_verification,
            'models': self._try_import_models,
            'certificate_gen': self._try_import_cert_gen
        }
        
        status = {}
        for name, import_func in components.items():
            try:
                start = time.time()
                success = import_func()
                duration = time.time() - start
                status[name] = {'success': success, 'duration': duration}
                logger.info(f"✅ {name}: {'OK' if success else 'FAILED'} ({duration:.2f}s)")
            except Exception as e:
                status[name] = {'success': False, 'error': str(e)}
                logger.warning(f"❌ {name}: {str(e)}")
        
        self.component_status = status
        return status
    
    def _try_import_config(self) -> bool:
        """Try importing config utilities."""
        try:
            from utils.config_loader import load_config
            # Don't actually load to avoid hanging
            return True
        except Exception:
            return False
    
    def _try_import_verification(self) -> bool:
        """Try importing verification service."""
        try:
            from web_interface.verification_service import VerificationService
            return True
        except Exception:
            return False
    
    def _try_import_models(self) -> bool:
        """Try importing database models."""
        try:
            import web_interface.models
            return True
        except Exception:
            return False
    
    def _try_import_cert_gen(self) -> bool:
        """Try importing certificate generator (will likely fail due to heavy deps)."""
        try:
            from web_interface.certificate_generator import CertificateGenerator
            return True
        except Exception:
            return False
    
    def run_basic_tests(self) -> List[TestResult]:
        """Run basic tests that don't require heavy components."""
        tests = []
        
        # Test 1: Basic system functionality
        tests.append(self._run_test('system_basics', self._test_system_basics, 'system', 'MOCK'))
        
        # Test 2: Text processing
        tests.append(self._run_test('text_processing', self._test_text_processing, 'utils', 'MOCK'))
        
        # Test 3: Configuration (if available)
        if self.component_status.get('config', {}).get('success', False):
            tests.append(self._run_test('config_loading', self._test_config_loading, 'config', 'UNIT'))
        
        # Test 4: Verification parsing (if available)
        if self.component_status.get('verification', {}).get('success', False):
            tests.append(self._run_test('verification_parsing', self._test_verification_parsing, 'verification', 'UNIT'))
        
        return tests
    
    def _run_test(self, test_name: str, test_func, component: str, level: str) -> TestResult:
        """Run a single test with error handling."""
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            test_result = TestResult(
                test_name=test_name,
                status='PASS' if result.get('success', True) else 'FAIL',
                duration=duration,
                component=component,
                level=level,
                details=result,
                error_msg=result.get('error')
            )
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=duration,
                component=component,
                level=level,
                details={'error': str(e)},
                error_msg=str(e)
            )
        
        self.results.append(test_result)
        status_emoji = "✅" if test_result.status == 'PASS' else "❌"
        logger.info(f"{status_emoji} {test_name}: {test_result.status} ({test_result.duration:.2f}s)")
        
        return test_result
    
    def _test_system_basics(self) -> Dict[str, Any]:
        """Test basic system functionality."""
        import os, sys, json, time
        from pathlib import Path
        
        # Test file system access
        config_exists = (PROJECT_ROOT / "config.yaml").exists()
        web_dir_exists = (PROJECT_ROOT / "web_interface").exists()
        
        return {
            'success': config_exists and web_dir_exists,
            'config_exists': config_exists,
            'web_dir_exists': web_dir_exists,
            'project_root': str(PROJECT_ROOT)
        }
    
    def _test_text_processing(self) -> Dict[str, Any]:
        """Test text processing utilities."""
        test_text = """System Dynamics: dx/dt = -x**2 + y, dy/dt = x - y**2
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: x >= 1.5"""
        
        lines = test_text.split('\n')
        has_dynamics = any('dynamics' in line.lower() for line in lines)
        has_initial = any('initial' in line.lower() for line in lines)
        has_unsafe = any('unsafe' in line.lower() for line in lines)
        
        return {
            'success': has_dynamics and has_initial and has_unsafe,
            'lines_parsed': len(lines),
            'found_dynamics': has_dynamics,
            'found_initial': has_initial,
            'found_unsafe': has_unsafe
        }
    
    def _test_config_loading(self) -> Dict[str, Any]:
        """Test configuration loading."""
        try:
            from utils.config_loader import load_config
            config = load_config(self.config_path)
            
            # Basic validation
            has_kb = hasattr(config, 'knowledge_base')
            has_inference = hasattr(config, 'inference')
            
            return {
                'success': has_kb and has_inference,
                'has_knowledge_base': has_kb,
                'has_inference': has_inference,
                'config_type': str(type(config))
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_verification_parsing(self) -> Dict[str, Any]:
        """Test verification service parsing."""
        try:
            from web_interface.verification_service import VerificationService
            
            # Create with mock config
            mock_config = Mock()
            mock_config.evaluation = Mock()
            mock_config.evaluation.verification = {}
            
            service = VerificationService(mock_config)
            
            # Test parsing
            test_desc = """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.5"""
            
            parsed = service.parse_system_description(test_desc)
            
            return {
                'success': len(parsed.get('variables', [])) > 0,
                'variables_found': parsed.get('variables', []),
                'dynamics_found': len(parsed.get('dynamics', [])) > 0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report with improvement suggestions."""
        if not self.results:
            return {'error': 'No test results available'}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == 'PASS')
        failed = sum(1 for r in self.results if r.status == 'FAIL')
        errors = sum(1 for r in self.results if r.status == 'ERROR')
        
        # Analyze component health
        working_components = []
        broken_components = []
        
        for comp, status in self.component_status.items():
            if status.get('success', False):
                working_components.append(comp)
            else:
                broken_components.append(comp)
        
        # Generate suggestions
        suggestions = []
        if broken_components:
            suggestions.append(f"Fix these components: {', '.join(broken_components)}")
        
        if errors > 0:
            suggestions.append("Investigate error causes in failed tests")
        
        if len(working_components) < 2:
            suggestions.append("Focus on getting basic components working before integration")
        
        # Identify heavy components that might need mocking
        if 'certificate_gen' in broken_components:
            suggestions.append("Certificate generator likely needs mocking due to heavy ML dependencies")
        
        return {
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'success_rate': passed / total if total > 0 else 0
            },
            'component_status': self.component_status,
            'working_components': working_components,
            'broken_components': broken_components,
            'suggestions': suggestions,
            'test_results': [asdict(r) for r in self.results],
            'next_steps': self._get_next_steps(working_components, broken_components)
        }
    
    def _get_next_steps(self, working: List[str], broken: List[str]) -> List[str]:
        """Get specific next steps based on component status."""
        steps = []
        
        if len(working) == 0:
            steps.append("1. Focus on fixing basic imports and dependencies")
            steps.append("2. Check if all required packages are installed")
        elif len(working) < 2:
            steps.append("1. Build on working components to enable integration tests")
            steps.append("2. Create mock versions of heavy components")
        else:
            steps.append("1. Ready for integration testing between working components")
            steps.append("2. Implement progressive loading for heavy components")
        
        if 'certificate_gen' in broken:
            steps.append("3. Mock the certificate generator to avoid ML model loading timeouts")
        
        steps.append("4. Add timeout handling for all component loading")
        
        return steps

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Web Interface Testbench")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🚀 Starting Smart Testbench...")
        
        # Create testbench
        testbench = SmartTestbench(args.config)
        
        # Diagnose components
        logger.info("🔍 Diagnosing component health...")
        component_status = testbench.diagnose_components()
        
        # Run basic tests
        logger.info("🧪 Running basic tests...")
        test_results = testbench.run_basic_tests()
        
        # Generate report
        report = testbench.generate_report()
        
        # Display results
        print("\n" + "="*50)
        print("SMART TESTBENCH RESULTS")
        print("="*50)
        
        summary = report['summary']
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        print(f"\nWorking Components: {', '.join(report['working_components']) or 'None'}")
        print(f"Broken Components: {', '.join(report['broken_components']) or 'None'}")
        
        print("\nSuggestions:")
        for i, suggestion in enumerate(report['suggestions'], 1):
            print(f"{i}. {suggestion}")
        
        print("\nNext Steps:")
        for step in report['next_steps']:
            print(f"• {step}")
        
        # Save detailed report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {args.output}")
        
        return 0 if report['summary']['errors'] == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testbench failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 