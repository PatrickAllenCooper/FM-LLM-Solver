#!/usr/bin/env python3
"""
Main Test Entry Point for FM-LLM Solver.

This script serves as the primary entry point for all testing in the project.
It automatically detects your environment (MacBook, desktop, deployed) and
runs the most appropriate test suite.

Usage Examples:
    # Auto-detect environment and run appropriate tests
    python test_runner.py
    
    # Force environment type
    python test_runner.py --environment macbook
    python test_runner.py --environment desktop
    python test_runner.py --environment deployed
    
    # Override test scope
    python test_runner.py --scope essential     # Quick essential tests
    python test_runner.py --scope comprehensive # Full test suite
    python test_runner.py --scope production    # Production-focused tests
    
    # Run specific test categories
    python test_runner.py --include unit_tests security_tests
    python test_runner.py --exclude load_tests gpu_tests
    
    # Preview what will be run
    python test_runner.py --dry-run
    
    # Legacy mode (original test runner)
    python test_runner.py --legacy --unit
    python test_runner.py --legacy --integration
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def show_environment_info():
    """Show detected environment information."""
    try:
        from fm_llm_solver.core.environment_detector import get_environment_detector
        detector = get_environment_detector()
        
        print("🔍 Environment Detection Results")
        print("=" * 50)
        print(f"📍 {detector.get_summary()}")
        print()
        
        info = detector.get_full_info()
        hardware = info["hardware"]
        capabilities = info["testing_capabilities"]
        
        print("🖥️  Hardware Details:")
        print(f"  Platform: {hardware['platform']}")
        print(f"  CPU: {hardware['cpu_cores']} cores ({hardware['cpu_cores_physical']} physical)")
        print(f"  Memory: {hardware['memory_total_gb']:.1f}GB total, {hardware['memory_available_gb']:.1f}GB available")
        
        if hardware["gpu"]["has_cuda_gpu"]:
            print(f"  GPU: {', '.join(hardware['gpu']['gpu_names'])} ({hardware['gpu']['gpu_memory_gb']:.1f}GB)")
        elif hardware["gpu"]["has_mps"]:
            print(f"  GPU: Apple Metal Performance Shaders")
        else:
            print(f"  GPU: None detected")
        
        print(f"\n🧪 Testing Capabilities:")
        print(f"  Recommended Scope: {capabilities['recommended_test_scope']}")
        print(f"  Max Parallel Jobs: {capabilities['max_parallel_jobs']}")
        print(f"  GPU Tests: {'✅' if capabilities['can_run_gpu_tests'] else '❌'}")
        print(f"  Load Tests: {'✅' if capabilities['can_run_load_tests'] else '❌'}")
        print(f"  End-to-End Tests: {'✅' if capabilities['can_run_end_to_end_tests'] else '❌'}")
        
        print(f"\n⚙️  Optimization Settings:")
        memory = capabilities['memory_constraints']
        print(f"  Max Memory per Test: {memory['max_memory_per_test_gb']:.1f}GB")
        print(f"  Memory Monitoring: {'Enabled' if memory['enable_memory_monitoring'] else 'Disabled'}")
        print(f"  Timeout Multiplier: {capabilities['timeout_multiplier']:.1f}x")
        
    except ImportError:
        print("❌ Environment detection not available")
        print("   Adaptive testing features require the environment detector module")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FM-LLM Solver Test Runner - Adaptive testing for all environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Types:
  macbook     Local development on MacBook (conservative, essential tests)
  desktop     Local development on high-powered desktop (comprehensive tests)
  deployed    Production/staging environment (production-focused tests)

Test Scopes:
  essential      Core functionality tests (fast, good for development)
  comprehensive  Full test suite including performance and GPU tests
  production     Production-readiness tests (security, deployment, monitoring)

Examples:
  %(prog)s                          # Auto-detect and run appropriate tests
  %(prog)s --environment macbook    # Force MacBook mode (fast, essential)
  %(prog)s --scope comprehensive    # Run full test suite
  %(prog)s --dry-run                # Show what would be run
  %(prog)s --info                   # Show environment detection results
        """
    )
    
    # Main options
    parser.add_argument("--environment", choices=["macbook", "desktop", "deployed"],
                       help="Force specific environment type")
    parser.add_argument("--scope", choices=["essential", "comprehensive", "production"],
                       help="Override test scope")
    parser.add_argument("--include", nargs="+", 
                       help="Include specific test categories")
    parser.add_argument("--exclude", nargs="+",
                       help="Exclude specific test categories")
    
    # Control options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show test strategy without running tests")
    parser.add_argument("--info", action="store_true",
                       help="Show environment detection results and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    # Legacy mode options
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy test runner (non-adaptive)")
    parser.add_argument("--unit", action="store_true",
                       help="Run only unit tests (legacy mode)")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests (legacy mode)")
    parser.add_argument("--benchmarks", action="store_true",
                       help="Run only benchmarks (legacy mode)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only (legacy mode)")
    
    args = parser.parse_args()
    
    # Handle info request
    if args.info:
        show_environment_info()
        return 0
    
    # Determine which test runner to use
    if args.legacy or any([args.unit, args.integration, args.benchmarks]):
        # Use legacy test runner
        print("🔄 Using Legacy Test Runner")
        from tests.run_tests import main as legacy_main
        
        # Convert arguments for legacy runner
        legacy_args = []
        if args.unit:
            legacy_args.append("--unit")
        if args.integration:
            legacy_args.append("--integration")
        if args.benchmarks:
            legacy_args.append("--benchmarks")
        if args.quick:
            legacy_args.append("--quick")
        if args.verbose:
            legacy_args.append("--verbose")
        
        # Override sys.argv for legacy runner
        original_argv = sys.argv[:]
        sys.argv = ["run_tests.py"] + legacy_args + ["--no-adaptive"]
        
        try:
            return legacy_main()
        finally:
            sys.argv = original_argv
    
    else:
        # Use adaptive test runner
        try:
            from tests.adaptive_test_runner import main as adaptive_main
            
            # Convert arguments for adaptive runner
            adaptive_args = []
            if args.environment:
                adaptive_args.extend(["--environment", args.environment])
            if args.scope:
                adaptive_args.extend(["--scope", args.scope])
            if args.include:
                adaptive_args.extend(["--include"] + args.include)
            if args.exclude:
                adaptive_args.extend(["--exclude"] + args.exclude)
            if args.dry_run:
                adaptive_args.append("--dry-run")
            if args.verbose:
                adaptive_args.append("--verbose")
            
            # Override sys.argv for adaptive runner
            original_argv = sys.argv[:]
            sys.argv = ["adaptive_test_runner.py"] + adaptive_args
            
            try:
                return adaptive_main()
            finally:
                sys.argv = original_argv
                
        except ImportError:
            print("❌ Adaptive test runner not available")
            print("🔄 Falling back to legacy test runner...")
            
            from tests.run_tests import main as legacy_main
            original_argv = sys.argv[:]
            sys.argv = ["run_tests.py", "--no-adaptive"]
            if args.verbose:
                sys.argv.append("--verbose")
            
            try:
                return legacy_main()
            finally:
                sys.argv = original_argv

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1) 