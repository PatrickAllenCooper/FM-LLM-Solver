#!/usr/bin/env python3
"""Simple integration test runner with immediate feedback."""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("🚀 Starting Advanced Integration Tests...")
    print("=" * 50)
    
    try:
        from tests.advanced_integration_tests import AdvancedIntegrationTester
        
        # Initialize tester
        print("📋 Initializing tester...")
        tester = AdvancedIntegrationTester()
        
        # Run tests
        print("🧪 Running integration tests...")
        start_time = time.time()
        results = tester.run_all_integration_tests()
        duration = time.time() - start_time
        
        # Generate report
        print("📊 Generating report...")
        report = tester.generate_report()
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 INTEGRATION TEST RESULTS")
        print("=" * 50)
        
        summary = report['summary']
        print(f"✅ Tests Passed: {summary['passed']}")
        print(f"❌ Tests Failed: {summary['failed']}")
        print(f"🚫 Tests Errored: {summary['errors']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total Duration: {duration:.2f}s")
        print(f"🎚️  Readiness Level: {report['readiness_level']}")
        
        if report.get('suggestions'):
            print("\n💡 Key Insights:")
            for i, suggestion in enumerate(report['suggestions'][:3], 1):
                print(f"   {i}. {suggestion}")
        
        # Determine overall status
        if report['readiness_level'] in ['INTEGRATION_READY', 'PRODUCTION_READY']:
            print("\n🎉 System is ready for advanced testing!")
            return 0
        else:
            print(f"\n⚠️  System needs improvement (Level: {report['readiness_level']})")
            return 1
            
    except Exception as e:
        print(f"\n❌ Integration testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 