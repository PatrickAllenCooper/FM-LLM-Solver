@echo off
REM Phase 2 Test Runner Batch Script
REM ================================

echo.
echo Phase 2 Comprehensive Testing Suite
echo ===================================
echo.

REM Check if we're in the right directory
if not exist "tests\phase2" (
    echo Error: Phase 2 test directory not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo Running Phase 2 tests...
echo.

REM Run Phase 2 tests with comprehensive reporting
python tests\phase2\run_phase2_tests.py --all --output phase2_test_results.json

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Phase 2 tests completed successfully!
    echo.
    echo Test Summary:
    echo - Unit Tests: Comprehensive validation strategy tests
    echo - Integration Tests: Phase 1 compatibility verification
    echo - Performance Tests: Strategy and orchestrator benchmarks
    echo - Compatibility Tests: Backward compatibility validation
    echo.
    echo Results saved to: phase2_test_results.json
    echo.
    echo Phase 2 testing suite includes:
    echo - 4 validation strategies (Sampling, Symbolic, Interval, SMT)
    echo - Intelligent orchestrator with parallel execution
    echo - Performance monitoring and optimization
    echo - Seamless integration with Phase 1 components
    echo - Comprehensive error handling and fallback mechanisms
    echo.
    echo Ready for Phase 2 development!
) else (
    echo.
    echo ❌ Phase 2 tests failed!
    echo.
    echo Please check the test output above for details.
    echo Common issues:
    echo - Missing dependencies (install required packages)
    echo - Configuration issues (check test_config.yaml)
    echo - Import errors (ensure Phase 2 modules are available)
    echo.
)

echo.
echo Phase 2 Test Files Created:
echo - tests\phase2\conftest.py (Test configuration and fixtures)
echo - tests\phase2\test_validation_strategies_comprehensive.py (Strategy tests)
echo - tests\phase2\test_validation_orchestrator_comprehensive.py (Orchestrator tests)
echo - tests\phase2\test_phase2_integration.py (Integration tests)
echo - tests\phase2\run_phase2_tests.py (Test runner)
echo - tests\phase2\test_config.yaml (Test configuration)
echo - tests\phase2\README.md (Documentation)
echo.

echo Integration with main test runner:
echo - Updated tests\run_tests.py to include Phase 2 tests
echo - Use: python tests\run_tests.py --phase2
echo - Use: python tests\run_tests.py --all (includes Phase 2)
echo.

pause 