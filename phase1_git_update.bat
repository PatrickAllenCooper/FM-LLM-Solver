@echo off
echo Phase 1 Git Update Script
echo ========================

echo Adding Phase 1 files to git...

REM Add all new Phase 1 files
git add utils/level_set_tracker.py
git add utils/set_membership.py
git add utils/adaptive_tolerance.py
git add utils/certificate_extraction_improved.py

git add evaluation/verify_certificate_unsafe_fix.py

git add tests/ground_truth/barrier_certificates.json
git add tests/unit/barrier_theory_fix.py
git add tests/unit/test_extraction_edge_cases.py
git add tests/integration/test_validation_pipeline.py
git add tests/test_theory_compliance.py
git add tests/test_harness.py
git add tests/run_phase1_tests.py
git add tests/report_generator.py
git add tests/metrics.py
git add tests/benchmarks/profiler.py
git add tests/benchmarks/optimization_targets.py

git add docs/PHASE1_DOCUMENTATION.md
git add PHASE1_COMPLETION_SUMMARY.md
git add PHASE1_DAY5_SUMMARY.md

REM Add updated files
git add web_interface/verification_service.py
git add tests/unit/test_certificate_validation_accuracy.py

echo.
echo Files added. Creating commit...

git commit -m "Phase 1: Complete barrier certificate validation overhaul

Major improvements:
- Fixed mathematical theory (correct unsafe set checking)
- Implemented proper level set computation (c1 < c2 separation)
- Enhanced certificate extraction with format support
- Added 22 ground truth test cases
- Created comprehensive test infrastructure
- Added performance profiling and metrics
- Complete documentation and migration guide

Key fixes:
- Unsafe set: Now checks B(x) >= c2 for points INSIDE unsafe set
- Level sets: c1 = max(B) on initial, c2 = min(B) on unsafe
- Extraction: Handles decimals, scientific notation, LaTeX/Unicode
- Testing: Automated harness, HTML reports, metrics calculation

All 20 Phase 1 tasks completed successfully."

echo.
echo Commit created. You can now push to remote with:
echo git push origin main

pause 