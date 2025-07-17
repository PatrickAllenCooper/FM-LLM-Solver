#!/usr/bin/env python3
"""
Mock vs Real LLM Comparison Test

CRITICAL DEMONSTRATION: Shows why testing with real LLM outputs is essential.
Mock outputs are clean and predictable, while real LLM outputs have:
- Inconsistent formatting
- Unexpected text artifacts
- Edge cases that break extraction
- Template vs specific coefficient ambiguity
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.certificate_extraction import extract_certificate_from_llm_output, is_template_expression
from utils.certificate_extraction import clean_and_validate_expression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockVsRealComparison:
    """Compare mock vs real LLM outputs for certificate extraction."""

    def __init__(self):
        self.test_system = {
            "description": "System: dx/dt = -x, dy/dt = -y\nInitial set: x² + y² ≤ 0.25\nUnsafe set: x² + y² ≥ 4.0",
            "variables": ["x", "y"],
        }

    def get_mock_outputs(self) -> List[Dict[str, Any]]:
        """Get clean, predictable mock outputs (current test approach)."""
        return [
            {
                "name": "clean_mock_1",
                "output": """BARRIER_CERTIFICATE_START
x**2 + y**2 - 1.0
BARRIER_CERTIFICATE_END""",
                "expected_extraction": "x**2 + y**2 - 1.0",
            },
            {
                "name": "clean_mock_2",
                "output": "B(x,y) = x**2 + y**2 - 1.5",
                "expected_extraction": "x**2 + y**2 - 1.5",
            },
            {
                "name": "clean_mock_3",
                "output": "The barrier certificate is: 2*x**2 + 2*y**2 - 3.0",
                "expected_extraction": "2*x**2 + 2*y**2 - 3.0",
            },
        ]

    def get_realistic_llm_outputs(self) -> List[Dict[str, Any]]:
        """Get realistic outputs that actual LLMs produce (with all their quirks)."""
        return [
            {
                "name": "real_qwen_messy",
                "output": """For this linear system with stable eigenvalues, I'll construct a quadratic barrier certificate.

BARRIER_CERTIFICATE_START
B(x, y) = 0.8*x^2 + 1.2*y^2 - 0.75
BARRIER_CERTIFICATE_END

This certificate ensures that:
- For the initial set where x² + y² ≤ 0.25, we have B ≤ 0
- For the unsafe set where x² + y² ≥ 4.0, we have B > 0""",
                "expected_extraction": "0.8*x**2 + 1.2*y**2 - 0.75",
                "challenges": ["LaTeX notation", "Explanatory text", "Mathematical symbols"],
            },
            {
                "name": "real_chatgpt_style",
                "output": """I'll solve this step by step:

1. Identify system dynamics: dx/dt = -x, dy/dt = -y
2. Construct barrier certificate

The barrier certificate is:
```
B(x,y) = x² + y² - 1.0
```

Verification:
- Initial set: B(x,y) ≤ 0 ✓
- Unsafe set: B(x,y) > 0 ✓
- Lie derivative: ∇B·f ≤ 0 ✓""",
                "expected_extraction": "x**2 + y**2 - 1.0",
                "challenges": ["Code blocks", "Unicode symbols", "Step-by-step format"],
            },
            {
                "name": "real_claude_academic",
                "output": """To construct a barrier certificate for this system, I need to find a function B: ℝ² → ℝ such that the level set B(x,y) = 0 separates the initial and unsafe regions.

Given the system dynamics and the circular geometry of both sets, a natural choice is:

B(x,y) := x² + y² - c

where c ∈ (0.25, 4.0). Let me choose c = 1.2 for a good margin:

**Proposed Certificate**: B(x,y) = x² + y² - 1.2

This satisfies all barrier conditions.""",
                "expected_extraction": "x**2 + y**2 - 1.2",
                "challenges": ["Mathematical notation", "Academic writing", "Variable definitions"],
            },
            {
                "name": "real_llm_with_templates",
                "output": """I'll construct a general quadratic barrier certificate:

BARRIER_CERTIFICATE_START
B(x,y) = ax² + by² + cxy + dx + ey + f
BARRIER_CERTIFICATE_END

For this specific system, suitable values would be:
a = 1, b = 1, c = 0, d = 0, e = 0, f = -1

So: B(x,y) = x² + y² - 1""",
                "expected_extraction": None,  # Should be rejected as template
                "challenges": [
                    "Template with variables",
                    "Two-step process",
                    "Parameter specification",
                ],
            },
            {
                "name": "real_llm_incomplete",
                "output": """For this stable linear system, the certificate should be quadratic. Based on the initial set radius of 0.5 and unsafe set radius of 2.0, I propose:

B(x,y) = x² + y² - β

where β should be chosen appropriately. A good choice might be β = 0.8, giving us

B(x,y) = x² + y² - 0.""",
                "expected_extraction": None,  # Incomplete/corrupted
                "challenges": ["Incomplete output", "Cut-off text", "Parameter placeholders"],
            },
            {
                "name": "real_llm_multiple_candidates",
                "output": """Here are several barrier certificate options:

Option 1: B₁(x,y) = x² + y² - 0.5 (too conservative)
Option 2: B₂(x,y) = x² + y² - 1.5 (good choice)
Option 3: B₃(x,y) = 2x² + 2y² - 3.0 (equivalent to Option 2)

I recommend **Option 2**: B(x,y) = x² + y² - 1.5""",
                "expected_extraction": "x**2 + y**2 - 1.5",
                "challenges": ["Multiple options", "Subscripts", "Recommendation logic"],
            },
            {
                "name": "real_llm_with_errors",
                "output": """Let me analyze this system:

The eigenvalues are λ₁ = -1, λ₂ = -1, so the system is stable.

For the barrier certificate, I'll use:
BARRIER_CERTIFICATE_START
B(x, y) = x**2 + y**2 - 1.0, but this could fail if x becomes negative. Therefore, we'll opt
BARRIER_CERTIFICATE_END""",
                "expected_extraction": None,  # Corrupted/incomplete
                "challenges": ["Text corruption", "Incomplete thoughts", "Mid-sentence cutoffs"],
            },
        ]

    def test_extraction_robustness(self, outputs: List[Dict], test_type: str) -> Dict:
        """Test extraction robustness on a set of outputs."""
        logger.info(f"\n🧪 Testing {test_type} outputs...")

        results = []
        for output_data in outputs:
            name = output_data["name"]
            output = output_data["output"]
            expected = output_data.get("expected_extraction")

            logger.info(f"\n📝 Testing: {name}")
            logger.info(f"   Output preview: '{output[:100]}...'")

            # Test extraction
            try:
                extracted_result = extract_certificate_from_llm_output(
                    output, self.test_system["variables"]
                )
                extracted = (
                    extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
                )
                extraction_success = extracted is not None

                logger.info(f"   Extracted: '{extracted}'")

            except Exception as e:
                logger.error(f"   Extraction failed: {e}")
                extracted = None
                extraction_success = False

            # Test template detection
            is_template = is_template_expression(extracted) if extracted else True
            logger.info(f"   Is template: {is_template}")

            # Test cleaning
            cleaned = None
            if extracted and not is_template:
                try:
                    cleaned = clean_and_validate_expression(
                        extracted, self.test_system["variables"]
                    )
                    logger.info(f"   Cleaned: '{cleaned}'")
                except Exception as e:
                    logger.warning(f"   Cleaning failed: {e}")

            # Evaluate result
            if expected is None:
                # Should be rejected
                success = not extraction_success or is_template or cleaned is None
                logger.info(f"   Expected rejection: {'✅ PASS' if success else '❌ FAIL'}")
            else:
                # Should be extracted successfully
                success = extraction_success and not is_template and cleaned is not None
                logger.info(f"   Expected success: {'✅ PASS' if success else '❌ FAIL'}")

            results.append(
                {
                    "name": name,
                    "output_length": len(output),
                    "extracted": extracted,
                    "extraction_success": extraction_success,
                    "is_template": is_template,
                    "cleaned": cleaned,
                    "expected": expected,
                    "test_passed": success,
                    "challenges": output_data.get("challenges", []),
                }
            )

        # Calculate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["test_passed"])
        extraction_rate = sum(1 for r in results if r["extraction_success"]) / total_tests
        template_rejection_rate = sum(1 for r in results if r["is_template"]) / total_tests

        summary = {
            "test_type": test_type,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests,
            "extraction_rate": extraction_rate,
            "template_rejection_rate": template_rejection_rate,
            "results": results,
        }

        logger.info(f"\n📊 {test_type} Summary:")
        logger.info(f"   Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"   Extraction Rate: {extraction_rate:.1%}")
        logger.info(f"   Template Rejection Rate: {template_rejection_rate:.1%}")

        return summary

    def run_comparison(self) -> Dict:
        """Run full comparison between mock and real outputs."""
        print("🔬 MOCK vs REAL LLM OUTPUT COMPARISON")
        print("=" * 60)
        print("This test demonstrates why real LLM testing is CRITICAL!")
        print()

        # Test mock outputs (current approach)
        mock_results = self.test_extraction_robustness(self.get_mock_outputs(), "MOCK")

        # Test realistic LLM outputs
        real_results = self.test_extraction_robustness(self.get_realistic_llm_outputs(), "REAL LLM")

        # Compare results
        print("\n" + "=" * 60)
        print("🎯 COMPARISON RESULTS")
        print("=" * 60)
        print(f"Mock Success Rate:     {mock_results['success_rate']:.1%}")
        print(f"Real LLM Success Rate: {real_results['success_rate']:.1%}")
        print(
            f"Performance Gap:       {mock_results['success_rate'] - real_results['success_rate']:.1%}"
        )

        # Analysis
        print("\n🔍 CRITICAL INSIGHTS:")
        if mock_results["success_rate"] > real_results["success_rate"] + 0.2:
            print("❌ MAJOR ISSUE: Mock tests are giving false confidence!")
            print("   Real LLM outputs expose significant extraction failures.")
            print("   Your filtering logic needs improvement for production use.")
        elif mock_results["success_rate"] > real_results["success_rate"] + 0.1:
            print("⚠️  MODERATE ISSUE: Real LLMs show some extraction challenges.")
            print("   Consider improving robustness for edge cases.")
        else:
            print("✅ GOOD: Extraction logic handles real LLM quirks well.")
            print("   Your system is robust to real-world LLM outputs.")

        # Specific challenges found
        real_challenges = []
        for result in real_results["results"]:
            if not result["test_passed"] and result["challenges"]:
                real_challenges.extend(result["challenges"])

        if real_challenges:
            print("\n🚨 Real LLM challenges found:")
            for challenge in set(real_challenges):
                print(f"   - {challenge}")

        return {
            "mock_results": mock_results,
            "real_results": real_results,
            "performance_gap": mock_results["success_rate"] - real_results["success_rate"],
            "real_challenges": list(set(real_challenges)),
        }


def main():
    """Run the comparison."""
    comparison = MockVsRealComparison()
    results = comparison.run_comparison()

    # Save results
    import json

    output_path = Path("test_results") / "mock_vs_real_comparison.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Return exit code based on findings
    if results["performance_gap"] > 0.2:
        print("\n❌ CRITICAL: Mock tests hiding major issues!")
        return 2
    elif results["performance_gap"] > 0.1:
        print("\n⚠️  WARNING: Some real LLM challenges found")
        return 1
    else:
        print("\n✅ SUCCESS: Robust to real LLM outputs")
        return 0


if __name__ == "__main__":
    exit(main())
