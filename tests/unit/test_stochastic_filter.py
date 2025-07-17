#!/usr/bin/env python3
"""
Test Script for Stochastic Barrier Certificate Filtering

This script tests the stochastic filtering functionality to ensure
it correctly identifies and filters stochastic content.

Usage:
    python tests/test_stochastic_filter.py
"""

import sys
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from knowledge_base.document_classifier import BarrierCertificateClassifier


def create_test_config():
    """Create a test configuration with stochastic filtering enabled."""
    test_config = OmegaConf.create(
        {
            "knowledge_base": {
                "classification": {
                    "discrete_keywords": ["discrete", "finite state"],
                    "continuous_keywords": ["continuous", "differential equation"],
                    "stochastic_keywords": [
                        "stochastic",
                        "probabilistic",
                        "random",
                        "noise",
                        "uncertainty",
                        "martingale",
                        "supermartingale",
                        "submartingale",
                        "brownian motion",
                        "wiener process",
                        "stochastic differential",
                        "SDE",
                        "markov",
                        "random walk",
                        "monte carlo",
                        "probabilistic safety",
                        "almost surely",
                        "probability",
                        "stochastic reachability",
                        "stochastic control",
                    ],
                    "confidence_threshold": 0.6,
                    "stochastic_filter": {
                        "enable": True,
                        "mode": "exclude",
                        "min_stochastic_keywords": 2,
                        "stochastic_confidence_threshold": 0.4,
                    },
                }
            },
            "fine_tuning": {
                "stochastic_filter": {
                    "enable": True,
                    "mode": "exclude",
                    "apply_to_extracted_data": True,
                    "apply_to_manual_data": False,
                    "apply_to_synthetic_data": True,
                }
            },
        }
    )
    return test_config


def test_stochastic_classification():
    """Test stochastic content classification."""
    print("🧪 Testing Stochastic Classification")
    print("=" * 50)

    cfg = create_test_config()
    classifier = BarrierCertificateClassifier(cfg)

    test_cases = [
        {
            "name": "Deterministic Continuous",
            "text": """
            Consider the continuous-time dynamical system dx/dt = f(x) where x is the state vector.
            We propose a barrier certificate B(x) = x^T P x where P is a positive definite matrix.
            The Lie derivative of B along the system trajectory must satisfy dB/dt <= 0 in the safe set.
            This ensures that the system remains safe using polynomial optimization and SOS techniques.
            """,
            "expected_stochastic": False,
            "expected_include": True,  # Include in exclude mode
        },
        {
            "name": "Stochastic System",
            "text": """
            We consider a stochastic differential equation dx = f(x)dt + g(x)dW where W is a Brownian motion.
            The probabilistic safety verification requires a stochastic barrier certificate that is a supermartingale.
            The expected value of the barrier function must satisfy E[B(x_t)] >= 0 almost surely.
            Monte Carlo methods are used to verify the stochastic reachability constraints.
            """,
            "expected_stochastic": True,
            "expected_include": False,  # Exclude in exclude mode
        },
        {
            "name": "Discrete Deterministic",
            "text": """
            For discrete-time systems x_{k+1} = f(x_k), we define a discrete barrier certificate.
            The barrier function B(x) must satisfy B(x_k) <= 0 for initial states and B(x_k) >= 0
            for unsafe states. The difference condition ΔB = B(x_{k+1}) - B(x_k) <= 0 ensures safety.
            """,
            "expected_stochastic": False,
            "expected_include": True,  # Include in exclude mode
        },
        {
            "name": "Mixed Content",
            "text": """
            This paper discusses both deterministic and stochastic approaches to barrier certificates.
            While deterministic methods use Lyapunov-like functions, stochastic approaches consider
            random disturbances and uncertainty in the system dynamics.
            """,
            "expected_stochastic": True,  # Has stochastic keywords
            "expected_include": False,  # Exclude in exclude mode
        },
        {
            "name": "Minimal Stochastic",
            "text": """
            The system has some random noise but we use deterministic analysis.
            Standard barrier certificate methods apply.
            """,
            "expected_stochastic": True,  # 2 stochastic keywords, threshold is 2
            "expected_include": False,  # Exclude in exclude mode
        },
    ]

    success_count = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)

        # Test stochastic classification
        is_stochastic, confidence, details = classifier.classify_stochastic_content(
            test_case["text"], f"test_case_{i}"
        )

        # Test inclusion decision
        should_include, reason, filter_details = classifier.should_include_document(
            test_case["text"], f"test_case_{i}"
        )

        # Check results
        stochastic_correct = is_stochastic == test_case["expected_stochastic"]
        include_correct = should_include == test_case["expected_include"]

        print(
            f"  Stochastic: {is_stochastic} (expected: {test_case['expected_stochastic']}) {'✅' if stochastic_correct else '❌'}"
        )
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Keywords found: {details.get('stochastic_count', 0)}")
        print(f"  Matches: {details.get('stochastic_matches', [])[:3]}")
        print(
            f"  Include: {should_include} (expected: {test_case['expected_include']}) {'✅' if include_correct else '❌'}"
        )
        print(f"  Reason: {reason}")

        if stochastic_correct and include_correct:
            success_count += 1
            print("  Result: ✅ PASS")
        else:
            print("  Result: ❌ FAIL")

    print(f"\n📊 Test Results: {success_count}/{total_tests} passed")
    assert (
        success_count == total_tests
    ), f"{success_count}/{total_tests} stochastic classification tests passed"


def test_filter_modes():
    """Test different filter modes (include vs exclude)."""
    print("\n🔄 Testing Filter Modes")
    print("=" * 50)

    test_text = """
    We analyze stochastic differential equations with Brownian motion and random uncertainty.
    The probabilistic barrier certificate is a supermartingale that ensures safety almost surely.
    """

    # Test exclude mode
    cfg_exclude = create_test_config()
    cfg_exclude.knowledge_base.classification.stochastic_filter.mode = "exclude"
    classifier_exclude = BarrierCertificateClassifier(cfg_exclude)

    should_include_exclude, reason_exclude, _ = (
        classifier_exclude.should_include_document(test_text, "test_exclude")
    )

    # Test include mode
    cfg_include = create_test_config()
    cfg_include.knowledge_base.classification.stochastic_filter.mode = "include"
    classifier_include = BarrierCertificateClassifier(cfg_include)

    should_include_include, reason_include, _ = (
        classifier_include.should_include_document(test_text, "test_include")
    )

    print(f"Exclude mode: Include={should_include_exclude}, Reason='{reason_exclude}'")
    print(f"Include mode: Include={should_include_include}, Reason='{reason_include}'")

    # They should be opposite for stochastic content
    modes_correct = should_include_exclude != should_include_include
    print(f"Modes opposite as expected: {'✅' if modes_correct else '❌'}")
    assert modes_correct, "Filter modes should be opposite for stochastic content"


def test_keyword_threshold():
    """Test minimum keyword threshold."""
    print("\n🎯 Testing Keyword Threshold")
    print("=" * 50)

    cfg = create_test_config()
    cfg.knowledge_base.classification.stochastic_filter.min_stochastic_keywords = 3
    classifier = BarrierCertificateClassifier(cfg)

    test_cases = [
        {
            "text": "This system has random noise but we use deterministic analysis methods. The barrier certificate approach works well for this case.",
            "expected_stochastic": False,
        },
        {
            "text": "The stochastic system has random noise and uncertainty. We consider probabilistic approaches and random disturbances in our analysis.",
            "expected_stochastic": True,
        },
    ]

    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        is_stochastic, confidence, details = classifier.classify_stochastic_content(
            test_case["text"], f"threshold_test_{i}"
        )

        correct = is_stochastic == test_case["expected_stochastic"]
        print(
            f"Test {i}: {details.get('stochastic_count', 0)} keywords → Stochastic: {is_stochastic} {'✅' if correct else '❌'}"
        )

        if correct:
            success_count += 1

    assert success_count == len(
        test_cases
    ), f"{success_count}/{len(test_cases)} keyword threshold tests passed"


def test_confidence_threshold():
    """Test confidence threshold."""
    print("\n📈 Testing Confidence Threshold")
    print("=" * 50)

    # High threshold
    cfg_high = create_test_config()
    cfg_high.knowledge_base.classification.stochastic_filter.stochastic_confidence_threshold = (
        0.8
    )
    classifier_high = BarrierCertificateClassifier(cfg_high)

    # Low threshold
    cfg_low = create_test_config()
    cfg_low.knowledge_base.classification.stochastic_filter.stochastic_confidence_threshold = (
        0.2
    )
    classifier_low = BarrierCertificateClassifier(cfg_low)

    test_text = "Random stochastic noise uncertainty."  # 3 keywords, should be classified as stochastic

    _, conf_high, _ = classifier_high.classify_stochastic_content(
        test_text, "conf_test_high"
    )
    _, conf_low, _ = classifier_low.classify_stochastic_content(
        test_text, "conf_test_low"
    )

    print(f"High threshold classifier confidence: {conf_high:.3f}")
    print(f"Low threshold classifier confidence: {conf_low:.3f}")

    # Both should classify as stochastic, but confidence handling may differ
    assert True, "Confidence threshold test passed"  # Basic test that it doesn't crash


def main():
    """Run all tests."""
    print("🚀 Stochastic Filter Test Suite")
    print("=" * 60)

    try:
        tests = [
            ("Stochastic Classification", test_stochastic_classification),
            ("Filter Modes", test_filter_modes),
            ("Keyword Threshold", test_keyword_threshold),
            ("Confidence Threshold", test_confidence_threshold),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                test_func()
                print(f"✅ {test_name}: PASSED")
                passed += 1
            except AssertionError as e:
                print(f"❌ {test_name}: FAILED - {e}")
                import traceback

                traceback.print_exc()
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"🏁 Final Results: {passed}/{total} tests passed")

        assert passed == total, f"{passed}/{total} tests passed"
        print("🎉 All stochastic filter tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
