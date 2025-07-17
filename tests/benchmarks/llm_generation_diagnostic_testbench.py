#!/usr/bin/env python3
"""
LLM Generation Diagnostic Testbench
===================================

This testbench diagnoses and fixes the specific LLM generation issues:
- Incomplete responses ("Therefore, we'll opt")
- Placeholder variables (α, β, C)
- Poor certificate extraction
- Overly complex prompts

Run with: python tests/llm_generation_diagnostic_testbench.py
"""

import json
import logging
import os
import sys
import threading
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.generate_certificate import format_prompt_with_context
from utils.config_loader import load_config
from web_interface.certificate_generator import CertificateGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMGenerationDiagnosticTestbench:
    """Diagnostic testbench for LLM generation issues"""

    def __init__(self):
        self.config = load_config()
        self.certificate_generator = None
        self.generation_in_progress = False

        logger.info("🔧 Initializing Certificate Generator...")
        try:
            self.certificate_generator = CertificateGenerator(self.config)
            logger.info("✅ Certificate generator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize certificate generator: {e}")

    def diagnose_current_system(self):
        """Diagnose the current system with the user's problematic case"""
        logger.info("🔍 DIAGNOSING CURRENT SYSTEM")
        logger.info("=" * 50)

        # Test case from user
        system_desc = "System Dynamics: dx/dt = -x, dy/dt = -y\nInitial Set: x**2 + y**2 <= 0.25\nUnsafe Set: x**2 + y**2 >= 4.0"
        domain_bounds = {"x": [-3, 3], "y": [-3, 3]}

        logger.info(f"📋 Test case: {system_desc}")
        logger.info(f"🎯 Domain bounds: {domain_bounds}")

        if self.certificate_generator:
            try:
                logger.info("🚀 Starting LLM generation...")
                logger.info("⏳ This may take 30-60 seconds with the 14B model...")

                # Show progress during generation
                self.generation_in_progress = True
                progress_thread = threading.Thread(target=self._show_progress)
                progress_thread.daemon = True
                progress_thread.start()

                start_time = time.time()
                result = self.certificate_generator.generate_certificate(
                    system_desc, "base", rag_k=0, domain_bounds=domain_bounds
                )

                self.generation_in_progress = False
                generation_time = time.time() - start_time

                logger.info(f"✅ Generation completed in {generation_time:.1f} seconds")
                self._analyze_result(result)
                return result

            except Exception as e:
                self.generation_in_progress = False
                logger.error(f"❌ Generation failed: {e}")
                return None
        else:
            logger.warning("⚠️  No certificate generator - using mock analysis")
            # Mock the problematic output from user
            mock_result = {
                "success": True,
                "llm_output": "B(x, y) = x + αy - C, but this could fail if x becomes negative. Therefore, we'll opt",
                "certificate": None,
                "prompt_length": 3500,
            }
            self._analyze_result(mock_result)
            return mock_result

    def _show_progress(self):
        """Show progress indicators during generation"""
        dots = 0
        while self.generation_in_progress:
            time.sleep(2)
            if self.generation_in_progress:
                dots = (dots + 1) % 4
                progress_str = "🔄 Generating" + "." * dots + " " * (3 - dots)
                print(f"\r{progress_str}", end="", flush=True)
        print()  # New line when done

    def _analyze_result(self, result):
        """Analyze the generation result for issues"""
        logger.info("\n🔍 DETAILED ANALYSIS RESULTS:")
        logger.info("-" * 40)

        llm_output = result.get("llm_output", "")
        certificate = result.get("certificate", "")
        prompt_length = result.get("prompt_length", 0)

        logger.info(f"📝 LLM Output Length: {len(llm_output)} characters")
        logger.info(
            f"📝 LLM Output Preview: {llm_output[:200]}{'...' if len(llm_output) > 200 else ''}"
        )
        logger.info(f"🔧 Extracted Certificate: {certificate}")
        logger.info(f"📏 Prompt Length: {prompt_length} characters")

        # Issue Analysis
        issues_found = []

        # Issue 1: Placeholder variables
        if self._has_placeholders(llm_output):
            issues_found.append("PLACEHOLDERS")
            logger.warning("⚠️  ISSUE 1: Contains placeholder variables (α, β, C)")
            placeholders = self._find_placeholders(llm_output)
            logger.warning(f"   Found placeholders: {placeholders}")

        # Issue 2: Incomplete generation
        if self._is_incomplete(llm_output):
            issues_found.append("INCOMPLETE")
            logger.warning("⚠️  ISSUE 2: Generation is incomplete")
            incomplete_indicators = self._find_incomplete_indicators(llm_output)
            logger.warning(f"   Incomplete indicators: {incomplete_indicators}")

        # Issue 3: Extraction failure
        if not certificate:
            issues_found.append("EXTRACTION_FAILED")
            logger.warning("⚠️  ISSUE 3: Certificate extraction failed")

        # Issue 4: Prompt too long
        if prompt_length > 2500:
            issues_found.append("PROMPT_TOO_LONG")
            logger.warning(f"⚠️  ISSUE 4: Prompt is too long ({prompt_length} chars)")

        if not issues_found:
            logger.info("✅ No major issues detected!")
        else:
            logger.warning(f"⚠️  Issues found: {', '.join(issues_found)}")

        return issues_found

    def _has_placeholders(self, text):
        """Check for placeholder variables"""
        placeholders = ["α", "β", "γ", "C", "\\alpha", "\\beta"]
        return any(p in text for p in placeholders)

    def _find_placeholders(self, text):
        """Find specific placeholders in text"""
        placeholders = ["α", "β", "γ", "C", "\\alpha", "\\beta"]
        found = [p for p in placeholders if p in text]
        return found

    def _is_incomplete(self, text):
        """Check for incomplete generation"""
        incomplete_patterns = [
            "Therefore, we'll opt",
            "but this could fail",
            "we need to consider",
            "However, we",
        ]
        return any(p in text for p in incomplete_patterns)

    def _find_incomplete_indicators(self, text):
        """Find specific incomplete indicators"""
        incomplete_patterns = [
            "Therefore, we'll opt",
            "but this could fail",
            "we need to consider",
            "However, we",
        ]
        found = [p for p in incomplete_patterns if p in text]
        return found

    def test_improved_prompts(self):
        """Test improved prompting strategies"""
        logger.info("\n🧪 TESTING IMPROVED PROMPTS")
        logger.info("=" * 50)

        system_desc = "System Dynamics: dx/dt = -x, dy/dt = -y\nInitial Set: x**2 + y**2 <= 0.25\nUnsafe Set: x**2 + y**2 >= 4.0"
        domain_bounds = {"x": [-3, 3], "y": [-3, 3]}

        # Current prompt (problematic)
        logger.info("📊 Analyzing current prompt structure...")
        current_prompt = format_prompt_with_context(
            system_desc, "", "unified", domain_bounds
        )
        logger.info(f"📏 Current prompt length: {len(current_prompt)} chars")

        # Count specific elements in current prompt
        instruction_count = current_prompt.count("IMPORTANT:")
        example_count = current_prompt.count("Example")
        warning_count = current_prompt.count("⚠️")

        logger.info("📊 Current prompt analysis:")
        logger.info(f"   - Instructions: {instruction_count}")
        logger.info(f"   - Examples: {example_count}")
        logger.info(f"   - Warnings: {warning_count}")

        # Generate improved prompts
        logger.info("\n🔧 Generating improved prompt alternatives...")
        improved_prompts = {
            "minimal": self._create_minimal_prompt(system_desc, domain_bounds),
            "direct": self._create_direct_prompt(system_desc, domain_bounds),
            "constrained": self._create_constrained_prompt(system_desc, domain_bounds),
        }

        for name, prompt in improved_prompts.items():
            logger.info(
                f"📏 {name.capitalize()} prompt: {len(prompt)} chars ({len(current_prompt) - len(prompt):+d} vs current)"
            )
            logger.info(f"   Preview: {prompt[:100]}...")

        return improved_prompts

    def _create_minimal_prompt(self, system_desc, domain_bounds):
        """Minimal prompt - just the essentials"""
        domain_text = ""
        if domain_bounds:
            domain_desc = ", ".join(
                [
                    f"{var} ∈ [{bounds[0]}, {bounds[1]}]"
                    for var, bounds in domain_bounds.items()
                ]
            )
            domain_text = f"Domain: {domain_desc}\n"

        return """<s>[INST] Generate a barrier certificate:

{system_desc}
{domain_text}
Use concrete numbers only (no α, β, C). Format: B(x, y) = <expression>

BARRIER_CERTIFICATE_START
B(x, y) = [/INST]"""

    def _create_direct_prompt(self, system_desc, domain_bounds):
        """Direct prompt with clear constraints"""
        domain_text = ""
        if domain_bounds:
            domain_desc = ", ".join(
                [
                    f"{var} ∈ [{bounds[0]}, {bounds[1]}]"
                    for var, bounds in domain_bounds.items()
                ]
            )
            domain_text = f"Domain: {domain_desc}\n"

        return """<s>[INST] Generate a barrier certificate for:

{system_desc}
{domain_text}
CRITICAL: Use only real numbers (1.0, -2.5, 3.14). No variables like α, β, C.
Complete the expression fully.

Examples:
- B(x, y) = x**2 + y**2 - 1.0
- B(x, y) = 2*x**2 + 0.5*y**2 - 3.0

BARRIER_CERTIFICATE_START
B(x, y) = [/INST]"""

    def _create_constrained_prompt(self, system_desc, domain_bounds):
        """Constrained prompt that forces completion"""
        domain_text = ""
        if domain_bounds:
            domain_desc = ", ".join(
                [
                    f"{var} ∈ [{bounds[0]}, {bounds[1]}]"
                    for var, bounds in domain_bounds.items()
                ]
            )
            domain_text = f"Domain: {domain_desc}\n"

        return """<s>[INST] Complete the barrier certificate:

{system_desc}
{domain_text}
BARRIER_CERTIFICATE_START
B(x, y) = x**2 + y**2 - [/INST]"""

    def suggest_fixes(self):
        """Suggest specific fixes for the identified issues"""
        logger.info("\n💡 SPECIFIC FIXES TO IMPLEMENT")
        logger.info("=" * 50)

        logger.info("1. 📝 PROMPT OPTIMIZATION:")
        logger.info("   ✓ Replace current 3500+ char prompt with 500-800 char version")
        logger.info("   ✓ Remove verbose discrete-time warnings and examples")
        logger.info("   ✓ Focus on essential constraint: 'Use concrete numbers only'")

        logger.info("\n2. 🔧 EXTRACTION IMPROVEMENTS:")
        logger.info("   ✓ Add fallback regex for incomplete expressions")
        logger.info("   ✓ Handle partial mathematical expressions")
        logger.info("   ✓ Detect and reject placeholder variables")

        logger.info("\n3. ⚙️  GENERATION PARAMETER TUNING:")
        logger.info("   ✓ Increase max_new_tokens from current to 200+")
        logger.info("   ✓ Lower temperature to 0.3 for more focused output")
        logger.info("   ✓ Add stop tokens: ['Therefore', 'However', 'But']")

        logger.info("\n4. 🎯 IMMEDIATE IMPLEMENTATION:")
        logger.info("   ✓ Update format_prompt_with_context() function")
        logger.info("   ✓ Enhance extract_certificate_from_output() method")
        logger.info("   ✓ Add placeholder detection and rejection")


def main():
    """Main execution with verbose progress reporting"""
    logger.info("🎯 LLM Generation Diagnostic Testbench")
    logger.info("=====================================")
    logger.info("⏳ This will take 2-3 minutes with the 14B model")
    logger.info("🔍 Analyzing current system behavior...")

    testbench = LLMGenerationDiagnosticTestbench()

    # Run diagnostics with progress reporting
    logger.info("\n🔬 Phase 1: Current System Diagnosis")
    result = testbench.diagnose_current_system()

    logger.info("\n🔬 Phase 2: Prompt Analysis")
    improved_prompts = testbench.test_improved_prompts()

    logger.info("\n🔬 Phase 3: Fix Recommendations")
    testbench.suggest_fixes()

    # Save results
    logger.info("\n💾 Saving diagnostic results...")
    results = {
        "current_result": result,
        "improved_prompts": {k: v for k, v in improved_prompts.items()},
        "timestamp": time.time(),
    }

    with open("llm_generation_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "✅ Diagnosis complete! Results saved to llm_generation_diagnostics.json"
    )
    logger.info("🚀 Ready to implement fixes to the certificate generation system")
    logger.info("\n📊 SUMMARY:")
    if result:
        logger.info(f"   - Generation successful: {result.get('success', False)}")
        logger.info(f"   - Certificate extracted: {bool(result.get('certificate'))}")
        logger.info(f"   - Output length: {len(result.get('llm_output', ''))}")
    logger.info("   - Improved prompts generated: 3 variants")
    logger.info("   - Fix recommendations: 4 categories")


if __name__ == "__main__":
    main()
