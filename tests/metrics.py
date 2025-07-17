"""
Metrics Calculator for Barrier Certificate Validation (Phase 1 Day 8)
Computes precision, recall, F1 scores and other validation metrics
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    # Additional metrics
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient

    # Level set metrics
    level_set_accuracy: float = 0.0
    level_set_mae: float = 0.0  # Mean Absolute Error

    # Performance metrics
    avg_validation_time: float = 0.0
    total_validation_time: float = 0.0

    # Agreement metrics
    validator_agreement_rate: float = 0.0

    def calculate_derived_metrics(self):
        """Calculate precision, recall, F1, etc. from confusion matrix"""
        tp, tn, fp, fn = (
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )

        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            self.precision = tp / (tp + fp)
        else:
            self.precision = 0.0

        # Recall (Sensitivity): TP / (TP + FN)
        if tp + fn > 0:
            self.recall = tp / (tp + fn)
        else:
            self.recall = 0.0

        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = tp + tn + fp + fn
        if total > 0:
            self.accuracy = (tp + tn) / total
        else:
            self.accuracy = 0.0

        # Specificity: TN / (TN + FP)
        if tn + fp > 0:
            self.specificity = tn / (tn + fp)
        else:
            self.specificity = 0.0

        # Balanced Accuracy: (Sensitivity + Specificity) / 2
        self.balanced_accuracy = (self.recall + self.specificity) / 2

        # Matthews Correlation Coefficient
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator > 0:
            self.mcc = numerator / denominator
        else:
            self.mcc = 0.0


class MetricsCalculator:
    """Calculate and analyze metrics for barrier certificate validation"""

    def __init__(self):
        self.results = []
        self.metrics = ValidationMetrics()

    def load_results(self, results_file: str):
        """Load test results from JSON file"""
        with open(results_file, "r") as f:
            data = json.load(f)
            self.results = data.get("results", [])

    def calculate_metrics(self) -> ValidationMetrics:
        """Calculate all metrics from loaded results"""
        if not self.results:
            return self.metrics

        # Reset metrics
        self.metrics = ValidationMetrics()

        # Count confusion matrix values
        for result in self.results:
            expected = result.get("expected_valid", None)
            predicted = result.get("new_validator_result", None)

            if expected is None or predicted is None:
                continue

            if expected and predicted:
                self.metrics.true_positives += 1
            elif expected and not predicted:
                self.metrics.false_negatives += 1
            elif not expected and predicted:
                self.metrics.false_positives += 1
            else:  # not expected and not predicted
                self.metrics.true_negatives += 1

        # Calculate derived metrics
        self.metrics.calculate_derived_metrics()

        # Calculate level set metrics
        self._calculate_level_set_metrics()

        # Calculate performance metrics
        self._calculate_performance_metrics()

        # Calculate agreement metrics
        self._calculate_agreement_metrics()

        return self.metrics

    def _calculate_level_set_metrics(self):
        """Calculate metrics related to level set computation accuracy"""
        level_set_errors = []
        correct_level_sets = 0
        total_level_sets = 0

        for result in self.results:
            expected_ls = result.get("level_sets_expected")
            computed_ls = result.get("level_sets_computed")

            if expected_ls and computed_ls:
                total_level_sets += 1

                # Check c1 and c2
                c1_expected = expected_ls.get("c1")
                c1_computed = computed_ls.get("c1")
                c2_expected = expected_ls.get("c2")
                c2_computed = computed_ls.get("c2")

                if all(x is not None for x in [c1_expected, c1_computed, c2_expected, c2_computed]):
                    c1_error = abs(c1_expected - c1_computed)
                    c2_error = abs(c2_expected - c2_computed)

                    level_set_errors.extend([c1_error, c2_error])

                    # Consider correct if both errors < 0.1
                    if c1_error < 0.1 and c2_error < 0.1:
                        correct_level_sets += 1

        if total_level_sets > 0:
            self.metrics.level_set_accuracy = correct_level_sets / total_level_sets

        if level_set_errors:
            self.metrics.level_set_mae = np.mean(level_set_errors)

    def _calculate_performance_metrics(self):
        """Calculate performance-related metrics"""
        validation_times = []

        for result in self.results:
            time = result.get("new_validator_time")
            if time is not None and time > 0:
                validation_times.append(time)

        if validation_times:
            self.metrics.avg_validation_time = np.mean(validation_times)
            self.metrics.total_validation_time = sum(validation_times)

    def _calculate_agreement_metrics(self):
        """Calculate validator agreement metrics"""
        agreements = 0
        total_comparisons = 0

        for result in self.results:
            new_result = result.get("new_validator_result")
            old_result = result.get("old_validator_result")

            if new_result is not None and old_result is not None:
                total_comparisons += 1
                if new_result == old_result:
                    agreements += 1

        if total_comparisons > 0:
            self.metrics.validator_agreement_rate = agreements / total_comparisons

    def generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix"""
        return np.array(
            [
                [self.metrics.true_negatives, self.metrics.false_positives],
                [self.metrics.false_negatives, self.metrics.true_positives],
            ]
        )

    def generate_classification_report(self) -> str:
        """Generate a detailed classification report"""
        report = []
        report.append("BARRIER CERTIFICATE VALIDATION METRICS")
        report.append("=" * 60)
        report.append("")

        # Confusion Matrix
        report.append("Confusion Matrix:")
        report.append("                 Predicted")
        report.append("                 Invalid  Valid")
        report.append(
            f"Actual Invalid     {self.metrics.true_negatives:4d}   {self.metrics.false_positives:4d}"
        )
        report.append(
            f"       Valid       {self.metrics.false_negatives:4d}   {self.metrics.true_positives:4d}"
        )
        report.append("")

        # Classification Metrics
        report.append("Classification Metrics:")
        report.append(f"  Accuracy:          {self.metrics.accuracy:.3f}")
        report.append(f"  Precision:         {self.metrics.precision:.3f}")
        report.append(f"  Recall:            {self.metrics.recall:.3f}")
        report.append(f"  F1 Score:          {self.metrics.f1_score:.3f}")
        report.append(f"  Specificity:       {self.metrics.specificity:.3f}")
        report.append(f"  Balanced Accuracy: {self.metrics.balanced_accuracy:.3f}")
        report.append(f"  MCC:               {self.metrics.mcc:.3f}")
        report.append("")

        # Level Set Metrics
        report.append("Level Set Metrics:")
        report.append(f"  Accuracy:          {self.metrics.level_set_accuracy:.3f}")
        report.append(f"  MAE:               {self.metrics.level_set_mae:.3f}")
        report.append("")

        # Performance Metrics
        report.append("Performance Metrics:")
        report.append(f"  Avg Validation Time: {self.metrics.avg_validation_time:.3f}s")
        report.append(f"  Total Time:          {self.metrics.total_validation_time:.1f}s")
        report.append("")

        # Agreement Metrics
        report.append("Validator Agreement:")
        report.append(f"  Agreement Rate:      {self.metrics.validator_agreement_rate:.3f}")

        return "\n".join(report)

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in validation failures"""
        failures = {"false_positives": [], "false_negatives": [], "patterns": {}}

        for result in self.results:
            expected = result.get("expected_valid")
            predicted = result.get("new_validator_result")

            if expected is not None and predicted is not None:
                if expected and not predicted:
                    # False negative
                    failures["false_negatives"].append(
                        {
                            "test_id": result.get("test_id"),
                            "system": result.get("system_name"),
                            "certificate": result.get("certificate"),
                            "notes": result.get("notes", ""),
                        }
                    )
                elif not expected and predicted:
                    # False positive
                    failures["false_positives"].append(
                        {
                            "test_id": result.get("test_id"),
                            "system": result.get("system_name"),
                            "certificate": result.get("certificate"),
                            "notes": result.get("notes", ""),
                        }
                    )

        # Analyze patterns
        if failures["false_negatives"]:
            failures["patterns"]["false_negative_rate"] = len(failures["false_negatives"]) / len(
                self.results
            )

        if failures["false_positives"]:
            failures["patterns"]["false_positive_rate"] = len(failures["false_positives"]) / len(
                self.results
            )

        return failures

    def export_metrics(self, output_file: str = "validation_metrics.json"):
        """Export metrics to JSON file"""
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "metrics": asdict(self.metrics),
            "confusion_matrix": self.generate_confusion_matrix().tolist(),
            "failure_analysis": self.analyze_failure_patterns(),
        }

        with open(output_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Metrics exported to: {output_file}")

    def print_summary(self):
        """Print a summary of key metrics"""
        print("\nVALIDATION METRICS SUMMARY")
        print("=" * 40)
        print(f"Total Tests: {len(self.results)}")
        print(f"Accuracy: {self.metrics.accuracy:.3f}")
        print(f"F1 Score: {self.metrics.f1_score:.3f}")
        print(f"Precision: {self.metrics.precision:.3f}")
        print(f"Recall: {self.metrics.recall:.3f}")

        if self.metrics.false_positives > 0 or self.metrics.false_negatives > 0:
            print("\nErrors:")
            print(f"  False Positives: {self.metrics.false_positives}")
            print(f"  False Negatives: {self.metrics.false_negatives}")


def calculate_metrics_from_file(results_file: str, output_file: Optional[str] = None):
    """Convenience function to calculate metrics from a results file"""
    calculator = MetricsCalculator()
    calculator.load_results(results_file)
    metrics = calculator.calculate_metrics()

    # Print report
    print(calculator.generate_classification_report())

    # Export if requested
    if output_file:
        calculator.export_metrics(output_file)

    return metrics


def compare_validator_metrics(results_files: List[str], labels: List[str]):
    """Compare metrics across multiple validator results"""
    print("\nVALIDATOR COMPARISON")
    print("=" * 60)

    all_metrics = []

    for i, (file, label) in enumerate(zip(results_files, labels)):
        calculator = MetricsCalculator()
        calculator.load_results(file)
        metrics = calculator.calculate_metrics()
        all_metrics.append((label, metrics))

        print(f"\n{label}:")
        print(f"  Accuracy: {metrics.accuracy:.3f}")
        print(f"  F1 Score: {metrics.f1_score:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  Avg Time: {metrics.avg_validation_time:.3f}s")

    # Find best performer
    best_f1 = max(all_metrics, key=lambda x: x[1].f1_score)
    best_speed = min(all_metrics, key=lambda x: x[1].avg_validation_time)

    print(f"\nBest F1 Score: {best_f1[0]} ({best_f1[1].f1_score:.3f})")
    print(f"Fastest: {best_speed[0]} ({best_speed[1].avg_validation_time:.3f}s)")


def main():
    """Main entry point for metrics calculation"""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate validation metrics")
    parser.add_argument("results_file", help="JSON file containing test results")
    parser.add_argument("-o", "--output", help="Output file for metrics export")
    parser.add_argument("--compare", nargs="+", help="Compare multiple result files")
    parser.add_argument("--labels", nargs="+", help="Labels for comparison")

    args = parser.parse_args()

    if args.compare:
        # Compare multiple validators
        labels = (
            args.labels if args.labels else [f"Validator {i+1}" for i in range(len(args.compare))]
        )
        compare_validator_metrics(args.compare, labels)
    else:
        # Single file analysis
        if not os.path.exists(args.results_file):
            print(f"Error: Results file '{args.results_file}' not found")
            return

        calculate_metrics_from_file(args.results_file, args.output)

        # Analyze failures
        calculator = MetricsCalculator()
        calculator.load_results(args.results_file)
        calculator.calculate_metrics()

        failures = calculator.analyze_failure_patterns()
        if failures["false_positives"] or failures["false_negatives"]:
            print("\nFAILURE ANALYSIS")
            print("=" * 40)

            if failures["false_positives"]:
                print(f"\nFalse Positives ({len(failures['false_positives'])}):")
                for fp in failures["false_positives"][:5]:  # Show first 5
                    print(f"  - {fp['test_id']}: {fp['notes']}")

            if failures["false_negatives"]:
                print(f"\nFalse Negatives ({len(failures['false_negatives'])}):")
                for fn in failures["false_negatives"][:5]:  # Show first 5
                    print(f"  - {fn['test_id']}: {fn['notes']}")


if __name__ == "__main__":
    main()
