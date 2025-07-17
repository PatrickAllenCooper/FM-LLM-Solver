"""
HTML Report Generator for Phase 1 Test Results
Generates beautiful, interactive HTML reports with metrics and visualizations
"""

import base64
import io
import json
import os
from typing import Any, Dict, List

# Try to import matplotlib for charts
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Charts will be disabled.")


class HTMLReportGenerator:
    """Generates comprehensive HTML reports for test results"""

    def __init__(self, results_file: str, output_file: str = "phase1_report.html"):
        self.results_file = results_file
        self.output_file = output_file
        self.results = None
        self.load_results()

    def load_results(self):
        """Load test results from JSON file"""
        with open(self.results_file, "r") as f:
            self.results = json.load(f)

    def generate_pie_chart(self, passed: int, failed: int, errors: int) -> str:
        """Generate a pie chart as base64 encoded image"""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        fig, ax = plt.subplots(figsize=(6, 6))

        # Data
        sizes = []
        labels = []
        colors = []

        if passed > 0:
            sizes.append(passed)
            labels.append(f"Passed ({passed})")
            colors.append("#28a745")

        if failed > 0:
            sizes.append(failed)
            labels.append(f"Failed ({failed})")
            colors.append("#dc3545")

        if errors > 0:
            sizes.append(errors)
            labels.append(f"Errors ({errors})")
            colors.append("#ffc107")

        if not sizes:
            return ""

        # Create pie chart
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def generate_timeline_chart(self, test_results: List[Dict]) -> str:
        """Generate a timeline chart showing test execution times"""
        if not MATPLOTLIB_AVAILABLE or not test_results:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract timing data
        test_names = []
        times = []
        colors = []

        for result in test_results:
            if "new_validator_time" in result and result["new_validator_time"]:
                test_names.append(
                    result["test_id"][:20] + "..."
                    if len(result["test_id"]) > 20
                    else result["test_id"]
                )
                times.append(result["new_validator_time"])
                colors.append("#28a745" if result.get("correct", False) else "#dc3545")

        if not times:
            return ""

        # Create horizontal bar chart
        y_pos = range(len(test_names))
        ax.barh(y_pos, times, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names)
        ax.set_xlabel("Execution Time (seconds)")
        ax.set_title("Test Execution Times")

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def generate_html(self):
        """Generate the complete HTML report"""
        # Calculate statistics
        self.results.get("total_tests", 0)
        summary = self.results.get("summary", {})
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        errors = summary.get("errors", 0)

        # Generate charts
        pie_chart = self.generate_pie_chart(passed, failed, errors)
        timeline_chart = self.generate_timeline_chart(self.results.get("results", []))

        # HTML template
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 1 Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background-color: #2c3e50;
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}
        .timestamp {{
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .summary-card .number {{
            font-size: 3rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .errors {{ color: #ffc107; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }}
        .results-table {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .status-pass {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-fail {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .status-error {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .level-sets {{
            font-family: monospace;
            font-size: 0.9rem;
        }}
        .notes {{
            font-size: 0.85rem;
            color: #666;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
        .collapsible {{
            cursor: pointer;
            padding: 10px;
            background-color: #f0f0f0;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            width: 100%;
            margin-top: 10px;
        }}
        .collapsible:hover {{
            background-color: #e0e0e0;
        }}
        .content {{
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Phase 1 Barrier Certificate Test Report</h1>
        <div class="timestamp">Generated: {self.results.get('timestamp', datetime.now().isoformat())}</div>
    </header>

    <div class="container">
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="number">{total_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Passed</h3>
                <div class="number passed">{passed}</div>
                <div>{100*passed/total_tests:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="number failed">{failed}</div>
                <div>{100*failed/total_tests:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Errors</h3>
                <div class="number errors">{errors}</div>
                <div>{100*errors/total_tests:.1f}%</div>
            </div>
        </div>
"""

        # Add pie chart if available
        if pie_chart:
            html += """
        <div class="chart-container">
            <h2>Test Results Distribution</h2>
            <img src="{pie_chart}" alt="Test Results Pie Chart" style="max-width: 500px;">
        </div>
"""

        # Add timeline chart if available
        if timeline_chart:
            html += """
        <div class="chart-container">
            <h2>Test Execution Timeline</h2>
            <img src="{timeline_chart}" alt="Test Execution Timeline" style="max-width: 100%;">
        </div>
"""

        # Add detailed results table
        html += """
        <div class="results-table">
            <h2 style="padding: 20px 20px 10px 20px;">Detailed Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>System</th>
                        <th>Expected</th>
                        <th>Result</th>
                        <th>Status</th>
                        <th>Level Sets</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add individual test results
        for result in self.results.get("results", []):
            status = (
                "pass"
                if result.get("correct")
                else "error" if result.get("correct") is None else "fail"
            )
            status_text = (
                "PASS" if status == "pass" else "ERROR" if status == "error" else "FAIL"
            )

            level_sets = ""
            if result.get("level_sets_computed"):
                c1 = result["level_sets_computed"].get("c1", "N/A")
                c2 = result["level_sets_computed"].get("c2", "N/A")
                level_sets = (
                    f"c₁={c1:.3f}, c₂={c2:.3f}"
                    if isinstance(c1, (int, float))
                    else "N/A"
                )

            time_str = (
                f"{result.get('new_validator_time', 0):.3f}"
                if result.get("new_validator_time")
                else "N/A"
            )

            html += """
                    <tr>
                        <td><strong>{result.get('test_id', 'Unknown')}</strong></td>
                        <td>{result.get('system_name', 'Unknown')}</td>
                        <td>{result.get('expected_valid', 'Unknown')}</td>
                        <td>{result.get('new_validator_result', 'N/A')}</td>
                        <td><span class="status-badge status-{status}">{status_text}</span></td>
                        <td class="level-sets">{level_sets}</td>
                        <td>{time_str}</td>
                    </tr>
"""

            # Add notes if present
            if result.get("notes"):
                html += """
                    <tr>
                        <td colspan="7" class="notes">Note: {result['notes']}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
"""

        # Add failed tests summary if any
        failed_tests = [
            r for r in self.results.get("results", []) if r.get("correct") == False
        ]
        if failed_tests:
            html += """
        <div class="results-table">
            <h2 style="padding: 20px 20px 10px 20px; color: #dc3545;">Failed Tests Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Expected Valid</th>
                        <th>Got Result</th>
                        <th>Certificate</th>
                    </tr>
                </thead>
                <tbody>
"""
            for result in failed_tests:
                html += """
                    <tr>
                        <td><strong>{result.get('test_id', 'Unknown')}</strong></td>
                        <td>{result.get('expected_valid', 'Unknown')}</td>
                        <td>{result.get('new_validator_result', 'N/A')}</td>
                        <td style="font-family: monospace; font-size: 0.9rem;">{result.get('certificate', 'N/A')}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""

        # Add footer
        html += """
    </div>

    <footer class="footer">
        <p>Phase 1 Barrier Certificate Validation Test Report</p>
        <p>Generated by HTMLReportGenerator</p>
    </footer>

    <script>
        // Add collapsible functionality if needed
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
    </script>
</body>
</html>
"""

        # Write to file
        with open(self.output_file, "w") as f:
            f.write(html)

        print(f"HTML report generated: {self.output_file}")

    def generate_metrics_summary(self) -> Dict[str, Any]:
        """Generate metrics summary for the report"""
        results = self.results.get("results", [])

        # Calculate various metrics
        metrics = {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.get("correct") == True),
            "failed": sum(1 for r in results if r.get("correct") == False),
            "errors": sum(1 for r in results if r.get("correct") is None),
            "avg_execution_time": 0,
            "validator_agreement_rate": 0,
            "level_set_accuracy": 0,
        }

        # Average execution time
        times = [
            r.get("new_validator_time", 0)
            for r in results
            if r.get("new_validator_time")
        ]
        if times:
            metrics["avg_execution_time"] = sum(times) / len(times)

        # Validator agreement rate
        agreements = [r for r in results if r.get("agreement") is not None]
        if agreements:
            agreed = sum(1 for r in agreements if r.get("agreement"))
            metrics["validator_agreement_rate"] = agreed / len(agreements) * 100

        # Level set accuracy
        level_set_matches = [r for r in results if r.get("level_set_match") is not None]
        if level_set_matches:
            matches = sum(1 for r in level_set_matches if r.get("level_set_match"))
            metrics["level_set_accuracy"] = matches / len(level_set_matches) * 100

        return metrics


def main():
    """Main entry point for report generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HTML report from test results"
    )
    parser.add_argument("results_file", help="JSON file containing test results")
    parser.add_argument(
        "-o", "--output", default="phase1_report.html", help="Output HTML file name"
    )

    args = parser.parse_args()

    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found")
        return

    # Generate report
    generator = HTMLReportGenerator(args.results_file, args.output)
    generator.generate_html()

    # Print metrics summary
    metrics = generator.generate_metrics_summary()
    print("\nTest Metrics Summary:")
    print(f"  Total Tests: {metrics['total_tests']}")
    print(
        f"  Passed: {metrics['passed']} ({metrics['passed']/metrics['total_tests']*100:.1f}%)"
    )
    print(
        f"  Failed: {metrics['failed']} ({metrics['failed']/metrics['total_tests']*100:.1f}%)"
    )
    print(
        f"  Errors: {metrics['errors']} ({metrics['errors']/metrics['total_tests']*100:.1f}%)"
    )
    print(f"  Avg Execution Time: {metrics['avg_execution_time']:.3f}s")
    print(f"  Validator Agreement: {metrics['validator_agreement_rate']:.1f}%")
    print(f"  Level Set Accuracy: {metrics['level_set_accuracy']:.1f}%")


if __name__ == "__main__":
    main()
