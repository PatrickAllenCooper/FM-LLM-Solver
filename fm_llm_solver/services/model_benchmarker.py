"""
Model benchmarking service for code generation models.

Provides comprehensive performance comparison across multiple coding tasks.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.exceptions import ModelError
from fm_llm_solver.services.model_provider import ModelProviderFactory
from fm_llm_solver.services.model_downloader import get_model_downloader


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    name: str
    description: str
    prompt: str
    expected_output: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    programming_language: str = "python"
    max_tokens: int = 512
    temperature: float = 0.1  # Low temperature for consistent results


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""
    task_id: str
    model_id: str
    generated_output: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    tokens_generated: int = 0
    bleu_score: Optional[float] = None
    code_compiles: Optional[bool] = None
    test_passes: Optional[bool] = None


@dataclass
class ModelBenchmarkSummary:
    """Summary of benchmark results for a model."""
    model_id: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    average_execution_time: float
    average_bleu_score: Optional[float]
    code_compilation_rate: Optional[float]
    test_pass_rate: Optional[float]
    total_tokens_generated: int
    benchmark_date: str


class ModelBenchmarker:
    """Comprehensive benchmarking system for code generation models."""

    def __init__(self, cache_dir: str = "benchmark_cache"):
        """
        Initialize the model benchmarker.
        
        Args:
            cache_dir: Directory to cache benchmark results
        """
        self.logger = get_logger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark tasks
        self.tasks: List[BenchmarkTask] = []
        self.results: Dict[str, List[BenchmarkResult]] = {}
        
        # Load default benchmark tasks
        self._load_default_tasks()
        
        self.logger.info(f"Model benchmarker initialized with {len(self.tasks)} tasks")

    def _load_default_tasks(self):
        """Load default benchmark tasks."""
        # Code generation tasks
        code_gen_tasks = [
            BenchmarkTask(
                task_id="fibonacci",
                name="Fibonacci Sequence",
                description="Generate a function to calculate fibonacci numbers",
                prompt="Write a Python function to calculate the nth Fibonacci number:",
                expected_output="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                category="algorithms",
                difficulty="easy",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="quicksort",
                name="Quick Sort Implementation",
                description="Implement the quicksort algorithm",
                prompt="Write a Python function that implements the quicksort algorithm:",
                category="algorithms",
                difficulty="medium",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="binary_search",
                name="Binary Search",
                description="Implement binary search algorithm",
                prompt="Write a Python function that performs binary search on a sorted array:",
                category="algorithms",
                difficulty="easy",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="merge_sort",
                name="Merge Sort Implementation",
                description="Implement the merge sort algorithm",
                prompt="Write a Python function that implements the merge sort algorithm:",
                category="algorithms",
                difficulty="medium",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="palindrome_check",
                name="Palindrome Check",
                description="Check if a string is a palindrome",
                prompt="Write a Python function to check if a given string is a palindrome:",
                category="string_processing",
                difficulty="easy",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="reverse_linked_list",
                name="Reverse Linked List",
                description="Reverse a singly linked list",
                prompt="Write a Python function to reverse a singly linked list:",
                category="data_structures",
                difficulty="medium",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="bst_insertion",
                name="BST Insertion",
                description="Insert a node into a binary search tree",
                prompt="Write a Python function to insert a node into a binary search tree:",
                category="data_structures",
                difficulty="medium",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="dfs_traversal",
                name="DFS Graph Traversal",
                description="Implement depth-first search traversal",
                prompt="Write a Python function that performs depth-first search on a graph:",
                category="graph_algorithms",
                difficulty="medium",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="dijkstra_algorithm",
                name="Dijkstra's Algorithm",
                description="Implement Dijkstra's shortest path algorithm",
                prompt="Write a Python function that implements Dijkstra's algorithm for finding shortest paths:",
                category="graph_algorithms",
                difficulty="hard",
                programming_language="python"
            ),
            BenchmarkTask(
                task_id="regex_email_validation",
                name="Email Validation with Regex",
                description="Validate email addresses using regular expressions",
                prompt="Write a Python function that validates email addresses using regular expressions:",
                category="regex",
                difficulty="medium",
                programming_language="python"
            )
        ]
        
        # JavaScript tasks
        js_tasks = [
            BenchmarkTask(
                task_id="js_array_filter",
                name="JavaScript Array Filter",
                description="Filter array elements based on condition",
                prompt="Write a JavaScript function that filters an array of numbers to return only even numbers:",
                category="functional_programming",
                difficulty="easy",
                programming_language="javascript"
            ),
            BenchmarkTask(
                task_id="js_promise_handling",
                name="JavaScript Promise Handling",
                description="Handle promises with async/await",
                prompt="Write a JavaScript function that fetches data from an API using async/await:",
                category="async_programming",
                difficulty="medium",
                programming_language="javascript"
            )
        ]
        
        # C++ tasks
        cpp_tasks = [
            BenchmarkTask(
                task_id="cpp_vector_operations",
                name="C++ Vector Operations",
                description="Perform operations on C++ vectors",
                prompt="Write a C++ function that finds the maximum element in a vector:",
                category="stl",
                difficulty="easy",
                programming_language="cpp"
            ),
            BenchmarkTask(
                task_id="cpp_class_inheritance",
                name="C++ Class Inheritance",
                description="Implement class inheritance in C++",
                prompt="Write C++ classes that demonstrate inheritance with a base Shape class and derived Circle class:",
                category="oop",
                difficulty="medium",
                programming_language="cpp"
            )
        ]
        
        # SQL tasks
        sql_tasks = [
            BenchmarkTask(
                task_id="sql_join_query",
                name="SQL Join Query",
                description="Write a SQL query with joins",
                prompt="Write a SQL query that joins two tables 'users' and 'orders' to get user names with their order counts:",
                category="database",
                difficulty="medium",
                programming_language="sql"
            )
        ]
        
        self.tasks.extend(code_gen_tasks + js_tasks + cpp_tasks + sql_tasks)

    def add_custom_task(self, task: BenchmarkTask):
        """Add a custom benchmark task."""
        self.tasks.append(task)
        self.logger.info(f"Added custom task: {task.name}")

    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate text."""
        try:
            # Simple implementation - for production, use proper BLEU implementation
            ref_tokens = reference.lower().split()
            cand_tokens = candidate.lower().split()
            
            if not cand_tokens:
                return 0.0
            
            # Calculate precision for different n-grams
            matches = 0
            total = len(cand_tokens)
            
            for token in cand_tokens:
                if token in ref_tokens:
                    matches += 1
            
            return matches / total if total > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"BLEU score calculation failed: {e}")
            return 0.0

    def _check_code_compilation(self, code: str, language: str) -> bool:
        """Check if generated code compiles/is syntactically correct."""
        try:
            if language == "python":
                import ast
                ast.parse(code)
                return True
            elif language == "javascript":
                # Basic syntax check - in production, use a proper JS parser
                return "function" in code or "=>" in code
            elif language == "cpp":
                # Basic syntax check - in production, use a proper C++ compiler
                return "#include" in code or "int main" in code or "class" in code
            elif language == "sql":
                # Basic syntax check
                return any(keyword in code.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"])
            else:
                return True  # Assume valid for unknown languages
                
        except Exception as e:
            self.logger.debug(f"Compilation check failed for {language}: {e}")
            return False

    async def _run_single_benchmark(self, model_id: str, task: BenchmarkTask, model_provider) -> BenchmarkResult:
        """Run a single benchmark task."""
        start_time = time.time()
        
        try:
            # Generate code
            generated_output = model_provider.generate_text(
                prompt=task.prompt,
                max_tokens=task.max_tokens,
                temperature=task.temperature
            )
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            bleu_score = None
            if task.expected_output:
                bleu_score = self._calculate_bleu_score(task.expected_output, generated_output)
            
            code_compiles = self._check_code_compilation(generated_output, task.programming_language)
            
            # Count tokens (approximation)
            tokens_generated = len(generated_output.split())
            
            return BenchmarkResult(
                task_id=task.task_id,
                model_id=model_id,
                generated_output=generated_output,
                execution_time=execution_time,
                success=True,
                tokens_generated=tokens_generated,
                bleu_score=bleu_score,
                code_compiles=code_compiles,
                test_passes=None  # Would require actual test execution
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Benchmark failed for {model_id} on {task.task_id}: {e}")
            
            return BenchmarkResult(
                task_id=task.task_id,
                model_id=model_id,
                generated_output="",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    async def benchmark_model(self, model_id: str, model_config: Dict[str, Any], 
                            tasks: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """
        Benchmark a single model on specified tasks.
        
        Args:
            model_id: Model identifier
            model_config: Model configuration
            tasks: List of task IDs to run (if None, run all tasks)
            
        Returns:
            List of benchmark results
        """
        self.logger.info(f"Starting benchmark for model {model_id}")
        
        # Filter tasks if specified
        tasks_to_run = self.tasks
        if tasks:
            tasks_to_run = [task for task in self.tasks if task.task_id in tasks]
        
        if not tasks_to_run:
            raise ModelError(f"No valid tasks found for benchmarking")
        
        # Load model
        try:
            provider = model_config['provider']
            model_provider = ModelProviderFactory.create(provider, model_config)
            
            # Check if model is downloaded
            downloader = get_model_downloader()
            if not downloader.is_model_downloaded(model_id):
                raise ModelError(f"Model {model_id} is not downloaded")
            
            # Load the model
            model_provider.load_model(model_config)
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise ModelError(f"Failed to load model: {e}")
        
        # Run benchmarks
        results = []
        
        try:
            # Run tasks sequentially to avoid memory issues
            for task in tasks_to_run:
                self.logger.info(f"Running task {task.task_id} for model {model_id}")
                result = await self._run_single_benchmark(model_id, task, model_provider)
                results.append(result)
                
                # Small delay between tasks
                await asyncio.sleep(0.1)
            
        finally:
            # Unload model to free memory
            try:
                model_provider.unload_model()
            except Exception as e:
                self.logger.warning(f"Failed to unload model {model_id}: {e}")
        
        # Cache results
        self._cache_results(model_id, results)
        
        self.logger.info(f"Completed benchmark for model {model_id}: {len(results)} tasks")
        return results

    def _cache_results(self, model_id: str, results: List[BenchmarkResult]):
        """Cache benchmark results to disk."""
        try:
            cache_file = self.cache_dir / f"{model_id}_results.json"
            
            results_data = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "results": [asdict(result) for result in results]
            }
            
            with open(cache_file, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            self.logger.info(f"Cached results for {model_id} to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache results for {model_id}: {e}")

    def load_cached_results(self, model_id: str) -> Optional[List[BenchmarkResult]]:
        """Load cached benchmark results."""
        try:
            cache_file = self.cache_dir / f"{model_id}_results.json"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            results = [BenchmarkResult(**result_data) for result_data in data['results']]
            
            self.logger.info(f"Loaded {len(results)} cached results for {model_id}")
            return results
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached results for {model_id}: {e}")
            return None

    def calculate_model_summary(self, results: List[BenchmarkResult]) -> ModelBenchmarkSummary:
        """Calculate summary statistics for a model's benchmark results."""
        if not results:
            raise ValueError("No results provided for summary calculation")
        
        model_id = results[0].model_id
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate averages for successful tasks only
        successful_results = [r for r in results if r.success]
        
        average_execution_time = 0.0
        average_bleu_score = None
        code_compilation_rate = None
        total_tokens_generated = 0
        
        if successful_results:
            average_execution_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            total_tokens_generated = sum(r.tokens_generated for r in successful_results)
            
            # BLEU score average
            bleu_scores = [r.bleu_score for r in successful_results if r.bleu_score is not None]
            if bleu_scores:
                average_bleu_score = sum(bleu_scores) / len(bleu_scores)
            
            # Code compilation rate
            compile_results = [r.code_compiles for r in successful_results if r.code_compiles is not None]
            if compile_results:
                code_compilation_rate = sum(compile_results) / len(compile_results)
        
        return ModelBenchmarkSummary(
            model_id=model_id,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            success_rate=success_rate,
            average_execution_time=average_execution_time,
            average_bleu_score=average_bleu_score,
            code_compilation_rate=code_compilation_rate,
            test_pass_rate=None,  # Would require actual test execution
            total_tokens_generated=total_tokens_generated,
            benchmark_date=datetime.now().isoformat()
        )

    async def benchmark_multiple_models(self, model_configs: Dict[str, Dict[str, Any]], 
                                      tasks: Optional[List[str]] = None,
                                      use_cached: bool = True) -> Dict[str, ModelBenchmarkSummary]:
        """
        Benchmark multiple models and return comparison results.
        
        Args:
            model_configs: Dictionary of model_id -> model_config
            tasks: List of task IDs to run (if None, run all tasks)
            use_cached: Whether to use cached results if available
            
        Returns:
            Dictionary of model_id -> ModelBenchmarkSummary
        """
        self.logger.info(f"Starting benchmark for {len(model_configs)} models")
        
        summaries = {}
        
        for model_id, model_config in model_configs.items():
            try:
                # Check for cached results first
                results = None
                if use_cached:
                    results = self.load_cached_results(model_id)
                
                # Run benchmark if no cached results
                if results is None:
                    results = await self.benchmark_model(model_id, model_config, tasks)
                
                # Calculate summary
                summary = self.calculate_model_summary(results)
                summaries[model_id] = summary
                
                self.logger.info(f"Model {model_id}: {summary.success_rate:.1%} success rate")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for model {model_id}: {e}")
                # Create empty summary for failed benchmarks
                summaries[model_id] = ModelBenchmarkSummary(
                    model_id=model_id,
                    total_tasks=0,
                    successful_tasks=0,
                    success_rate=0.0,
                    average_execution_time=0.0,
                    average_bleu_score=None,
                    code_compilation_rate=None,
                    test_pass_rate=None,
                    total_tokens_generated=0,
                    benchmark_date=datetime.now().isoformat()
                )
        
        self.logger.info(f"Completed benchmarking {len(summaries)} models")
        return summaries

    def generate_comparison_report(self, summaries: Dict[str, ModelBenchmarkSummary]) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        if not summaries:
            return {"error": "No benchmark summaries provided"}
        
        # Sort models by success rate
        sorted_models = sorted(summaries.items(), key=lambda x: x[1].success_rate, reverse=True)
        
        # Calculate overall statistics
        total_models = len(summaries)
        avg_success_rate = sum(s.success_rate for s in summaries.values()) / total_models
        avg_execution_time = sum(s.average_execution_time for s in summaries.values()) / total_models
        
        # Find best performers
        best_success_rate = sorted_models[0][1] if sorted_models else None
        fastest_model = min(summaries.items(), key=lambda x: x[1].average_execution_time) if summaries else None
        
        # Generate rankings
        rankings = {
            "by_success_rate": [(model_id, summary.success_rate) for model_id, summary in sorted_models],
            "by_speed": sorted(summaries.items(), key=lambda x: x[1].average_execution_time),
            "by_code_quality": sorted(
                [(k, v) for k, v in summaries.items() if v.code_compilation_rate is not None],
                key=lambda x: x[1].code_compilation_rate or 0,
                reverse=True
            )
        }
        
        return {
            "summary": {
                "total_models_tested": total_models,
                "average_success_rate": avg_success_rate,
                "average_execution_time": avg_execution_time,
                "best_performer": {
                    "model_id": best_success_rate[0] if best_success_rate else None,
                    "success_rate": best_success_rate[1].success_rate if best_success_rate else None
                },
                "fastest_model": {
                    "model_id": fastest_model[0] if fastest_model else None,
                    "execution_time": fastest_model[1].average_execution_time if fastest_model else None
                }
            },
            "rankings": rankings,
            "detailed_results": {model_id: asdict(summary) for model_id, summary in summaries.items()},
            "benchmark_date": datetime.now().isoformat()
        }

    def export_results(self, summaries: Dict[str, ModelBenchmarkSummary], 
                      output_file: str = "benchmark_results.json"):
        """Export benchmark results to a file."""
        try:
            report = self.generate_comparison_report(summaries)
            
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Exported benchmark results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise


# Global benchmarker instance
_benchmarker: Optional[ModelBenchmarker] = None


def get_model_benchmarker(cache_dir: str = "benchmark_cache") -> ModelBenchmarker:
    """Get the global model benchmarker instance."""
    global _benchmarker
    
    if _benchmarker is None:
        _benchmarker = ModelBenchmarker(cache_dir)
    
    return _benchmarker 