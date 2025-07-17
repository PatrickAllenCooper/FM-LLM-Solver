"""
Asynchronous Operations Manager for FM-LLM-Solver.

This module provides utilities for managing async operations, connection pooling,
request queuing, and concurrent task execution throughout the system.
"""

import asyncio
import logging
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
from queue import Queue, Empty
import threading
from functools import wraps, partial

from .exceptions import PerformanceError, ServiceError
from .monitoring import MonitoringManager


@dataclass
class TaskResult:
    """Result of an async task execution."""

    task_id: str
    result: Any
    success: bool
    duration: float
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PoolConfig:
    """Configuration for executor pools."""

    max_workers: int = 4
    thread_name_prefix: str = "AsyncWorker"
    initializer: Optional[Callable] = None
    initargs: tuple = ()


class RequestQueue:
    """High-performance request queue with priority support."""

    def __init__(self, maxsize: int = 1000, priority_levels: int = 3):
        """Initialize the request queue."""
        self.maxsize = maxsize
        self.priority_levels = priority_levels
        self._queues = [Queue(maxsize=maxsize // priority_levels) for _ in range(priority_levels)]
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False

    def put(self, item: Any, priority: int = 1, timeout: Optional[float] = None) -> bool:
        """Put item in queue with specified priority (0=highest, 2=lowest)."""
        if self._closed:
            return False

        priority = max(0, min(priority, self.priority_levels - 1))

        try:
            with self._condition:
                self._queues[priority].put(item, timeout=timeout)
                self._condition.notify_all()
            return True
        except Exception:
            return False

    def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue, respecting priority order."""
        if self._closed:
            raise Empty()

        with self._condition:
            start_time = time.time()

            while True:
                # Check queues in priority order
                for queue in self._queues:
                    try:
                        return queue.get_nowait()
                    except Empty:
                        continue

                if self._closed:
                    raise Empty()

                # Calculate remaining timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise Empty()
                    remaining_timeout = timeout - elapsed
                    self._condition.wait(remaining_timeout)
                else:
                    self._condition.wait()

    def qsize(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self._queues)

    def empty(self) -> bool:
        """Check if queue is empty."""
        return all(q.empty() for q in self._queues)

    def close(self):
        """Close the queue."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class AsyncManager:
    """Main async operations manager."""

    def __init__(self, config_manager, monitoring_manager: Optional[MonitoringManager] = None):
        """Initialize the async manager."""
        self.config_manager = config_manager
        self.monitoring = monitoring_manager
        self.logger = logging.getLogger(__name__)

        # Get configuration
        async_config = config_manager.get("performance.async", {})
        self.max_thread_workers = async_config.get("max_thread_workers", 8)
        self.max_process_workers = async_config.get("max_process_workers", 4)
        self.default_timeout = async_config.get("default_timeout", 30.0)
        self.queue_size = async_config.get("queue_size", 1000)

        # Initialize executors
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_thread_workers, thread_name_prefix="AsyncThread"
        )
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_process_workers)

        # Initialize request queue
        self.request_queue = RequestQueue(maxsize=self.queue_size)

        # Task tracking
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_counter = 0
        self._lock = asyncio.Lock()

        # Performance metrics
        self._metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_duration": 0.0,
            "queue_size_max": 0,
        }

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter}_{int(time.time())}"

    def _record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record performance metric."""
        if self.monitoring:
            self.monitoring.record_metric(metric_name, value, tags or {})

    async def run_async(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        priority: int = 1,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """Run an async function with monitoring and error handling."""
        task_id = task_id or self._generate_task_id()
        timeout = timeout or self.default_timeout
        start_time = time.time()

        try:
            async with self._lock:
                # Create and track task
                if asyncio.iscoroutinefunction(func):
                    task = asyncio.create_task(func(*args, **kwargs))
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(self.thread_pool, partial(func, *args, **kwargs))

                self._active_tasks[task_id] = task

            # Wait for completion with timeout
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                duration = time.time() - start_time

                # Update metrics
                self._metrics["tasks_completed"] += 1
                self._metrics["average_duration"] = (
                    self._metrics["average_duration"] * (self._metrics["tasks_completed"] - 1)
                    + duration
                ) / self._metrics["tasks_completed"]

                self._record_metric("async_task_duration", duration, {"status": "success"})

                return TaskResult(
                    task_id=task_id,
                    result=result,
                    success=True,
                    duration=duration,
                    metadata={"priority": priority},
                )

            except asyncio.TimeoutError:
                task.cancel()
                duration = time.time() - start_time
                self._metrics["tasks_failed"] += 1

                self._record_metric("async_task_duration", duration, {"status": "timeout"})

                return TaskResult(
                    task_id=task_id,
                    result=None,
                    success=False,
                    duration=duration,
                    error=asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s"),
                )

        except Exception as e:
            duration = time.time() - start_time
            self._metrics["tasks_failed"] += 1

            self._record_metric("async_task_duration", duration, {"status": "error"})

            return TaskResult(
                task_id=task_id, result=None, success=False, duration=duration, error=e
            )

        finally:
            # Remove from active tasks
            async with self._lock:
                self._active_tasks.pop(task_id, None)

    async def batch_execute(
        self, tasks: List[Callable], max_concurrent: int = 10, timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """Execute multiple tasks concurrently with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        timeout = timeout or self.default_timeout

        async def execute_with_semaphore(task_func, task_index):
            async with semaphore:
                return await self.run_async(
                    task_func, timeout=timeout, task_id=f"batch_{task_index}"
                )

        # Create tasks
        coroutines = [execute_with_semaphore(task, i) for i, task in enumerate(tasks)]

        # Execute all tasks
        try:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            # Convert exceptions to failed TaskResults
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(
                        TaskResult(
                            task_id=f"batch_{i}",
                            result=None,
                            success=False,
                            duration=0.0,
                            error=result,
                        )
                    )
                else:
                    final_results.append(result)

            return final_results

        except Exception as e:
            raise PerformanceError(f"Batch execution failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        queue_size = self.request_queue.qsize()
        self._metrics["queue_size_max"] = max(self._metrics["queue_size_max"], queue_size)

        metrics = self._metrics.copy()
        metrics.update(
            {
                "active_tasks": len(self._active_tasks),
                "queue_size": queue_size,
                "thread_pool_threads": (
                    len(self.thread_pool._threads) if hasattr(self.thread_pool, "_threads") else 0
                ),
                "process_pool_processes": getattr(self.process_pool, "_processes", 0),
            }
        )

        return metrics

    def shutdown(self):
        """Shutdown the async manager."""
        self.logger.info("Shutting down async manager...")

        # Close request queue
        self.request_queue.close()

        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        self.logger.info("Async manager shutdown complete")


def async_cached(cache_manager, ttl: int = 300, key_func: Optional[Callable] = None):
    """Decorator for async functions with caching support."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def rate_limited(max_calls: int, period: float = 60.0):
    """Decorator for rate limiting async functions."""
    calls = []
    lock = asyncio.Lock()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with lock:
                now = time.time()
                # Remove old calls outside the period
                while calls and calls[0] <= now - period:
                    calls.pop(0)

                # Check rate limit
                if len(calls) >= max_calls:
                    raise PerformanceError(f"Rate limit exceeded: {max_calls} calls per {period}s")

                calls.append(now)

            return await func(*args, **kwargs)

        return wrapper

    return decorator
