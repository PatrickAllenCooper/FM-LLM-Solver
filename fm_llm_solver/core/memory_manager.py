"""
Memory Management for FM-LLM-Solver.

This module provides utilities for memory optimization, garbage collection,
object pooling, and memory usage monitoring.
"""

import gc
import logging
import sys
import time
import weakref
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable
import psutil
import os

from .exceptions import MemoryError as CustomMemoryError
from .monitoring import MonitoringManager


T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    available_mb: float  # Available memory in MB
    gc_objects: int  # Number of objects tracked by GC
    gc_collections: Dict[int, int]  # GC collections per generation


class ObjectPool(Generic[T]):
    """Generic object pool for memory optimization."""
    
    def __init__(
        self,
        factory: Callable[[], T],
        reset_func: Optional[Callable[[T], None]] = None,
        max_size: int = 100,
        cleanup_interval: float = 300.0
    ):
        """Initialize object pool."""
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self._pool: deque = deque()
        self._lock = threading.RLock()
        self._created_count = 0
        self._reused_count = 0
        self._last_cleanup = time.time()
        
        # Weak references to track objects in use
        self._in_use: weakref.WeakSet = weakref.WeakSet()
    
    def get(self) -> T:
        """Get an object from the pool."""
        with self._lock:
            self._cleanup_if_needed()
            
            if self._pool:
                obj = self._pool.popleft()
                self._reused_count += 1
            else:
                obj = self.factory()
                self._created_count += 1
            
            self._in_use.add(obj)
            return obj
    
    def put(self, obj: T) -> bool:
        """Return an object to the pool."""
        with self._lock:
            if len(self._pool) >= self.max_size:
                return False
            
            # Reset object if reset function provided
            if self.reset_func:
                try:
                    self.reset_func(obj)
                except Exception:
                    return False
            
            self._pool.append(obj)
            
            # Remove from in-use tracking
            try:
                self._in_use.discard(obj)
            except TypeError:
                pass  # Object may not be hashable
            
            return True
    
    def _cleanup_if_needed(self):
        """Cleanup pool if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self.cleanup_interval:
            self._last_cleanup = now
            
            # Remove objects that haven't been used recently
            # This is a simple implementation - in practice you might want
            # to track last access time per object
            if len(self._pool) > self.max_size // 2:
                excess = len(self._pool) - self.max_size // 2
                for _ in range(excess):
                    if self._pool:
                        self._pool.popleft()
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'in_use_count': len(self._in_use),
                'hit_rate': self._reused_count / max(self._created_count + self._reused_count, 1)
            }
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self._pool.clear()


class MemoryTracker:
    """Track memory usage of specific objects or operations."""
    
    def __init__(self, name: str, monitoring_manager: Optional[MonitoringManager] = None):
        """Initialize memory tracker."""
        self.name = name
        self.monitoring = monitoring_manager
        self.logger = logging.getLogger(__name__)
        
        self._start_memory = 0
        self._peak_memory = 0
        self._allocations: List[Dict[str, Any]] = []
        self._tracking = False
    
    def start_tracking(self):
        """Start memory tracking."""
        self._tracking = True
        self._start_memory = self._get_memory_usage()
        self._peak_memory = self._start_memory
        self._allocations.clear()
        
        self.logger.debug(f"Started memory tracking for {self.name}")
    
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop memory tracking and return statistics."""
        if not self._tracking:
            return {}
        
        end_memory = self._get_memory_usage()
        memory_diff = end_memory - self._start_memory
        
        stats = {
            'name': self.name,
            'start_memory_mb': self._start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self._peak_memory,
            'memory_diff_mb': memory_diff,
            'allocations': len(self._allocations),
            'tracking_duration': time.time()
        }
        
        if self.monitoring:
            self.monitoring.record_metric(
                f'memory_usage_{self.name}',
                memory_diff,
                {'type': 'memory_diff'}
            )
        
        self._tracking = False
        self.logger.debug(f"Stopped memory tracking for {self.name}: {memory_diff:.2f} MB diff")
        
        return stats
    
    def track_allocation(self, obj_type: str, size_bytes: int):
        """Track a memory allocation."""
        if not self._tracking:
            return
        
        current_memory = self._get_memory_usage()
        self._peak_memory = max(self._peak_memory, current_memory)
        
        self._allocations.append({
            'type': obj_type,
            'size_bytes': size_bytes,
            'timestamp': time.time(),
            'memory_mb': current_memory
        })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager for tracking memory usage of an operation."""
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self._get_memory_usage()
            duration = time.time() - start_time
            memory_diff = end_memory - start_memory
            
            if self.monitoring:
                self.monitoring.record_metric(
                    f'operation_memory_{operation_name}',
                    memory_diff,
                    {'duration': str(duration)}
                )
            
            self.logger.debug(
                f"Operation {operation_name} used {memory_diff:.2f} MB "
                f"in {duration:.2f}s"
            )


class MemoryManager:
    """Main memory management system."""
    
    def __init__(self, config_manager, monitoring_manager: Optional[MonitoringManager] = None):
        """Initialize memory manager."""
        self.config_manager = config_manager
        self.monitoring = monitoring_manager
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        memory_config = config_manager.get('performance.memory', {})
        self.gc_threshold = memory_config.get('gc_threshold_mb', 100)
        self.monitoring_interval = memory_config.get('monitoring_interval', 60)
        self.enable_object_pools = memory_config.get('enable_object_pools', True)
        self.max_pool_size = memory_config.get('max_pool_size', 100)
        
        # Object pools
        self._object_pools: Dict[str, ObjectPool] = {}
        
        # Memory trackers
        self._trackers: Dict[str, MemoryTracker] = {}
        
        # Monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._last_gc_time = time.time()
        
        # Memory thresholds
        self._memory_warning_threshold = memory_config.get('warning_threshold_mb', 1000)
        self._memory_critical_threshold = memory_config.get('critical_threshold_mb', 2000)
        
        # Start monitoring if enabled
        if self.monitoring and memory_config.get('enable_monitoring', True):
            self.start_monitoring()
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            # GC statistics
            gc_stats = gc.get_stats()
            gc_collections = {i: stat['collections'] for i, stat in enumerate(gc_stats)}
            
            return MemoryStats(
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=process.memory_percent(),
                available_mb=system_memory.available / 1024 / 1024,
                gc_objects=len(gc.get_objects()),
                gc_collections=gc_collections
            )
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, {})
    
    def create_object_pool(
        self,
        name: str,
        factory: Callable[[], T],
        reset_func: Optional[Callable[[T], None]] = None,
        max_size: Optional[int] = None
    ) -> ObjectPool[T]:
        """Create a named object pool."""
        if not self.enable_object_pools:
            raise CustomMemoryError("Object pools are disabled")
        
        max_size = max_size or self.max_pool_size
        pool = ObjectPool(factory, reset_func, max_size)
        self._object_pools[name] = pool
        
        self.logger.debug(f"Created object pool '{name}' with max size {max_size}")
        return pool
    
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get an object pool by name."""
        return self._object_pools.get(name)
    
    def create_tracker(self, name: str) -> MemoryTracker:
        """Create a memory tracker."""
        tracker = MemoryTracker(name, self.monitoring)
        self._trackers[name] = tracker
        return tracker
    
    def get_tracker(self, name: str) -> Optional[MemoryTracker]:
        """Get a memory tracker by name."""
        return self._trackers.get(name)
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        start_time = time.time()
        
        # Collect statistics before
        before_stats = self.get_memory_stats()
        
        # Force full garbage collection
        collected = {}
        for generation in range(3):
            collected[generation] = gc.collect(generation)
        
        # Statistics after
        after_stats = self.get_memory_stats()
        duration = time.time() - start_time
        
        memory_freed = before_stats.rss_mb - after_stats.rss_mb
        
        gc_stats = {
            'generation_0': collected.get(0, 0),
            'generation_1': collected.get(1, 0),
            'generation_2': collected.get(2, 0),
            'total_collected': sum(collected.values()),
            'memory_freed_mb': memory_freed,
            'duration_ms': duration * 1000,
            'objects_before': before_stats.gc_objects,
            'objects_after': after_stats.gc_objects
        }
        
        if self.monitoring:
            self.monitoring.record_metric('gc_memory_freed', memory_freed, {'type': 'forced'})
            self.monitoring.record_metric('gc_duration', duration, {'type': 'forced'})
        
        self._last_gc_time = time.time()
        self.logger.info(f"Forced GC freed {memory_freed:.2f} MB in {duration*1000:.1f} ms")
        
        return gc_stats
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if system is under memory pressure."""
        stats = self.get_memory_stats()
        
        pressure_info = {
            'memory_mb': stats.rss_mb,
            'percent': stats.percent,
            'available_mb': stats.available_mb,
            'pressure_level': 'normal',
            'should_gc': False,
            'recommendations': []
        }
        
        # Determine pressure level
        if stats.rss_mb > self._memory_critical_threshold:
            pressure_info['pressure_level'] = 'critical'
            pressure_info['should_gc'] = True
            pressure_info['recommendations'].extend([
                'Force garbage collection',
                'Clear object pools',
                'Reduce cache sizes',
                'Consider reducing concurrent operations'
            ])
        elif stats.rss_mb > self._memory_warning_threshold:
            pressure_info['pressure_level'] = 'warning'
            pressure_info['should_gc'] = True
            pressure_info['recommendations'].extend([
                'Run garbage collection',
                'Clear unused caches'
            ])
        elif stats.rss_mb > self.gc_threshold:
            pressure_info['should_gc'] = True
        
        # Check if GC should be triggered based on time
        if time.time() - self._last_gc_time > 300:  # 5 minutes
            pressure_info['should_gc'] = True
            pressure_info['recommendations'].append('Periodic garbage collection due')
        
        return pressure_info
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_time = time.time()
        optimization_actions = []
        
        # Check memory pressure
        pressure_info = self.check_memory_pressure()
        
        # Force GC if needed
        if pressure_info['should_gc']:
            gc_stats = self.force_gc()
            optimization_actions.append({
                'action': 'garbage_collection',
                'result': gc_stats
            })
        
        # Clear object pools if under pressure
        if pressure_info['pressure_level'] in ['warning', 'critical']:
            for name, pool in self._object_pools.items():
                pool_stats = pool.stats()
                pool.clear()
                optimization_actions.append({
                    'action': 'clear_object_pool',
                    'pool_name': name,
                    'cleared_objects': pool_stats['pool_size']
                })
        
        # Optimize Python internals
        if pressure_info['pressure_level'] == 'critical':
            # Clear type cache
            sys._clear_type_cache()
            optimization_actions.append({'action': 'clear_type_cache'})
        
        duration = time.time() - start_time
        
        optimization_result = {
            'duration_ms': duration * 1000,
            'actions_taken': optimization_actions,
            'pressure_before': pressure_info,
            'pressure_after': self.check_memory_pressure()
        }
        
        if self.monitoring:
            self.monitoring.record_metric(
                'memory_optimization_duration',
                duration,
                {'actions_count': str(len(optimization_actions))}
            )
        
        return optimization_result
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitoring_thread.start()
        
        self.logger.info("Started memory monitoring")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped memory monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Get memory statistics
                stats = self.get_memory_stats()
                
                # Record metrics
                if self.monitoring:
                    self.monitoring.record_metric('memory_rss', stats.rss_mb, {'type': 'rss'})
                    self.monitoring.record_metric('memory_vms', stats.vms_mb, {'type': 'vms'})
                    self.monitoring.record_metric('memory_percent', stats.percent, {'type': 'percent'})
                    self.monitoring.record_metric('gc_objects', stats.gc_objects, {'type': 'count'})
                
                # Check for memory pressure
                pressure_info = self.check_memory_pressure()
                
                # Auto-optimize if needed
                if pressure_info['pressure_level'] == 'critical':
                    self.logger.warning(f"Critical memory pressure detected: {stats.rss_mb:.1f} MB")
                    self.optimize_memory()
                elif pressure_info['should_gc']:
                    self.force_gc()
                
                # Record object pool statistics
                for name, pool in self._object_pools.items():
                    pool_stats = pool.stats()
                    if self.monitoring:
                        self.monitoring.record_metric(
                            f'object_pool_{name}_size',
                            pool_stats['pool_size'],
                            {'pool': name}
                        )
                        self.monitoring.record_metric(
                            f'object_pool_{name}_hit_rate',
                            pool_stats['hit_rate'],
                            {'pool': name}
                        )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive memory performance report."""
        stats = self.get_memory_stats()
        pressure_info = self.check_memory_pressure()
        
        # Object pool statistics
        pool_stats = {}
        for name, pool in self._object_pools.items():
            pool_stats[name] = pool.stats()
        
        # Tracker statistics
        tracker_stats = {}
        for name, tracker in self._trackers.items():
            if hasattr(tracker, '_allocations'):
                tracker_stats[name] = {
                    'allocations': len(tracker._allocations),
                    'tracking_active': tracker._tracking
                }
        
        return {
            'memory_stats': {
                'rss_mb': stats.rss_mb,
                'vms_mb': stats.vms_mb,
                'percent': stats.percent,
                'available_mb': stats.available_mb,
                'gc_objects': stats.gc_objects
            },
            'pressure_info': pressure_info,
            'object_pools': pool_stats,
            'trackers': tracker_stats,
            'gc_collections': stats.gc_collections,
            'monitoring_active': self._monitoring_active
        }
    
    def shutdown(self):
        """Shutdown memory manager."""
        self.logger.info("Shutting down memory manager...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clear object pools
        for pool in self._object_pools.values():
            pool.clear()
        
        # Final cleanup
        self.force_gc()
        
        self.logger.info("Memory manager shutdown complete")


@contextmanager
def track_memory(name: str, memory_manager: MemoryManager):
    """Context manager for tracking memory usage."""
    tracker = memory_manager.create_tracker(name)
    tracker.start_tracking()
    
    try:
        yield tracker
    finally:
        stats = tracker.stop_tracking()
        memory_manager.logger.debug(f"Memory tracking for {name}: {stats}")


def memory_limit(max_mb: float):
    """Decorator to enforce memory limits on functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            try:
                result = func(*args, **kwargs)
                
                end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_used = end_memory - start_memory
                
                if memory_used > max_mb:
                    raise CustomMemoryError(
                        f"Function {func.__name__} exceeded memory limit: "
                        f"{memory_used:.1f} MB > {max_mb} MB"
                    )
                
                return result
                
            except Exception as e:
                # Cleanup on error
                gc.collect()
                raise
        
        return wrapper
    return decorator
