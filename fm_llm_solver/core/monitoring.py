"""
Comprehensive monitoring and metrics system for FM-LLM-Solver.

Provides Prometheus metrics, health checks, performance monitoring,
and system observability features.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
from enum import Enum

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from .config_manager import ConfigurationManager
from .logging_manager import get_logger
from .database_manager import get_database_manager
from .cache_manager import get_cache_manager


class HealthStatus(Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Metric types for custom metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class HealthCheck:
    """Health check definition."""

    name: str
    check_func: Callable[[], Dict[str, Any]]
    interval: int = 30  # seconds
    timeout: int = 10  # seconds
    critical: bool = False
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric data."""

    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Metrics collection and aggregation system.

    Features:
    - Prometheus metrics integration
    - Custom metric tracking
    - Performance monitoring
    - System resource monitoring
    - Business metrics tracking
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = get_logger(__name__)

        # Metrics storage
        self.custom_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Prometheus metrics
        if HAS_PROMETHEUS:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()

        # Performance tracking
        self.request_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, int] = defaultdict(int)

        # System monitoring
        self.system_stats = {}
        self.last_system_check = None

        self.logger.info("Metrics collector initialized")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # HTTP request metrics
        self.http_requests_total = Counter(
            "fm_llm_http_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self.http_request_duration_seconds = Histogram(
            "fm_llm_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # Model operation metrics
        self.model_operations_total = Counter(
            "fm_llm_model_operations_total",
            "Total number of model operations",
            ["operation", "model_name", "status"],
            registry=self.registry,
        )

        self.model_operation_duration_seconds = Histogram(
            "fm_llm_model_operation_duration_seconds",
            "Model operation duration in seconds",
            ["operation", "model_name"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_operations_total = Counter(
            "fm_llm_cache_operations_total",
            "Total number of cache operations",
            ["operation", "status"],
            registry=self.registry,
        )

        self.cache_hit_rate = Gauge(
            "fm_llm_cache_hit_rate", "Cache hit rate", registry=self.registry
        )

        # Database metrics
        self.database_operations_total = Counter(
            "fm_llm_database_operations_total",
            "Total number of database operations",
            ["operation", "table", "status"],
            registry=self.registry,
        )

        self.database_connection_pool_size = Gauge(
            "fm_llm_database_connection_pool_size",
            "Database connection pool size",
            registry=self.registry,
        )

        # System metrics
        if HAS_PSUTIL:
            self.system_cpu_usage = Gauge(
                "fm_llm_system_cpu_usage_percent",
                "System CPU usage percentage",
                registry=self.registry,
            )

            self.system_memory_usage = Gauge(
                "fm_llm_system_memory_usage_bytes",
                "System memory usage in bytes",
                registry=self.registry,
            )

            self.system_disk_usage = Gauge(
                "fm_llm_system_disk_usage_percent",
                "System disk usage percentage",
                registry=self.registry,
            )

        # Application metrics
        self.active_users = Gauge(
            "fm_llm_active_users", "Number of active users", registry=self.registry
        )

        self.certificate_generations_total = Counter(
            "fm_llm_certificate_generations_total",
            "Total number of certificate generations",
            ["status"],
            registry=self.registry,
        )

        self.verification_attempts_total = Counter(
            "fm_llm_verification_attempts_total",
            "Total number of verification attempts",
            ["method", "status"],
            registry=self.registry,
        )

        # Application info
        self.app_info = Info("fm_llm_app_info", "Application information", registry=self.registry)

        # Set application info
        try:
            config = self.config_manager.load_config()
            self.app_info.info(
                {
                    "version": config.get("version", "unknown"),
                    "environment": self.config_manager.environment.value,
                    "build_date": config.get("build_date", "unknown"),
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to set app info: {e}")

    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        if HAS_PROMETHEUS:
            self.http_requests_total.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()

            self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                duration
            )

        # Store in custom metrics
        self.request_durations[endpoint].append(duration)

        if status_code >= 400:
            self.error_counts[f"{method}:{endpoint}"] += 1

    def record_model_operation(
        self, operation: str, model_name: str, status: str, duration: Optional[float] = None
    ):
        """Record model operation metrics."""
        if HAS_PROMETHEUS:
            self.model_operations_total.labels(
                operation=operation, model_name=model_name, status=status
            ).inc()

            if duration is not None:
                self.model_operation_duration_seconds.labels(
                    operation=operation, model_name=model_name
                ).observe(duration)

    def record_cache_operation(self, operation: str, status: str):
        """Record cache operation metrics."""
        if HAS_PROMETHEUS:
            self.cache_operations_total.labels(operation=operation, status=status).inc()

    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate metric."""
        if HAS_PROMETHEUS:
            self.cache_hit_rate.set(hit_rate)

    def record_database_operation(self, operation: str, table: str, status: str):
        """Record database operation metrics."""
        if HAS_PROMETHEUS:
            self.database_operations_total.labels(
                operation=operation, table=table, status=status
            ).inc()

    def record_certificate_generation(self, status: str):
        """Record certificate generation metrics."""
        if HAS_PROMETHEUS:
            self.certificate_generations_total.labels(status=status).inc()

    def record_verification_attempt(self, method: str, status: str):
        """Record verification attempt metrics."""
        if HAS_PROMETHEUS:
            self.verification_attempts_total.labels(method=method, status=status).inc()

    def update_active_users(self, count: int):
        """Update active users metric."""
        if HAS_PROMETHEUS:
            self.active_users.set(count)

    def record_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record custom metric."""
        metric = PerformanceMetric(
            name=name, value=value, timestamp=datetime.utcnow(), tags=tags or {}
        )

        self.custom_metrics[name].append(metric)
        self.metric_windows[name].append((time.time(), value))

        # Keep only last 24 hours of custom metrics
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.custom_metrics[name] = [m for m in self.custom_metrics[name] if m.timestamp > cutoff]

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not HAS_PSUTIL:
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if HAS_PROMETHEUS:
                self.system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            if HAS_PROMETHEUS:
                self.system_memory_usage.set(memory.used)

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            if HAS_PROMETHEUS:
                self.system_disk_usage.set(disk_percent)

            # Store system stats
            self.system_stats = {
                "cpu_percent": cpu_percent,
                "memory_used": memory.used,
                "memory_percent": memory.percent,
                "disk_used": disk.used,
                "disk_percent": disk_percent,
                "last_updated": datetime.utcnow(),
            }

            self.last_system_check = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "system": self.system_stats,
            "custom_metrics": {},
            "error_counts": dict(self.error_counts),
            "request_stats": {},
        }

        # Custom metrics summary
        for name, metrics in self.custom_metrics.items():
            if metrics:
                values = [m.value for m in metrics[-100:]]  # Last 100 values
                summary["custom_metrics"][name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0,
                }

        # Request stats
        for endpoint, durations in self.request_durations.items():
            if durations:
                summary["request_stats"][endpoint] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                }

        return summary

    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return "# Prometheus not available\n"

        return generate_latest(self.registry)

    def get_prometheus_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST if HAS_PROMETHEUS else "text/plain"


class HealthMonitor:
    """
    Health monitoring system.

    Features:
    - Component health checks
    - Dependency monitoring
    - System health aggregation
    - Health status reporting
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = get_logger(__name__)

        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Setup default health checks
        self._setup_default_health_checks()

        self.logger.info("Health monitor initialized")

    def _setup_default_health_checks(self):
        """Setup default health checks."""

        def database_health_check() -> Dict[str, Any]:
            """Check database health."""
            try:
                db_manager = get_database_manager()
                stats = db_manager.get_database_stats()

                if isinstance(stats, dict) and "error" not in stats:
                    return {
                        "status": HealthStatus.HEALTHY,
                        "response_time": 0.1,  # Placeholder
                        "details": stats,
                    }
                else:
                    return {
                        "status": HealthStatus.UNHEALTHY,
                        "error": stats.get("error", "Unknown database error"),
                        "details": stats,
                    }
            except Exception as e:
                return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

        def cache_health_check() -> Dict[str, Any]:
            """Check cache health."""
            try:
                cache_manager = get_cache_manager()
                health = cache_manager.health_check()

                if health.get("healthy"):
                    return {"status": HealthStatus.HEALTHY, "details": health}
                else:
                    return {
                        "status": HealthStatus.DEGRADED,
                        "error": health.get("error", "Cache not healthy"),
                        "details": health,
                    }
            except Exception as e:
                return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

        def system_resources_check() -> Dict[str, Any]:
            """Check system resources."""
            if not HAS_PSUTIL:
                return {"status": HealthStatus.UNKNOWN, "error": "psutil not available"}

            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                status = HealthStatus.HEALTHY
                warnings = []

                if cpu_percent > 90:
                    status = HealthStatus.DEGRADED
                    warnings.append(f"High CPU usage: {cpu_percent}%")

                if memory.percent > 90:
                    status = HealthStatus.DEGRADED
                    warnings.append(f"High memory usage: {memory.percent}%")

                return {
                    "status": status,
                    "warnings": warnings,
                    "details": {"cpu_percent": cpu_percent, "memory_percent": memory.percent},
                }
            except Exception as e:
                return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

        def basic_health_check() -> Dict[str, Any]:
            """Basic application health check."""
            return {
                "status": HealthStatus.HEALTHY,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {"basic_check": "passed"},
            }

        # Register health checks
        self.register_health_check("database", database_health_check, interval=30, critical=True)

        self.register_health_check("cache", cache_health_check, interval=60, critical=False)

        if HAS_PSUTIL:
            self.register_health_check(
                "system_resources", system_resources_check, interval=30, critical=False
            )

        self.register_health_check("basic", basic_health_check, interval=30, critical=True)

    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], Dict[str, Any]],
        interval: int = 30,
        timeout: int = 10,
        critical: bool = False,
    ):
        """Register a health check."""
        health_check = HealthCheck(
            name=name, check_func=check_func, interval=interval, timeout=timeout, critical=critical
        )

        self.health_checks[name] = health_check
        self.logger.info(f"Registered health check: {name}")

    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {"status": HealthStatus.UNKNOWN, "error": f"Health check not found: {name}"}

        health_check = self.health_checks[name]
        start_time = time.time()

        try:
            result = health_check.check_func()
            duration = time.time() - start_time

            # Update health check
            health_check.last_check = datetime.utcnow()
            health_check.last_status = result.get("status", HealthStatus.UNKNOWN)
            health_check.last_result = result

            # Add metadata
            result["duration"] = duration
            result["timestamp"] = health_check.last_check.isoformat()
            result["check_name"] = name

            # Store in history
            self.health_history[name].append(
                {
                    "timestamp": health_check.last_check,
                    "status": health_check.last_status,
                    "duration": duration,
                    "result": result,
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_result = {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "check_name": name,
            }

            health_check.last_check = datetime.utcnow()
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.last_result = error_result

            self.health_history[name].append(
                {
                    "timestamp": health_check.last_check,
                    "status": HealthStatus.UNHEALTHY,
                    "duration": duration,
                    "result": error_result,
                }
            )

            self.logger.error(f"Health check failed for {name}: {e}")
            return error_result

    def run_all_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        critical_failures = []

        for name in self.health_checks:
            result = self.run_health_check(name)
            results[name] = result

            status = result.get("status", HealthStatus.UNKNOWN)

            # Determine overall status
            if status == HealthStatus.UNHEALTHY:
                if self.health_checks[name].critical:
                    overall_status = HealthStatus.UNHEALTHY
                    critical_failures.append(name)
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            elif status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        return {
            "overall_status": overall_status,
            "critical_failures": critical_failures,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results,
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        summary = {
            "overall_status": HealthStatus.UNKNOWN,
            "total_checks": len(self.health_checks),
            "healthy_checks": 0,
            "degraded_checks": 0,
            "unhealthy_checks": 0,
            "last_check_time": None,
            "checks": {},
        }

        latest_check_time = None
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        for name, health_check in self.health_checks.items():
            if health_check.last_check:
                if latest_check_time is None or health_check.last_check > latest_check_time:
                    latest_check_time = health_check.last_check

                if health_check.last_status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif health_check.last_status == HealthStatus.DEGRADED:
                    degraded_count += 1
                elif health_check.last_status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1

                summary["checks"][name] = {
                    "status": health_check.last_status.value,
                    "last_check": health_check.last_check.isoformat(),
                    "critical": health_check.critical,
                    "interval": health_check.interval,
                }

        summary["healthy_checks"] = healthy_count
        summary["degraded_checks"] = degraded_count
        summary["unhealthy_checks"] = unhealthy_count
        summary["last_check_time"] = latest_check_time.isoformat() if latest_check_time else None

        # Determine overall status
        if unhealthy_count > 0:
            summary["overall_status"] = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            summary["overall_status"] = HealthStatus.DEGRADED
        elif healthy_count > 0:
            summary["overall_status"] = HealthStatus.HEALTHY

        return summary


class MonitoringManager:
    """
    Main monitoring manager that coordinates all monitoring activities.

    Features:
    - Metrics collection and aggregation
    - Health monitoring
    - Performance tracking
    - Alerting capabilities
    - Monitoring dashboard data
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = get_logger(__name__)

        # Initialize components
        self.metrics_collector = MetricsCollector(config_manager)
        self.health_monitor = HealthMonitor(config_manager)

        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False

        self.logger.info("Monitoring manager initialized")

    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Background monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Background monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        last_system_update = 0
        last_health_check = 0

        while self.monitoring_active:
            try:
                current_time = time.time()

                # Update system metrics every 30 seconds
                if current_time - last_system_update > 30:
                    self.metrics_collector.update_system_metrics()
                    last_system_update = current_time

                # Run health checks every 60 seconds
                if current_time - last_health_check > 60:
                    self.health_monitor.run_all_health_checks()
                    last_health_check = current_time

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error

    @contextmanager
    def measure_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to measure operation duration."""
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.metrics_collector.record_custom_metric(
                f"{operation_name}_duration", duration, tags
            )
        except Exception:
            duration = time.time() - start_time
            self.metrics_collector.record_custom_metric(
                f"{operation_name}_error_duration", duration, tags
            )
            raise

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "health": self.health_monitor.get_health_summary(),
            "metrics": self.metrics_collector.get_metrics_summary(),
            "system": self.metrics_collector.system_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics."""
        return self.metrics_collector.export_prometheus_metrics()

    def get_prometheus_content_type(self) -> str:
        """Get Prometheus content type."""
        return self.metrics_collector.get_prometheus_content_type()


# Global monitoring manager instance
_monitoring_manager = None


def get_monitoring_manager() -> MonitoringManager:
    """Get the global monitoring manager instance."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager


def start_monitoring():
    """Start global monitoring."""
    get_monitoring_manager().start_monitoring()


def stop_monitoring():
    """Stop global monitoring."""
    if _monitoring_manager:
        _monitoring_manager.stop_monitoring()
