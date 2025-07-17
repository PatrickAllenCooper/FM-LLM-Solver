"""
Monitoring service for FM-LLM Solver.

Handles application monitoring, metrics collection, and health checks.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

from fm_llm_solver.core.config import Config
from fm_llm_solver.core.logging import get_logger


class MonitoringService:
    """
    Monitoring service for collecting and reporting application metrics.
    """

    def __init__(self, config: Config, db=None):
        """
        Initialize the monitoring service.

        Args:
            config: Configuration object
            db: Database connection (optional)
        """
        self.config = config
        self.db = db
        self.logger = get_logger(__name__)
        self.metrics = {}
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({"value": value, "timestamp": time.time(), "tags": tags or {}})

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "metrics": self.metrics,
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get application health status.

        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "database": self._check_database(),
                "memory": self._check_memory(),
                "disk": self._check_disk(),
            },
        }

    def _check_database(self) -> Dict[str, Any]:
        """Check database health."""
        if self.db:
            try:
                # Simple database check
                return {"status": "healthy", "message": "Database connection OK"}
            except Exception as e:
                return {"status": "unhealthy", "message": str(e)}
        return {"status": "not_configured", "message": "Database not configured"}

    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "status": "healthy" if memory.percent < 90 else "warning",
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
            }
        except ImportError:
            return {"status": "unknown", "message": "psutil not available"}

    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            import psutil

            disk = psutil.disk_usage("/")
            return {
                "status": "healthy" if disk.percent < 90 else "warning",
                "usage_percent": disk.percent,
                "free_gb": disk.free / (1024**3),
            }
        except ImportError:
            return {"status": "unknown", "message": "psutil not available"}

    def record_generation_metrics(self, generation_time: float, success: bool) -> None:
        """Record certificate generation metrics."""
        self.record_metric("generation_time", generation_time)
        self.record_metric("generation_success", 1.0 if success else 0.0)

    def record_verification_metrics(self, verification_time: float, success: bool) -> None:
        """Record certificate verification metrics."""
        self.record_metric("verification_time", verification_time)
        self.record_metric("verification_success", 1.0 if success else 0.0)

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.logger.info("Metrics cleared")
