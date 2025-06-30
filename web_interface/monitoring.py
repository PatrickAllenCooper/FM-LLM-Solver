"""
Monitoring module for FM-LLM Solver web interface.
Tracks usage, costs, certificate generation history, and system metrics.
"""

import os
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import psutil
import GPUtil

from flask import current_app
from sqlalchemy import func, and_, or_, extract
from web_interface.models import db, User, QueryLog, RateLimitLog, SecurityLog, VerificationResult

@dataclass
class UsageMetrics:
    """Container for usage metrics."""
    total_requests: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    total_users: int = 0
    active_users_today: int = 0
    active_users_week: int = 0
    active_users_month: int = 0
    avg_generation_time: float = 0.0
    avg_verification_time: float = 0.0
    total_api_calls: int = 0
    total_web_calls: int = 0

@dataclass
class CostMetrics:
    """Container for cost tracking."""
    gpu_hours: float = 0.0
    gpu_cost: float = 0.0
    api_calls: int = 0
    api_cost: float = 0.0
    storage_gb: float = 0.0
    storage_cost: float = 0.0
    bandwidth_gb: float = 0.0
    bandwidth_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_user: float = 0.0
    cost_per_generation: float = 0.0

@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_utilization: float = 0.0
    active_sessions: int = 0
    queue_size: int = 0
    error_rate: float = 0.0

class MonitoringService:
    """Service for monitoring and analytics."""
    
    def __init__(self, config):
        """Initialize monitoring service."""
        self.config = config
        self.cost_config = {
            'gpu_cost_per_hour': 0.50,  # Default GPU cost per hour
            'api_cost_per_1k': 0.02,    # Cost per 1000 API calls
            'storage_cost_per_gb_month': 0.023,  # S3 standard pricing
            'bandwidth_cost_per_gb': 0.09,  # Data transfer cost
        }
        
        # Update with config if available
        if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'costs'):
            self.cost_config.update(config.monitoring.costs)
        
        # Track GPU usage sessions
        self.gpu_sessions = {}
        
    def get_usage_metrics(self, time_range: str = 'all') -> UsageMetrics:
        """Get usage metrics for specified time range."""
        metrics = UsageMetrics()
        
        # Time range filters
        now = datetime.utcnow()
        if time_range == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_range == 'week':
            start_date = now - timedelta(days=7)
        elif time_range == 'month':
            start_date = now - timedelta(days=30)
        else:  # 'all'
            start_date = None
        
        # Build query filters
        query_filter = []
        if start_date:
            query_filter.append(QueryLog.timestamp >= start_date)
        
        # Total requests
        metrics.total_requests = db.session.query(QueryLog).filter(*query_filter).count()
        
        # Successful vs failed generations
        metrics.successful_generations = db.session.query(QueryLog).filter(
            *query_filter,
            QueryLog.status == 'completed',
            QueryLog.generated_certificate.isnot(None)
        ).count()
        
        metrics.failed_generations = db.session.query(QueryLog).filter(
            *query_filter,
            or_(QueryLog.status == 'failed', QueryLog.generated_certificate.is_(None))
        ).count()
        
        # User metrics
        metrics.total_users = db.session.query(User).count()
        
        # Active users (made at least one request)
        today = date.today()
        metrics.active_users_today = db.session.query(func.count(func.distinct(QueryLog.user_id))).filter(
            func.date(QueryLog.timestamp) == today
        ).scalar() or 0
        
        week_ago = datetime.utcnow() - timedelta(days=7)
        metrics.active_users_week = db.session.query(func.count(func.distinct(QueryLog.user_id))).filter(
            QueryLog.timestamp >= week_ago
        ).scalar() or 0
        
        month_ago = datetime.utcnow() - timedelta(days=30)
        metrics.active_users_month = db.session.query(func.count(func.distinct(QueryLog.user_id))).filter(
            QueryLog.timestamp >= month_ago
        ).scalar() or 0
        
        # Average times
        completed_queries = db.session.query(QueryLog).filter(
            *query_filter,
            QueryLog.status == 'completed',
            QueryLog.processing_start.isnot(None),
            QueryLog.processing_end.isnot(None)
        ).all()
        
        if completed_queries:
            total_gen_time = sum((q.processing_end - q.processing_start).total_seconds() 
                               for q in completed_queries)
            metrics.avg_generation_time = total_gen_time / len(completed_queries)
        
        # Verification times
        verifications = db.session.query(VerificationResult).filter(
            VerificationResult.verification_time_seconds.isnot(None)
        ).all()
        
        if verifications:
            metrics.avg_verification_time = sum(v.verification_time_seconds for v in verifications) / len(verifications)
        
        # API vs Web calls (simplified - you might track this differently)
        # For now, assume queries with conversation_id are web, others are API
        metrics.total_web_calls = db.session.query(QueryLog).filter(
            *query_filter,
            QueryLog.conversation_id.isnot(None)
        ).count()
        
        metrics.total_api_calls = metrics.total_requests - metrics.total_web_calls
        
        return metrics
    
    def get_cost_metrics(self, time_range: str = 'month') -> CostMetrics:
        """Calculate cost metrics."""
        metrics = CostMetrics()
        
        # Time range
        if time_range == 'today':
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_range == 'week':
            start_date = datetime.utcnow() - timedelta(days=7)
        else:  # month
            start_date = datetime.utcnow() - timedelta(days=30)
        
        # GPU hours (estimate based on generation times)
        completed_queries = db.session.query(QueryLog).filter(
            QueryLog.timestamp >= start_date,
            QueryLog.status == 'completed',
            QueryLog.processing_start.isnot(None),
            QueryLog.processing_end.isnot(None)
        ).all()
        
        total_gpu_seconds = sum((q.processing_end - q.processing_start).total_seconds() 
                               for q in completed_queries)
        metrics.gpu_hours = total_gpu_seconds / 3600
        metrics.gpu_cost = metrics.gpu_hours * self.cost_config['gpu_cost_per_hour']
        
        # API calls
        metrics.api_calls = db.session.query(QueryLog).filter(
            QueryLog.timestamp >= start_date,
            QueryLog.conversation_id.is_(None)  # Assuming API calls don't have conversation
        ).count()
        metrics.api_cost = (metrics.api_calls / 1000) * self.cost_config['api_cost_per_1k']
        
        # Storage (estimate)
        # Count KB size and documents
        kb_size_gb = self._estimate_kb_storage()
        db_size_gb = self._estimate_db_storage()
        metrics.storage_gb = kb_size_gb + db_size_gb
        
        # Monthly storage cost (prorated for shorter periods)
        days_in_period = (datetime.utcnow() - start_date).days or 1
        metrics.storage_cost = (metrics.storage_gb * self.cost_config['storage_cost_per_gb_month'] * 
                               days_in_period / 30)
        
        # Bandwidth (rough estimate based on queries)
        # Assume ~1MB per query response
        total_queries = db.session.query(QueryLog).filter(
            QueryLog.timestamp >= start_date
        ).count()
        metrics.bandwidth_gb = (total_queries * 1) / 1024  # 1MB per query in GB
        metrics.bandwidth_cost = metrics.bandwidth_gb * self.cost_config['bandwidth_cost_per_gb']
        
        # Total cost
        metrics.total_cost = (metrics.gpu_cost + metrics.api_cost + 
                            metrics.storage_cost + metrics.bandwidth_cost)
        
        # Cost per user/generation
        active_users = db.session.query(func.count(func.distinct(QueryLog.user_id))).filter(
            QueryLog.timestamp >= start_date
        ).scalar() or 1
        
        total_generations = len(completed_queries) or 1
        
        metrics.cost_per_user = metrics.total_cost / active_users
        metrics.cost_per_generation = metrics.total_cost / total_generations
        
        return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        metrics = SystemMetrics()
        
        # CPU and Memory
        metrics.cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        metrics.memory_percent = memory.percent
        metrics.memory_used_gb = memory.used / (1024**3)
        metrics.memory_total_gb = memory.total / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        metrics.disk_percent = disk.percent
        metrics.disk_used_gb = disk.used / (1024**3)
        metrics.disk_total_gb = disk.total / (1024**3)
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                metrics.gpu_utilization = gpu.load * 100
                metrics.gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                metrics.gpu_memory_used_gb = gpu.memoryUsed / 1024
        except:
            pass  # GPU not available
        
        # Active sessions (from database)
        metrics.active_sessions = db.session.query(User).filter(
            User.last_login >= datetime.utcnow() - timedelta(minutes=30)
        ).count()
        
        # Error rate (last hour)
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        total_recent = db.session.query(QueryLog).filter(
            QueryLog.timestamp >= hour_ago
        ).count() or 1
        
        failed_recent = db.session.query(QueryLog).filter(
            QueryLog.timestamp >= hour_ago,
            QueryLog.status == 'failed'
        ).count()
        
        metrics.error_rate = (failed_recent / total_recent) * 100
        
        return metrics
    
    def get_certificate_history(self, user_id: Optional[int] = None, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get certificate generation history."""
        query = db.session.query(QueryLog).order_by(QueryLog.timestamp.desc())
        
        if user_id:
            query = query.filter(QueryLog.user_id == user_id)
        
        queries = query.limit(limit).all()
        
        history = []
        for q in queries:
            # Get verification result if exists
            verification = db.session.query(VerificationResult).filter_by(
                query_id=q.id
            ).first()
            
            # Get user info
            user = db.session.query(User).filter_by(id=q.user_id).first() if q.user_id else None
            
            history.append({
                'id': q.id,
                'timestamp': q.timestamp.isoformat(),
                'user': user.username if user else 'Anonymous',
                'user_id': q.user_id,
                'system_description': q.system_description[:100] + '...' if len(q.system_description) > 100 else q.system_description,
                'model_config': q.model_config,
                'status': q.status,
                'certificate': q.generated_certificate,
                'processing_time': q.processing_time,
                'verification': {
                    'overall_success': verification.overall_success if verification else None,
                    'numerical': verification.numerical_check_passed if verification else None,
                    'symbolic': verification.symbolic_check_passed if verification else None,
                    'sos': verification.sos_check_passed if verification else None,
                } if verification else None,
                'error': q.error_message
            })
        
        return history
    
    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get detailed statistics for a specific user."""
        # Basic user info
        user = db.session.query(User).filter_by(id=user_id).first()
        if not user:
            return {}
        
        # Query statistics
        total_queries = db.session.query(QueryLog).filter_by(user_id=user_id).count()
        successful_queries = db.session.query(QueryLog).filter_by(
            user_id=user_id,
            status='completed'
        ).count()
        
        # Time-based stats
        today = date.today()
        queries_today = db.session.query(QueryLog).filter(
            QueryLog.user_id == user_id,
            func.date(QueryLog.timestamp) == today
        ).count()
        
        # Rate limit history
        rate_limit_violations = db.session.query(RateLimitLog).filter_by(
            user_id=user_id,
            was_blocked=True
        ).count()
        
        # Average processing time
        completed = db.session.query(QueryLog).filter(
            QueryLog.user_id == user_id,
            QueryLog.status == 'completed',
            QueryLog.processing_start.isnot(None),
            QueryLog.processing_end.isnot(None)
        ).all()
        
        avg_processing_time = 0
        if completed:
            total_time = sum((q.processing_end - q.processing_start).total_seconds() 
                           for q in completed)
            avg_processing_time = total_time / len(completed)
        
        # Model usage breakdown
        model_usage = db.session.query(
            QueryLog.model_config,
            func.count(QueryLog.id)
        ).filter_by(user_id=user_id).group_by(QueryLog.model_config).all()
        
        return {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'daily_limit': user.daily_request_limit,
                'is_active': user.is_active
            },
            'statistics': {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                'queries_today': queries_today,
                'queries_remaining_today': user.daily_request_limit - user.daily_request_count,
                'rate_limit_violations': rate_limit_violations,
                'avg_processing_time': avg_processing_time,
                'model_usage': dict(model_usage)
            }
        }
    
    def get_trending_systems(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently queried system types."""
        since = datetime.utcnow() - timedelta(days=days)
        
        # This is simplified - in practice you might want to cluster similar systems
        trending = db.session.query(
            QueryLog.system_description,
            func.count(QueryLog.id).label('count')
        ).filter(
            QueryLog.timestamp >= since
        ).group_by(
            QueryLog.system_description
        ).order_by(
            func.count(QueryLog.id).desc()
        ).limit(limit).all()
        
        return [
            {
                'system': t[0][:100] + '...' if len(t[0]) > 100 else t[0],
                'count': t[1]
            }
            for t in trending
        ]
    
    def start_gpu_session(self, session_id: str):
        """Start tracking a GPU usage session."""
        self.gpu_sessions[session_id] = {
            'start_time': time.time(),
            'end_time': None
        }
    
    def end_gpu_session(self, session_id: str) -> float:
        """End GPU usage session and return duration in hours."""
        if session_id in self.gpu_sessions:
            session = self.gpu_sessions[session_id]
            session['end_time'] = time.time()
            duration_hours = (session['end_time'] - session['start_time']) / 3600
            del self.gpu_sessions[session_id]
            return duration_hours
        return 0.0
    
    def _estimate_kb_storage(self) -> float:
        """Estimate knowledge base storage in GB."""
        kb_path = self.config.paths.kb_output_dir
        total_size = 0
        
        if os.path.exists(kb_path):
            for dirpath, dirnames, filenames in os.walk(kb_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        
        return total_size / (1024**3)  # Convert to GB
    
    def _estimate_db_storage(self) -> float:
        """Estimate database storage in GB."""
        db_path = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
        if 'sqlite:///' in db_path:
            db_file = db_path.replace('sqlite:///', '')
            if os.path.exists(db_file):
                return os.path.getsize(db_file) / (1024**3)
        
        # For other databases, use table counts as rough estimate
        # Assume ~1KB per row average
        total_rows = (
            db.session.query(User).count() +
            db.session.query(QueryLog).count() +
            db.session.query(VerificationResult).count() +
            db.session.query(RateLimitLog).count() +
            db.session.query(SecurityLog).count()
        )
        
        return (total_rows * 1024) / (1024**3)  # Rough estimate in GB
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export all metrics in specified format."""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'usage': asdict(self.get_usage_metrics()),
            'costs': asdict(self.get_cost_metrics()),
            'system': asdict(self.get_system_metrics()),
            'trending_systems': self.get_trending_systems(),
            'recent_certificates': self.get_certificate_history(limit=20)
        }
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            # Could add CSV, Prometheus format, etc.
            return json.dumps(data, indent=2, default=str) 