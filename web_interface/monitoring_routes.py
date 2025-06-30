"""
Monitoring routes for FM-LLM Solver web interface.
"""

from flask import Blueprint, render_template, request, jsonify, abort, g
from flask_login import login_required, current_user
from web_interface.models import db
from web_interface.monitoring import MonitoringService
from web_interface.auth import admin_required, require_api_key
from utils.config_loader import load_config
import json

monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')

# Initialize monitoring service
config = load_config()
monitoring_service = MonitoringService(config)

@monitoring_bp.route('/dashboard')
@login_required
def dashboard():
    """Main monitoring dashboard."""
    # Check if user has access (admin or premium)
    if current_user.role not in ['admin', 'premium']:
        # Regular users see their own stats
        return render_template('monitoring/user_dashboard.html',
                             user_stats=monitoring_service.get_user_statistics(current_user.id))
    
    # Admin/premium users see full dashboard
    return render_template('monitoring/dashboard.html')

@monitoring_bp.route('/api/metrics/usage')
@login_required
def api_usage_metrics():
    """Get usage metrics."""
    time_range = request.args.get('range', 'month')
    
    # Regular users only see their own stats
    if current_user.role not in ['admin', 'premium']:
        user_stats = monitoring_service.get_user_statistics(current_user.id)
        return jsonify(user_stats)
    
    metrics = monitoring_service.get_usage_metrics(time_range)
    return jsonify({
        'time_range': time_range,
        'metrics': {
            'total_requests': metrics.total_requests,
            'successful_generations': metrics.successful_generations,
            'failed_generations': metrics.failed_generations,
            'success_rate': (metrics.successful_generations / metrics.total_requests * 100) 
                          if metrics.total_requests > 0 else 0,
            'total_users': metrics.total_users,
            'active_users': {
                'today': metrics.active_users_today,
                'week': metrics.active_users_week,
                'month': metrics.active_users_month
            },
            'avg_generation_time': round(metrics.avg_generation_time, 2),
            'avg_verification_time': round(metrics.avg_verification_time, 2),
            'api_vs_web': {
                'api_calls': metrics.total_api_calls,
                'web_calls': metrics.total_web_calls
            }
        }
    })

@monitoring_bp.route('/api/metrics/costs')
@admin_required
def api_cost_metrics():
    """Get cost metrics (admin only)."""
    time_range = request.args.get('range', 'month')
    metrics = monitoring_service.get_cost_metrics(time_range)
    
    return jsonify({
        'time_range': time_range,
        'metrics': {
            'gpu': {
                'hours': round(metrics.gpu_hours, 2),
                'cost': round(metrics.gpu_cost, 2)
            },
            'api': {
                'calls': metrics.api_calls,
                'cost': round(metrics.api_cost, 2)
            },
            'storage': {
                'gb': round(metrics.storage_gb, 2),
                'cost': round(metrics.storage_cost, 2)
            },
            'bandwidth': {
                'gb': round(metrics.bandwidth_gb, 2),
                'cost': round(metrics.bandwidth_cost, 2)
            },
            'total_cost': round(metrics.total_cost, 2),
            'cost_per_user': round(metrics.cost_per_user, 2),
            'cost_per_generation': round(metrics.cost_per_generation, 2)
        }
    })

@monitoring_bp.route('/api/metrics/system')
@admin_required
def api_system_metrics():
    """Get system performance metrics (admin only)."""
    metrics = monitoring_service.get_system_metrics()
    
    return jsonify({
        'cpu': {
            'percent': round(metrics.cpu_percent, 1)
        },
        'memory': {
            'percent': round(metrics.memory_percent, 1),
            'used_gb': round(metrics.memory_used_gb, 2),
            'total_gb': round(metrics.memory_total_gb, 2)
        },
        'disk': {
            'percent': round(metrics.disk_percent, 1),
            'used_gb': round(metrics.disk_used_gb, 2),
            'total_gb': round(metrics.disk_total_gb, 2)
        },
        'gpu': {
            'utilization': round(metrics.gpu_utilization, 1),
            'memory_percent': round(metrics.gpu_memory_percent, 1),
            'memory_used_gb': round(metrics.gpu_memory_used_gb, 2)
        },
        'active_sessions': metrics.active_sessions,
        'error_rate': round(metrics.error_rate, 2)
    })

@monitoring_bp.route('/api/history')
@login_required
def api_certificate_history():
    """Get certificate generation history."""
    limit = request.args.get('limit', 50, type=int)
    
    # Regular users only see their own history
    if current_user.role not in ['admin', 'premium']:
        history = monitoring_service.get_certificate_history(
            user_id=current_user.id,
            limit=limit
        )
    else:
        # Admin/premium can see all or filter by user
        user_id = request.args.get('user_id', type=int)
        history = monitoring_service.get_certificate_history(
            user_id=user_id,
            limit=limit
        )
    
    return jsonify({
        'history': history,
        'count': len(history),
        'limit': limit
    })

@monitoring_bp.route('/api/trending')
@login_required
def api_trending_systems():
    """Get trending system types."""
    days = request.args.get('days', 7, type=int)
    limit = request.args.get('limit', 10, type=int)
    
    trending = monitoring_service.get_trending_systems(days, limit)
    
    return jsonify({
        'trending': trending,
        'period_days': days
    })

@monitoring_bp.route('/api/user/<int:user_id>')
@admin_required
def api_user_statistics(user_id):
    """Get detailed statistics for a specific user (admin only)."""
    stats = monitoring_service.get_user_statistics(user_id)
    
    if not stats:
        abort(404, 'User not found')
    
    return jsonify(stats)

@monitoring_bp.route('/api/export')
@admin_required
def api_export_metrics():
    """Export all metrics (admin only)."""
    format = request.args.get('format', 'json')
    
    data = monitoring_service.export_metrics(format)
    
    response = jsonify(json.loads(data)) if format == 'json' else data
    
    # Add download headers
    response.headers['Content-Disposition'] = f'attachment; filename=metrics_export.{format}'
    
    return response

# Public API endpoints with API key authentication

@monitoring_bp.route('/api/v1/metrics')
@require_api_key
def api_v1_metrics():
    """Public API endpoint for metrics (requires API key)."""
    user = g.current_api_user
    
    # API users get limited metrics
    return jsonify({
        'user': {
            'username': user.username,
            'daily_limit': user.daily_request_limit,
            'requests_today': user.daily_request_count,
            'requests_remaining': user.daily_request_limit - user.daily_request_count
        },
        'usage': monitoring_service.get_user_statistics(user.id)['statistics']
    })

@monitoring_bp.route('/api/v1/history')
@require_api_key
def api_v1_history():
    """Public API endpoint for certificate history (requires API key)."""
    user = g.current_api_user
    limit = request.args.get('limit', 20, type=int)
    
    history = monitoring_service.get_certificate_history(
        user_id=user.id,
        limit=min(limit, 100)  # Cap at 100
    )
    
    return jsonify({
        'history': history,
        'count': len(history)
    })

# Webhooks for monitoring alerts

@monitoring_bp.route('/webhook/alert', methods=['POST'])
def webhook_alert():
    """Webhook endpoint for monitoring alerts."""
    # This could be called by external monitoring services
    # or by internal alert mechanisms
    
    data = request.get_json()
    alert_type = data.get('type')
    severity = data.get('severity', 'info')
    message = data.get('message')
    
    # Log the alert
    from web_interface.auth import log_security_event
    log_security_event(f'monitoring_alert_{alert_type}', 
                      severity=severity,
                      description=message)
    
    # Could trigger additional actions here:
    # - Send email notifications
    # - Post to Slack
    # - Create incident tickets
    
    return jsonify({'status': 'received'}), 200

# Health check endpoint

@monitoring_bp.route('/health')
def health_check():
    """Health check endpoint for monitoring services."""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        db_status = 'healthy'
    except:
        db_status = 'unhealthy'
    
    # Get basic system metrics
    system_metrics = monitoring_service.get_system_metrics()
    
    health_status = {
        'status': 'healthy' if db_status == 'healthy' else 'degraded',
        'database': db_status,
        'cpu_percent': system_metrics.cpu_percent,
        'memory_percent': system_metrics.memory_percent,
        'error_rate': system_metrics.error_rate
    }
    
    # Return appropriate status code
    status_code = 200 if health_status['status'] == 'healthy' else 503
    
    return jsonify(health_status), status_code 