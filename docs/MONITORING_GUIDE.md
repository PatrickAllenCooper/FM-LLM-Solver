# Monitoring System Guide

## Overview

The FM-LLM Solver now includes a comprehensive monitoring solution that tracks usage, costs, certificate generation history, and system performance metrics.

## Features

### 1. Usage Metrics
- Total requests and success rates
- Active users (daily, weekly, monthly)
- Average generation and verification times
- API vs Web interface usage breakdown

### 2. Cost Tracking
- GPU hours and associated costs
- API call costs
- Storage costs
- Bandwidth costs
- Cost per user and per generation

### 3. Certificate History
- Complete history of all certificate generations
- User-specific query tracking
- Verification results
- Processing times and status

### 4. System Performance
- CPU, Memory, and Disk usage
- GPU utilization and memory
- Active sessions
- Error rates
- Health status monitoring

### 5. Trending Analysis
- Most frequently queried system types
- Usage patterns over time

## Access Levels

### Regular Users
- Access to personal dashboard at `/monitoring/dashboard`
- View own usage statistics
- See personal certificate generation history
- Monitor daily request limits and remaining quota

### Admin Users
- Full monitoring dashboard with all metrics
- Cost analysis and breakdown
- System-wide performance metrics
- Export capabilities for all data
- User-specific statistics viewing

## API Endpoints

### Web Interface Endpoints

#### Dashboard
- `GET /monitoring/dashboard` - Main monitoring dashboard

#### Metrics APIs
- `GET /monitoring/api/metrics/usage?range={today|week|month|all}` - Usage metrics
- `GET /monitoring/api/metrics/costs?range={today|week|month}` - Cost metrics (admin only)
- `GET /monitoring/api/metrics/system` - System performance metrics (admin only)

#### History and Analytics
- `GET /monitoring/api/history?limit=50&user_id=123` - Certificate generation history
- `GET /monitoring/api/trending?days=7&limit=10` - Trending systems
- `GET /monitoring/api/user/<user_id>` - User statistics (admin only)

#### Export and Health
- `GET /monitoring/api/export?format=json` - Export all metrics (admin only)
- `GET /monitoring/health` - Health check endpoint

### Public API Endpoints (with API Key)
- `GET /monitoring/api/v1/metrics` - Get personal metrics
- `GET /monitoring/api/v1/history` - Get personal certificate history

## Configuration

Add monitoring configuration to your `config.yaml`:

```yaml
monitoring:
  costs:
    gpu_cost_per_hour: 0.50  # Adjust based on your GPU provider
    api_cost_per_1k: 0.02    # Cost per 1000 API calls
    storage_cost_per_gb_month: 0.023  # Storage costs
    bandwidth_cost_per_gb: 0.09  # Data transfer costs
  
  refresh_interval: 30  # Dashboard auto-refresh in seconds
  
  alerts:
    error_rate_threshold: 10  # Alert if error rate exceeds 10%
    gpu_utilization_threshold: 90  # Alert if GPU usage exceeds 90%
```

## Dashboard Features

### Time Range Selection
- Today: Current day statistics
- Last 7 Days: Weekly overview
- Last 30 Days: Monthly summary
- All Time: Complete historical data

### Real-time Updates
- Dashboard auto-refreshes every 30 seconds
- Manual refresh button available
- Live system metrics

### Visual Analytics
- Line charts for generation trends
- Doughnut chart for cost breakdown
- Progress bars for system resources
- Status badges for certificate generations

## Cost Tracking

The monitoring system tracks costs based on:

1. **GPU Usage**: Calculated from actual processing times
2. **API Calls**: Number of programmatic API requests
3. **Storage**: Knowledge base and database storage
4. **Bandwidth**: Data transfer for responses

### Cost Optimization Tips
- Monitor peak usage times
- Identify inefficient queries
- Track cost per user to identify heavy users
- Use cost breakdown to optimize resources

## Integration with External Monitoring

### Prometheus Integration
Export metrics in Prometheus format by implementing a `/metrics` endpoint:

```python
# Example Prometheus metrics endpoint
@monitoring_bp.route('/metrics')
def prometheus_metrics():
    metrics = monitoring_service.get_usage_metrics()
    return f"""
# HELP llm_total_requests Total number of LLM requests
# TYPE llm_total_requests counter
llm_total_requests {metrics.total_requests}

# HELP llm_success_rate Success rate of LLM generations
# TYPE llm_success_rate gauge
llm_success_rate {metrics.successful_generations / metrics.total_requests}
"""
```

### Webhook Alerts
Configure webhooks for monitoring alerts:

```bash
curl -X POST http://your-domain.com/monitoring/webhook/alert \
  -H "Content-Type: application/json" \
  -d '{
    "type": "high_error_rate",
    "severity": "critical",
    "message": "Error rate exceeded 10%"
  }'
```

## Best Practices

1. **Regular Monitoring**
   - Check dashboard daily for anomalies
   - Monitor cost trends weekly
   - Review error rates after deployments

2. **Capacity Planning**
   - Use historical data for resource planning
   - Monitor growth trends
   - Plan for peak usage periods

3. **Performance Optimization**
   - Identify slow queries
   - Monitor GPU utilization
   - Track memory usage patterns

4. **Cost Management**
   - Set up cost alerts
   - Regular cost reviews
   - Optimize based on usage patterns

## Troubleshooting

### High Error Rates
1. Check system metrics for resource constraints
2. Review recent certificate history for patterns
3. Check health endpoint for system status

### Missing Metrics
1. Ensure monitoring service is initialized
2. Check database connectivity
3. Verify user permissions

### Performance Issues
1. Monitor system resources
2. Check for memory leaks
3. Review query optimization

## Security Considerations

- Monitoring data is access-controlled
- Regular users can only see their own data
- Admin access required for system-wide metrics
- API endpoints require authentication
- Sensitive cost data is admin-only

## Future Enhancements

Planned features for the monitoring system:
- Real-time alerting via email/SMS
- Advanced analytics and ML-based anomaly detection
- Custom dashboards and reports
- Integration with popular monitoring tools
- Predictive cost analysis 