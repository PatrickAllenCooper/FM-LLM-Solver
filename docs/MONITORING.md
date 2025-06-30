# Monitoring

FM-LLM Solver includes comprehensive monitoring for tracking usage, costs, and performance.

## Access

- **Dashboard**: http://localhost:5000/monitoring/dashboard
- **Regular Users**: See personal statistics only
- **Admins**: Full system metrics and cost analysis

## Metrics Tracked

### Usage
- Total requests and success rates
- Active users (daily, weekly, monthly)
- Average generation and verification times
- Certificate generation history

### Costs (Admin Only)
- GPU hours and costs
- API call costs
- Storage and bandwidth costs
- Cost per user/generation

### System Performance
- CPU, memory, disk usage
- GPU utilization and memory
- Error rates
- Health status

## Configuration

Add to `config/config.yaml`:

```yaml
monitoring:
  costs:
    gpu_cost_per_hour: 0.50
    api_cost_per_1k: 0.02
    storage_cost_per_gb_month: 0.023
    bandwidth_cost_per_gb: 0.09
```

## API Endpoints

### Metrics
- `GET /monitoring/api/metrics/usage?range={today|week|month|all}`
- `GET /monitoring/api/metrics/costs?range={today|week|month}` (admin)
- `GET /monitoring/api/metrics/system` (admin)

### Analytics
- `GET /monitoring/api/history?limit=50`
- `GET /monitoring/api/trending?days=7`
- `GET /monitoring/api/user/<id>` (admin)

### Health & Export
- `GET /monitoring/health`
- `GET /monitoring/api/export?format=json` (admin)

## Features

- **Auto-refresh**: Dashboard updates every 30 seconds
- **Time ranges**: Today, week, month, all-time views
- **Visual charts**: Line graphs for trends, pie charts for costs
- **Export**: Download all metrics as JSON

## Integration

### Prometheus
```python
@monitoring_bp.route('/metrics')
def prometheus_metrics():
    # Return metrics in Prometheus format
```

### Webhooks
```bash
curl -X POST /monitoring/webhook/alert \
  -d '{"type": "high_error_rate", "severity": "critical"}'
```

## Tips

1. Monitor daily for anomalies
2. Review costs weekly
3. Check error rates after deployments
4. Use export for detailed analysis 