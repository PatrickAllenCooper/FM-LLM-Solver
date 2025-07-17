#!/bin/bash

# GCP Cost Monitor for FM-LLM-Solver
# Monitors costs and sends alerts when approaching budget limits

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${PROJECT_ID:-fmgen-net-production}"
BUDGET_LIMIT="${BUDGET_LIMIT:-100}"  # Monthly budget in USD
ALERT_THRESHOLD="${ALERT_THRESHOLD:-80}"  # Alert at 80% of budget
REGION="${REGION:-us-central1}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get current billing data
get_billing_data() {
    log_info "Fetching billing data for project: $PROJECT_ID"
    
    # Get billing account ID
    BILLING_ACCOUNT=$(gcloud billing projects describe $PROJECT_ID --format="value(billingAccountName)" | sed 's|billingAccounts/||')
    
    if [ -z "$BILLING_ACCOUNT" ]; then
        log_error "No billing account found for project $PROJECT_ID"
        exit 1
    fi
    
    # Get current month's spend
    CURRENT_MONTH=$(date '+%Y-%m')
    START_DATE="${CURRENT_MONTH}-01"
    END_DATE=$(date '+%Y-%m-%d')
    
    log_info "Checking costs from $START_DATE to $END_DATE"
}

# Calculate current spend
calculate_current_spend() {
    log_info "Calculating current month spend..."
    
    # Create temporary query file
    cat > /tmp/cost_query.sql << EOF
SELECT
  service.description as service,
  location.location as location,
  ROUND(SUM(cost), 2) as cost_usd,
  currency
FROM \`${PROJECT_ID}.cloud_billing_export.gcp_billing_export_v1_${BILLING_ACCOUNT//[-]/}_\`
WHERE DATE(usage_start_time) >= DATE('${START_DATE}')
  AND DATE(usage_start_time) <= DATE('${END_DATE}')
  AND project.id = '${PROJECT_ID}'
GROUP BY service.description, location.location, currency
ORDER BY cost_usd DESC
EOF
    
    # Check if BigQuery export is enabled
    if ! gcloud projects get-iam-policy $PROJECT_ID --flatten="bindings[].members" --format="table(bindings.role)" --filter="bindings.members:*bigquery*" | grep -q "bigquery"; then
        log_warning "BigQuery billing export not enabled. Using alternative cost estimation..."
        estimate_costs_alternative
    else
        log_info "Running BigQuery cost analysis..."
        bq query --use_legacy_sql=false --format=prettyjson < /tmp/cost_query.sql > /tmp/billing_data.json
        rm /tmp/cost_query.sql
        
        # Parse results
        TOTAL_COST=$(jq -r 'map(.cost_usd | tonumber) | add // 0' /tmp/billing_data.json)
        log_info "Current month total: \$${TOTAL_COST}"
    fi
}

# Alternative cost estimation using resource pricing
estimate_costs_alternative() {
    log_info "Estimating costs based on current resources..."
    
    # Initialize total cost
    TOTAL_COST=0
    
    # GKE cluster costs (e2-small preemptible)
    GKE_NODES=$(kubectl get nodes --no-headers 2>/dev/null | wc -l || echo "2")
    GKE_HOURLY_COST=0.0134  # e2-small preemptible cost per hour
    HOURS_IN_MONTH=$(date -d "$(date +'%Y-%m-01') +1 month -1 day" +'%d')
    HOURS_IN_MONTH=$((HOURS_IN_MONTH * 24))
    GKE_MONTHLY_COST=$(echo "$GKE_NODES * $GKE_HOURLY_COST * $HOURS_IN_MONTH" | bc -l)
    
    # Cloud SQL costs (db-f1-micro)
    SQL_MONTHLY_COST=25.00  # Approximate cost for db-f1-micro
    
    # Redis costs (basic tier, 1GB)
    REDIS_MONTHLY_COST=15.00  # Approximate cost for basic Redis
    
    # Load Balancer costs
    LB_MONTHLY_COST=8.00  # Approximate cost for load balancer
    
    # Estimate total
    TOTAL_COST=$(echo "$GKE_MONTHLY_COST + $SQL_MONTHLY_COST + $REDIS_MONTHLY_COST + $LB_MONTHLY_COST" | bc -l)
    
    log_info "Estimated monthly costs:"
    log_info "  GKE Cluster ($GKE_NODES nodes): \$$(printf '%.2f' $GKE_MONTHLY_COST)"
    log_info "  Cloud SQL: \$$SQL_MONTHLY_COST"
    log_info "  Redis: \$$REDIS_MONTHLY_COST"
    log_info "  Load Balancer: \$$LB_MONTHLY_COST"
    log_info "  Estimated Total: \$$(printf '%.2f' $TOTAL_COST)"
}

# Check budget alerts
check_budget_alerts() {
    log_info "Checking budget alerts..."
    
    COST_PERCENTAGE=$(echo "scale=1; $TOTAL_COST / $BUDGET_LIMIT * 100" | bc -l)
    ALERT_AMOUNT=$(echo "scale=2; $BUDGET_LIMIT * $ALERT_THRESHOLD / 100" | bc -l)
    
    echo ""
    echo "💰 Cost Summary:"
    echo "   Monthly Budget: \$$BUDGET_LIMIT"
    echo "   Current Spend: \$$(printf '%.2f' $TOTAL_COST)"
    echo "   Percentage Used: $(printf '%.1f' $COST_PERCENTAGE)%"
    echo "   Alert Threshold: \$$(printf '%.2f' $ALERT_AMOUNT) (${ALERT_THRESHOLD}%)"
    echo ""
    
    # Check if we're over the alert threshold
    if (( $(echo "$TOTAL_COST > $ALERT_AMOUNT" | bc -l) )); then
        log_warning "🚨 BUDGET ALERT: Current spend (\$$(printf '%.2f' $TOTAL_COST)) exceeds ${ALERT_THRESHOLD}% of monthly budget!"
        
        # Suggest cost optimization actions
        echo ""
        echo "💡 Cost Optimization Suggestions:"
        echo "   1. Scale down deployments: kubectl scale deployment fm-llm-web --replicas=1 -n fm-llm-prod"
        echo "   2. Check resource usage: kubectl top pods -n fm-llm-prod"
        echo "   3. Review unused resources: gcloud compute disks list --filter='status:READY AND -users:*'"
        echo "   4. Consider preemptible instances for non-critical workloads"
        
        return 1
    else
        log_success "✅ Budget is within safe limits"
        return 0
    fi
}

# Get resource usage stats
get_resource_usage() {
    log_info "Getting current resource usage..."
    
    echo ""
    echo "📊 Resource Usage:"
    
    # Kubernetes resources
    if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        echo ""
        echo "🏗️  Kubernetes Cluster:"
        kubectl get nodes --no-headers | wc -l | xargs echo "   Nodes:"
        kubectl get pods -n fm-llm-prod --no-headers 2>/dev/null | wc -l | xargs echo "   Pods:"
        
        echo ""
        echo "📈 Pod Resource Usage:"
        kubectl top pods -n fm-llm-prod 2>/dev/null || echo "   Metrics not available"
        
        echo ""
        echo "🎯 HPA Status:"
        kubectl get hpa -n fm-llm-prod 2>/dev/null || echo "   HPA not found"
    fi
    
    # GCP resources
    echo ""
    echo "☁️  GCP Resources:"
    
    # Cloud SQL instances
    SQL_INSTANCES=$(gcloud sql instances list --format="value(name)" 2>/dev/null | wc -l)
    echo "   Cloud SQL Instances: $SQL_INSTANCES"
    
    # Redis instances
    REDIS_INSTANCES=$(gcloud redis instances list --region=$REGION --format="value(name)" 2>/dev/null | wc -l)
    echo "   Redis Instances: $REDIS_INSTANCES"
    
    # Load balancers
    LBS=$(gcloud compute forwarding-rules list --format="value(name)" 2>/dev/null | wc -l)
    echo "   Load Balancers: $LBS"
    
    # Persistent disks
    DISKS=$(gcloud compute disks list --format="value(name)" 2>/dev/null | wc -l)
    echo "   Persistent Disks: $DISKS"
}

# Optimize costs automatically
optimize_costs() {
    log_info "Running automatic cost optimization..."
    
    # Scale down if low usage
    if command -v kubectl &> /dev/null; then
        CURRENT_HOUR=$(date +%H)
        
        # Scale down during low traffic hours (2 AM - 6 AM)
        if [ "$CURRENT_HOUR" -ge 2 ] && [ "$CURRENT_HOUR" -le 6 ]; then
            log_info "Low traffic hours detected, scaling down..."
            kubectl scale deployment fm-llm-web --replicas=1 -n fm-llm-prod 2>/dev/null || true
        fi
    fi
    
    # Clean up unused resources
    log_info "Checking for unused resources..."
    
    # List unattached disks
    UNUSED_DISKS=$(gcloud compute disks list --filter="status:READY AND -users:*" --format="value(name)" 2>/dev/null)
    if [ ! -z "$UNUSED_DISKS" ]; then
        log_warning "Found unused disks: $UNUSED_DISKS"
        echo "   Consider deleting with: gcloud compute disks delete [DISK_NAME] --zone=[ZONE]"
    fi
    
    # List reserved IP addresses
    UNUSED_IPS=$(gcloud compute addresses list --filter="status:RESERVED AND -users:*" --format="value(name)" 2>/dev/null)
    if [ ! -z "$UNUSED_IPS" ]; then
        log_warning "Found unused IP addresses: $UNUSED_IPS"
        echo "   Consider releasing with: gcloud compute addresses delete [IP_NAME] --region=[REGION]"
    fi
}

# Generate cost report
generate_report() {
    local report_file="/tmp/gcp_cost_report_$(date +%Y%m%d_%H%M%S).json"
    
    log_info "Generating cost report: $report_file"
    
    cat > "$report_file" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project_id": "$PROJECT_ID",
  "billing_account": "$BILLING_ACCOUNT",
  "period": {
    "start_date": "$START_DATE",
    "end_date": "$END_DATE"
  },
  "budget": {
    "limit": $BUDGET_LIMIT,
    "current_spend": $(printf '%.2f' $TOTAL_COST),
    "percentage_used": $(printf '%.1f' $COST_PERCENTAGE),
    "alert_threshold": $ALERT_THRESHOLD,
    "over_threshold": $([ $(echo "$TOTAL_COST > $ALERT_AMOUNT" | bc -l) = 1 ] && echo "true" || echo "false")
  },
  "resources": {
    "gke_nodes": $GKE_NODES,
    "sql_instances": $SQL_INSTANCES,
    "redis_instances": $REDIS_INSTANCES,
    "load_balancers": $LBS,
    "persistent_disks": $DISKS
  },
  "recommendations": [
    "Monitor resource usage during peak hours",
    "Consider using preemptible instances for batch workloads",
    "Implement automatic scaling policies",
    "Regular cleanup of unused resources"
  ]
}
EOF
    
    echo "📄 Cost report saved to: $report_file"
    
    # Show summary
    jq '.' "$report_file" 2>/dev/null || cat "$report_file"
}

# Set up monitoring alerts
setup_monitoring() {
    log_info "Setting up cost monitoring alerts..."
    
    # Create alerting policy for high costs
    cat > /tmp/cost_alert_policy.yaml << EOF
displayName: "High Monthly Spend Alert"
documentation:
  content: "Alert when monthly spending exceeds ${ALERT_THRESHOLD}% of budget"
  mimeType: "text/markdown"
conditions:
  - displayName: "Monthly spend exceeds threshold"
    conditionThreshold:
      filter: 'resource.type="billing_account"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: $ALERT_AMOUNT
      duration: 300s
alertStrategy:
  notificationRateLimit:
    period: 3600s
enabled: true
EOF
    
    # Apply the alert policy
    gcloud alpha monitoring policies create --policy-from-file=/tmp/cost_alert_policy.yaml 2>/dev/null || log_warning "Could not create monitoring policy"
    
    rm /tmp/cost_alert_policy.yaml
}

# Main cost monitoring function
main() {
    echo "💰 FM-LLM-Solver GCP Cost Monitor"
    echo "Project: $PROJECT_ID | Budget: \$$BUDGET_LIMIT/month"
    echo ""
    
    get_billing_data
    calculate_current_spend
    get_resource_usage
    
    if check_budget_alerts; then
        log_success "Cost monitoring complete - all good!"
    else
        log_warning "Budget threshold exceeded - review recommended actions"
        optimize_costs
    fi
    
    if [ "${GENERATE_REPORT:-false}" = "true" ]; then
        generate_report
    fi
}

# Handle script arguments
case "${1:-}" in
    "setup")
        setup_monitoring
        ;;
    "report")
        GENERATE_REPORT=true
        main
        ;;
    "optimize")
        optimize_costs
        ;;
    "emergency-scale-down")
        log_warning "Emergency scale down initiated..."
        kubectl scale deployment fm-llm-web --replicas=0 -n fm-llm-prod 2>/dev/null || true
        gcloud run services update fm-llm-inference --min-instances=0 --region=$REGION 2>/dev/null || true
        log_success "Emergency scale down complete"
        ;;
    "scale-up")
        log_info "Scaling up for high traffic..."
        kubectl scale deployment fm-llm-web --replicas=3 -n fm-llm-prod 2>/dev/null || true
        log_success "Scale up complete"
        ;;
    *)
        main
        ;;
esac 