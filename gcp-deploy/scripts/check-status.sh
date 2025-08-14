#!/bin/bash

# Status checking script for FM-LLM Solver deployment
# Checks health of all GCP services

set -e

PROJECT_ID=${1:-"fm-llm-solver"}
REGION=${2:-"us-central1"}

echo "🔍 Checking FM-LLM Solver deployment status"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check gcloud auth
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud"
    exit 1
fi

gcloud config set project $PROJECT_ID

echo "📊 Cloud Run Services:"
echo "====================="

# Check backend service
if gcloud run services describe fmgen-api --region=$REGION --quiet 2>/dev/null; then
    API_URL=$(gcloud run services describe fmgen-api --region=$REGION --format="value(status.url)")
    API_READY=$(gcloud run services describe fmgen-api --region=$REGION --format="value(status.conditions[0].status)")
    echo "✅ fmgen-api: $API_READY"
    echo "   URL: $API_URL"
    
    # Test API health
    if curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" | grep -q "200"; then
        echo "   Health: ✅ OK"
    else
        echo "   Health: ❌ Failed"
    fi
else
    echo "❌ fmgen-api: Not deployed"
fi

echo ""

# Check frontend service  
if gcloud run services describe fmgen-ui --region=$REGION --quiet 2>/dev/null; then
    UI_URL=$(gcloud run services describe fmgen-ui --region=$REGION --format="value(status.url)")
    UI_READY=$(gcloud run services describe fmgen-ui --region=$REGION --format="value(status.conditions[0].status)")
    echo "✅ fmgen-ui: $UI_READY"
    echo "   URL: $UI_URL"
    
    # Test UI health
    if curl -s -o /dev/null -w "%{http_code}" "$UI_URL/health" | grep -q "200"; then
        echo "   Health: ✅ OK"
    else
        echo "   Health: ❌ Failed"
    fi
else
    echo "❌ fmgen-ui: Not deployed"
fi

echo ""
echo "🗄️ Database Services:"
echo "===================="

# Check Cloud SQL
if gcloud sql instances describe fmgen-postgres --quiet 2>/dev/null; then
    DB_STATE=$(gcloud sql instances describe fmgen-postgres --format="value(state)")
    DB_IP=$(gcloud sql instances describe fmgen-postgres --format="value(ipAddresses[0].ipAddress)")
    echo "✅ Cloud SQL: $DB_STATE"
    echo "   IP: $DB_IP"
else
    echo "❌ Cloud SQL: Not created"
fi

# Check Redis
if gcloud redis instances describe fmgen-redis --region=$REGION --quiet 2>/dev/null; then
    REDIS_STATE=$(gcloud redis instances describe fmgen-redis --region=$REGION --format="value(state)")
    REDIS_HOST=$(gcloud redis instances describe fmgen-redis --region=$REGION --format="value(host)")
    echo "✅ Redis: $REDIS_STATE"
    echo "   Host: $REDIS_HOST"
else
    echo "❌ Redis: Not created"
fi

echo ""
echo "🔐 Secrets:"
echo "==========="

# Check secrets
SECRETS=("db-user" "db-password" "jwt-secret" "anthropic-api-key" "redis-url")
for secret in "${SECRETS[@]}"; do
    if gcloud secrets describe $secret --quiet 2>/dev/null; then
        VERSION=$(gcloud secrets versions list $secret --limit=1 --format="value(name)")
        echo "✅ $secret: version $VERSION"
    else
        echo "❌ $secret: Not created"
    fi
done

echo ""
echo "🌐 Domain Mappings:"
echo "=================="

# Check domain mappings
if gcloud run domain-mappings describe --domain=fmgen.net --region=$REGION --quiet 2>/dev/null; then
    DOMAIN_STATUS=$(gcloud run domain-mappings describe --domain=fmgen.net --region=$REGION --format="value(status.conditions[0].status)")
    echo "✅ fmgen.net: $DOMAIN_STATUS"
    
    # Test domain
    if curl -s -o /dev/null -w "%{http_code}" "https://fmgen.net/health" | grep -q "200"; then
        echo "   HTTPS Health: ✅ OK"
    else
        echo "   HTTPS Health: ❌ Failed (DNS propagation may be pending)"
    fi
else
    echo "❌ fmgen.net: Not mapped"
fi

if gcloud run domain-mappings describe --domain=api.fmgen.net --region=$REGION --quiet 2>/dev/null; then
    API_DOMAIN_STATUS=$(gcloud run domain-mappings describe --domain=api.fmgen.net --region=$REGION --format="value(status.conditions[0].status)")
    echo "✅ api.fmgen.net: $API_DOMAIN_STATUS"
    
    # Test API domain
    if curl -s -o /dev/null -w "%{http_code}" "https://api.fmgen.net/health" | grep -q "200"; then
        echo "   HTTPS Health: ✅ OK"
    else
        echo "   HTTPS Health: ❌ Failed (DNS propagation may be pending)"
    fi
else
    echo "❌ api.fmgen.net: Not mapped"
fi

echo ""
echo "📈 Recent Activity:"
echo "=================="

echo "Last 5 Cloud Run deployments:"
gcloud run revisions list --service=fmgen-api --region=$REGION --limit=3 --format="table(metadata.name,status.conditions[0].lastTransitionTime,spec.containers[0].image)"

echo ""
echo "🔍 Quick Access URLs:"
echo "===================="
echo "Production: https://fmgen.net"
echo "API: https://api.fmgen.net"
echo "Cloud Console: https://console.cloud.google.com/run?project=$PROJECT_ID"

echo ""
echo "Status check completed! ✅"
