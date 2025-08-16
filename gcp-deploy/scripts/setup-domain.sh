#!/bin/bash

# Domain setup script for fmgen.net
# Run this after the main deployment is complete

set -e

PROJECT_ID=${1:-"fm-llm-solver"}
REGION=${2:-"us-central1"}
DOMAIN="fmgen.net"
API_DOMAIN="api.fmgen.net"

echo "🌐 Setting up custom domain: $DOMAIN"

# Check if domain is verified
echo "📋 Checking domain verification..."
if ! gcloud domains list-user-verified-domains | grep -q $DOMAIN; then
    echo "⚠️  Domain $DOMAIN needs to be verified"
    echo "Please verify your domain ownership first:"
    echo "https://console.cloud.google.com/apis/credentials/domainverification"
    echo ""
    echo "Add this TXT record to your domain DNS:"
    gcloud domains verify $DOMAIN
    echo ""
    echo "After verification, run this script again."
    exit 1
fi

# Create domain mappings
echo "🔗 Creating domain mappings..."

# Map main domain to frontend
if ! gcloud run domain-mappings describe --domain=$DOMAIN --region=$REGION --quiet 2>/dev/null; then
    gcloud run domain-mappings create \
        --service=fmgen-ui \
        --domain=$DOMAIN \
        --region=$REGION
fi

# Map API subdomain to backend
if ! gcloud run domain-mappings describe --domain=$API_DOMAIN --region=$REGION --quiet 2>/dev/null; then
    gcloud run domain-mappings create \
        --service=fmgen-api \
        --domain=$API_DOMAIN \
        --region=$REGION
fi

# Get DNS record requirements
echo "📝 DNS Configuration Required:"
echo ""
echo "Add these DNS records to your domain registrar:"
echo ""

# Get the required DNS records
UI_RECORDS=$(gcloud run domain-mappings describe --domain=$DOMAIN --region=$REGION --format="value(status.resourceRecords[].rrdata)")
API_RECORDS=$(gcloud run domain-mappings describe --domain=$API_DOMAIN --region=$REGION --format="value(status.resourceRecords[].rrdata)")

echo "For $DOMAIN (A records):"
for record in $UI_RECORDS; do
    echo "  A    @    $record"
done

echo ""
echo "For $API_DOMAIN (A records):"
for record in $API_RECORDS; do
    echo "  A    api    $record"
done

echo ""
echo "Additional recommended DNS records:"
echo "  CNAME  www    $DOMAIN"
echo ""

# Check SSL certificate status
echo "🔐 SSL Certificate Status:"
UI_CERT_STATUS=$(gcloud run domain-mappings describe --domain=$DOMAIN --region=$REGION --format="value(status.conditions[].message)")
API_CERT_STATUS=$(gcloud run domain-mappings describe --domain=$API_DOMAIN --region=$REGION --format="value(status.conditions[].message)")

echo "Main domain ($DOMAIN): $UI_CERT_STATUS"
echo "API domain ($API_DOMAIN): $API_CERT_STATUS"

echo ""
echo "✅ Domain mapping completed!"
echo ""
echo "📋 Next steps:"
echo "1. Add the DNS records shown above to your domain registrar"
echo "2. Wait for DNS propagation (up to 48 hours)"
echo "3. SSL certificates will be automatically provisioned once DNS is configured"
echo "4. Your site will be available at: https://$DOMAIN"
echo "5. API will be available at: https://$API_DOMAIN"
echo ""
echo "🔍 Monitor status with:"
echo "gcloud run domain-mappings describe --domain=$DOMAIN --region=$REGION"
echo "gcloud run domain-mappings describe --domain=$API_DOMAIN --region=$REGION"

