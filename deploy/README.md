# FM-LLM Solver Deployment Guide

This guide covers deploying the FM-LLM Solver to Google Cloud Platform (GCP) and local development setup.

## Prerequisites

### GCP Deployment
- Google Cloud SDK installed and configured
- Docker installed
- GCP project with billing enabled
- Anthropic API key

### Local Development
- Docker and Docker Compose installed
- Node.js 18+ and npm
- Anthropic API key

## Quick Start - Local Development

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd FM-LLM-Solver

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Start all services
cd deploy
docker-compose up -d
```

### 2. Initialize Database

```bash
# Run migrations
docker-compose exec backend npm run db:migrate

# Optional: Seed with test data
docker-compose exec backend npm run db:seed
```

### 3. Access the Application

- Frontend: http://localhost:3001
- Backend API: http://localhost:3000
- API Documentation: http://localhost:3000/api

## Production Deployment - GCP

### 1. Prerequisites Setup

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Set environment variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. Automated Deployment

```bash
cd deploy
./gcp-setup.sh
```

This script will:
- Enable required GCP APIs
- Create Cloud SQL PostgreSQL instance
- Set up Secret Manager secrets
- Build and deploy backend to Cloud Run
- Build and deploy frontend to Cloud Run
- Run database migrations
- Configure networking and security

### 3. Manual Deployment Steps

If you prefer manual deployment:

#### Create Secrets
```bash
# Anthropic API Key
echo -n "your-anthropic-api-key" | gcloud secrets create fm-llm-anthropic-api-key --data-file=-

# JWT Secret
openssl rand -base64 32 | gcloud secrets create fm-llm-jwt-secret --data-file=-

# Database Password
openssl rand -base64 20 | gcloud secrets create fm-llm-db-password --data-file=-
```

#### Create Cloud SQL Instance
```bash
gcloud sql instances create fm-llm-postgres \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --storage-type=SSD \
  --storage-size=10GB
```

#### Deploy Backend
```bash
cd ../backend

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/fm-llm-backend

# Deploy to Cloud Run
gcloud run deploy fm-llm-backend \
  --image gcr.io/$PROJECT_ID/fm-llm-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-secrets ANTHROPIC_API_KEY=fm-llm-anthropic-api-key:latest \
  --set-secrets JWT_SECRET=fm-llm-jwt-secret:latest \
  --add-cloudsql-instances $PROJECT_ID:us-central1:fm-llm-postgres
```

#### Deploy Frontend
```bash
cd ../frontend

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/fm-llm-frontend

# Deploy to Cloud Run
gcloud run deploy fm-llm-frontend \
  --image gcr.io/$PROJECT_ID/fm-llm-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi
```

## Configuration

### Environment Variables

#### Backend
- `NODE_ENV`: Environment (development/production)
- `PORT`: Server port (default: 3000)
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret for JWT token signing
- `ANTHROPIC_API_KEY`: Anthropic API key
- `FRONTEND_URL`: Frontend URL for CORS

#### Frontend
- `VITE_API_URL`: Backend API URL

### Database Configuration

The application uses PostgreSQL with the following schema:
- Users and authentication
- System specifications
- Experiments and runs
- Certificate candidates
- Verification results
- Audit logs

### Security Configuration

- JWT authentication with secure secrets
- CORS configured for frontend domain
- Rate limiting on API endpoints
- Helmet.js security headers
- Non-root container users

## Monitoring and Maintenance

### Health Checks

Both services include health check endpoints:
- Backend: `GET /health`
- Frontend: `GET /health`

### Logging

```bash
# View backend logs
gcloud run services logs tail fm-llm-backend --region=us-central1

# View frontend logs
gcloud run services logs tail fm-llm-frontend --region=us-central1
```

### Scaling

```bash
# Scale backend
gcloud run services update fm-llm-backend \
  --max-instances 20 \
  --min-instances 1 \
  --region us-central1

# Scale down for maintenance
gcloud run services update fm-llm-backend \
  --max-instances 0 \
  --region us-central1
```

### Database Maintenance

```bash
# Connect to database
gcloud sql connect fm-llm-postgres --user=postgres

# Create backup
gcloud sql backups create --instance=fm-llm-postgres

# View backups
gcloud sql backups list --instance=fm-llm-postgres
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Check Cloud SQL instance is running
   - Verify Cloud SQL proxy configuration
   - Check database credentials in secrets

2. **API Key Issues**
   - Verify Anthropic API key in Secret Manager
   - Check secret IAM permissions
   - Validate API key format

3. **CORS Issues**
   - Check FRONTEND_URL environment variable
   - Verify domain configuration
   - Check CORS middleware setup

4. **Migration Issues**
   - Run migrations manually: `npm run db:migrate`
   - Check database user permissions
   - Verify connection string format

### Debugging

```bash
# Check service status
gcloud run services describe fm-llm-backend --region=us-central1

# View recent logs
gcloud run services logs read fm-llm-backend --region=us-central1 --limit=50

# Check secrets
gcloud secrets versions list fm-llm-anthropic-api-key

# Test database connection
gcloud sql connect fm-llm-postgres --user=postgres
```

## Development Workflow

### Local Development
1. Start services: `docker-compose up -d`
2. Run tests: `cd backend && npm test`
3. Make changes and test
4. Commit and push

### Deployment
1. Test locally with `docker-compose`
2. Deploy to staging: `./gcp-setup.sh`
3. Run integration tests
4. Deploy to production

### Database Migrations
```bash
# Create new migration
cd backend
npx knex migrate:make migration_name

# Run migrations
npm run db:migrate

# Rollback if needed
npm run db:rollback
```

## Performance Optimization

### Backend Optimization
- Use connection pooling for database
- Implement caching with Redis
- Optimize LLM API calls
- Monitor memory usage

### Frontend Optimization
- Enable gzip compression
- Use CDN for static assets
- Implement service worker
- Optimize bundle size

### Database Optimization
- Add appropriate indexes
- Monitor query performance
- Regular maintenance and vacuuming
- Connection pooling

## Security Best Practices

1. **Secrets Management**
   - Use Secret Manager for all sensitive data
   - Rotate secrets regularly
   - Limit secret access permissions

2. **Network Security**
   - Use HTTPS only
   - Configure VPC if needed
   - Implement WAF rules

3. **Application Security**
   - Regular dependency updates
   - Security headers
   - Input validation
   - Rate limiting

4. **Database Security**
   - Private IP for Cloud SQL
   - Regular backups
   - Access logging
   - Encrypted connections

## Backup and Recovery

### Automated Backups
- Cloud SQL automatic backups (daily)
- Point-in-time recovery enabled
- Retention period: 7 days

### Manual Backup
```bash
# Export database
gcloud sql export sql fm-llm-postgres gs://bucket/backup.sql \
  --database=fm_llm_solver

# Import database
gcloud sql import sql fm-llm-postgres gs://bucket/backup.sql \
  --database=fm_llm_solver
```

### Disaster Recovery
1. Database restoration from backup
2. Container image redeployment
3. Secret recovery from backups
4. DNS and domain reconfiguration

## Cost Optimization

### Resource Management
- Use appropriate instance sizes
- Implement auto-scaling
- Monitor usage patterns
- Clean up unused resources

### Cost Monitoring
```bash
# View billing
gcloud billing accounts list
gcloud billing projects describe $PROJECT_ID

# Set budget alerts
gcloud billing budgets create --billing-account=ACCOUNT_ID
```

## Support and Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)

For technical support, please open an issue in the repository or contact the development team.
