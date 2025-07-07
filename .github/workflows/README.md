# GitHub Actions Workflows

This directory contains the CI/CD workflows for the FM-LLM Solver project.

## Workflows

### 1. CI (`ci.yml`)
- **Trigger**: Push to main/develop, PRs to main, daily schedule
- **Purpose**: Continuous Integration testing
- **Jobs**:
  - **Lint**: Code quality checks using pre-commit hooks
  - **Test**: Unit and integration tests across multiple Python versions (3.8-3.11) and OS (Ubuntu, Windows, macOS)
  - **Test GPU**: GPU-specific tests (Ubuntu only)
  - **Build**: Package distribution build and validation
  - **Docker**: Docker image build verification
  - **Docs**: Documentation build verification
  - **Security**: Security vulnerability scanning
  - **Release**: Automated releases on version tags

### 2. CI/CD Pipeline (`ci-cd.yml`)
- **Trigger**: Push to main/develop, PRs to main, releases
- **Purpose**: Full CI/CD pipeline with deployment
- **Jobs**:
  - **Test**: Comprehensive testing with database services
  - **Security**: Security scanning and vulnerability checks
  - **Build**: Multi-platform Docker image builds (web, cli, development)
  - **Deploy Staging**: Automatic deployment to staging on develop branch
  - **Deploy Production**: Deployment to production on releases
  - **Performance Test**: Load testing on staging deployments

### 3. Documentation (`docs.yml`)
- **Trigger**: Push to main, manual workflow dispatch
- **Purpose**: Build and deploy documentation to GitHub Pages
- **Jobs**:
  - **Build**: Sphinx documentation build
  - **Deploy**: GitHub Pages deployment

## Required Secrets

### General
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Deployment (ci-cd.yml)
- `KUBE_CONFIG_STAGING`: Kubernetes config for staging cluster
- `KUBE_CONFIG_PRODUCTION`: Kubernetes config for production cluster
- `SLACK_WEBHOOK_URL`: Slack webhook for deployment notifications

## Required Variables

### Deployment URLs (ci-cd.yml)
- `STAGING_URL`: Staging environment URL (default: https://staging.fm-llm-solver.example.com)
- `PRODUCTION_URL`: Production environment URL (default: https://fm-llm-solver.example.com)

## Setup Instructions

1. **Enable GitHub Actions**: Ensure Actions are enabled in your repository settings

2. **Configure Secrets**:
   ```bash
   # Add secrets via GitHub UI or CLI
   gh secret set KUBE_CONFIG_STAGING < ~/.kube/staging-config
   gh secret set KUBE_CONFIG_PRODUCTION < ~/.kube/production-config
   gh secret set SLACK_WEBHOOK_URL
   ```

3. **Configure Variables**:
   ```bash
   # Add variables via GitHub UI or CLI
   gh variable set STAGING_URL --body "https://staging.your-domain.com"
   gh variable set PRODUCTION_URL --body "https://your-domain.com"
   ```

4. **Enable GitHub Pages** (for documentation):
   - Go to Settings → Pages
   - Source: GitHub Actions
   - Branch: main

## Workflow Dependencies

### Python Dependencies
All workflows use the project's dependencies defined in:
- `requirements.txt`: Core dependencies
- `web_requirements.txt`: Web-specific dependencies
- `pyproject.toml`: Package metadata and optional dependencies

### Docker Images
- Base images: Python 3.10-slim
- Multi-stage builds for optimized images
- Platforms: linux/amd64, linux/arm64

## Troubleshooting

### Common Issues

1. **Test Failures**:
   - Check Python version compatibility
   - Ensure all dependencies are properly specified
   - Verify database services are running

2. **Documentation Build Failures**:
   - Ensure all .md files referenced in index.rst exist
   - Check for Sphinx syntax errors
   - Verify myst-parser is installed

3. **Deployment Failures**:
   - Verify Kubernetes credentials are valid
   - Check namespace and resource permissions
   - Ensure Docker images are successfully built

### Running Workflows Locally

You can test workflows locally using [act](https://github.com/nektos/act):

```bash
# Test CI workflow
act -j lint

# Test with specific Python version
act -j test -e '{"matrix": {"python-version": "3.10"}}'
```

## Maintenance

### Updating Dependencies
1. Update version in requirements files
2. Test locally
3. Create PR - CI will validate changes
4. Merge after approval

### Adding New Workflows
1. Create new .yml file in .github/workflows/
2. Follow existing patterns for consistency
3. Document in this README
4. Test thoroughly before merging 