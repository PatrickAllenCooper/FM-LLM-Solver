# FM-LLM Solver

A rigorous evaluation platform for Large Language Models (LLMs) in proposing Lyapunov functions and barrier certificates for dynamical systems analysis.

## Overview

FM-LLM Solver provides a comprehensive framework for:
- Evaluating LLM capabilities in formal verification tasks
- Generating and verifying mathematical certificates (Lyapunov functions, barrier certificates)
- Comparing LLM-generated solutions against classical baselines (SOS, SDP)
- Ensuring reproducible and traceable experiments with complete provenance

## Architecture

### Backend (Node.js/TypeScript)
- **API Server**: REST API with WebSocket support
- **Database**: PostgreSQL with JSONB for mathematical data
- **LLM Integration**: Anthropic Claude API with structured prompting
- **Verification**: Mathematical verification using symbolic and numerical methods
- **Authentication**: JWT-based auth with role-based access control

### Frontend (React/TypeScript)
- **SPA**: Modern React application with Tailwind CSS
- **State Management**: Zustand for auth and app state
- **API Client**: Axios with React Query for data fetching
- **UI Components**: Headless UI with custom components

### Database Schema
- `users` - User accounts with role-based permissions
- `system_specs` - Dynamical system specifications
- `candidates` - Generated certificate candidates
- `counterexamples` - Verification failures with context
- `audit_events` - Complete audit trail
- `experiment_runs` - Batch experiment tracking

## Prerequisites

- Node.js 18+ 
- PostgreSQL 12+
- Anthropic API key

## Quick Start

### 1. Database Setup

```bash
# Create PostgreSQL database
createdb fmgen_dev
createdb fmgen_test
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Copy environment configuration
cp env.sample .env

# Edit .env with your configuration:
# - Database credentials
# - ANTHROPIC_API_KEY
# - JWT_SECRET (change in production)

# Run database migrations
npm run db:migrate

# Start development server
npm run dev
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access the Application

- Frontend: http://localhost:3001
- Backend API: http://localhost:3000
- API Health: http://localhost:3000/health

## Environment Variables

### Backend (.env)
```bash
# Server
PORT=3000
NODE_ENV=development
FRONTEND_URL=http://localhost:3001

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fmgen_dev
DB_USER=postgres
DB_PASSWORD=postgres

# Authentication
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=24h

# LLM Integration
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Logging
LOG_LEVEL=info
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user
- `POST /api/auth/change-password` - Change password

### System Specifications
- `POST /api/system-specs` - Create system specification
- `GET /api/system-specs` - List system specifications

### Certificate Generation
- `POST /api/certificates/generate` - Generate certificate
- `GET /api/certificates` - List candidates
- `GET /api/certificates/:id` - Get candidate details

## Usage Examples

### 1. Create a System Specification

```json
{
  "name": "Van der Pol Oscillator",
  "description": "Classic nonlinear oscillator",
  "system_type": "continuous",
  "dimension": 2,
  "dynamics": {
    "type": "nonlinear",
    "variables": ["x1", "x2"],
    "equations": [
      "x2",
      "mu * (1 - x1^2) * x2 - x1"
    ],
    "domain": {
      "bounds": {
        "x1": {"min": -3, "max": 3},
        "x2": {"min": -3, "max": 3}
      }
    }
  }
}
```

### 2. Generate Lyapunov Function

```json
{
  "system_spec_id": "uuid-of-system-spec",
  "certificate_type": "lyapunov",
  "generation_method": "llm",
  "llm_config": {
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.0,
    "max_tokens": 2048,
    "mode": "direct_expression"
  }
}
```

## Development

### Running Tests
```bash
# Backend tests
cd backend && npm test

# Frontend tests  
cd frontend && npm test
```

### Code Quality
```bash
# Backend linting
cd backend && npm run lint

# Frontend linting
cd frontend && npm run lint
```

### Database Operations
```bash
cd backend

# Create new migration
npx knex migrate:make migration_name

# Run migrations
npm run db:migrate

# Rollback migration
npm run db:rollback

# Seed database
npm run db:seed
```

## Deployment

### Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
1. Set up PostgreSQL database
2. Configure environment variables
3. Build applications:
   ```bash
   npm run build:backend
   npm run build:frontend
   ```
4. Deploy to your hosting platform (GCP Cloud Run, etc.)

## LLM Integration

### Supported Providers
- **Anthropic Claude**: Primary provider with structured JSON output
- **OpenAI GPT**: Planned support
- **Local Models**: Planned support via Ollama

### Certificate Types
- **Lyapunov Functions**: For stability analysis
- **Barrier Functions**: For safety verification  
- **Inductive Invariants**: For invariant synthesis

### Verification Methods
- **Symbolic**: Pattern-based verification (MVP)
- **Numerical**: Monte Carlo sampling
- **SMT Solvers**: Z3 integration (planned)
- **SOS/SDP**: Classical baseline methods

## Security Features

- JWT-based authentication
- Role-based access control (Admin/Researcher/Viewer)
- API key management via secret manager
- Complete audit logging
- Input validation and sanitization
- Rate limiting

## Monitoring & Observability

- Health check endpoints
- Structured logging with Winston
- Performance metrics tracking
- Error reporting and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Research Context

This platform supports research in:
- LLM capabilities for formal verification
- Automated certificate synthesis
- Comparison of AI vs classical methods
- Reproducible formal methods research

## Support

For issues and questions:
- GitHub Issues: Report bugs and feature requests
- Documentation: See `/docs` directory
- API Reference: Available at `/api` endpoint when running

---

**Note**: This is an MVP implementation. Production deployment requires additional security hardening, monitoring, and optimization.
