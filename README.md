# FM-LLM Solver

A research platform for evaluating Large Language Models in formal verification tasks, specifically generating and verifying Lyapunov functions and barrier certificates for dynamical systems.

## Overview

FM-LLM Solver enables researchers to:
- Define dynamical systems through an intuitive web interface
- Generate formal verification certificates using LLM-based and traditional methods
- Compare LLM performance against baseline approaches
- Analyze verification results and system performance

## Architecture

**Frontend**: React with TypeScript, Material Design 3, Tailwind CSS
**Backend**: Node.js with Express, Firestore database (PostgreSQL planned)
**LLM Integration**: Anthropic Claude API with structured prompting
**Authentication**: JWT-based with Firebase Auth and role-based access control
**Deployment**: Docker containers with Docker Compose, GCP Cloud Run ready

## Prerequisites

- Docker and Docker Compose
- Anthropic API key

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp backend/env.sample backend/.env

# Edit backend/.env with your Anthropic API key:
ANTHROPIC_API_KEY=your-key-here
```

### 2. Deploy with Docker

```bash
cd deploy
docker compose up --build
```

### 3. Access the Application

- Frontend: http://localhost:3001
- Backend API: http://localhost:3000

## How to Use

### 1. Create System Specification
- Define system name, type (continuous/discrete/hybrid), and dimension
- Specify state variables and differential equations
- Set domain constraints and variable bounds
- Define initial and unsafe sets

### 2. Generate Certificates
- Choose certificate type (Lyapunov function or barrier certificate)
- Select generation method (LLM Direct, LLM SOS, or baseline methods)
- Configure template and generation parameters

### 3. Analyze Results
- Review verification status and mathematical validity
- Compare performance between different methods
- Export results for further analysis

## Key Features

- **Multi-step System Definition**: Intuitive wizard interface for complex systems
- **LLM Integration**: Direct integration with Anthropic Claude API with structured prompting
- **Intelligent Validation**: Comprehensive form validation with detailed error feedback
- **Material Design 3**: Modern, professional user interface with CU Boulder branding
- **Real-time Verification**: Automatic verification of generated certificates
- **Research-Grade Analysis**: Comprehensive technical details and experimental parameter controls
- **Provenance Tracking**: Complete audit trail showing who generated what and when

## Recent Improvements (v2.0.1)

- **Security Hardening**: All npm audit vulnerabilities resolved (backend and frontend)
- **Code Quality**: ESLint configurations added for TypeScript and React codebases
- **Production Ready**: Console.log statements replaced with proper Winston logging
- **Codebase Cleanup**: Removed unused backup files and dead code (6,500+ lines cleaned)
- **Build Optimization**: Replaced vulnerable tscpaths with secure tsc-alias
- **Deployment Reliability**: Enhanced Cloud Run startup probes for robust deployment

## Development

### Local Development Setup
```bash
# Backend
cd backend
npm install
cp env.sample .env
npm run db:migrate
npm run dev

# Frontend
cd frontend
npm install
npm run dev
```

### Docker Development
```bash
cd deploy
docker compose up --build
```

## Technology Stack

**Frontend**
- React 18 with TypeScript
- Material Design 3 with Tailwind CSS
- React Hook Form with Zod validation
- TanStack Query for state management
- Vite for build tooling

**Backend**
- Node.js 18+ with Express.js and TypeScript
- Firestore database (PostgreSQL migration planned)
- Anthropic Claude API integration with structured prompting
- JWT authentication with Firebase Auth and bcrypt
- Winston logging with comprehensive error handling
- ESLint with TypeScript rules for code quality

**Infrastructure**
- Docker containers with multi-stage builds
- Nginx for frontend serving in production
- GCP Cloud Run deployment ready
- ESLint and security vulnerability management

## Author

**Patrick Cooper**  
University of Colorado Boulder  
patrick.cooper@colorado.edu

## License

MIT License
