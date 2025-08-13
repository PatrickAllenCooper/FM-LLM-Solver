#!/bin/bash

# FM-LLM Solver Setup Script
# This script helps set up the development environment

set -e

echo "🚀 Setting up FM-LLM Solver Development Environment"
echo "=================================================="

# Check Node.js version
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//')
REQUIRED_VERSION="18.0.0"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Node.js version $NODE_VERSION is too old. Please upgrade to 18.0.0 or higher."
    exit 1
fi

echo "✅ Node.js version $NODE_VERSION is compatible"

# Check PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "⚠️  PostgreSQL is not installed. Please install PostgreSQL 12+ first."
    echo "   On macOS: brew install postgresql"
    echo "   On Ubuntu: sudo apt-get install postgresql postgresql-contrib"
    exit 1
fi

echo "✅ PostgreSQL is available"

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
if [ ! -f ".env" ]; then
    echo "📝 Creating backend environment file..."
    cp env.sample .env
    echo "⚠️  Please edit backend/.env with your configuration (database credentials, API keys, etc.)"
fi

npm install
echo "✅ Backend dependencies installed"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd ../frontend
npm install
echo "✅ Frontend dependencies installed"

# Create databases
echo "🗄️  Setting up databases..."
cd ..

# Check if databases exist and create them if they don't
if ! psql -lqt | cut -d \| -f 1 | grep -qw fmgen_dev; then
    echo "Creating development database..."
    createdb fmgen_dev
fi

if ! psql -lqt | cut -d \| -f 1 | grep -qw fmgen_test; then
    echo "Creating test database..."
    createdb fmgen_test
fi

echo "✅ Databases created"

# Run migrations
echo "🔄 Running database migrations..."
cd backend
npm run db:migrate
echo "✅ Database migrations completed"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your configuration:"
echo "   - Add your ANTHROPIC_API_KEY"
echo "   - Update database credentials if needed"
echo "   - Change JWT_SECRET for production"
echo ""
echo "2. Start the development servers:"
echo "   Terminal 1: cd backend && npm run dev"
echo "   Terminal 2: cd frontend && npm run dev"
echo ""
echo "3. Access the application:"
echo "   Frontend: http://localhost:3001"
echo "   Backend API: http://localhost:3000"
echo "   Health Check: http://localhost:3000/health"
echo ""
echo "📚 See README.md for detailed documentation"
