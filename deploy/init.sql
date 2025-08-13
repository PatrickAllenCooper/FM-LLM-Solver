-- Initial database setup for FM-LLM Solver
-- This script is run when the PostgreSQL container starts

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE fm_llm_solver'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'fm_llm_solver')\gexec

-- Create user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'fm_llm_user') THEN
        CREATE USER fm_llm_user WITH PASSWORD 'fm_llm_password';
    END IF;
END
$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE fm_llm_solver TO fm_llm_user;

-- Connect to the database
\c fm_llm_solver;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO fm_llm_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fm_llm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fm_llm_user;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO fm_llm_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO fm_llm_user;
