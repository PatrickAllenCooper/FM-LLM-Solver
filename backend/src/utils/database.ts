import knex from 'knex';

const config = {
  development: {
    client: 'postgresql',
    connection: {
      database: process.env.DATABASE_NAME || 'fm_llm_solver',
      user: process.env.DATABASE_USER || 'postgres',
      password: process.env.DATABASE_PASSWORD || 'postgres',
      host: process.env.DATABASE_HOST || 'localhost',
      port: parseInt(process.env.DATABASE_PORT || '5432'),
    },
    migrations: {
      directory: '../migrations',
    },
  },
  test: {
    client: 'postgresql',
    connection: {
      database: process.env.DATABASE_NAME || 'fm_llm_solver_test',
      user: process.env.DATABASE_USER || 'postgres',
      password: process.env.DATABASE_PASSWORD || 'postgres',
      host: process.env.DATABASE_HOST || 'localhost',
      port: parseInt(process.env.DATABASE_PORT || '5432'),
    },
    migrations: {
      directory: '../migrations',
    },
  },
  production: {
    client: 'postgresql',
    connection: process.env.DATABASE_URL,
    migrations: {
      directory: '../migrations',
    },
  }
};

const environment = (process.env.NODE_ENV as keyof typeof config) || 'development';
const dbConfig = config[environment];

if (!dbConfig) {
  throw new Error(`Database configuration not found for environment: ${environment}`);
}

export const db = knex(dbConfig);

// Test database connection
export async function testDbConnection(): Promise<boolean> {
  try {
    await db.raw('SELECT 1');
    console.log('Database connection successful');
    return true;
  } catch (error) {
    console.error('Database connection failed:', error);
    return false;
  }
}

// Graceful shutdown
export async function closeDbConnection(): Promise<void> {
  try {
    await db.destroy();
    console.log('Database connection closed');
  } catch (error) {
    console.error('Error closing database connection:', error);
  }
}

// Health check
export async function checkDbHealth(): Promise<{
  connected: boolean;
  latency?: number;
  error?: string;
}> {
  try {
    const start = Date.now();
    await db.raw('SELECT 1');
    const latency = Date.now() - start;
    return { connected: true, latency };
  } catch (error) {
    return { 
      connected: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}
