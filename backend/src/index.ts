import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import * as dotenv from 'dotenv';
import { testDbConnection, closeDbConnection, checkDbHealth } from '@/utils/database';
import { logger } from '@/utils/logger';
import { AuthController } from '@/controllers/auth.controller';
import { CertificateController } from '@/controllers/certificate.controller';
import { AuthMiddleware } from '@/middleware/auth.middleware';

// Load environment variables
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Initialize controllers and middleware
const authController = new AuthController();
const certificateController = new CertificateController();
const authMiddleware = new AuthMiddleware();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false, // Disable for API
}));

// CORS configuration
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3001',
  credentials: true,
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: process.env.NODE_ENV === 'production' ? 100 : 1000, // requests per window
  message: {
    success: false,
    error: 'Too many requests from this IP',
    timestamp: new Date().toISOString(),
  },
});
app.use('/api/', limiter);

// Body parsing middleware
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging middleware
app.use((req, res, next) => {
  logger.info('HTTP Request', {
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
  });
  next();
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const dbHealth = await checkDbHealth();
    
    const health = {
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      database: dbHealth,
      environment: process.env.NODE_ENV || 'development',
    };

    res.status(dbHealth.connected ? 200 : 503).json(health);
  } catch (error) {
    logger.error('Health check failed', {
      error: error instanceof Error ? error.message : 'Unknown error',
    });

    res.status(503).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
    });
  }
});

// API Routes

// Authentication routes
app.post('/api/auth/register', authController.register);
app.post('/api/auth/login', authController.login);
app.get('/api/auth/me', authMiddleware.authenticate, authController.me);
app.post('/api/auth/change-password', authMiddleware.authenticate, authController.changePassword);

// System specification routes
app.post('/api/system-specs', 
  authMiddleware.authenticate, 
  authMiddleware.authorize(['admin', 'researcher']),
  certificateController.createSystemSpec
);
app.get('/api/system-specs', 
  authMiddleware.optionalAuth,
  certificateController.getSystemSpecs
);

// Certificate generation routes
app.post('/api/certificates/generate',
  authMiddleware.authenticate,
  authMiddleware.authorize(['admin', 'researcher']),
  certificateController.generateCertificate
);
app.get('/api/certificates',
  authMiddleware.optionalAuth,
  certificateController.getCandidates
);
app.get('/api/certificates/:id',
  authMiddleware.optionalAuth,
  certificateController.getCandidateById
);

// API documentation endpoint
app.get('/api', (req, res) => {
  res.json({
    name: 'FM-LLM Solver API',
    version: '1.0.0',
    description: 'API for rigorous evaluation of LLMs for proposing Lyapunov functions and barrier certificates',
    endpoints: {
      authentication: {
        'POST /api/auth/register': 'Register a new user',
        'POST /api/auth/login': 'Login user',
        'GET /api/auth/me': 'Get current user info',
        'POST /api/auth/change-password': 'Change password',
      },
      systemSpecs: {
        'POST /api/system-specs': 'Create system specification',
        'GET /api/system-specs': 'List system specifications',
      },
      certificates: {
        'POST /api/certificates/generate': 'Generate certificate',
        'GET /api/certificates': 'List candidates',
        'GET /api/certificates/:id': 'Get candidate details',
      },
    },
    timestamp: new Date().toISOString(),
  });
});

// Error handling middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error', {
    error: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
  });

  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === 'production' ? 'Internal server error' : err.message,
    timestamp: new Date().toISOString(),
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    timestamp: new Date().toISOString(),
  });
});

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('Received SIGINT, shutting down gracefully');
  await closeDbConnection();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  await closeDbConnection();
  process.exit(0);
});

// Start server
async function startServer() {
  try {
    // Test database connection
    const dbConnected = await testDbConnection();
    if (!dbConnected) {
      logger.error('Failed to connect to database');
      process.exit(1);
    }

    app.listen(port, () => {
      logger.info(`FM-LLM Solver API server running on port ${port}`, {
        environment: process.env.NODE_ENV || 'development',
        port,
      });
    });
  } catch (error) {
    logger.error('Failed to start server', {
      error: error instanceof Error ? error.message : 'Unknown error',
    });
    process.exit(1);
  }
}

startServer();
