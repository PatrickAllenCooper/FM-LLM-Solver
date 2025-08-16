import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import * as dotenv from 'dotenv';
import { testDbConnection } from '@/utils/database';
import { logger } from '@/utils/logger';
import { SimpleController } from '@/controllers/simple.controller';
import { AuthController } from '@/controllers/auth.controller';
import { CertificateFirestoreController } from '@/controllers/certificate.firestore';
import { AuthMiddleware } from '@/middleware/auth.middleware';

// Load environment variables
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Initialize controllers and middleware
const simpleController = new SimpleController();
const authController = new AuthController();
const certificateController = new CertificateFirestoreController();
const authMiddleware = new AuthMiddleware();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false, // Disable for API
}));

// CORS configuration
app.use(cors({
  origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:5173'],
  credentials: true,
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Compression
app.use(compression());

// Request logging
app.use((req, res, next) => {
  logger.info('Request received', {
    method: req.method,
    url: req.url,
    userAgent: req.get('User-Agent'),
  });
  next();
});

// Health check endpoint using Firestore
app.get('/health', simpleController.health);

// Firestore test endpoint
app.get('/api/test', simpleController.test);

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
app.get('/api/system-specs/:id', 
  authMiddleware.optionalAuth,
  certificateController.getSystemSpecById
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
  certificateController.getCandidate
);

// API info endpoint
app.get('/api', (req, res) => {
  res.json({
    name: 'FM-LLM Solver API',
    version: '1.0.0',
    status: 'running',
    database: 'firestore',
    timestamp: new Date().toISOString(),
  });
});

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    timestamp: new Date().toISOString(),
  });
});

// Start server
async function startServer() {
  try {
    // Test Firestore connection
    const dbConnected = await testDbConnection();
    
    if (!dbConnected) {
      logger.error('Failed to connect to Firestore - stopping server');
      process.exit(1);
    }

    app.listen(port, () => {
      logger.info(`🚀 FM-LLM Solver API server started`, {
        port,
        environment: process.env.NODE_ENV || 'development',
        database: 'firestore',
        timestamp: new Date().toISOString()
      });
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});
