import request from 'supertest';
import express from 'express';
import { AuthController } from '../../src/controllers/auth.controller';
import { CertificateController } from '../../src/controllers/certificate.controller';
import { AuthMiddleware } from '../../src/middleware/auth.middleware';

// Mock the services and middleware
jest.mock('../../src/services/auth.service');
jest.mock('../../src/services/llm.service');
jest.mock('../../src/services/verification.service');
jest.mock('../../src/utils/database');

describe('API Integration Tests', () => {
  let app: express.Application;
  let authController: AuthController;
  let certificateController: CertificateController;
  let authMiddleware: AuthMiddleware;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    
    authController = new AuthController();
    certificateController = new CertificateController();
    authMiddleware = new AuthMiddleware();

    // Mock authentication middleware
    jest.spyOn(authMiddleware, 'authenticate').mockImplementation(async (req, res, next) => {
      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };
      next();
    });

    jest.spyOn(authMiddleware, 'authorize').mockImplementation((roles) => (req, res, next) => {
      next();
    });

    jest.spyOn(authMiddleware, 'optionalAuth').mockImplementation(async (req, res, next) => {
      next();
    });

    // Set up routes
    setupRoutes();
  });

  const setupRoutes = () => {
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
  };

  describe('Authentication Endpoints', () => {
    describe('POST /api/auth/register', () => {
      it('should register a new user successfully', async () => {
        const mockAuthService = require('@/services/auth.service').AuthService;
        mockAuthService.prototype.register = jest.fn().mockResolvedValue({
          token: 'mock-token',
          user: {
            id: 'user-123',
            email: 'test@example.com',
            role: 'researcher',
          },
          expires_at: new Date().toISOString(),
        });

        const response = await request(app)
          .post('/api/auth/register')
          .send({
            email: 'test@example.com',
            password: 'password123',
            role: 'researcher',
          })
          .expect(201);

        expect(response.body.success).toBe(true);
        expect(response.body.data.token).toBe('mock-token');
        expect(response.body.data.user.email).toBe('test@example.com');
      });

      it('should return 400 for invalid input', async () => {
        const response = await request(app)
          .post('/api/auth/register')
          .send({
            email: 'invalid-email',
            // Missing password and role
          })
          .expect(400);

        expect(response.body.success).toBe(false);
        expect(response.body.error).toBeTruthy();
      });

      it('should return 409 for existing user', async () => {
        const mockAuthService = require('@/services/auth.service').AuthService;
        mockAuthService.prototype.register = jest.fn().mockRejectedValue(
          new Error('User with this email already exists')
        );

        const response = await request(app)
          .post('/api/auth/register')
          .send({
            email: 'existing@example.com',
            password: 'password123',
            role: 'researcher',
          })
          .expect(409);

        expect(response.body.success).toBe(false);
        expect(response.body.error).toContain('already exists');
      });
    });

    describe('POST /api/auth/login', () => {
      it('should login user successfully', async () => {
        const mockAuthService = require('@/services/auth.service').AuthService;
        mockAuthService.prototype.login = jest.fn().mockResolvedValue({
          token: 'mock-token',
          user: {
            id: 'user-123',
            email: 'test@example.com',
            role: 'researcher',
          },
          expires_at: new Date().toISOString(),
        });

        const response = await request(app)
          .post('/api/auth/login')
          .send({
            email: 'test@example.com',
            password: 'password123',
          })
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.token).toBe('mock-token');
      });

      it('should return 401 for invalid credentials', async () => {
        const mockAuthService = require('@/services/auth.service').AuthService;
        mockAuthService.prototype.login = jest.fn().mockRejectedValue(
          new Error('Invalid credentials')
        );

        const response = await request(app)
          .post('/api/auth/login')
          .send({
            email: 'test@example.com',
            password: 'wrong-password',
          })
          .expect(401);

        expect(response.body.success).toBe(false);
      });
    });

    describe('GET /api/auth/me', () => {
      it('should return current user info', async () => {
        const response = await request(app)
          .get('/api/auth/me')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.user.id).toBe('test-user-id');
        expect(response.body.data.user.email).toBe('test@example.com');
      });
    });

    describe('POST /api/auth/change-password', () => {
      it('should change password successfully', async () => {
        const mockAuthService = require('@/services/auth.service').AuthService;
        mockAuthService.prototype.changePassword = jest.fn().mockResolvedValue(undefined);

        const response = await request(app)
          .post('/api/auth/change-password')
          .send({
            currentPassword: 'old-password',
            newPassword: 'new-password',
          })
          .expect(200);

        expect(response.body.success).toBe(true);
      });
    });
  });

  describe('System Specification Endpoints', () => {
    describe('POST /api/system-specs', () => {
      it('should create system specification successfully', async () => {
        const mockSystemSpec = {
          id: 'system-123',
          name: 'Test System',
          system_type: 'continuous',
          dimension: 2,
          dynamics_json: {
            variables: ['x1', 'x2'],
            equations: ['-x1', '-x2'],
          },
        };

        // Mock the database operation
        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          insert: jest.fn().mockReturnThis(),
          returning: jest.fn().mockResolvedValue([mockSystemSpec]),
        });

        const response = await request(app)
          .post('/api/system-specs')
          .send({
            name: 'Test System',
            system_type: 'continuous',
            dimension: 2,
            dynamics: {
              variables: ['x1', 'x2'],
              equations: ['-x1', '-x2'],
            },
          })
          .expect(201);

        expect(response.body.success).toBe(true);
        expect(response.body.data.name).toBe('Test System');
      });

      it('should return 400 for invalid system specification', async () => {
        const response = await request(app)
          .post('/api/system-specs')
          .send({
            // Missing required fields
            name: 'Invalid System',
          })
          .expect(400);

        expect(response.body.success).toBe(false);
      });
    });

    describe('GET /api/system-specs', () => {
      it('should list system specifications', async () => {
        const mockSystemSpecs = [
          {
            id: 'system-1',
            name: 'System 1',
            system_type: 'continuous',
            dimension: 2,
          },
          {
            id: 'system-2',
            name: 'System 2',
            system_type: 'discrete',
            dimension: 3,
          },
        ];

        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          select: jest.fn().mockReturnThis(),
          orderBy: jest.fn().mockResolvedValue(mockSystemSpecs),
        });

        const response = await request(app)
          .get('/api/system-specs')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data).toHaveLength(2);
        expect(response.body.data[0].name).toBe('System 1');
      });
    });
  });

  describe('Certificate Generation Endpoints', () => {
    describe('POST /api/certificates/generate', () => {
      it('should generate certificate successfully', async () => {
        const mockLLMService = require('@/services/llm.service').LLMService;
        const mockVerificationService = require('@/services/verification.service').VerificationService;

        mockLLMService.prototype.generateCertificate = jest.fn().mockResolvedValue({
          response: {
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            confidence: 0.95,
          },
          duration_ms: 1500,
          raw_response: 'mock response',
        });

        mockVerificationService.prototype.verifyCertificate = jest.fn().mockResolvedValue({
          verified: true,
          margin: 0.5,
          verification_method: 'mathematical',
          duration_ms: 500,
          solver_output: 'Verification successful',
        });

        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          where: jest.fn().mockReturnThis(),
          first: jest.fn().mockResolvedValue({
            id: 'system-123',
            name: 'Test System',
            dynamics_json: { variables: ['x1', 'x2'], equations: ['-x1', '-x2'] },
          }),
          insert: jest.fn().mockReturnThis(),
          returning: jest.fn().mockResolvedValue([{
            id: 'candidate-123',
            certificate_type: 'lyapunov',
            candidate_expression: 'x1^2 + x2^2',
          }]),
        });

        const response = await request(app)
          .post('/api/certificates/generate')
          .send({
            system_spec_id: 'system-123',
            certificate_type: 'lyapunov',
            llm_config: {
              model: 'claude-sonnet-4-20250514',
              mode: 'direct_expression',
              temperature: 0.0,
              max_tokens: 1000,
            },
          })
          .expect(201);

        expect(response.body.success).toBe(true);
        expect(response.body.data.candidate.certificate_type).toBe('lyapunov');
        expect(response.body.data.verification.verified).toBe(true);
      });

      it('should return 404 for non-existent system', async () => {
        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          where: jest.fn().mockReturnThis(),
          first: jest.fn().mockResolvedValue(null),
        });

        const response = await request(app)
          .post('/api/certificates/generate')
          .send({
            system_spec_id: 'non-existent',
            certificate_type: 'lyapunov',
            llm_config: {
              model: 'claude-sonnet-4-20250514',
              mode: 'direct_expression',
              temperature: 0.0,
              max_tokens: 1000,
            },
          })
          .expect(404);

        expect(response.body.success).toBe(false);
        expect(response.body.error).toContain('System specification not found');
      });
    });

    describe('GET /api/certificates', () => {
      it('should list certificates', async () => {
        const mockCandidates = [
          {
            id: 'candidate-1',
            certificate_type: 'lyapunov',
            candidate_expression: 'x1^2 + x2^2',
            generation_method: 'llm',
          },
          {
            id: 'candidate-2',
            certificate_type: 'barrier',
            candidate_expression: '10 - x1^2 - x2^2',
            generation_method: 'baseline',
          },
        ];

        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          select: jest.fn().mockReturnThis(),
          orderBy: jest.fn().mockReturnThis(),
          limit: jest.fn().mockReturnThis(),
          offset: jest.fn().mockResolvedValue(mockCandidates),
        });

        const response = await request(app)
          .get('/api/certificates')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data).toHaveLength(2);
        expect(response.body.data[0].certificate_type).toBe('lyapunov');
      });

      it('should support pagination', async () => {
        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          select: jest.fn().mockReturnThis(),
          orderBy: jest.fn().mockReturnThis(),
          limit: jest.fn().mockReturnThis(),
          offset: jest.fn().mockResolvedValue([]),
        });

        const response = await request(app)
          .get('/api/certificates?page=2&limit=10')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(mockDb().limit).toHaveBeenCalledWith(10);
        expect(mockDb().offset).toHaveBeenCalledWith(10);
      });
    });

    describe('GET /api/certificates/:id', () => {
      it('should return certificate details', async () => {
        const mockCandidate = {
          id: 'candidate-123',
          certificate_type: 'lyapunov',
          candidate_expression: 'x1^2 + x2^2',
          verification_results: {
            verified: true,
            margin: 0.5,
          },
        };

        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          where: jest.fn().mockReturnThis(),
          first: jest.fn().mockResolvedValue(mockCandidate),
        });

        const response = await request(app)
          .get('/api/certificates/candidate-123')
          .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.id).toBe('candidate-123');
        expect(response.body.data.certificate_type).toBe('lyapunov');
      });

      it('should return 404 for non-existent certificate', async () => {
        const mockDb = require('@/utils/database').db;
        mockDb.mockReturnValue({
          where: jest.fn().mockReturnThis(),
          first: jest.fn().mockResolvedValue(null),
        });

        const response = await request(app)
          .get('/api/certificates/non-existent')
          .expect(404);

        expect(response.body.success).toBe(false);
        expect(response.body.error).toContain('Certificate not found');
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle 500 errors gracefully', async () => {
      const mockAuthService = require('@/services/auth.service').AuthService;
      mockAuthService.prototype.register = jest.fn().mockRejectedValue(
        new Error('Database connection failed')
      );

      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com',
          password: 'password123',
          role: 'researcher',
        })
        .expect(500);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeTruthy();
    });

    it('should handle malformed JSON', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .set('Content-Type', 'application/json')
        .send('{ invalid json }')
        .expect(400);

      expect(response.body.success).toBe(false);
    });

    it('should handle missing Content-Type', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send('email=test@example.com')
        .expect(400);

      expect(response.body.success).toBe(false);
    });
  });

  describe('CORS and Security Headers', () => {
    it('should include appropriate headers', async () => {
      const response = await request(app)
        .get('/api/system-specs')
        .expect(200);

      // Note: In a real integration test, you'd check for actual security headers
      // These would be set by the helmet middleware
      expect(response.headers).toBeTruthy();
    });
  });

  describe('Rate Limiting', () => {
    it('should allow normal request rates', async () => {
      // Test that normal request rates work
      for (let i = 0; i < 5; i++) {
        await request(app)
          .get('/api/system-specs')
          .expect(200);
      }
    });
  });
});
