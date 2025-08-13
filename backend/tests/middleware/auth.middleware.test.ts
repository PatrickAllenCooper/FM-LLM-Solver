import { Request, Response, NextFunction } from 'express';
import { AuthMiddleware } from '../../src/middleware/auth.middleware';
import { AuthService } from '../../src/services/auth.service';
import { User } from '../../src/types/database';

// Mock AuthService
jest.mock('../../src/services/auth.service');

describe('AuthMiddleware', () => {
  let authMiddleware: AuthMiddleware;
  let mockAuthService: jest.Mocked<AuthService>;
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let mockNext: NextFunction;

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockAuthService = {
      verifyToken: jest.fn(),
    } as any;

    (AuthService as jest.MockedClass<typeof AuthService>).mockImplementation(() => mockAuthService);
    
    authMiddleware = new AuthMiddleware();
    
    mockRequest = {
      headers: {},
      user: undefined,
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
    
    mockNext = jest.fn();
  });

  describe('authenticate', () => {
    it('should authenticate user with valid Bearer token', async () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.headers = {
        authorization: 'Bearer valid-token',
      };

      mockAuthService.verifyToken.mockResolvedValue(mockUser);

      await authMiddleware.authenticate(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockAuthService.verifyToken).toHaveBeenCalledWith('valid-token');
      expect(mockRequest.user).toEqual(mockUser);
      expect(mockNext).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should reject request with missing authorization header', async () => {
      mockRequest.headers = {};

      await authMiddleware.authenticate(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Authorization header required',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should reject request with invalid authorization format', async () => {
      mockRequest.headers = {
        authorization: 'InvalidFormat token',
      };

      await authMiddleware.authenticate(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Invalid authorization format',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should reject request with invalid token', async () => {
      mockRequest.headers = {
        authorization: 'Bearer invalid-token',
      };

      mockAuthService.verifyToken.mockRejectedValue(new Error('Invalid token'));

      await authMiddleware.authenticate(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Invalid or expired token',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should handle token verification errors gracefully', async () => {
      mockRequest.headers = {
        authorization: 'Bearer some-token',
      };

      mockAuthService.verifyToken.mockRejectedValue(new Error('Database connection failed'));

      await authMiddleware.authenticate(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Invalid or expired token',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });
  });

  describe('optionalAuth', () => {
    it('should authenticate user with valid token', async () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.headers = {
        authorization: 'Bearer valid-token',
      };

      mockAuthService.verifyToken.mockResolvedValue(mockUser);

      await authMiddleware.optionalAuth(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockAuthService.verifyToken).toHaveBeenCalledWith('valid-token');
      expect(mockRequest.user).toEqual(mockUser);
      expect(mockNext).toHaveBeenCalled();
    });

    it('should continue without authentication when no token provided', async () => {
      mockRequest.headers = {};

      await authMiddleware.optionalAuth(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockAuthService.verifyToken).not.toHaveBeenCalled();
      expect(mockRequest.user).toBeUndefined();
      expect(mockNext).toHaveBeenCalled();
    });

    it('should continue without authentication when token is invalid', async () => {
      mockRequest.headers = {
        authorization: 'Bearer invalid-token',
      };

      mockAuthService.verifyToken.mockRejectedValue(new Error('Invalid token'));

      await authMiddleware.optionalAuth(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockRequest.user).toBeUndefined();
      expect(mockNext).toHaveBeenCalled();
    });
  });

  describe('authorize', () => {
    it('should allow access for users with required role', () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'admin@example.com',
        role: 'admin',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.user = mockUser;

      const middleware = authMiddleware.authorize(['admin', 'researcher']);
      middleware(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should deny access for users without required role', () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'viewer@example.com',
        role: 'viewer',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.user = mockUser;

      const middleware = authMiddleware.authorize(['admin']);
      middleware(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Insufficient permissions',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should deny access for unauthenticated users', () => {
      mockRequest.user = undefined;

      const middleware = authMiddleware.authorize(['admin']);
      middleware(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Authentication required',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should allow multiple roles', () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'researcher@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.user = mockUser;

      const middleware = authMiddleware.authorize(['admin', 'researcher']);
      middleware(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });
  });

  describe('requireRole', () => {
    it('should allow access for exact role match', () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'admin@example.com',
        role: 'admin',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.user = mockUser;

      const middleware = authMiddleware.requireRole('admin');
      middleware(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should deny access for role mismatch', () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'researcher@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.user = mockUser;

      const middleware = authMiddleware.requireRole('admin');
      middleware(mockRequest as Request, mockResponse as Response, mockNext);

      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Insufficient permissions',
      });
      expect(mockNext).not.toHaveBeenCalled();
    });
  });
});
