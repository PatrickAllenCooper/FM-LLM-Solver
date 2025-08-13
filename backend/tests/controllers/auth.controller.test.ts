import { Request, Response } from 'express';
import { AuthController } from '../../src/controllers/auth.controller';
import { AuthService } from '../../src/services/auth.service';
import { User } from '../../src/types/database';

// Mock AuthService
jest.mock('../../src/services/auth.service');

describe('AuthController', () => {
  let authController: AuthController;
  let mockAuthService: jest.Mocked<AuthService>;
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockAuthService = {
      register: jest.fn(),
      login: jest.fn(),
      verifyToken: jest.fn(),
      refreshToken: jest.fn(),
    } as any;

    (AuthService as jest.MockedClass<typeof AuthService>).mockImplementation(() => mockAuthService);
    
    authController = new AuthController();
    
    mockRequest = {
      body: {},
      user: undefined,
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
  });

  describe('register', () => {
    it('should register a new user successfully', async () => {
      const registerData = {
        email: 'test@example.com',
        password: 'password123',
        role: 'researcher' as const,
      };

      const mockAuthResponse = {
        user: {
          id: 'user-123',
          email: 'test@example.com',
          role: 'researcher',
        },
        token: 'jwt-token',
        expiresAt: '2024-12-31T23:59:59.000Z',
      };

      mockRequest.body = registerData;
      mockAuthService.register.mockResolvedValue(mockAuthResponse);

      await authController.register(mockRequest as Request, mockResponse as Response);

      expect(mockAuthService.register).toHaveBeenCalledWith({
        email: 'test@example.com',
        password_hash: 'password123',
        role: 'researcher',
      });
      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: mockAuthResponse,
      });
    });

    it('should handle validation errors', async () => {
      mockRequest.body = {
        email: 'invalid-email',
        password: '123', // too short
      };

      await authController.register(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Validation failed',
        details: expect.any(Array),
      });
    });

    it('should handle registration service errors', async () => {
      const registerData = {
        email: 'test@example.com',
        password: 'password123',
        role: 'researcher' as const,
      };

      mockRequest.body = registerData;
      mockAuthService.register.mockRejectedValue(new Error('Email already exists'));

      await authController.register(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Email already exists',
      });
    });

    it('should handle internal server errors', async () => {
      const registerData = {
        email: 'test@example.com',
        password: 'password123',
        role: 'researcher' as const,
      };

      mockRequest.body = registerData;
      mockAuthService.register.mockRejectedValue(new Error('Database connection failed'));

      await authController.register(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Database connection failed',
      });
    });
  });

  describe('login', () => {
    it('should login user successfully', async () => {
      const loginData = {
        email: 'test@example.com',
        password: 'password123',
      };

      const mockAuthResponse = {
        user: {
          id: 'user-123',
          email: 'test@example.com',
          role: 'researcher',
        },
        token: 'jwt-token',
        expiresAt: '2024-12-31T23:59:59.000Z',
      };

      mockRequest.body = loginData;
      mockAuthService.login.mockResolvedValue(mockAuthResponse);

      await authController.login(mockRequest as Request, mockResponse as Response);

      expect(mockAuthService.login).toHaveBeenCalledWith('test@example.com', 'password123');
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: mockAuthResponse,
      });
    });

    it('should handle invalid credentials', async () => {
      const loginData = {
        email: 'test@example.com',
        password: 'wrong-password',
      };

      mockRequest.body = loginData;
      mockAuthService.login.mockRejectedValue(new Error('Invalid credentials'));

      await authController.login(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Invalid credentials',
      });
    });

    it('should handle validation errors', async () => {
      mockRequest.body = {
        email: 'invalid-email',
      };

      await authController.login(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Validation failed',
        details: expect.any(Array),
      });
    });
  });

  describe('profile', () => {
    it('should return user profile successfully', async () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.user = mockUser;

      await authController.profile(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: {
          id: 'user-123',
          email: 'test@example.com',
          role: 'researcher',
          created_at: mockUser.created_at,
          updated_at: mockUser.updated_at,
        },
      });
    });

    it('should handle missing user in request', async () => {
      mockRequest.user = undefined;

      await authController.profile(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'User not authenticated',
      });
    });
  });

  describe('refreshToken', () => {
    it('should refresh token successfully', async () => {
      const refreshData = {
        refreshToken: 'valid-refresh-token',
      };

      const mockAuthResponse = {
        user: {
          id: 'user-123',
          email: 'test@example.com',
          role: 'researcher',
        },
        token: 'new-jwt-token',
        expiresAt: '2024-12-31T23:59:59.000Z',
      };

      mockRequest.body = refreshData;
      mockAuthService.refreshToken.mockResolvedValue(mockAuthResponse);

      await authController.refreshToken(mockRequest as Request, mockResponse as Response);

      expect(mockAuthService.refreshToken).toHaveBeenCalledWith('valid-refresh-token');
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: mockAuthResponse,
      });
    });

    it('should handle invalid refresh token', async () => {
      const refreshData = {
        refreshToken: 'invalid-refresh-token',
      };

      mockRequest.body = refreshData;
      mockAuthService.refreshToken.mockRejectedValue(new Error('Invalid refresh token'));

      await authController.refreshToken(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Invalid refresh token',
      });
    });

    it('should handle validation errors', async () => {
      mockRequest.body = {};

      await authController.refreshToken(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Validation failed',
        details: expect.any(Array),
      });
    });
  });

  describe('logout', () => {
    it('should logout successfully', async () => {
      await authController.logout(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        message: 'Logged out successfully',
      });
    });
  });

  describe('validateToken', () => {
    it('should validate token successfully', async () => {
      const mockUser: User = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      };

      mockRequest.body = { token: 'valid-token' };
      mockAuthService.verifyToken.mockResolvedValue(mockUser);

      await authController.validateToken(mockRequest as Request, mockResponse as Response);

      expect(mockAuthService.verifyToken).toHaveBeenCalledWith('valid-token');
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: {
          valid: true,
          user: {
            id: 'user-123',
            email: 'test@example.com',
            role: 'researcher',
            created_at: mockUser.created_at,
            updated_at: mockUser.updated_at,
          },
        },
      });
    });

    it('should handle invalid token', async () => {
      mockRequest.body = { token: 'invalid-token' };
      mockAuthService.verifyToken.mockRejectedValue(new Error('Invalid token'));

      await authController.validateToken(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: {
          valid: false,
        },
      });
    });

    it('should handle validation errors', async () => {
      mockRequest.body = {};

      await authController.validateToken(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Validation failed',
        details: expect.any(Array),
      });
    });
  });
});
