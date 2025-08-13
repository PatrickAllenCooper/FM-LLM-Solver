import { AuthService } from '../../src/services/auth.service';
import { CreateUser } from '../../src/types/database';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

// Mock dependencies
jest.mock('../../src/utils/database', () => ({
  db: jest.fn(() => mockDb),
}));

jest.mock('bcryptjs');
jest.mock('jsonwebtoken');

const mockDb = {
  where: jest.fn().mockReturnThis(),
  first: jest.fn(),
  insert: jest.fn().mockReturnThis(),
  returning: jest.fn(),
  update: jest.fn(),
};

describe('AuthService', () => {
  let authService: AuthService;
  let mockBcrypt: jest.Mocked<typeof bcrypt>;
  let mockJwt: jest.Mocked<typeof jwt>;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Set test environment
    process.env.NODE_ENV = 'test';
    process.env.JWT_SECRET = 'test-secret-key';
    process.env.JWT_EXPIRES_IN = '24h';
    
    authService = new AuthService();
    mockBcrypt = bcrypt as jest.Mocked<typeof bcrypt>;
    mockJwt = jwt as jest.Mocked<typeof jwt>;
    
    // Reset the mock database
    Object.keys(mockDb).forEach(key => {
      const method = (mockDb as any)[key];
      if (typeof method === 'function') {
        method.mockClear();
        if (key !== 'where' && key !== 'insert') {
          method.mockReturnThis();
        }
      }
    });
    mockDb.where.mockReturnThis();
    mockDb.insert.mockReturnThis();
  });

  afterEach(() => {
    delete process.env.JWT_SECRET;
    delete process.env.JWT_EXPIRES_IN;
  });

  describe('constructor', () => {
    it('should throw error if JWT_SECRET not set in production', () => {
      process.env.NODE_ENV = 'production';
      delete process.env.JWT_SECRET;
      
      expect(() => new AuthService()).toThrow('JWT_SECRET must be set in production');
    });

    it('should use default secret in development', () => {
      delete process.env.JWT_SECRET;
      expect(() => new AuthService()).not.toThrow();
    });
  });

  describe('register', () => {
    const mockUserData: CreateUser = {
      email: 'test@example.com',
      password_hash: 'plaintext-password',
      role: 'researcher',
    };

    it('should register new user successfully', async () => {
      // Mock user doesn't exist
      mockDb.first.mockResolvedValueOnce(null);
      
      // Mock password hashing
      mockBcrypt.hash.mockResolvedValue('hashed-password' as never);
      
      // Mock user creation
      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
        created_at: new Date(),
      };
      mockDb.returning.mockResolvedValue([mockUser]);
      
      // Mock JWT generation
      mockJwt.sign.mockReturnValue('mock-jwt-token' as never);

      const result = await authService.register(mockUserData);

      expect(result.token).toBe('mock-jwt-token');
      expect(result.user.email).toBe('test@example.com');
      expect(result.user.role).toBe('researcher');
      expect(result.expires_at).toBeTruthy();

      // Verify bcrypt was called with correct parameters
      expect(mockBcrypt.hash).toHaveBeenCalledWith('plaintext-password', 12);
      
      // Verify database operations
      expect(mockDb.where).toHaveBeenCalledWith('email', 'test@example.com');
      expect(mockDb.insert).toHaveBeenCalledWith({
        email: 'test@example.com',
        password_hash: 'hashed-password',
        role: 'researcher',
      });
    });

    it('should throw error if user already exists', async () => {
      // Mock existing user
      mockDb.first.mockResolvedValueOnce({
        id: 'existing-user',
        email: 'test@example.com',
      });

      await expect(authService.register(mockUserData)).rejects.toThrow(
        'User with this email already exists'
      );

      expect(mockBcrypt.hash).not.toHaveBeenCalled();
      expect(mockDb.insert).not.toHaveBeenCalled();
    });

    it('should handle database errors gracefully', async () => {
      mockDb.first.mockResolvedValueOnce(null);
      mockBcrypt.hash.mockResolvedValue('hashed-password' as never);
      mockDb.returning.mockRejectedValue(new Error('Database error'));

      await expect(authService.register(mockUserData)).rejects.toThrow('Database error');
    });
  });

  describe('login', () => {
    const mockUser = {
      id: 'user-123',
      email: 'test@example.com',
      password_hash: 'hashed-password',
      role: 'researcher',
    };

    it('should login user successfully', async () => {
      // Mock user found
      mockDb.first.mockResolvedValueOnce(mockUser);
      
      // Mock password verification
      mockBcrypt.compare.mockResolvedValue(true as never);
      
      // Mock last login update
      mockDb.update.mockResolvedValue(undefined);
      
      // Mock JWT generation
      mockJwt.sign.mockReturnValue('mock-jwt-token' as never);

      const result = await authService.login('test@example.com', 'correct-password');

      expect(result.token).toBe('mock-jwt-token');
      expect(result.user.email).toBe('test@example.com');
      expect(result.user.id).toBe('user-123');

      // Verify password comparison
      expect(mockBcrypt.compare).toHaveBeenCalledWith('correct-password', 'hashed-password');
      
      // Verify last login update
      expect(mockDb.update).toHaveBeenCalledWith({
        last_login_at: expect.any(Date)
      });
    });

    it('should throw error for non-existent user', async () => {
      mockDb.first.mockResolvedValueOnce(null);

      await expect(authService.login('nonexistent@example.com', 'password')).rejects.toThrow(
        'Invalid credentials'
      );

      expect(mockBcrypt.compare).not.toHaveBeenCalled();
    });

    it('should throw error for incorrect password', async () => {
      mockDb.first.mockResolvedValueOnce(mockUser);
      mockBcrypt.compare.mockResolvedValue(false as never);

      await expect(authService.login('test@example.com', 'wrong-password')).rejects.toThrow(
        'Invalid credentials'
      );

      expect(mockDb.update).not.toHaveBeenCalled();
    });
  });

  describe('verifyToken', () => {
    it('should verify valid token successfully', async () => {
      const mockDecodedToken = {
        userId: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
      };

      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
      };

      mockJwt.verify.mockReturnValue(mockDecodedToken as never);
      mockDb.first.mockResolvedValue(mockUser);

      const result = await authService.verifyToken('valid-token');

      expect(result).toEqual(mockUser);
      expect(mockJwt.verify).toHaveBeenCalledWith('valid-token', 'test-secret-key');
      expect(mockDb.where).toHaveBeenCalledWith('id', 'user-123');
    });

    it('should return null for invalid token', async () => {
      mockJwt.verify.mockImplementation(() => {
        throw new Error('Invalid token');
      });

      const result = await authService.verifyToken('invalid-token');

      expect(result).toBeNull();
    });

    it('should return null if user not found', async () => {
      const mockDecodedToken = { userId: 'nonexistent-user' };
      mockJwt.verify.mockReturnValue(mockDecodedToken as never);
      mockDb.first.mockResolvedValue(null);

      const result = await authService.verifyToken('valid-token');

      expect(result).toBeNull();
    });

    it('should handle database errors gracefully', async () => {
      const mockDecodedToken = { userId: 'user-123' };
      mockJwt.verify.mockReturnValue(mockDecodedToken as never);
      mockDb.first.mockRejectedValue(new Error('Database error'));

      const result = await authService.verifyToken('valid-token');

      expect(result).toBeNull();
    });
  });

  describe('getUserById', () => {
    it('should return user if found', async () => {
      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
      };

      mockDb.first.mockResolvedValue(mockUser);

      const result = await authService.getUserById('user-123');

      expect(result).toEqual(mockUser);
      expect(mockDb.where).toHaveBeenCalledWith('id', 'user-123');
    });

    it('should return null if user not found', async () => {
      mockDb.first.mockResolvedValue(null);

      const result = await authService.getUserById('nonexistent-user');

      expect(result).toBeNull();
    });

    it('should handle database errors gracefully', async () => {
      mockDb.first.mockRejectedValue(new Error('Database error'));

      const result = await authService.getUserById('user-123');

      expect(result).toBeNull();
    });
  });

  describe('updateUserRole', () => {
    it('should update user role successfully', async () => {
      mockDb.update.mockResolvedValue(undefined);

      await authService.updateUserRole('user-123', 'admin');

      expect(mockDb.where).toHaveBeenCalledWith('id', 'user-123');
      expect(mockDb.update).toHaveBeenCalledWith({
        role: 'admin',
        updated_at: expect.any(Date),
      });
    });

    it('should handle database errors', async () => {
      mockDb.update.mockRejectedValue(new Error('Database error'));

      await expect(authService.updateUserRole('user-123', 'admin')).rejects.toThrow(
        'Database error'
      );
    });
  });

  describe('changePassword', () => {
    const mockUser = {
      id: 'user-123',
      email: 'test@example.com',
      password_hash: 'old-hashed-password',
    };

    it('should change password successfully', async () => {
      mockDb.first.mockResolvedValue(mockUser);
      mockBcrypt.compare.mockResolvedValue(true as never);
      mockBcrypt.hash.mockResolvedValue('new-hashed-password' as never);
      mockDb.update.mockResolvedValue(undefined);

      await authService.changePassword('user-123', 'old-password', 'new-password');

      expect(mockBcrypt.compare).toHaveBeenCalledWith('old-password', 'old-hashed-password');
      expect(mockBcrypt.hash).toHaveBeenCalledWith('new-password', 12);
      expect(mockDb.update).toHaveBeenCalledWith({
        password_hash: 'new-hashed-password',
        updated_at: expect.any(Date),
      });
    });

    it('should throw error if user not found', async () => {
      mockDb.first.mockResolvedValue(null);

      await expect(
        authService.changePassword('nonexistent-user', 'old-password', 'new-password')
      ).rejects.toThrow('User not found');

      expect(mockBcrypt.compare).not.toHaveBeenCalled();
    });

    it('should throw error if current password is incorrect', async () => {
      mockDb.first.mockResolvedValue(mockUser);
      mockBcrypt.compare.mockResolvedValue(false as never);

      await expect(
        authService.changePassword('user-123', 'wrong-password', 'new-password')
      ).rejects.toThrow('Current password is incorrect');

      expect(mockBcrypt.hash).not.toHaveBeenCalled();
      expect(mockDb.update).not.toHaveBeenCalled();
    });
  });

  describe('JWT token generation', () => {
    it('should generate tokens with correct payload', async () => {
      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher' as const,
        password_hash: 'hashed-password',
      };

      mockDb.first.mockResolvedValueOnce(null); // User doesn't exist for registration
      mockBcrypt.hash.mockResolvedValue('hashed-password' as never);
      mockDb.returning.mockResolvedValue([mockUser]);
      mockJwt.sign.mockReturnValue('mock-jwt-token' as never);

      await authService.register({
        email: 'test@example.com',
        password_hash: 'password',
        role: 'researcher',
      });

      expect(mockJwt.sign).toHaveBeenCalledWith(
        {
          userId: 'user-123',
          email: 'test@example.com',
          role: 'researcher',
        },
        'test-secret-key',
        { expiresIn: '24h' }
      );
    });

    it('should generate token expiration correctly', async () => {
      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher' as const,
        password_hash: 'hashed-password',
      };

      mockDb.first.mockResolvedValueOnce(null);
      mockBcrypt.hash.mockResolvedValue('hashed-password' as never);
      mockDb.returning.mockResolvedValue([mockUser]);
      mockJwt.sign.mockReturnValue('mock-jwt-token' as never);

      const beforeTime = new Date();
      const result = await authService.register({
        email: 'test@example.com',
        password_hash: 'password',
        role: 'researcher',
      });
      const afterTime = new Date();

      const expirationTime = new Date(result.expires_at);
      const expectedMin = new Date(beforeTime.getTime() + 24 * 60 * 60 * 1000);
      const expectedMax = new Date(afterTime.getTime() + 24 * 60 * 60 * 1000);

      expect(expirationTime.getTime()).toBeGreaterThanOrEqual(expectedMin.getTime());
      expect(expirationTime.getTime()).toBeLessThanOrEqual(expectedMax.getTime());
    });
  });

  describe('Error handling and edge cases', () => {
    it('should handle bcrypt errors during registration', async () => {
      mockDb.first.mockResolvedValueOnce(null);
      (mockBcrypt.hash as jest.Mock).mockRejectedValue(new Error('Bcrypt error'));

      await expect(authService.register({
        email: 'test@example.com',
        password_hash: 'password',
        role: 'researcher',
      })).rejects.toThrow('Bcrypt error');
    });

    it('should handle bcrypt errors during login', async () => {
      const mockUser = { password_hash: 'hashed-password' };
      mockDb.first.mockResolvedValueOnce(mockUser);
      (mockBcrypt.compare as jest.Mock).mockRejectedValue(new Error('Bcrypt error'));

      await expect(
        authService.login('test@example.com', 'password')
      ).rejects.toThrow('Bcrypt error');
    });

    it('should handle JWT signing errors', async () => {
      mockDb.first.mockResolvedValueOnce(null);
      mockBcrypt.hash.mockResolvedValue('hashed-password' as never);
      mockDb.returning.mockResolvedValue([{ id: 'user-123', email: 'test@example.com', role: 'researcher' }]);
      mockJwt.sign.mockImplementation(() => {
        throw new Error('JWT error');
      });

      await expect(authService.register({
        email: 'test@example.com',
        password_hash: 'password',
        role: 'researcher',
      })).rejects.toThrow('JWT error');
    });

    it('should handle empty or invalid email addresses', async () => {
      mockDb.first.mockResolvedValueOnce(null);

      // This would typically be caught by input validation, but test the service behavior
      await expect(authService.register({
        email: '',
        password_hash: 'password',
        role: 'researcher',
      })).rejects.toThrow();
    });
  });

  describe('Performance and security', () => {
    it('should use appropriate bcrypt rounds', async () => {
      mockDb.first.mockResolvedValueOnce(null);
      mockBcrypt.hash.mockResolvedValue('hashed-password' as never);
      mockDb.returning.mockResolvedValue([{ id: 'user-123', email: 'test@example.com', role: 'researcher' }]);
      mockJwt.sign.mockReturnValue('token' as never);

      await authService.register({
        email: 'test@example.com',
        password_hash: 'password',
        role: 'researcher',
      });

      expect(mockBcrypt.hash).toHaveBeenCalledWith('password', 12);
    });

    it('should not expose sensitive information in errors', async () => {
      mockDb.first.mockResolvedValueOnce({ email: 'test@example.com' });

      try {
        await authService.register({
          email: 'test@example.com',
          password_hash: 'password',
          role: 'researcher',
        });
      } catch (error) {
        expect((error as Error).message).not.toContain('hashed-password');
        expect((error as Error).message).not.toContain('database');
      }
    });
  });
});
