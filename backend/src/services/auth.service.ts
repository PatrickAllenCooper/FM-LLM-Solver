import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { db } from '../utils/database';
import { User, CreateUser } from '../types/database';
import { AuthResponse } from '../types/api';
import { logger } from '../utils/logger';
import { EmailAuthorizationService } from './email-authorization.service';

export class AuthService {
  private readonly jwtSecret: string;
  private readonly jwtExpiresIn: string;
  private readonly emailAuthService: EmailAuthorizationService;

  constructor() {
    this.jwtSecret = process.env.JWT_SECRET || 'development-secret-key';
    this.jwtExpiresIn = process.env.JWT_EXPIRES_IN || '24h';
    this.emailAuthService = new EmailAuthorizationService();
    
    if (process.env.NODE_ENV === 'production' && this.jwtSecret === 'development-secret-key') {
      throw new Error('JWT_SECRET must be set in production');
    }
  }

  async register(userData: CreateUser): Promise<AuthResponse> {
    try {
      // Check if user already exists
      const usersRef = db.collection('users');
      const existingUserQuery = await usersRef.where('email', '==', userData.email).limit(1).get();

      if (!existingUserQuery.empty) {
        throw new Error('User with this email already exists');
      }

      // Check if email is authorized for registration
      const isAuthorized = await this.emailAuthService.isEmailAuthorized(userData.email);
      if (!isAuthorized) {
        throw new Error('Your email is not authorized to create an account. Please contact Patrick at patrick.cooper@colorado.edu for access.');
      }

      // Hash password
      const password_hash = await bcrypt.hash(userData.password_hash, 12);

      // Create user document
      const newUserData = {
        ...userData,
        password_hash,
        created_at: new Date(),
        updated_at: new Date(),
      };

      const docRef = await usersRef.add(newUserData);
      const user = { 
        id: docRef.id, 
        email: userData.email,
        role: userData.role,
        created_at: newUserData.created_at
      };

      logger.info('User registered successfully', {
        userId: user.id,
        email: user.email,
        role: user.role,
      });

      // Generate token
      const token = this.generateToken(user);
      const expires_at = this.getTokenExpiration();

      return {
        token,
        user: {
          id: user.id,
          email: user.email,
          role: user.role,
        },
        expires_at,
      };
    } catch (error) {
      logger.error('User registration failed', {
        email: userData.email,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    try {
      // Find user
      const usersRef = db.collection('users');
      const userQuery = await usersRef.where('email', '==', email).limit(1).get();

      if (userQuery.empty) {
        throw new Error('Invalid credentials');
      }

      const userDoc = userQuery.docs[0];
      const userData = userDoc.data();
      const user = { id: userDoc.id, ...userData } as any;

      // Verify password
      const isValidPassword = await bcrypt.compare(password, user.password_hash);
      if (!isValidPassword) {
        throw new Error('Invalid credentials');
      }

      // Update last login
      await userDoc.ref.update({ 
        last_login_at: new Date(),
        updated_at: new Date()
      });

      logger.info('User logged in successfully', {
        userId: user.id,
        email: user.email,
      });

      // Generate token
      const token = this.generateToken(user);
      const expires_at = this.getTokenExpiration();

      return {
        token,
        user: {
          id: user.id,
          email: user.email,
          role: user.role,
        },
        expires_at,
      };
    } catch (error) {
      logger.error('User login failed', {
        email,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  async verifyToken(token: string): Promise<User | null> {
    try {
      const decoded = jwt.verify(token, this.jwtSecret) as any;
      
      const userDoc = await db.collection('users').doc(decoded.userId).get();

      if (!userDoc.exists) {
        return null;
      }

      return { id: userDoc.id, ...userDoc.data() } as User;
    } catch (error) {
      logger.debug('Token validation failed', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      return null;
    }
  }

  async getUserById(userId: string): Promise<User | null> {
    try {
      const userDoc = await db.collection('users').doc(userId).get();

      if (!userDoc.exists) {
        return null;
      }

      return { id: userDoc.id, ...userDoc.data() } as User;
    } catch (error) {
      logger.error('Failed to get user by ID', {
        userId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      return null;
    }
  }

  async updateUserRole(userId: string, role: User['role']): Promise<void> {
    try {
      await db.collection('users').doc(userId).update({ 
        role, 
        updated_at: new Date() 
      });

      logger.info('User role updated', { userId, role });
    } catch (error) {
      logger.error('Failed to update user role', {
        userId,
        role,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  private generateToken(user: Pick<User, 'id' | 'email' | 'role'>): string {
    return jwt.sign(
      {
        userId: user.id,
        email: user.email,
        role: user.role,
      },
      this.jwtSecret,
      { expiresIn: this.jwtExpiresIn } as jwt.SignOptions
    );
  }

  private getTokenExpiration(): string {
    const now = new Date();
    const expirationTime = new Date(now.getTime() + 24 * 60 * 60 * 1000); // 24 hours
    return expirationTime.toISOString();
  }

  async changePassword(userId: string, currentPassword: string, newPassword: string): Promise<void> {
    try {
      const userDoc = await db.collection('users').doc(userId).get();

      if (!userDoc.exists) {
        throw new Error('User not found');
      }

      const user = userDoc.data() as any;

      // Verify current password
      const isValidPassword = await bcrypt.compare(currentPassword, user.password_hash);
      if (!isValidPassword) {
        throw new Error('Current password is incorrect');
      }

      // Hash new password
      const newPasswordHash = await bcrypt.hash(newPassword, 12);

      // Update password
      await userDoc.ref.update({ 
        password_hash: newPasswordHash,
        updated_at: new Date()
      });

      logger.info('Password changed successfully', { userId });
    } catch (error) {
      logger.error('Password change failed', {
        userId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }
}
