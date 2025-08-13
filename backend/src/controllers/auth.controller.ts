import { Request, Response } from 'express';
import { AuthService } from '../services/auth.service';
import { AuthRequest, ApiResponse, AuthResponse } from '../types/api';
import { CreateUser } from '../types/database';
import { logger } from '../utils/logger';
import { z } from 'zod';

const RegisterSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  role: z.enum(['admin', 'researcher', 'viewer']).optional().default('researcher'),
});

const LoginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

const ChangePasswordSchema = z.object({
  currentPassword: z.string().min(1),
  newPassword: z.string().min(8),
});

export class AuthController {
  private authService: AuthService;

  constructor() {
    this.authService = new AuthService();
  }

  register = async (req: Request, res: Response): Promise<void> => {
    try {
      const validatedData = RegisterSchema.parse(req.body);
      
      const userData: CreateUser = {
        email: validatedData.email,
        password_hash: validatedData.password, // Will be hashed in service
        role: validatedData.role,
      };

      const authResponse = await this.authService.register(userData);

      const response: ApiResponse<AuthResponse> = {
        success: true,
        data: authResponse,
        message: 'User registered successfully',
        timestamp: new Date().toISOString(),
      };

      res.status(201).json(response);
    } catch (error) {
      if (error instanceof z.ZodError) {
        const response: ApiResponse = {
          success: false,
          error: 'Validation error',
          timestamp: new Date().toISOString(),
        };
        res.status(400).json(response);
        return;
      }

      logger.error('Registration error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        email: req.body?.email,
      });

      const response: ApiResponse = {
        success: false,
        error: error instanceof Error ? error.message : 'Registration failed',
        timestamp: new Date().toISOString(),
      };

      res.status(400).json(response);
    }
  };

  login = async (req: Request, res: Response): Promise<void> => {
    try {
      const validatedData = LoginSchema.parse(req.body);
      
      const authResponse = await this.authService.login(
        validatedData.email,
        validatedData.password
      );

      const response: ApiResponse<AuthResponse> = {
        success: true,
        data: authResponse,
        message: 'Login successful',
        timestamp: new Date().toISOString(),
      };

      res.status(200).json(response);
    } catch (error) {
      if (error instanceof z.ZodError) {
        const response: ApiResponse = {
          success: false,
          error: 'Validation error',
          timestamp: new Date().toISOString(),
        };
        res.status(400).json(response);
        return;
      }

      logger.error('Login error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        email: req.body?.email,
      });

      const response: ApiResponse = {
        success: false,
        error: 'Invalid credentials',
        timestamp: new Date().toISOString(),
      };

      res.status(401).json(response);
    }
  };

  me = async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.user) {
        const response: ApiResponse = {
          success: false,
          error: 'User not authenticated',
          timestamp: new Date().toISOString(),
        };
        res.status(401).json(response);
        return;
      }

      const { password_hash, ...userWithoutPassword } = req.user;

      const response: ApiResponse = {
        success: true,
        data: userWithoutPassword,
        timestamp: new Date().toISOString(),
      };

      res.status(200).json(response);
    } catch (error) {
      logger.error('Me endpoint error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        userId: req.user?.id,
      });

      const response: ApiResponse = {
        success: false,
        error: 'Failed to get user information',
        timestamp: new Date().toISOString(),
      };

      res.status(500).json(response);
    }
  };

  changePassword = async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.user) {
        const response: ApiResponse = {
          success: false,
          error: 'User not authenticated',
          timestamp: new Date().toISOString(),
        };
        res.status(401).json(response);
        return;
      }

      const validatedData = ChangePasswordSchema.parse(req.body);

      await this.authService.changePassword(
        req.user.id,
        validatedData.currentPassword,
        validatedData.newPassword
      );

      const response: ApiResponse = {
        success: true,
        message: 'Password changed successfully',
        timestamp: new Date().toISOString(),
      };

      res.status(200).json(response);
    } catch (error) {
      if (error instanceof z.ZodError) {
        const response: ApiResponse = {
          success: false,
          error: 'Validation error',
          timestamp: new Date().toISOString(),
        };
        res.status(400).json(response);
        return;
      }

      logger.error('Change password error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        userId: req.user?.id,
      });

      const response: ApiResponse = {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to change password',
        timestamp: new Date().toISOString(),
      };

      res.status(400).json(response);
    }
  };
}
