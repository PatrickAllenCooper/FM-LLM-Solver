import { Request, Response } from 'express';
import { EmailAuthorizationService } from '../services/email-authorization.service';
import { logger } from '../utils/logger';
import { z } from 'zod';

const AddEmailSchema = z.object({
  email: z.string().email(),
});

const RemoveEmailSchema = z.object({
  email: z.string().email(),
});

export class AdminController {
  private emailAuthService: EmailAuthorizationService;

  constructor() {
    this.emailAuthService = new EmailAuthorizationService();
  }

  /**
   * Get all authorized emails
   */
  getAuthorizedEmails = async (req: Request, res: Response): Promise<void> => {
    try {
      const authorizedEmails = await this.emailAuthService.getAuthorizedEmails();

      res.status(200).json({
        success: true,
        data: authorizedEmails,
        message: 'Authorized emails retrieved successfully',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      logger.error('Failed to get authorized emails', {
        error: error instanceof Error ? error.message : 'Unknown error',
        userId: (req as any).user?.id,
      });

      res.status(500).json({
        success: false,
        error: 'Failed to retrieve authorized emails',
        timestamp: new Date().toISOString(),
      });
    }
  };

  /**
   * Add an email to the authorized list
   */
  addAuthorizedEmail = async (req: Request, res: Response): Promise<void> => {
    try {
      const validatedData = AddEmailSchema.parse(req.body);
      const addedBy = (req as any).user?.email || 'admin';

      await this.emailAuthService.addAuthorizedEmail(validatedData.email, addedBy);

      res.status(201).json({
        success: true,
        message: `Email ${validatedData.email} added to authorized list`,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({
          success: false,
          error: 'Invalid email format',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      logger.error('Failed to add authorized email', {
        email: req.body?.email,
        error: error instanceof Error ? error.message : 'Unknown error',
        userId: (req as any).user?.id,
      });

      const statusCode = error instanceof Error && error.message.includes('already authorized') ? 409 : 500;
      
      res.status(statusCode).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to add authorized email',
        timestamp: new Date().toISOString(),
      });
    }
  };

  /**
   * Remove an email from the authorized list
   */
  removeAuthorizedEmail = async (req: Request, res: Response): Promise<void> => {
    try {
      const validatedData = RemoveEmailSchema.parse(req.body);

      await this.emailAuthService.removeAuthorizedEmail(validatedData.email);

      res.status(200).json({
        success: true,
        message: `Email ${validatedData.email} removed from authorized list`,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({
          success: false,
          error: 'Invalid email format',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      logger.error('Failed to remove authorized email', {
        email: req.body?.email,
        error: error instanceof Error ? error.message : 'Unknown error',
        userId: (req as any).user?.id,
      });

      const statusCode = error instanceof Error && error.message.includes('not in the authorized list') ? 404 : 500;
      
      res.status(statusCode).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to remove authorized email',
        timestamp: new Date().toISOString(),
      });
    }
  };

  /**
   * Create initial admin user (temporary endpoint)
   */
  createInitialAdmin = async (req: Request, res: Response): Promise<void> => {
    try {
      const bcrypt = require('bcryptjs');
      const { db } = require('../utils/database');
      
      const email = 'patrick.cooper@colorado.edu';
      const password = 'admin123'; // You can change this after login
      const role = 'admin';

      // Check if user already exists
      const usersRef = db.collection('users');
      const existingUserQuery = await usersRef.where('email', '==', email).limit(1).get();

      if (!existingUserQuery.empty) {
        // Update existing user to admin
        const userDoc = existingUserQuery.docs[0];
        await userDoc.ref.update({
          role: 'admin',
          updated_at: new Date(),
        });
        
        res.status(200).json({
          success: true,
          message: `Updated existing user ${email} to admin role`,
          data: { email, role: 'admin' },
          timestamp: new Date().toISOString(),
        });
      } else {
        // Create new admin user
        const password_hash = await bcrypt.hash(password, 12);
        
        const newUserData = {
          email,
          password_hash,
          role,
          created_at: new Date(),
          updated_at: new Date(),
        };

        const docRef = await usersRef.add(newUserData);
        
        res.status(201).json({
          success: true,
          message: `Created new admin user ${email}`,
          data: { 
            id: docRef.id, 
            email, 
            role: 'admin',
            defaultPassword: password 
          },
          timestamp: new Date().toISOString(),
        });
      }
    } catch (error) {
      logger.error('Failed to create initial admin', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      res.status(500).json({
        success: false,
        error: 'Failed to create initial admin',
        timestamp: new Date().toISOString(),
      });
    }
  };

  /**
   * Check if an email is authorized (for admin validation)
   */
  checkEmailAuthorization = async (req: Request, res: Response): Promise<void> => {
    try {
      const { email } = req.query;

      if (!email || typeof email !== 'string') {
        res.status(400).json({
          success: false,
          error: 'Email parameter is required',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const isAuthorized = await this.emailAuthService.isEmailAuthorized(email);

      res.status(200).json({
        success: true,
        data: {
          email,
          isAuthorized,
        },
        message: `Email authorization status checked`,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      logger.error('Failed to check email authorization', {
        email: req.query?.email,
        error: error instanceof Error ? error.message : 'Unknown error',
        userId: (req as any).user?.id,
      });

      res.status(500).json({
        success: false,
        error: 'Failed to check email authorization',
        timestamp: new Date().toISOString(),
      });
    }
  };
}
