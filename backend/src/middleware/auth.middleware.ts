import { Request, Response, NextFunction } from 'express';
import { AuthService } from '../services/auth.service';
import { User } from '../types/database';
import { logger } from '../utils/logger';

// Extend Express Request type to include user
declare global {
  namespace Express {
    interface Request {
      user?: User;
    }
  }
}

export class AuthMiddleware {
  private authService: AuthService;

  constructor() {
    this.authService = new AuthService();
  }

  authenticate = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const authHeader = req.headers.authorization;
      
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        res.status(401).json({
          success: false,
          error: 'No token provided',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const token = authHeader.substring(7); // Remove 'Bearer ' prefix
      const user = await this.authService.verifyToken(token);

      if (!user) {
        res.status(401).json({
          success: false,
          error: 'Invalid or expired token',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      req.user = user;
      next();
    } catch (error) {
      logger.error('Authentication middleware error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        path: req.path,
        method: req.method,
      });

      res.status(500).json({
        success: false,
        error: 'Authentication error',
        timestamp: new Date().toISOString(),
      });
    }
  };

  authorize = (roles: User['role'][]) => {
    return (req: Request, res: Response, next: NextFunction): void => {
      if (!req.user) {
        res.status(401).json({
          success: false,
          error: 'Authentication required',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      if (!roles.includes(req.user.role)) {
        logger.warn('Unauthorized access attempt', {
          userId: req.user.id,
          userRole: req.user.role,
          requiredRoles: roles,
          path: req.path,
          method: req.method,
        });

        res.status(403).json({
          success: false,
          error: 'Insufficient permissions',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      next();
    };
  };

  // Optional authentication - sets user if token is valid but doesn't fail if missing
  optionalAuth = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const authHeader = req.headers.authorization;
      
      if (authHeader && authHeader.startsWith('Bearer ')) {
        const token = authHeader.substring(7);
        const user = await this.authService.verifyToken(token);
        
        if (user) {
          req.user = user;
        }
      }

      next();
    } catch (error) {
      // Log error but continue without authentication
      logger.debug('Optional authentication failed', {
        error: error instanceof Error ? error.message : 'Unknown error',
        path: req.path,
      });
      next();
    }
  };
}
