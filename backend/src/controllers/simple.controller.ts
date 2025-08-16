import { Request, Response } from 'express';
import { db } from '../utils/database';
import { logger } from '../utils/logger';

export class SimpleController {
  // Simple health check with Firestore
  health = async (req: Request, res: Response): Promise<void> => {
    try {
      // Test Firestore connection
      await db.collection('health_check').limit(1).get();
      
      res.json({
        status: 'healthy',
        service: 'fm-llm-solver-api',
        database: 'firestore',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    } catch (error) {
      logger.error('Health check failed:', error);
      res.status(503).json({
        status: 'unhealthy',
        service: 'fm-llm-solver-api',
        database: 'firestore',
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      });
    }
  };

  // Simple test endpoint
  test = async (req: Request, res: Response): Promise<void> => {
    try {
      // Write a test document
      const testDoc = {
        message: 'Firestore connection working!',
        timestamp: new Date(),
        test: true
      };
      
      const docRef = await db.collection('test').add(testDoc);
      
      // Read it back
      const doc = await docRef.get();
      const data = doc.data();
      
      res.json({
        success: true,
        message: 'Firestore read/write test successful',
        data: { id: doc.id, ...data },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Test endpoint failed:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      });
    }
  };
}

