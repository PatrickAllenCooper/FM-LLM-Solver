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
        version: '2.0.0'
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

  // DIAGNOSTIC: Test mathematical evaluation pipeline step by step
  testMathEvaluation = async (req: Request, res: Response): Promise<void> => {
    try {
      const { expression = "x1**2 + x2**2", variables = { x1: -1.5, x2: -2.4 } } = req.body;
      
      // Import MathService for testing
      const { MathService } = await import('../services/math.service');
      const mathService = new MathService();
      
      // Test step by step
      const results: any = {
        input: { expression, variables },
        expected_result: Math.pow(variables.x1, 2) + Math.pow(variables.x2, 2),
      };
      
      try {
        // Step 1: Parse expression
        results.step1_parse = mathService.parseExpression(expression);
      } catch (error) {
        results.step1_parse = { error: error instanceof Error ? error.message : 'Unknown error' };
      }
      
      try {
        // Step 2: Evaluate expression
        results.step2_evaluate = mathService.evaluate(expression, variables);
      } catch (error) {
        results.step2_evaluate = { error: error instanceof Error ? error.message : 'Unknown error' };
      }
      
      // Step 3: Expected vs actual
      results.comparison = {
        expected: results.expected_result,
        actual: results.step2_evaluate?.value,
        correct: Math.abs(results.expected_result - (results.step2_evaluate?.value || 0)) < 1e-6,
        difference: results.expected_result - (results.step2_evaluate?.value || 0),
      };
      
      res.status(200).json({
        success: true,
        data: results,
        message: 'Mathematical evaluation pipeline test completed',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Test failed',
        timestamp: new Date().toISOString(),
      });
    }
  };
}

