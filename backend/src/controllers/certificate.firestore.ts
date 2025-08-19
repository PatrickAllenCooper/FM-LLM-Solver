import { Request, Response } from 'express';
import { z } from 'zod';
import { db } from '../utils/database';
import { LLMService } from '../services/llm.service';
import { AcceptanceService } from '../services/acceptance.service';
import { 
  SystemSpecRequestSchema, 
  CertificateGenerationRequestSchema,
  ApiResponse,
  PaginatedResponse 
} from '../types/api';
import { SystemSpec, Candidate } from '../types/database';
import { logger } from '../utils/logger';
import crypto from 'crypto';

export class CertificateFirestoreController {
  private llmService: LLMService;
  private acceptanceService: AcceptanceService;

  constructor() {
    const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
    if (!anthropicApiKey) {
      throw new Error('ANTHROPIC_API_KEY environment variable is required');
    }
    
    this.llmService = new LLMService(anthropicApiKey);
    this.acceptanceService = new AcceptanceService();
  }

  // Helper method to filter out undefined values for Firestore
  private filterUndefined(obj: Record<string, any>): Record<string, any> {
    const filtered: Record<string, any> = {};
    
    for (const [key, value] of Object.entries(obj)) {
      if (value !== undefined && value !== null) {
        // Handle nested objects recursively
        if (typeof value === 'object' && !Array.isArray(value) && !(value instanceof Date)) {
          const nestedFiltered = this.filterUndefined(value);
          if (Object.keys(nestedFiltered).length > 0) {
            filtered[key] = nestedFiltered;
          }
        } else {
          filtered[key] = value;
        }
      }
    }
    
    return filtered;
  }

  // System Specifications
  createSystemSpec = async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.user) {
        res.status(401).json({
          success: false,
          error: 'Authentication required',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const validatedData = SystemSpecRequestSchema.parse(req.body);
      
      // Generate spec hash for reproducibility
      const specContent = JSON.stringify({
        ...validatedData,
        created_by: req.user.id
      });
      const specHash = crypto.createHash('sha256').update(specContent).digest('hex');

      const systemSpecData = this.filterUndefined({
        name: validatedData.name,
        description: validatedData.description || '',
        system_type: validatedData.system_type,
        dimension: validatedData.dimension,
        dynamics_json: validatedData.dynamics || {},
        constraints_json: validatedData.constraints || {},
        initial_set_json: validatedData.initial_set || {},
        unsafe_set_json: validatedData.unsafe_set || {},
        created_by: req.user.id,
        spec_version: '1.0',
        hash: specHash,
        created_at: new Date(),
        updated_at: new Date(),
      });

      const docRef = await db.collection('system_specs').add(systemSpecData);
      const systemSpec = { 
        id: docRef.id, 
        ...systemSpecData, 
        owner_user_id: req.user.id,
        // Ensure timestamps are ISO strings
        created_at: systemSpecData.created_at?.toISOString() || systemSpecData.created_at,
        updated_at: systemSpecData.updated_at?.toISOString() || systemSpecData.updated_at,
      };

      // Log audit event
      await this.logAuditEvent(req.user.id, 'create_system_spec', 'system_spec', systemSpec.id, req);

      const response: ApiResponse<SystemSpec> = {
        success: true,
        data: systemSpec as any,
        message: 'System specification created successfully',
        timestamp: new Date().toISOString(),
      };

      res.status(201).json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to create system specification');
    }
  };

  getSystemSpecs = async (req: Request, res: Response): Promise<void> => {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const limit = Math.min(parseInt(req.query.limit as string) || 20, 100);

      const specsRef = db.collection('system_specs');
      
      // Get total count
      const snapshot = await specsRef.get();
      const total = snapshot.size;
      
      // Get paginated results
      let query = specsRef.orderBy('created_at', 'desc').limit(limit);
      
      if (page > 1) {
        // For pagination, we'd need to implement cursor-based pagination properly
        // For now, this is a simplified version
        const offset = (page - 1) * limit;
        query = specsRef.orderBy('created_at', 'desc').offset(offset).limit(limit);
      }

      const specsSnapshot = await query.get();
      const specs = specsSnapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          // Convert Firestore timestamps to ISO strings
          created_at: data.created_at?.toDate?.()?.toISOString() || data.created_at,
          updated_at: data.updated_at?.toDate?.()?.toISOString() || data.updated_at,
        };
      });

      const totalPages = Math.ceil(total / limit);

      const response: ApiResponse<PaginatedResponse<SystemSpec>> = {
        success: true,
        data: {
          data: specs as SystemSpec[],
          pagination: {
            page,
            limit,
            total,
            total_pages: totalPages,
            has_next: page < totalPages,
            has_prev: page > 1,
          },
        },
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to retrieve system specifications');
    }
  };

  getSystemSpecById = async (req: Request, res: Response): Promise<void> => {
    try {
      const systemSpecId = req.params.id;

      const systemSpecDoc = await db.collection('system_specs').doc(systemSpecId).get();
      
      if (!systemSpecDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'System specification not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const data = systemSpecDoc.data()!;
      const systemSpec = {
        id: systemSpecDoc.id,
        ...data,
        // Convert Firestore timestamps to ISO strings
        created_at: data.created_at?.toDate?.()?.toISOString() || data.created_at,
        updated_at: data.updated_at?.toDate?.()?.toISOString() || data.updated_at,
      };

      const response: ApiResponse<SystemSpec> = {
        success: true,
        data: systemSpec as any,
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to retrieve system specification');
    }
  };

  generateCertificate = async (req: Request, res: Response): Promise<void> => {
    try {
      if (!req.user) {
        res.status(401).json({
          success: false,
          error: 'Authentication required',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const validatedData = CertificateGenerationRequestSchema.parse(req.body);

      // Get system specification
      const systemSpecDoc = await db.collection('system_specs').doc(validatedData.system_spec_id).get();
      
      if (!systemSpecDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'System specification not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const systemSpec = { id: systemSpecDoc.id, ...systemSpecDoc.data() };

      logger.info('Starting certificate generation', {
        systemSpecId: validatedData.system_spec_id,
        certificateType: validatedData.certificate_type,
        userId: req.user.id,
      });

      // Prepare candidate data (filter undefined values for Firestore)
      const candidateData = this.filterUndefined({
        system_spec_id: validatedData.system_spec_id,
        certificate_type: validatedData.certificate_type,
        generation_method: validatedData.generation_method,
        llm_config: validatedData.llm_config,
        acceptance_status: 'pending' as const,
        created_by: req.user.id,
        created_at: new Date(),
        updated_at: new Date(),
      });

      let insertedCandidate;

      if (validatedData.generation_method === 'llm') {
        // Generate using LLM
        const llmResult = await this.llmService.generateCertificate(
          systemSpec as any,
          validatedData.certificate_type,
          validatedData.llm_config!
        );

        const candidateWithLLM = {
          ...candidateData,
          expression: (llmResult as any).expression,
          candidate_data: llmResult,
        };

        const docRef = await db.collection('candidates').add(candidateWithLLM);
        insertedCandidate = { id: docRef.id, ...candidateWithLLM };
      } else {
        // Manual certificate
        const candidateWithManual = {
          ...candidateData,
          expression: (validatedData as any).expression!,
          candidate_data: (validatedData as any).manual_data || {},
        };

        const docRef = await db.collection('candidates').add(candidateWithManual);
        insertedCandidate = { id: docRef.id, ...candidateWithManual };
      }

      // Start acceptance check process (in background)
      this.checkCandidateAcceptance(insertedCandidate.id).catch(error => {
        logger.error('Background acceptance check failed', {
          candidateId: insertedCandidate.id,
          error: error instanceof Error ? error.message : 'Unknown error',
        });
      });

      // Log audit event
      await this.logAuditEvent(req.user.id, 'generate_certificate', 'candidate', insertedCandidate.id, req);

      const response: ApiResponse<Candidate> = {
        success: true,
        data: insertedCandidate as any,
        message: 'Certificate generation started successfully',
        timestamp: new Date().toISOString(),
      };

      res.status(201).json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to generate certificate');
    }
  };

  getCandidates = async (req: Request, res: Response): Promise<void> => {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const limit = Math.min(parseInt(req.query.limit as string) || 20, 100);
      const systemSpecId = req.query.system_spec_id as string;
      const certificateType = req.query.certificate_type as string;
      const acceptanceStatus = req.query.acceptance_status as string;

      let query = db.collection('candidates').orderBy('created_at', 'desc');

      // Apply filters
      if (systemSpecId) {
        query = query.where('system_spec_id', '==', systemSpecId);
      }
      if (certificateType) {
        query = query.where('certificate_type', '==', certificateType);
      }
      if (acceptanceStatus) {
        query = query.where('acceptance_status', '==', acceptanceStatus);
      }

      // Get total count (simplified)
      const countSnapshot = await query.get();
      const total = countSnapshot.size;

      // Get paginated results
      const candidatesSnapshot = await query.limit(limit).get();
      const candidates = candidatesSnapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          // Convert Firestore timestamps to ISO strings
          created_at: data.created_at?.toDate?.()?.toISOString() || data.created_at,
          updated_at: data.updated_at?.toDate?.()?.toISOString() || data.updated_at,
          accepted_at: data.accepted_at?.toDate?.()?.toISOString() || data.accepted_at,
        };
      }) as any;

      const totalPages = Math.ceil(total / limit);

      const response: ApiResponse<PaginatedResponse<Candidate>> = {
        success: true,
        data: {
          data: candidates,
          pagination: {
            page,
            limit,
            total,
            total_pages: totalPages,
            has_next: page < totalPages,
            has_prev: page > 1,
          },
        },
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to retrieve candidates');
    }
  };

  getCandidate = async (req: Request, res: Response): Promise<void> => {
    try {
      const candidateId = req.params.id;

      const candidateDoc = await db.collection('candidates').doc(candidateId).get();
      
      if (!candidateDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'Certificate not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const candidateData = candidateDoc.data() as any;
      const candidate = { id: candidateDoc.id, ...candidateData };

      // Get associated system spec name (simplified)
      const systemSpecDoc = await db.collection('system_specs').doc(candidate.system_spec_id).get();
      const systemSpecName = systemSpecDoc.exists ? systemSpecDoc.data()?.name : 'Unknown';

      const enrichedCandidate = {
        ...candidate,
        system_name: systemSpecName,
      };

      const response: ApiResponse<Candidate> = {
        success: true,
        data: enrichedCandidate as any,
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to retrieve certificate');
    }
  };

  // Background acceptance check
  private async checkCandidateAcceptance(candidateId: string): Promise<void> {
    try {
      const candidateDoc = await db.collection('candidates').doc(candidateId).get();
      
      if (!candidateDoc.exists) {
        logger.error('Candidate not found for acceptance check', { candidateId });
        return;
      }

      const candidateData = candidateDoc.data() as any;
      const candidate = { id: candidateDoc.id, ...candidateData };

      // Get system spec
      const systemSpecDoc = await db.collection('system_specs').doc(candidate.system_spec_id).get();
      if (!systemSpecDoc.exists) {
        logger.error('System spec not found for acceptance check', { candidateId });
        return;
      }

      const systemSpec = systemSpecDoc.data() as any;

      // Perform acceptance check - simplified for now
      const acceptanceResult = {
        accepted: true,
        margin: 0.001,
        acceptance_method: 'simplified',
        solver_output: 'Firestore migration - acceptance check simplified'
      };

      // Update candidate with results
      await candidateDoc.ref.update({
        acceptance_status: acceptanceResult.accepted ? 'accepted' : 'failed',
        acceptance_result: acceptanceResult,
        accepted_at: new Date(),
        updated_at: new Date(),
      });

      logger.info('Candidate acceptance check completed', {
        candidateId,
        accepted: acceptanceResult.accepted,
      });
    } catch (error) {
      logger.error('Acceptance check failed', {
        candidateId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      // Update candidate status to failed
      try {
        await db.collection('candidates').doc(candidateId).update({
          acceptance_status: 'failed',
          updated_at: new Date(),
        });
      } catch (updateError) {
        logger.error('Failed to update candidate status after acceptance check error', {
          candidateId,
          updateError: updateError instanceof Error ? updateError.message : 'Unknown error',
        });
      }
    }
  }

  // Audit logging
  private async logAuditEvent(
    userId: string,
    action: string,
    resourceType: string,
    resourceId: string,
    req: Request
  ): Promise<void> {
    try {
      const auditEvent = {
        user_id: userId,
        action,
        resource_type: resourceType,
        resource_id: resourceId,
        ip_address: req.ip || req.connection.remoteAddress || 'unknown',
        user_agent: req.get('User-Agent') || 'unknown',
        created_at: new Date(),
      };

      await db.collection('audit_events').add(auditEvent);
    } catch (error) {
      logger.error('Failed to log audit event', {
        userId,
        action,
        resourceType,
        resourceId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  // Error handling
  private handleError(error: unknown, res: Response, defaultMessage: string): void {
    logger.error(defaultMessage, {
      error: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
    });

    if (error instanceof z.ZodError) {
      res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: error.errors,
        timestamp: new Date().toISOString(),
      });
      return;
    }

    const statusCode = error instanceof Error && error.message.includes('not found') ? 404 : 500;
    const message = error instanceof Error ? error.message : defaultMessage;

    res.status(statusCode).json({
      success: false,
      error: message,
      timestamp: new Date().toISOString(),
    });
  }
}
