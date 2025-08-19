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
import { SystemSpec, Candidate, CreateCandidate, CreateAuditEvent } from '../types/database';
import { logger } from '../utils/logger';
import crypto from 'crypto';

export class CertificateController {
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

      const systemSpecData = {
        name: validatedData.name,
        description: validatedData.description,
        system_type: validatedData.system_type,
        dimension: validatedData.dimension,
        dynamics_json: validatedData.dynamics,
        constraints_json: validatedData.constraints,
        initial_set_json: validatedData.initial_set,
        unsafe_set_json: validatedData.unsafe_set,
        created_by: req.user.id,
        spec_version: '1.0',
        hash: specHash,
        created_at: new Date(),
        updated_at: new Date(),
      };

      const docRef = await db.collection('system_specs').add(systemSpecData);
      const systemSpec = { id: docRef.id, ...systemSpecData };

      // Log audit event
      await this.logAuditEvent(req.user.id, 'create_system_spec', 'system_spec', systemSpec.id, req);

      const response: ApiResponse<SystemSpec> = {
        success: true,
        data: systemSpec,
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
      const offset = (page - 1) * limit;

      const query = db('system_specs')
        .select('*')
        .orderBy('created_at', 'desc');

      const [specs, countResult] = await Promise.all([
        query.clone().limit(limit).offset(offset),
        db('system_specs').count('* as count').first()
      ]);

      const total = parseInt(countResult?.count as string) || 0;
      const totalPages = Math.ceil(total / limit);

      const response: ApiResponse<PaginatedResponse<SystemSpec>> = {
        success: true,
        data: {
          data: specs,
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

      res.status(200).json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to fetch system specifications');
    }
  };

  // Certificate Generation
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
      const systemSpec = await db('system_specs')
        .where('id', validatedData.system_spec_id)
        .first();

      if (!systemSpec) {
        res.status(404).json({
          success: false,
          error: 'System specification not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      let candidate: Candidate;

      if (validatedData.generation_method === 'llm' && validatedData.llm_config) {
        // Generate using LLM
        const llmResult = await this.llmService.generateCertificate(
          systemSpec,
          validatedData.certificate_type,
          validatedData.llm_config
        );

        const candidateData: CreateCandidate = {
          system_spec_id: validatedData.system_spec_id,
          certificate_type: validatedData.certificate_type,
          generation_method: 'llm',
          llm_provider: 'anthropic',
          llm_model: validatedData.llm_config.model,
          llm_mode: validatedData.llm_config.mode,
          llm_config_json: validatedData.llm_config,
          candidate_expression: llmResult.response.expression,
          candidate_json: llmResult.response,
          verification_status: 'pending',
          created_by: req.user.id,
          generation_duration_ms: llmResult.duration_ms,
        };

        const [insertedCandidate] = await db('candidates')
          .insert(candidateData)
          .returning('*');

        candidate = insertedCandidate;
      } else {
        // Generate using baseline method (simplified for MVP)
        const baselineResult = await this.generateBaseline(
          systemSpec,
          validatedData.certificate_type,
          validatedData.generation_method as 'sos' | 'sdp' | 'quadratic_template'
        );

        const candidateData: CreateCandidate = {
          system_spec_id: validatedData.system_spec_id,
          certificate_type: validatedData.certificate_type,
          generation_method: validatedData.generation_method,
          candidate_expression: baselineResult.expression,
          candidate_json: baselineResult,
          verification_status: 'pending',
          created_by: req.user.id,
          generation_duration_ms: baselineResult.duration_ms,
        };

        const [insertedCandidate] = await db('candidates')
          .insert(candidateData)
          .returning('*');

        candidate = insertedCandidate;
      }

      // Start acceptance check asynchronously
      this.checkCandidateAcceptance(candidate.id, systemSpec);

      // Log audit event
      await this.logAuditEvent(req.user.id, 'generate_certificate', 'candidate', candidate.id, req);

      const response: ApiResponse<Candidate> = {
        success: true,
        data: candidate,
        message: 'Certificate generation started',
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
      const offset = (page - 1) * limit;
      
      const systemSpecId = req.query.system_spec_id as string;
      const certificateType = req.query.certificate_type as string;
      const verificationStatus = req.query.verification_status as string;

      let query = db('candidates')
        .select('candidates.*', 'system_specs.name as system_name')
        .leftJoin('system_specs', 'candidates.system_spec_id', 'system_specs.id')
        .orderBy('candidates.created_at', 'desc');

      if (systemSpecId) {
        query = query.where('candidates.system_spec_id', systemSpecId);
      }
      
      if (certificateType) {
        query = query.where('candidates.certificate_type', certificateType);
      }
      
      if (verificationStatus) {
        query = query.where('candidates.verification_status', verificationStatus);
      }

      // Create a separate count query without joins and selects
      let countQuery = db('candidates');
      
      if (systemSpecId) {
        countQuery = countQuery.where('system_spec_id', systemSpecId);
      }
      
      if (certificateType) {
        countQuery = countQuery.where('certificate_type', certificateType);
      }
      
      if (verificationStatus) {
        countQuery = countQuery.where('verification_status', verificationStatus);
      }

      const [candidates, countResult] = await Promise.all([
        query.clone().limit(limit).offset(offset),
        countQuery.count('* as count').first()
      ]);

      const total = parseInt(countResult?.count as string) || 0;
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

      res.status(200).json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to fetch candidates');
    }
  };

  getCandidateById = async (req: Request, res: Response): Promise<void> => {
    try {
      const candidateId = req.params.id;
      
      const candidate = await db('candidates')
        .select('candidates.*', 'system_specs.name as system_name', 'system_specs.dynamics_json')
        .leftJoin('system_specs', 'candidates.system_spec_id', 'system_specs.id')
        .where('candidates.id', candidateId)
        .first();

      if (!candidate) {
        res.status(404).json({
          success: false,
          error: 'Candidate not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      // Get counterexamples if any
      const counterexamples = await db('counterexamples')
        .where('candidate_id', candidateId)
        .orderBy('created_at', 'desc');

      const response: ApiResponse = {
        success: true,
        data: {
          ...candidate,
          counterexamples,
        },
        timestamp: new Date().toISOString(),
      };

      res.status(200).json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to fetch candidate');
    }
  };

  // Acceptance check
  private async checkCandidateAcceptance(candidateId: string, systemSpec: SystemSpec): Promise<void> {
    try {
      const candidate = await db('candidates')
        .where('id', candidateId)
        .first();

      if (!candidate) {
        logger.error('Candidate not found for acceptance check', { candidateId });
        return;
      }

      logger.info('Starting acceptance check', { candidateId });

      const acceptanceResult = await this.acceptanceService.acceptCandidate(
        candidate,
        systemSpec
      );

      // Update candidate with acceptance results
      const updateData: any = {
        acceptance_status: acceptanceResult.accepted ? 'accepted' : 'failed',
        accepted_at: new Date(),
        acceptance_duration_ms: acceptanceResult.duration_ms,
      };

      if (acceptanceResult.margin !== undefined) {
        updateData.margin = acceptanceResult.margin;
      }

      await db('candidates')
        .where('id', candidateId)
        .update(updateData);

      // Create counterexample if acceptance check failed
      if (!acceptanceResult.accepted && acceptanceResult.counterexample) {
        await db('counterexamples').insert({
          candidate_id: candidateId,
          x_json: verificationResult.counterexample.state,
          context: `Acceptance check failed: ${acceptanceResult.counterexample.violation_type}`,
          violation_metrics_json: {
            type: acceptanceResult.counterexample.violation_type,
            magnitude: acceptanceResult.counterexample.violation_magnitude,
          },
        });
      }

      logger.info('Acceptance check completed', {
        candidateId,
        accepted: acceptanceResult.accepted,
        duration_ms: acceptanceResult.duration_ms,
      });
    } catch (error) {
      logger.error('Acceptance check failed', {
        candidateId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      // Update candidate status to timeout/error
      await db('candidates')
        .where('id', candidateId)
        .update({
          verification_status: 'timeout',
          verified_at: new Date(),
        });
    }
  }

  private async generateBaseline(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    method: 'sos' | 'sdp' | 'quadratic_template'
  ): Promise<{
    expression: string;
    duration_ms: number;
    [key: string]: any;
  }> {
    const startTime = Date.now();

    // Simplified baseline generation for MVP
    const dimension = systemSpec.dimension;
    const variables = Array.from({ length: dimension }, (_, i) => `x${i + 1}`);

    let expression: string;

    switch (method) {
      case 'quadratic_template':
        // Generate simple quadratic form
        if (certificateType === 'lyapunov') {
          const terms = variables.map((v, i) => `${v}^2`);
          expression = terms.join(' + ');
        } else {
          expression = variables.join(' + ');
        }
        break;
      
      case 'sos':
        // Simplified SOS form
        expression = variables.map(v => `${v}^2`).join(' + ');
        break;
      
      case 'sdp':
        // Simplified SDP form
        expression = `0.5 * (${variables.map(v => `${v}^2`).join(' + ')})`;
        break;
      
      default:
        throw new Error(`Unsupported baseline method: ${method}`);
    }

    const duration_ms = Date.now() - startTime;

    return {
      certificate_type: certificateType,
      expression,
      variables,
      duration_ms,
      method,
      confidence: 0.9,
    };
  }

  private async logAuditEvent(
    userId: string,
    action: string,
    entityType: string,
    entityId: string,
    req: Request
  ): Promise<void> {
    try {
      const auditEvent: CreateAuditEvent = {
        user_id: userId,
        action,
        entity_type: entityType,
        entity_id: entityId,
        ip: req.ip,
        user_agent: req.get('User-Agent'),
      };

      await db('audit_events').insert(auditEvent);
    } catch (error) {
      logger.error('Failed to log audit event', {
        userId,
        action,
        entityType,
        entityId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  private handleError(error: unknown, res: Response, defaultMessage: string): void {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        success: false,
        error: 'Validation error',
        details: error.errors,
        timestamp: new Date().toISOString(),
      });
      return;
    }

    logger.error(defaultMessage, {
      error: error instanceof Error ? error.message : 'Unknown error',
    });

    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : defaultMessage,
      timestamp: new Date().toISOString(),
    });
  }
}
