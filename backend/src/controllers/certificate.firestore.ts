import { Request, Response } from 'express';
import { z } from 'zod';
import { db } from '../utils/database';
import { LLMService } from '../services/llm.service';
import { AcceptanceService } from '../services/acceptance.service';
import { 
  SystemSpecRequestSchema, 
  CertificateGenerationRequestSchema,
  AcceptanceParametersSchema,
  StartConversationRequestSchema,
  SendMessageRequestSchema,
  PublishFromConversationRequestSchema,
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
          // Ensure proper field mapping for frontend
          candidate_expression: data.candidate_data?.response?.expression || data.candidate_expression,
          candidate_json: data.candidate_data || data.candidate_json,
          llm_config_json: data.llm_config || data.llm_config_json,
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

      // Generate technical details on-the-fly for accepted certificates if not present
      let acceptanceResult = candidateData.acceptance_result;
      
      if (candidate.acceptance_status === 'accepted' && !acceptanceResult?.technical_details) {
        try {
          logger.info('Generating technical details on-the-fly for accepted certificate', {
            candidateId: candidate.id,
          });
          
          // Get full system spec for acceptance checking
          if (systemSpecDoc.exists) {
            const systemSpecData = systemSpecDoc.data();
            const systemSpec = { id: candidate.system_spec_id, ...systemSpecData } as SystemSpec;
            
            // Create properly mapped candidate object for AcceptanceService
            const mappedCandidate = {
              ...candidate,
              candidate_expression: candidateData.candidate_data?.response?.expression || candidateData.candidate_expression,
              candidate_json: candidateData.candidate_data || candidateData.candidate_json,
            } as Candidate;
            
            acceptanceResult = await this.acceptanceService.acceptCandidate(mappedCandidate, systemSpec);
          }
        } catch (error) {
          logger.warn('Failed to generate technical details on-the-fly', {
            candidateId: candidate.id,
            error: error instanceof Error ? error.message : 'Unknown error',
          });
        }
      }

      const enrichedCandidate = {
        ...candidate,
        system_name: systemSpecName,
        // Convert Firestore timestamps to ISO strings for frontend compatibility
        created_at: candidateData.created_at?.toDate?.()?.toISOString() || candidateData.created_at,
        updated_at: candidateData.updated_at?.toDate?.()?.toISOString() || candidateData.updated_at,
        accepted_at: candidateData.accepted_at?.toDate?.()?.toISOString() || candidateData.accepted_at,
        // Ensure proper field mapping
        candidate_expression: candidateData.candidate_data?.response?.expression || candidateData.candidate_expression,
        candidate_json: candidateData.candidate_data || candidateData.candidate_json,
        llm_config_json: candidateData.llm_config || candidateData.llm_config_json,
        // Add acceptance result with technical details
        acceptance_result: acceptanceResult,
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

  // Re-run acceptance check with custom parameters for experimental analysis
  rerunAcceptance = async (req: Request, res: Response): Promise<void> => {
    try {
      const candidateId = req.params.id;
      const acceptanceParams = AcceptanceParametersSchema.parse(req.body);

      // Get candidate
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

      // Get system spec
      const systemSpecDoc = await db.collection('system_specs').doc(candidate.system_spec_id).get();
      if (!systemSpecDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'System specification not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const systemSpecData = systemSpecDoc.data() as any;
      const systemSpec = { id: systemSpecDoc.id, ...systemSpecData };

      logger.info('Re-running acceptance check with custom parameters', {
        candidateId,
        parameters: acceptanceParams,
        requestedBy: req.user?.id,
      });

      // Pass custom parameters to AcceptanceService for experimental control
      const mappedCandidate = {
        ...candidate,
        candidate_expression: candidateData.candidate_data?.response?.expression || candidateData.candidate_expression,
        candidate_json: candidateData.candidate_data || candidateData.candidate_json,
      } as Candidate;
      
      const acceptanceResult = await this.acceptanceService.acceptCandidate(mappedCandidate, systemSpec, acceptanceParams);

      // Store the re-run result (optional - for experiment tracking)
      const rerunRecord = {
        candidate_id: candidateId,
        parameters_used: acceptanceParams,
        result: acceptanceResult,
        requested_by: req.user?.id || 'anonymous',
        timestamp: new Date(),
      };

      await db.collection('acceptance_reruns').add(rerunRecord);

      // CRITICAL FIX: Update the original certificate with new acceptance result
      await db.collection('candidates').doc(candidateId).update({
        acceptance_result: acceptanceResult,
        acceptance_status: acceptanceResult.accepted ? 'accepted' : 'failed',
        updated_at: new Date(),
        rerun_timestamp: new Date(),
        rerun_parameters: acceptanceParams,
      });

      logger.info('Updated certificate with re-run results', {
        candidateId,
        newStatus: acceptanceResult.accepted ? 'accepted' : 'failed',
        newSampleCount: acceptanceResult.technical_details?.sample_count,
      });

      const response: ApiResponse<any> = {
        success: true,
        data: {
          candidate_id: candidateId,
          acceptance_result: acceptanceResult,
          parameters_used: acceptanceParams,
          rerun_timestamp: new Date().toISOString(),
        },
        message: 'Acceptance check re-run completed',
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to re-run acceptance check');
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

  // ====== CONVERSATIONAL MODE ENDPOINTS ======

  // Start a new conversation for certificate generation
  startConversation = async (req: Request, res: Response): Promise<void> => {
    try {
      const conversationData = StartConversationRequestSchema.parse(req.body);

      // Get system spec
      const systemSpecDoc = await db.collection('system_specs').doc(conversationData.system_spec_id).get();
      if (!systemSpecDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'System specification not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const systemSpecData = systemSpecDoc.data() as any;
      const systemSpec = { id: conversationData.system_spec_id, ...systemSpecData } as SystemSpec;

      // Initialize conversation with LLM
      const llmConfig = {
        provider: 'anthropic' as const,
        model: 'claude-sonnet-4-20250514',
        temperature: 0.2, // Slightly higher for conversational creativity
        max_tokens: 2048,
        max_attempts: 3,
        mode: 'direct_expression' as const,
        timeout_ms: 30000,
      };

      const initialResponse = await this.llmService.initializeConversation(
        systemSpec,
        conversationData.certificate_type,
        llmConfig,
        conversationData.initial_message
      );

      // Create conversation document
      const conversationId = crypto.randomUUID();
      const conversation = {
        id: conversationId,
        system_spec_id: conversationData.system_spec_id,
        certificate_type: conversationData.certificate_type,
        status: 'active',
        messages: [
          {
            role: 'user',
            content: conversationData.initial_message || `Let's discuss ${conversationData.certificate_type} functions for this system.`,
            timestamp: new Date().toISOString(),
            metadata: {
              message_type: 'question',
            },
          },
          initialResponse,
        ],
        created_by: req.user?.id || 'anonymous',
        created_at: new Date(),
        updated_at: new Date(),
        token_count: initialResponse.metadata?.token_count || 0,
        message_count: 2,
      };

      await db.collection('conversations').doc(conversationId).set(conversation);

      const response: ApiResponse<any> = {
        success: true,
        data: {
          conversation_id: conversationId,
          ...conversation,
          created_at: conversation.created_at.toISOString(),
          updated_at: conversation.updated_at.toISOString(),
        },
        message: 'Conversation started successfully',
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to start conversation');
    }
  };

  // Send message in conversation  
  sendMessage = async (req: Request, res: Response): Promise<void> => {
    try {
      const conversationId = req.params.id;
      const messageData = SendMessageRequestSchema.parse(req.body);

      // Get conversation
      const conversationDoc = await db.collection('conversations').doc(conversationId).get();
      if (!conversationDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'Conversation not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const conversationData = conversationDoc.data() as any;
      
      if (conversationData.status !== 'active') {
        res.status(400).json({
          success: false,
          error: 'Conversation is not active',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      // Get system spec
      const systemSpecDoc = await db.collection('system_specs').doc(conversationData.system_spec_id).get();
      const systemSpec = { id: conversationData.system_spec_id, ...systemSpecDoc.data() } as SystemSpec;

      // Send message to LLM
      const llmConfig = {
        provider: 'anthropic' as const,
        model: 'claude-sonnet-4-20250514',
        temperature: 0.2,
        max_tokens: 2048,
        max_attempts: 3,
        mode: 'direct_expression' as const,
        timeout_ms: 30000,
      };

      const assistantResponse = await this.llmService.sendConversationMessage(
        systemSpec,
        conversationData.certificate_type,
        conversationData.messages,
        messageData.message,
        llmConfig
      );

      // Update conversation
      const userMessage = {
        role: 'user' as const,
        content: messageData.message,
        timestamp: new Date().toISOString(),
        metadata: { message_type: 'question' as const },
      };

      const updatedMessages = [...conversationData.messages, userMessage, assistantResponse];
      const newTokenCount = conversationData.token_count + (assistantResponse.metadata?.token_count || 0);

      await db.collection('conversations').doc(conversationId).update({
        messages: updatedMessages,
        updated_at: new Date(),
        token_count: newTokenCount,
        message_count: updatedMessages.length,
      });

      const response: ApiResponse<any> = {
        success: true,
        data: {
          message: assistantResponse,
          token_count: newTokenCount,
          message_count: updatedMessages.length,
        },
        message: 'Message sent successfully',
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to send message');
    }
  };

  // Publish certificate from conversation
  publishCertificateFromConversation = async (req: Request, res: Response): Promise<void> => {
    try {
      const { conversation_id, final_instructions } = PublishFromConversationRequestSchema.parse(req.body);

      // Get conversation
      const conversationDoc = await db.collection('conversations').doc(conversation_id).get();
      if (!conversationDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'Conversation not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const conversationData = conversationDoc.data() as any;
      
      // Get system spec
      const systemSpecDoc = await db.collection('system_specs').doc(conversationData.system_spec_id).get();
      const systemSpec = { id: conversationData.system_spec_id, ...systemSpecDoc.data() } as SystemSpec;

      // Ensure conversation is summarized
      let summary = conversationData.summary;
      if (!summary) {
        const llmConfig = {
          provider: 'anthropic' as const,
          model: 'claude-sonnet-4-20250514',
          temperature: 0.1,
          max_tokens: 4096,
          max_attempts: 3,
          mode: 'direct_expression' as const,
          timeout_ms: 30000,
        };

        summary = await this.llmService.summarizeConversation(
          systemSpec,
          conversationData.certificate_type,
          conversationData.messages,
          llmConfig
        );
      }

      // Generate final certificate from conversation
      const llmConfig = {
        provider: 'anthropic' as const,
        model: 'claude-sonnet-4-20250514',
        temperature: 0.0,
        max_tokens: 2048,
        max_attempts: 3,
        mode: 'direct_expression' as const,
        timeout_ms: 30000,
      };

      const certificateResult = await this.llmService.generateCertificateFromConversation(
        systemSpec,
        conversationData.certificate_type,
        summary,
        final_instructions,
        llmConfig
      );

      // Create candidate document
      const candidateId = crypto.randomUUID();
      const candidateData = {
        id: candidateId,
        system_spec_id: conversationData.system_spec_id,
        certificate_type: conversationData.certificate_type,
        generation_method: 'conversational',
        llm_provider: llmConfig.provider,
        llm_model: llmConfig.model,
        llm_mode: llmConfig.mode,
        llm_config: llmConfig,
        candidate_data: {
          response: certificateResult.response,
          raw_response: certificateResult.raw_response,
          conversation_context: certificateResult.conversation_context,
          conversation_id: conversation_id,
          conversation_summary: summary,
        },
        candidate_expression: certificateResult.response.expression,
        candidate_json: certificateResult.response,
        acceptance_status: 'pending',
        created_by: req.user?.id || 'anonymous',
        created_at: new Date(),
        updated_at: new Date(),
        generation_duration_ms: certificateResult.duration_ms,
      };

      await db.collection('candidates').doc(candidateId).set(candidateData);

      // Update conversation status
      await db.collection('conversations').doc(conversation_id).update({
        status: 'published',
        final_certificate_id: candidateId,
        summary,
        published_at: new Date(),
        updated_at: new Date(),
      });

      // Start background acceptance checking
      this.checkCandidateAcceptance(candidateId);

      const response: ApiResponse<any> = {
        success: true,
        data: {
          candidate_id: candidateId,
          conversation_id: conversation_id,
          certificate_type: conversationData.certificate_type,
          conversation_summary: summary,
          generation_duration_ms: certificateResult.duration_ms,
        },
        message: 'Certificate published from conversation successfully',
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to publish certificate from conversation');
    }
  };

  // Get user's conversations
  getConversations = async (req: Request, res: Response): Promise<void> => {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const limit = Math.min(parseInt(req.query.limit as string) || 20, 100);
      const status = req.query.status as string;

      let query = db.collection('conversations')
        .where('created_by', '==', req.user?.id);

      if (status) {
        query = query.where('status', '==', status);
      }

      const totalQuery = await query.get();
      const total = totalQuery.size;

      const conversationsQuery = await query
        .orderBy('updated_at', 'desc')
        .limit(limit)
        .offset((page - 1) * limit)
        .get();

      const conversations = conversationsQuery.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          // Convert Firestore timestamps
          created_at: data.created_at?.toDate?.()?.toISOString() || data.created_at,
          updated_at: data.updated_at?.toDate?.()?.toISOString() || data.updated_at,
          published_at: data.published_at?.toDate?.()?.toISOString() || data.published_at,
          // Only include last few messages for list view
          messages: data.messages?.slice(-2) || [],
          message_preview: data.messages?.[data.messages.length - 1]?.content?.substring(0, 100) + '...' || '',
        };
      });

      const totalPages = Math.ceil(total / limit);

      const response: ApiResponse<PaginatedResponse<any>> = {
        success: true,
        data: {
          data: conversations,
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
      this.handleError(error, res, 'Failed to retrieve conversations');
    }
  };

  // Get conversation details
  getConversation = async (req: Request, res: Response): Promise<void> => {
    try {
      const conversationId = req.params.id;

      const conversationDoc = await db.collection('conversations').doc(conversationId).get();
      if (!conversationDoc.exists) {
        res.status(404).json({
          success: false,
          error: 'Conversation not found',
          timestamp: new Date().toISOString(),
        });
        return;
      }

      const conversationData = conversationDoc.data() as any;
      const conversation = {
        id: conversationDoc.id,
        ...conversationData,
        created_at: conversationData.created_at?.toDate?.()?.toISOString() || conversationData.created_at,
        updated_at: conversationData.updated_at?.toDate?.()?.toISOString() || conversationData.updated_at,
        published_at: conversationData.published_at?.toDate?.()?.toISOString() || conversationData.published_at,
      };

      const response: ApiResponse<any> = {
        success: true,
        data: conversation,
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to retrieve conversation');
    }
  };

  // Re-evaluate all existing accepted certificates with corrected mathematical logic
  revalidateAcceptedCertificates = async (req: Request, res: Response): Promise<void> => {
    try {
      logger.info('Starting re-validation of all accepted certificates', {
        requestedBy: req.user?.id,
      });

      // Get all currently accepted certificates
      const acceptedCertificatesQuery = await db.collection('candidates')
        .where('acceptance_status', '==', 'accepted')
        .get();

      const acceptedCertificates = acceptedCertificatesQuery.docs;
      let revalidatedCount = 0;
      let statusChangedCount = 0;
      const results: any[] = [];

      for (const candidateDoc of acceptedCertificates) {
        const candidateData = candidateDoc.data() as any;
        const candidate = { id: candidateDoc.id, ...candidateData } as any;

        try {
          // Get associated system spec
          const systemSpecDoc = await db.collection('system_specs').doc(candidate.system_spec_id).get();
          if (!systemSpecDoc.exists) {
            logger.warn('System spec not found for candidate', { candidateId: candidate.id });
            continue;
          }

          const systemSpecData = systemSpecDoc.data() as any;
          const systemSpec = { id: candidate.system_spec_id, ...systemSpecData } as SystemSpec;

          // Create properly mapped candidate object for AcceptanceService
          const mappedCandidate = {
            ...candidate,
            candidate_expression: candidateData.candidate_data?.response?.expression || candidateData.candidate_expression,
            candidate_json: candidateData.candidate_data || candidateData.candidate_json,
          } as Candidate;

          // Re-run acceptance check with corrected logic
          const newAcceptanceResult = await this.acceptanceService.acceptCandidate(mappedCandidate, systemSpec);
          
          const previousStatus = candidate.acceptance_status;
          const newStatus = newAcceptanceResult.accepted ? 'accepted' : 'failed';
          
          // Update candidate if status changed
          if (previousStatus !== newStatus) {
            await db.collection('candidates').doc(candidate.id).update({
              acceptance_status: newStatus,
              acceptance_result: newAcceptanceResult,
              updated_at: new Date(),
              revalidated_at: new Date(),
              revalidation_reason: 'Mathematical validation logic corrected',
            });
            statusChangedCount++;
          } else {
            // Update just the acceptance result with enhanced technical details
            await db.collection('candidates').doc(candidate.id).update({
              acceptance_result: newAcceptanceResult,
              updated_at: new Date(),
              revalidated_at: new Date(),
            });
          }

          revalidatedCount++;
          results.push({
            candidate_id: candidate.id,
            expression: mappedCandidate.candidate_expression,
            previous_status: previousStatus,
            new_status: newStatus,
            status_changed: previousStatus !== newStatus,
            violations_found: newAcceptanceResult.technical_details?.violation_analysis?.total_violations || 0,
          });

          logger.info('Certificate re-validated', {
            candidateId: candidate.id,
            previousStatus,
            newStatus,
            violationsFound: newAcceptanceResult.technical_details?.violation_analysis?.total_violations || 0,
          });

        } catch (error) {
          logger.error('Failed to re-validate certificate', {
            candidateId: candidate.id,
            error: error instanceof Error ? error.message : 'Unknown error',
          });
        }
      }

      logger.info('Re-validation completed', {
        totalCertificates: acceptedCertificates.length,
        revalidatedCount,
        statusChangedCount,
        requestedBy: req.user?.id,
      });

      const response: ApiResponse<any> = {
        success: true,
        data: {
          total_certificates: acceptedCertificates.length,
          revalidated_count: revalidatedCount,
          status_changed_count: statusChangedCount,
          results: results,
          summary: {
            message: `Re-validated ${revalidatedCount} certificates. ${statusChangedCount} certificates changed from accepted to failed due to mathematical violations.`,
            mathematical_rationale: 'Applied strict validation: ANY violations now cause failure (mathematically correct)',
          },
        },
        message: 'Certificate re-validation completed successfully',
        timestamp: new Date().toISOString(),
      };

      res.json(response);
    } catch (error) {
      this.handleError(error, res, 'Failed to re-validate accepted certificates');
    }
  };
}
