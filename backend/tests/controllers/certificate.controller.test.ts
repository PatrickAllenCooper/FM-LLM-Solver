import { Request, Response } from 'express';
import { CertificateController } from '../../src/controllers/certificate.controller';
import { LLMService } from '../../src/services/llm.service';
import { VerificationService } from '../../src/services/verification.service';
import { db } from '../../src/utils/database';
import { User, SystemSpec, Candidate } from '../../src/types/database';

// Mock services and database
jest.mock('../../src/services/llm.service');
jest.mock('../../src/services/verification.service');
jest.mock('../../src/utils/database');

describe('CertificateController', () => {
  let certificateController: CertificateController;
  let mockLLMService: jest.Mocked<LLMService>;
  let mockVerificationService: jest.Mocked<VerificationService>;
  let mockDb: any;
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockLLMService = {
      generateCertificate: jest.fn(),
      testConnection: jest.fn(),
    } as any;

    mockVerificationService = {
      verifyCertificate: jest.fn(),
      generateBaseline: jest.fn(),
    } as any;

    mockDb = {
      select: jest.fn().mockReturnThis(),
      where: jest.fn().mockReturnThis(),
      first: jest.fn(),
      insert: jest.fn().mockReturnThis(),
      returning: jest.fn(),
      update: jest.fn().mockReturnThis(),
      orderBy: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
      offset: jest.fn().mockReturnThis(),
    };

    (LLMService as jest.MockedClass<typeof LLMService>).mockImplementation(() => mockLLMService);
    (VerificationService as jest.MockedClass<typeof VerificationService>).mockImplementation(() => mockVerificationService);
    (db as jest.MockedFunction<typeof db>).mockImplementation((table: string) => mockDb);
    
    certificateController = new CertificateController();
    
    mockRequest = {
      body: {},
      params: {},
      query: {},
      user: {
        id: 'user-123',
        email: 'test@example.com',
        role: 'researcher',
        password_hash: 'hash',
        created_at: new Date(),
        updated_at: new Date(),
      },
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
  });

  describe('createSystemSpec', () => {
    it('should create system specification successfully', async () => {
      const systemSpecData = {
        name: 'Test System',
        description: 'A test linear system',
        system_type: 'continuous' as const,
        dimension: 2,
        dynamics: {
          type: 'linear',
          variables: ['x1', 'x2'],
          equations: ['-x1', '-x2'],
        },
      };

      const createdSpec = {
        id: 'spec-123',
        ...systemSpecData,
        owner_user_id: 'user-123',
        created_by: 'user-123',
        created_at: new Date(),
        updated_at: new Date(),
        spec_version: '1.0',
        hash: 'abc123',
      };

      mockRequest.body = systemSpecData;
      mockDb.returning.mockResolvedValue([createdSpec]);

      await certificateController.createSystemSpec(mockRequest as Request, mockResponse as Response);

      expect(mockDb.insert).toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: createdSpec,
      });
    });

    it('should handle validation errors', async () => {
      mockRequest.body = {
        name: '', // invalid
        dimension: -1, // invalid
      };

      await certificateController.createSystemSpec(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Validation failed',
        details: expect.any(Array),
      });
    });

    it('should handle database errors', async () => {
      const systemSpecData = {
        name: 'Test System',
        system_type: 'continuous' as const,
        dimension: 2,
        dynamics: {
          type: 'linear',
          variables: ['x1', 'x2'],
          equations: ['-x1', '-x2'],
        },
      };

      mockRequest.body = systemSpecData;
      mockDb.returning.mockRejectedValue(new Error('Database error'));

      await certificateController.createSystemSpec(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(500);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Internal server error',
      });
    });
  });

  describe('generateCertificate', () => {
    it('should generate certificate successfully', async () => {
      const generateData = {
        system_spec_id: 'spec-123',
        certificate_type: 'lyapunov' as const,
        llm_config: {
          provider: 'anthropic' as const,
          model: 'claude-3-sonnet-20240229',
          temperature: 0.0,
          max_tokens: 1000,
          max_attempts: 3,
          mode: 'direct_expression' as const,
          timeout_ms: 30000,
        },
      };

      const mockSystemSpec: SystemSpec = {
        id: 'spec-123',
        owner_user_id: 'user-123',
        name: 'Test System',
        system_type: 'continuous',
        dimension: 2,
        dynamics_json: {},
        created_by: 'user-123',
        created_at: new Date(),
        updated_at: new Date(),
        spec_version: '1.0',
        hash: 'abc123',
      };

      const mockLLMResponse = {
        response: {
          certificate_type: 'lyapunov',
          method: 'direct_expression',
          form: 'polynomial',
          expression: 'x1^2 + x2^2',
          properties: {},
        },
        duration_ms: 1500,
      };

      const mockVerificationResult = {
        verified: true,
        verification_method: 'mathematical' as const,
        margin: 0.5,
        solver_output: 'Verification successful',
        duration_ms: 500,
      };

      const mockCandidate: Candidate = {
        id: 'candidate-123',
        run_id: 'run-123',
        certificate_type: 'lyapunov',
        generation_method: 'llm',
        candidate_expression: 'x1^2 + x2^2',
        created_at: new Date(),
      };

      mockRequest.body = generateData;
      mockDb.first.mockResolvedValue(mockSystemSpec);
      mockDb.returning.mockResolvedValue([mockCandidate]);
      mockLLMService.generateCertificate.mockResolvedValue(mockLLMResponse);
      mockVerificationService.verifyCertificate.mockResolvedValue(mockVerificationResult);

      await certificateController.generateCertificate(mockRequest as Request, mockResponse as Response);

      expect(mockLLMService.generateCertificate).toHaveBeenCalledWith(
        mockSystemSpec,
        'lyapunov',
        generateData.llm_config
      );
      expect(mockVerificationService.verifyCertificate).toHaveBeenCalledWith(
        mockCandidate,
        mockSystemSpec
      );
      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: {
          candidate: mockCandidate,
          verification: mockVerificationResult,
          generation_time_ms: 1500,
          verification_time_ms: 500,
        },
      });
    });

    it('should handle system not found error', async () => {
      const generateData = {
        system_spec_id: 'nonexistent-spec',
        certificate_type: 'lyapunov' as const,
        llm_config: {
          provider: 'anthropic' as const,
          model: 'claude-3-sonnet-20240229',
          temperature: 0.0,
          max_tokens: 1000,
          max_attempts: 3,
          mode: 'direct_expression' as const,
          timeout_ms: 30000,
        },
      };

      mockRequest.body = generateData;
      mockDb.first.mockResolvedValue(null);

      await certificateController.generateCertificate(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(404);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'System specification not found',
      });
    });

    it('should handle LLM generation errors', async () => {
      const generateData = {
        system_spec_id: 'spec-123',
        certificate_type: 'lyapunov' as const,
        llm_config: {
          provider: 'anthropic' as const,
          model: 'claude-3-sonnet-20240229',
          temperature: 0.0,
          max_tokens: 1000,
          max_attempts: 3,
          mode: 'direct_expression' as const,
          timeout_ms: 30000,
        },
      };

      const mockSystemSpec: SystemSpec = {
        id: 'spec-123',
        owner_user_id: 'user-123',
        name: 'Test System',
        system_type: 'continuous',
        dimension: 2,
        dynamics_json: {},
        created_by: 'user-123',
        created_at: new Date(),
        updated_at: new Date(),
        spec_version: '1.0',
        hash: 'abc123',
      };

      mockRequest.body = generateData;
      mockDb.first.mockResolvedValue(mockSystemSpec);
      mockLLMService.generateCertificate.mockRejectedValue(new Error('LLM API error'));

      await certificateController.generateCertificate(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(500);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Certificate generation failed',
        details: 'LLM API error',
      });
    });
  });

  describe('getCertificates', () => {
    it('should return paginated certificates', async () => {
      const mockCertificates = [
        {
          id: 'candidate-1',
          run_id: 'run-1',
          certificate_type: 'lyapunov',
          generation_method: 'llm',
          candidate_expression: 'x1^2 + x2^2',
          created_at: new Date(),
        },
        {
          id: 'candidate-2',
          run_id: 'run-2',
          certificate_type: 'barrier',
          generation_method: 'sos',
          candidate_expression: '1 - x1^2 - x2^2',
          created_at: new Date(),
        },
      ];

      mockRequest.query = { page: '1', limit: '10' };
      mockDb.limit.mockResolvedValue(mockCertificates);

      await certificateController.getCertificates(mockRequest as Request, mockResponse as Response);

      expect(mockDb.limit).toHaveBeenCalledWith(10);
      expect(mockDb.offset).toHaveBeenCalledWith(0);
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: {
          certificates: mockCertificates,
          pagination: {
            page: 1,
            limit: 10,
            total: mockCertificates.length,
            hasMore: false,
          },
        },
      });
    });

    it('should handle database errors', async () => {
      mockRequest.query = { page: '1', limit: '10' };
      mockDb.limit.mockRejectedValue(new Error('Database error'));

      await certificateController.getCertificates(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(500);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Internal server error',
      });
    });
  });

  describe('getCertificateById', () => {
    it('should return certificate by ID', async () => {
      const mockCandidate = {
        id: 'candidate-123',
        run_id: 'run-123',
        certificate_type: 'lyapunov',
        generation_method: 'llm',
        candidate_expression: 'x1^2 + x2^2',
        created_at: new Date(),
      };

      mockRequest.params = { id: 'candidate-123' };
      mockDb.first.mockResolvedValue(mockCandidate);

      await certificateController.getCertificateById(mockRequest as Request, mockResponse as Response);

      expect(mockDb.where).toHaveBeenCalledWith('id', 'candidate-123');
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: mockCandidate,
      });
    });

    it('should handle certificate not found', async () => {
      mockRequest.params = { id: 'nonexistent-candidate' };
      mockDb.first.mockResolvedValue(null);

      await certificateController.getCertificateById(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(404);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Certificate not found',
      });
    });
  });

  describe('generateBaseline', () => {
    it('should generate baseline certificate successfully', async () => {
      const baselineData = {
        system_spec_id: 'spec-123',
        certificate_type: 'lyapunov' as const,
        method: 'quadratic_template' as const,
      };

      const mockSystemSpec: SystemSpec = {
        id: 'spec-123',
        owner_user_id: 'user-123',
        name: 'Test System',
        system_type: 'continuous',
        dimension: 2,
        dynamics_json: {},
        created_by: 'user-123',
        created_at: new Date(),
        updated_at: new Date(),
        spec_version: '1.0',
        hash: 'abc123',
      };

      const mockBaselineResult = {
        success: true,
        expression: 'x1^2 + x2^2',
        margin: 0.5,
        executionTime: 100,
        method: 'quadratic_template',
      };

      mockRequest.body = baselineData;
      mockDb.first.mockResolvedValue(mockSystemSpec);
      mockVerificationService.generateBaseline.mockResolvedValue(mockBaselineResult);

      await certificateController.generateBaseline(mockRequest as Request, mockResponse as Response);

      expect(mockVerificationService.generateBaseline).toHaveBeenCalledWith(
        mockSystemSpec,
        'lyapunov',
        'quadratic_template'
      );
      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: true,
        data: mockBaselineResult,
      });
    });

    it('should handle baseline generation errors', async () => {
      const baselineData = {
        system_spec_id: 'spec-123',
        certificate_type: 'lyapunov' as const,
        method: 'quadratic_template' as const,
      };

      const mockSystemSpec: SystemSpec = {
        id: 'spec-123',
        owner_user_id: 'user-123',
        name: 'Test System',
        system_type: 'continuous',
        dimension: 2,
        dynamics_json: {},
        created_by: 'user-123',
        created_at: new Date(),
        updated_at: new Date(),
        spec_version: '1.0',
        hash: 'abc123',
      };

      mockRequest.body = baselineData;
      mockDb.first.mockResolvedValue(mockSystemSpec);
      mockVerificationService.generateBaseline.mockRejectedValue(new Error('Baseline generation failed'));

      await certificateController.generateBaseline(mockRequest as Request, mockResponse as Response);

      expect(mockResponse.status).toHaveBeenCalledWith(500);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: 'Baseline generation failed',
        details: 'Baseline generation failed',
      });
    });
  });
});
