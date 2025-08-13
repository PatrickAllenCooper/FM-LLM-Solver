import { VerificationService } from '../../src/services/verification.service';
import { SystemSpec, Candidate } from '../../src/types/database';
import { MathService } from '../../src/services/math.service';
import { BaselineService } from '../../src/services/baseline.service';

// Mock the services
jest.mock('../../src/services/math.service');
jest.mock('../../src/services/baseline.service');

describe('VerificationService', () => {
  let verificationService: VerificationService;
  let mockMathService: jest.Mocked<MathService>;
  let mockBaselineService: jest.Mocked<BaselineService>;
  let mockSystemSpec: SystemSpec;
  let mockCandidate: Candidate;

  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    
    verificationService = new VerificationService();
    mockMathService = (verificationService as any).mathService;
    mockBaselineService = (verificationService as any).baselineService;
    
    // Create mock system specification
    mockSystemSpec = {
      id: 'test-system-1',
      owner_user_id: 'user-1',
      name: 'Test System',
      system_type: 'continuous',
      dimension: 2,
      spec_version: '1.0',
      created_by: 'test-user',
      constraints_json: {},
      dynamics_json: {
        variables: ['x1', 'x2'],
        equations: ['-x1', '-x2'],
      },
      initial_set_json: {
        type: 'box',
        bounds: { x1: { min: -1, max: 1 }, x2: { min: -1, max: 1 } }
      },
      safe_set_json: {
        type: 'box',
        bounds: { x1: { min: -5, max: 5 }, x2: { min: -5, max: 5 } }
      },
      unsafe_set_json: {
        type: 'box',
        bounds: { x1: { min: 10, max: 15 }, x2: { min: 10, max: 15 } }
      },
      hash: 'test-hash',
      description: 'Test system',
      created_at: new Date(),
      updated_at: new Date(),
    };

    // Create mock candidate
    mockCandidate = {
      id: 'test-candidate-1',
      run_id: 'test-run-1',
      system_spec_id: 'test-system-1',
      certificate_type: 'lyapunov',
      generation_method: 'llm',
      candidate_expression: 'x1^2 + x2^2',
      canonical_expression: 'x1^2 + x2^2',
      latex_expression: 'x_1^2 + x_2^2',
      ast_json: {},
      degree: 2,
      term_count: 2,
      generation_time_ms: 1000,
      created_at: new Date(),
    };
  });

  describe('verifyCertificate', () => {
    it('should verify valid Lyapunov certificate', async () => {
      // Mock successful Lyapunov verification
      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: true,
        decreasing: true,
        margin: 0.5,
        violations: [],
      });

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(true);
      expect(result.margin).toBe(0.5);
      expect(result.verification_method).toBe('mathematical');
      expect(result.duration_ms).toBeGreaterThan(0);
      expect(mockMathService.verifyLyapunovConditions).toHaveBeenCalledWith(
        'x1^2 + x2^2',
        { x1: '-x1', x2: '-x2' },
        expect.any(Object)
      );
    });

    it('should verify valid barrier certificate', async () => {
      mockCandidate.certificate_type = 'barrier';
      mockCandidate.candidate_expression = 'x1 + x2 - 5';

      // Mock successful barrier verification
      mockMathService.verifyBarrierConditions.mockReturnValue({
        separatesRegions: true,
        nonIncreasing: true,
        margin: 0.3,
        violations: [],
      });

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(true);
      expect(result.margin).toBe(0.3);
      expect(result.verification_method).toBe('mathematical');
      expect(mockMathService.verifyBarrierConditions).toHaveBeenCalled();
    });

    it('should verify inductive invariant', async () => {
      mockCandidate.certificate_type = 'inductive_invariant';
      mockCandidate.candidate_expression = 'x1^2 + x2^2 - 1';

      // Mock evaluation calls
      mockMathService.evaluate
        .mockReturnValueOnce({ value: 0.5 }) // Initial set check
        .mockReturnValueOnce({ value: 0.3 }) // More initial set checks
        .mockReturnValueOnce({ value: -0.2 }); // Unsafe set check

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(true);
      expect(result.verification_method).toBe('mathematical');
    });

    it('should handle failed Lyapunov verification', async () => {
      // Mock failed verification with violations
      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: false,
        decreasing: true,
        margin: -0.1,
        violations: [{
          point: { x1: 1, x2: 0 },
          condition: 'positive_definite' as const,
          value: -0.5,
        }],
      });

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(false);
      expect(result.counterexample).toBeDefined();
      expect(result.counterexample!.violation_type).toBe('positive_definite');
      expect(result.counterexample!.violation_magnitude).toBe(0.5);
    });

    it('should handle failed barrier verification', async () => {
      mockCandidate.certificate_type = 'barrier';

      mockMathService.verifyBarrierConditions.mockReturnValue({
        separatesRegions: false,
        nonIncreasing: true,
        margin: -0.2,
        violations: [{
          point: { x1: 2, x2: 3 },
          condition: 'separation' as const,
          value: -0.8,
        }],
      });

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(false);
      expect(result.counterexample).toBeDefined();
      expect(result.counterexample!.violation_type).toBe('separation');
    });

    it('should handle missing safe/unsafe sets for barrier', async () => {
      mockCandidate.certificate_type = 'barrier';
      mockSystemSpec.safe_set_json = undefined;

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(false);
      expect(result.solver_output).toMatch(/Safe and unsafe sets must be defined/);
    });

    it('should handle verification errors gracefully', async () => {
      mockMathService.verifyLyapunovConditions.mockImplementation(() => {
        throw new Error('Mathematical error');
      });

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(false);
      expect(result.solver_output).toMatch(/Verification error: Mathematical error/);
    });

    it('should handle unsupported certificate types', async () => {
      mockCandidate.certificate_type = 'unsupported' as any;

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);

      expect(result.verified).toBe(false);
      expect(result.solver_output).toMatch(/Unsupported certificate type/);
    });
  });

  describe('generateBaseline', () => {
    it('should generate Lyapunov baseline successfully', async () => {
      mockBaselineService.generateLyapunovBaseline.mockResolvedValue({
        method: 'quadratic_template',
        success: true,
        certificate_expression: 'x1^2 + x2^2',
        execution_time_ms: 50,
        solver_output: 'Success',
        margin: 0.5,
      });

      const result = await verificationService.generateBaseline(
        mockSystemSpec,
        'lyapunov',
        'quadratic_template'
      );

      expect(result.success).toBe(true);
      expect(result.expression).toBe('x1^2 + x2^2');
      expect(result.margin).toBe(0.5);
      expect(result.executionTime).toBe(50);
      expect(result.method).toBe('quadratic_template');
    });

    it('should generate barrier baseline successfully', async () => {
      mockBaselineService.generateBarrierBaseline.mockResolvedValue({
        method: 'linear_template',
        success: true,
        certificate_expression: 'x1 + x2 - 5',
        execution_time_ms: 30,
        solver_output: 'Success',
        margin: 0.3,
      });

      const result = await verificationService.generateBaseline(
        mockSystemSpec,
        'barrier',
        'linear_template'
      );

      expect(result.success).toBe(true);
      expect(result.expression).toBe('x1 + x2 - 5');
      expect(result.margin).toBe(0.3);
      expect(result.executionTime).toBe(30);
      expect(result.method).toBe('linear_template');
    });

    it('should handle baseline generation failures', async () => {
      mockBaselineService.generateLyapunovBaseline.mockResolvedValue({
        method: 'sos',
        success: false,
        execution_time_ms: 100,
        solver_output: 'SOS solver failed',
      });

      const result = await verificationService.generateBaseline(
        mockSystemSpec,
        'lyapunov',
        'sos'
      );

      expect(result.success).toBe(false);
      expect(result.executionTime).toBe(100);
      expect(result.method).toBe('sos');
    });

    it('should handle baseline service errors', async () => {
      mockBaselineService.generateLyapunovBaseline.mockRejectedValue(
        new Error('Baseline service error')
      );

      const result = await verificationService.generateBaseline(
        mockSystemSpec,
        'lyapunov'
      );

      expect(result.success).toBe(false);
      expect(result.executionTime).toBe(0);
    });
  });

  describe('Helper methods', () => {
    describe('extractDomain', () => {
      it('should extract domain from safe set bounds', () => {
        const extractDomain = (verificationService as any).extractDomain.bind(verificationService);
        const domain = extractDomain(mockSystemSpec);

        expect(domain).toEqual({
          x1: { min: -5, max: 5 },
          x2: { min: -5, max: 5 },
        });
      });

      it('should provide default bounds for missing variables', () => {
        const systemWithoutBounds = {
          ...mockSystemSpec,
          safe_set_json: null,
          dimension: 3,
        };

        const extractDomain = (verificationService as any).extractDomain.bind(verificationService);
        const domain = extractDomain(systemWithoutBounds);

        expect(domain.x1).toEqual({ min: -10, max: 10 });
        expect(domain.x2).toEqual({ min: -10, max: 10 });
        expect(domain.x3).toEqual({ min: -10, max: 10 });
      });
    });

    describe('extractDynamicsMap', () => {
      it('should extract dynamics from equations array', () => {
        const extractDynamicsMap = (verificationService as any).extractDynamicsMap.bind(verificationService);
        const dynamicsMap = extractDynamicsMap(mockSystemSpec.dynamics_json);

        expect(dynamicsMap).toEqual({
          x1: '-x1',
          x2: '-x2',
        });
      });

      it('should extract dynamics from variables and rhs', () => {
        const dynamicsWithRhs = {
          variables: ['y1', 'y2'],
          rhs: ['sin(y1)', 'cos(y2)'],
        };

        const extractDynamicsMap = (verificationService as any).extractDynamicsMap.bind(verificationService);
        const dynamicsMap = extractDynamicsMap(dynamicsWithRhs);

        expect(dynamicsMap).toEqual({
          y1: 'sin(y1)',
          y2: 'cos(y2)',
        });
      });
    });

    describe('sampleFromSet', () => {
      it('should sample from box sets', () => {
        const boxSet = {
          type: 'box',
          bounds: { x1: { min: 0, max: 1 }, x2: { min: -1, max: 1 } }
        };

        const sampleFromSet = (verificationService as any).sampleFromSet.bind(verificationService);
        const samples = sampleFromSet(boxSet, 10);

        expect(samples).toHaveLength(10);
        for (const sample of samples) {
          expect(sample.x1).toBeGreaterThanOrEqual(0);
          expect(sample.x1).toBeLessThanOrEqual(1);
          expect(sample.x2).toBeGreaterThanOrEqual(-1);
          expect(sample.x2).toBeLessThanOrEqual(1);
        }
      });

      it('should handle ellipsoid sets', () => {
        const ellipsoidSet = {
          type: 'ellipsoid',
          center: { x1: 0, x2: 0 },
          matrix: [[1, 0], [0, 1]]
        };

        const sampleFromSet = (verificationService as any).sampleFromSet.bind(verificationService);
        const samples = sampleFromSet(ellipsoidSet, 10);

        expect(samples).toHaveLength(10);
        // All samples should be within reasonable bounds around center
        for (const sample of samples) {
          const distance = Math.sqrt(sample.x1 * sample.x1 + sample.x2 * sample.x2);
          expect(distance).toBeLessThan(5); // Reasonable bound
        }
      });

      it('should handle empty or invalid sets', () => {
        const sampleFromSet = (verificationService as any).sampleFromSet.bind(verificationService);
        
        expect(sampleFromSet(null, 10)).toHaveLength(0);
        expect(sampleFromSet({}, 10)).toHaveLength(0);
        expect(sampleFromSet({ type: 'unknown' }, 10)).toHaveLength(0);
      });
    });
  });

  describe('Performance and edge cases', () => {
    it('should complete verification within reasonable time', async () => {
      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: true,
        decreasing: true,
        margin: 0.5,
        violations: [],
      });

      const startTime = Date.now();
      await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(1000); // Should complete within 1 second
    });

    it('should handle large expressions', async () => {
      const largeExpression = Array.from({ length: 100 }, (_, i) => `x${i % 2 + 1}^2`).join(' + ');
      mockCandidate.candidate_expression = largeExpression;

      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: true,
        decreasing: true,
        margin: 0.1,
        violations: [],
      });

      const result = await verificationService.verifyCertificate(mockCandidate, mockSystemSpec);
      expect(result.verified).toBe(true);
    });

    it('should handle high-dimensional systems', async () => {
      const highDimSystem = {
        ...mockSystemSpec,
        dimension: 10,
        dynamics_json: {
          variables: Array.from({ length: 10 }, (_, i) => `x${i + 1}`),
          equations: Array.from({ length: 10 }, (_, i) => `-x${i + 1}`),
        },
      };

      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: true,
        decreasing: true,
        margin: 0.1,
        violations: [],
      });

      const result = await verificationService.verifyCertificate(mockCandidate, highDimSystem);
      expect(result.verified).toBe(true);
    });
  });
});
