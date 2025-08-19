import { AcceptanceService } from '../../src/services/acceptance.service';
import { SystemSpec, Candidate } from '../../src/types/database';
import { MathService } from '../../src/services/math.service';
import { BaselineService } from '../../src/services/baseline.service';

// Mock the services
jest.mock('../../src/services/math.service');
jest.mock('../../src/services/baseline.service');

describe('AcceptanceService', () => {
  let acceptanceService: AcceptanceService;
  let mockMathService: jest.Mocked<MathService>;
  let mockBaselineService: jest.Mocked<BaselineService>;
  let mockSystemSpec: SystemSpec;
  let mockCandidate: Candidate;

  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    
    acceptanceService = new AcceptanceService();
    mockMathService = (acceptanceService as any).mathService;
    mockBaselineService = (acceptanceService as any).baselineService;
    
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

  describe('acceptCandidate', () => {
    it('should accept valid Lyapunov candidate', async () => {
      // Mock successful Lyapunov acceptance check
      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: true,
        decreasing: true,
        margin: 0.5,
        violations: [],
      });

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(true);
      expect(result.margin).toBe(0.5);
      expect(result.acceptance_method).toBe('mathematical');
      expect(result.duration_ms).toBeGreaterThan(0);
      expect(mockMathService.verifyLyapunovConditions).toHaveBeenCalledWith(
        'x1^2 + x2^2',
        { x1: '-x1', x2: '-x2' },
        expect.any(Object)
      );
    });

    it('should accept valid barrier candidate', async () => {
      mockCandidate.certificate_type = 'barrier';
      mockCandidate.candidate_expression = 'x1 + x2 - 5';

      // Mock successful barrier acceptance check
      mockMathService.verifyBarrierConditions.mockReturnValue({
        separatesRegions: true,
        nonIncreasing: true,
        margin: 0.3,
        violations: [],
      });

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(true);
      expect(result.margin).toBe(0.3);
      expect(result.acceptance_method).toBe('mathematical');
      expect(mockMathService.verifyBarrierConditions).toHaveBeenCalled();
    });

    it('should check inductive invariant candidate', async () => {
      mockCandidate.certificate_type = 'inductive_invariant';
      mockCandidate.candidate_expression = 'x1^2 + x2^2 - 1';

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.acceptance_method).toBe('mathematical');
      expect(result.duration_ms).toBeGreaterThan(0);
    });

    it('should reject invalid Lyapunov candidate', async () => {
      // Mock failed Lyapunov acceptance check
      mockMathService.verifyLyapunovConditions.mockReturnValue({
        positiveDefinite: false,
        decreasing: true,
        margin: -0.1,
        violations: [{
          point: { x1: 1, x2: 0 },
          condition: 'positive_definite' as const,
          value: -0.1
        }],
      });

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(false);
      expect(result.counterexample).toBeDefined();
      expect(result.counterexample?.violation_type).toBe('positive_definite');
      expect(result.acceptance_method).toBe('mathematical');
    });

    it('should reject invalid barrier candidate', async () => {
      mockCandidate.certificate_type = 'barrier';

      // Mock failed barrier acceptance check
      mockMathService.verifyBarrierConditions.mockReturnValue({
        separatesRegions: false,
        nonIncreasing: true,
        margin: -0.2,
        violations: [{
          point: { x1: 2, x2: 2 },
          condition: 'separation' as const,
          value: -0.2
        }],
      });

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(false);
      expect(result.counterexample).toBeDefined();
      expect(result.counterexample?.violation_type).toBe('separation');
    });

    it('should handle acceptance check errors gracefully', async () => {
      // Mock service throwing an error
      mockMathService.verifyLyapunovConditions.mockImplementation(() => {
        throw new Error('Math service error');
      });

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(false);
      expect(result.acceptance_method).toBe('mathematical');
      expect(result.solver_output).toContain('Math service error');
    });

    it('should handle unsupported certificate type', async () => {
      // @ts-ignore - Testing invalid certificate type
      mockCandidate.certificate_type = 'invalid_type';

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(false);
      expect(result.acceptance_method).toBe('symbolic');
      expect(result.solver_output).toContain('Unsupported certificate type');
    });

    it('should require safe and unsafe sets for barrier candidates', async () => {
      mockCandidate.certificate_type = 'barrier';
      mockSystemSpec.safe_set_json = null;

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(false);
      expect(result.solver_output).toContain('Safe and unsafe sets must be defined');
    });

    it('should require initial set for inductive invariant candidates', async () => {
      mockCandidate.certificate_type = 'inductive_invariant';
      mockSystemSpec.initial_set_json = null;

      const result = await acceptanceService.acceptCandidate(mockCandidate, mockSystemSpec);

      expect(result.accepted).toBe(false);
      expect(result.solver_output).toContain('Initial set must be defined');
    });
  });

  describe('generateBaseline', () => {
    it('should generate Lyapunov baseline successfully', async () => {
      mockBaselineService.generateLyapunovBaseline.mockResolvedValue({
        success: true,
        certificate_expression: 'x1^2 + x2^2',
        margin: 0.4,
        execution_time_ms: 150,
        method: 'quadratic_template',
      });

      const result = await acceptanceService.generateBaseline(
        mockSystemSpec,
        'lyapunov',
        'quadratic_template'
      );

      expect(result.success).toBe(true);
      expect(result.expression).toBe('x1^2 + x2^2');
      expect(result.margin).toBe(0.4);
      expect(result.executionTime).toBe(150);
      expect(result.method).toBe('quadratic_template');
      expect(mockBaselineService.generateLyapunovBaseline).toHaveBeenCalledWith(
        mockSystemSpec,
        'quadratic_template'
      );
    });

    it('should generate barrier baseline successfully', async () => {
      mockBaselineService.generateBarrierBaseline.mockResolvedValue({
        success: true,
        certificate_expression: 'x1 + x2 - 3',
        margin: 0.2,
        execution_time_ms: 200,
        method: 'linear_template',
      });

      const result = await acceptanceService.generateBaseline(
        mockSystemSpec,
        'barrier',
        'linear_template'
      );

      expect(result.success).toBe(true);
      expect(result.expression).toBe('x1 + x2 - 3');
      expect(result.margin).toBe(0.2);
      expect(result.executionTime).toBe(200);
      expect(result.method).toBe('linear_template');
      expect(mockBaselineService.generateBarrierBaseline).toHaveBeenCalledWith(
        mockSystemSpec,
        'linear_template'
      );
    });

    it('should handle baseline generation failure', async () => {
      mockBaselineService.generateLyapunovBaseline.mockRejectedValue(
        new Error('Baseline generation failed')
      );

      const result = await acceptanceService.generateBaseline(
        mockSystemSpec,
        'lyapunov'
      );

      expect(result.success).toBe(false);
      expect(result.executionTime).toBe(0);
      expect(result.method).toBe('quadratic_template'); // default method
    });

    it('should use default methods when not specified', async () => {
      mockBaselineService.generateLyapunovBaseline.mockResolvedValue({
        success: true,
        certificate_expression: 'x1^2 + x2^2',
        margin: 0.1,
        execution_time_ms: 100,
        method: 'quadratic_template',
      });

      await acceptanceService.generateBaseline(mockSystemSpec, 'lyapunov');

      expect(mockBaselineService.generateLyapunovBaseline).toHaveBeenCalledWith(
        mockSystemSpec,
        'quadratic_template'
      );
    });
  });
});
