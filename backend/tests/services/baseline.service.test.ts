import { BaselineService } from '../../src/services/baseline.service';
import { SystemSpec } from '../../src/types/database';

describe('BaselineService', () => {
  let baselineService: BaselineService;
  let mockSystemSpec: SystemSpec;

  beforeEach(() => {
    baselineService = new BaselineService();
    
    // Create mock system specification
    mockSystemSpec = {
      id: 'test-system-1',
      owner_user_id: 'user-1',
      name: 'Test Linear System',
      system_type: 'continuous',
      dimension: 2,
      spec_version: '1.0',
      created_by: 'test-user',
      constraints_json: {},
      dynamics_json: {
        variables: ['x1', 'x2'],
        equations: ['-x1 + 0.1*x2', '-x2 - 0.1*x1'],
        matrix_A: [[-1, 0.1], [-0.1, -1]],
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
      description: 'Test system for unit tests',
      created_at: new Date(),
      updated_at: new Date(),
    };
  });

  describe('generateLyapunovBaseline', () => {
    it('should generate quadratic Lyapunov function for stable linear system', async () => {
      const result = await baselineService.generateLyapunovBaseline(mockSystemSpec, 'quadratic_template');
      
      expect(result.method).toBe('quadratic_template');
      expect(result.success).toBe(true);
      expect(result.certificate_expression).toBeDefined();
      expect(result.certificate_expression).toMatch(/x1|x2/); // Should contain variables
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
      expect(result.margin).toBeGreaterThan(0);
    });

    it('should generate SOS Lyapunov function', async () => {
      const result = await baselineService.generateLyapunovBaseline(mockSystemSpec, 'sos');
      
      expect(result.method).toBe('sos');
      expect(result.success).toBe(true);
      expect(result.certificate_expression).toBeDefined();
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
      expect(result.iterations).toBeDefined();
    });

    it('should handle unstable systems gracefully', async () => {
      // Create unstable system
      const unstableSystem = {
        ...mockSystemSpec,
        dynamics_json: {
          ...mockSystemSpec.dynamics_json,
          equations: ['x1', 'x2'], // Unstable dynamics
          matrix_A: [[1, 0], [0, 1]], // Positive eigenvalues
        },
      };
      
      const result = await baselineService.generateLyapunovBaseline(unstableSystem, 'quadratic_template');
      
      expect(result.method).toBe('quadratic_template');
      // May succeed or fail depending on implementation
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
    });

    it('should handle systems without matrix_A', async () => {
      const systemWithoutMatrix = {
        ...mockSystemSpec,
        dynamics_json: {
          variables: ['x1', 'x2'],
          equations: ['-2*x1', '-3*x2'],
        },
      };
      
      const result = await baselineService.generateLyapunovBaseline(systemWithoutMatrix, 'quadratic_template');
      
      expect(result.method).toBe('quadratic_template');
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
      // Should either succeed by parsing equations or fail gracefully
    });
  });

  describe('generateBarrierBaseline', () => {
    it('should generate linear barrier function', async () => {
      const result = await baselineService.generateBarrierBaseline(mockSystemSpec, 'linear_template');
      
      expect(result.method).toBe('linear_template');
      expect(result.success).toBe(true);
      expect(result.certificate_expression).toBeDefined();
      expect(result.certificate_expression).toMatch(/x1|x2/);
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
      expect(result.margin).toBeGreaterThan(0);
    });

    it('should generate SDP barrier function', async () => {
      const result = await baselineService.generateBarrierBaseline(mockSystemSpec, 'sdp');
      
      expect(result.method).toBe('sdp');
      expect(result.success).toBe(true);
      expect(result.certificate_expression).toBeDefined();
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
    });

    it('should handle missing safe/unsafe sets', async () => {
      const systemWithoutSets = {
        ...mockSystemSpec,
        safe_set_json: undefined,
        unsafe_set_json: undefined,
      };
      
      const result = await baselineService.generateBarrierBaseline(systemWithoutSets, 'linear_template');
      
      expect(result.method).toBe('linear_template');
      expect(result.success).toBe(false);
      expect(result.solver_output).toMatch(/Safe and unsafe sets/);
    });
  });

  describe('Matrix operations', () => {
    it('should solve 2x2 Lyapunov equation for stable system', () => {
      const A = [[-1, 0.1], [-0.1, -1]];
      // Access private method through reflection for testing
      const solveLyapunov = (baselineService as any).solveLyapunovEquation.bind(baselineService);
      
      const P = solveLyapunov(A);
      
      expect(P).toBeTruthy();
      expect(P).toHaveLength(2);
      expect(P[0]).toHaveLength(2);
      
      // Check positive definiteness: P[0][0] > 0 and det(P) > 0
      expect(P[0][0]).toBeGreaterThan(0);
      const det = P[0][0] * P[1][1] - P[0][1] * P[1][0];
      expect(det).toBeGreaterThan(0);
    });

    it('should detect unstable systems', () => {
      const unstableA = [[1, 0], [0, 1]]; // Positive eigenvalues
      const solveLyapunov = (baselineService as any).solveLyapunovEquation.bind(baselineService);
      
      const P = solveLyapunov(unstableA);
      
      expect(P).toBeNull(); // Should fail for unstable system
    });
  });

  describe('Expression construction', () => {
    it('should construct valid quadratic expressions', () => {
      const P = [[1, 0.1], [0.1, 2]];
      const constructQuadratic = (baselineService as any).constructQuadraticExpression.bind(baselineService);
      
      const expression = constructQuadratic(P, 2);
      
      expect(expression).toBeTruthy();
      expect(expression).toMatch(/x1.*x2/); // Should contain both variables
      expect(expression).toMatch(/\^2/); // Should contain squared terms
    });

    it('should construct valid linear expressions', () => {
      const normal = [1, 2];
      const offset = 0.5;
      const constructLinear = (baselineService as any).constructLinearExpression.bind(baselineService);
      
      const expression = constructLinear(normal, offset);
      
      expect(expression).toBeTruthy();
      expect(expression).toMatch(/x1.*x2/);
      expect(expression).toContain('0.500000'); // Should contain offset
    });
  });

  describe('Separating hyperplane finder', () => {
    it('should find separating hyperplane for separable box sets', () => {
      const safeSet = {
        type: 'box',
        bounds: { x1: { min: -1, max: 1 }, x2: { min: -1, max: 1 } }
      };
      const unsafeSet = {
        type: 'box',
        bounds: { x1: { min: 3, max: 5 }, x2: { min: 3, max: 5 } }
      };
      
      const findSeparator = (baselineService as any).findSeparatingHyperplane.bind(baselineService);
      const separator = findSeparator(safeSet, unsafeSet);
      
      expect(separator).toBeTruthy();
      expect(separator.normal).toBeDefined();
      expect(separator.margin).toBeGreaterThan(0);
    });

    it('should handle non-separable sets', () => {
      const overlappingSafeSet = {
        type: 'box',
        bounds: { x1: { min: -2, max: 2 }, x2: { min: -2, max: 2 } }
      };
      const overlappingUnsafeSet = {
        type: 'box',
        bounds: { x1: { min: -1, max: 1 }, x2: { min: -1, max: 1 } }
      };
      
      const findSeparator = (baselineService as any).findSeparatingHyperplane.bind(baselineService);
      const separator = findSeparator(overlappingSafeSet, overlappingUnsafeSet);
      
      // Should still return a separator (may be trivial) or null for non-separable
      expect(separator).toBeDefined();
    });
  });

  describe('Monomial generation', () => {
    it('should generate correct monomials for given degree', () => {
      const generateMonomials = (baselineService as any).generateMonomials.bind(baselineService);
      const monomials = generateMonomials(2, 2); // 2 variables, degree 2
      
      expect(monomials).toBeTruthy();
      expect(monomials.length).toBeGreaterThan(0);
      
      // Check that we have constant, linear, and quadratic terms
      const degrees = monomials.map((m: any) => m.powers.reduce((sum: number, p: number) => sum + p, 0));
      expect(degrees).toContain(0); // Constant term
      expect(degrees).toContain(2); // Quadratic terms
    });

    it('should respect degree constraints', () => {
      const generateMonomials = (baselineService as any).generateMonomials.bind(baselineService);
      const monomials = generateMonomials(2, 4);
      
      // All degrees should be even and <= 4
      for (const monomial of monomials) {
        const degree = monomial.powers.reduce((sum: number, p: number) => sum + p, 0);
        expect(degree % 2).toBe(0); // Even degrees only
        expect(degree).toBeLessThanOrEqual(4);
      }
    });
  });

  describe('Error handling', () => {
    it('should handle invalid system specifications', async () => {
      const invalidSystem = {
        ...mockSystemSpec,
        dynamics_json: {},
      };
      
      const result = await baselineService.generateLyapunovBaseline(invalidSystem, 'quadratic_template');
      
      expect(result.success).toBe(false);
      expect(result.execution_time_ms).toBeGreaterThanOrEqual(0);
    });

    it('should handle unsupported methods gracefully', async () => {
      const result = await baselineService.generateLyapunovBaseline(
        mockSystemSpec, 
        'unsupported_method' as any
      );
      
      expect(result.success).toBe(false);
      expect(result.solver_output).toMatch(/Unsupported baseline method/);
    });
  });

  describe('Performance', () => {
    it('should complete baseline generation within reasonable time', async () => {
      const startTime = Date.now();
      
      await baselineService.generateLyapunovBaseline(mockSystemSpec, 'quadratic_template');
      
      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
    });

    it('should handle large dimension systems', async () => {
      const largeDimensionSystem = {
        ...mockSystemSpec,
        dimension: 4,
        dynamics_json: {
          variables: ['x1', 'x2', 'x3', 'x4'],
          equations: ['-x1', '-x2', '-x3', '-x4'],
          matrix_A: [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
          ],
        },
      };
      
      const result = await baselineService.generateLyapunovBaseline(largeDimensionSystem, 'quadratic_template');
      
      expect(result.execution_time_ms).toBeLessThan(10000); // Should handle larger systems
    });
  });
});
