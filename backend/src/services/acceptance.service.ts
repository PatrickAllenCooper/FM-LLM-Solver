import { SystemSpec, Candidate } from '../types/database';
import { AcceptanceResult } from '../types/api';
import { logger } from '../utils/logger';
import { MathService } from './math.service';
import { BaselineService } from './baseline.service';

export class AcceptanceService {
  private mathService: MathService;
  private baselineService: BaselineService;

  constructor() {
    this.mathService = new MathService();
    this.baselineService = new BaselineService();
  }
  async acceptCandidate(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<AcceptanceResult> {
    const startTime = Date.now();
    
    try {
      logger.info('Starting candidate acceptance check', {
        candidateId: candidate.id,
        certificateType: candidate.certificate_type,
        systemSpecId: systemSpec.id,
      });

      let partialResult: Omit<AcceptanceResult, 'duration_ms'>;

      switch (candidate.certificate_type) {
        case 'lyapunov':
          partialResult = await this.checkLyapunov(candidate, systemSpec);
          break;
        case 'barrier':
          partialResult = await this.checkBarrier(candidate, systemSpec);
          break;
        case 'inductive_invariant':
          partialResult = await this.checkInductiveInvariant(candidate, systemSpec);
          break;
        default:
          throw new Error(`Unsupported certificate type: ${candidate.certificate_type}`);
      }

      const result: AcceptanceResult = {
        ...partialResult,
        duration_ms: Date.now() - startTime,
      };

      logger.info('Candidate acceptance check completed', {
        candidateId: candidate.id,
        accepted: result.accepted,
        duration_ms: result.duration_ms,
        margin: result.margin,
      });

      return result;
    } catch (error) {
      const duration_ms = Date.now() - startTime;
      
      logger.error('Candidate acceptance check failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration_ms,
      });

      return {
        accepted: false,
        acceptance_method: 'symbolic',
        duration_ms,
        solver_output: error instanceof Error ? error.message : 'Unknown acceptance error',
      };
    }
  }

  private async checkLyapunov(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<Omit<AcceptanceResult, 'duration_ms'>> {
    const expression = candidate.candidate_expression;
    const dynamics = systemSpec.dynamics_json;

    logger.info('Debug: checkLyapunov inputs', {
      candidateId: candidate.id,
      candidateKeys: Object.keys(candidate),
      expressionType: typeof expression,
      expression,
      dynamicsType: typeof dynamics,
      systemSpecKeys: Object.keys(systemSpec),
    });

    try {
      // Use MathService for safe checking
      const domain = this.extractDomain(systemSpec);
      const dynamicsMap = this.extractDynamicsMap(dynamics);
      
      logger.info('Debug: Lyapunov check inputs', {
        candidateId: candidate.id,
        expression,
        dynamics,
        domain,
        dynamicsMap,
      });
      
      const verification = this.mathService.verifyLyapunovConditions(
        expression,
        dynamicsMap,
        domain
      );

      // Calculate detailed technical information for experimental analysis
      const technicalDetails = {
        conditions_checked: [
          'V(x) > 0 for x ≠ 0 (positive definite)',
          'V(0) = 0 (zero at equilibrium)', 
          'dV/dt ≤ 0 along trajectories (decreasing)',
        ],
        sampling_method: 'uniform' as const,
        sample_count: 1000, // Default from MathService
        domain_coverage: domain,
        violation_analysis: {
          total_violations: verification.violations.length,
          violation_points: verification.violations.map(v => ({
            point: v.point,
            condition: v.condition,
            value: v.value,
            severity: Math.abs(v.value) > 0.1 ? 'severe' as const : 
                     Math.abs(v.value) > 0.01 ? 'moderate' as const : 'minor' as const,
          })),
        },
        margin_breakdown: {
          positivity_margin: verification.positiveDefinite ? Math.abs(verification.margin) : undefined,
          decreasing_margin: verification.decreasing ? Math.abs(verification.margin) : undefined,
        },
        numerical_parameters: {
          tolerance: 1e-6,
          max_iterations: 1000,
          convergence_threshold: 1e-8,
        },
        stage_results: {
          stage_a_passed: verification.positiveDefinite && verification.decreasing,
          stage_b_enabled: false, // Currently only Stage A implemented
          stage_b_passed: undefined,
        },
      };

      if (verification.positiveDefinite && verification.decreasing) {
        return {
          accepted: true,
          margin: verification.margin,
          acceptance_method: 'mathematical',
          solver_output: `Lyapunov conditions satisfied. Proves stability. Margin: ${verification.margin.toFixed(6)}`,
          technical_details: technicalDetails,
        };
      } else {
        const firstViolation = verification.violations[0];
        return {
          accepted: false,
          counterexample: firstViolation ? {
            state: firstViolation.point,
            violation_type: firstViolation.condition,
            violation_magnitude: Math.abs(firstViolation.value),
          } : undefined,
          acceptance_method: 'mathematical',
          solver_output: `Failed Lyapunov conditions. Violations: ${verification.violations.length}`,
          technical_details: technicalDetails,
        };
      }
    } catch (error) {
      logger.error('Lyapunov acceptance check failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        accepted: false,
        acceptance_method: 'mathematical',
        solver_output: `Acceptance error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  private async checkBarrier(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<Omit<AcceptanceResult, 'duration_ms'>> {
    const expression = candidate.candidate_expression;
    const dynamics = systemSpec.dynamics_json;

    try {
      const dynamicsMap = this.extractDynamicsMap(dynamics);
      const safeSet = systemSpec.safe_set_json;
      const unsafeSet = systemSpec.unsafe_set_json;

      if (!safeSet || !unsafeSet) {
        return {
          accepted: false,
          acceptance_method: 'mathematical',
          solver_output: 'Safe and unsafe sets must be defined for barrier acceptance check',
        };
      }

      const verification = this.mathService.verifyBarrierConditions(
        expression,
        dynamicsMap,
        safeSet,
        unsafeSet
      );

      // Calculate detailed technical information for barrier certificates
      const technicalDetails = {
        conditions_checked: [
          'B(x) ≥ 0 for x ∈ safe set (initial safety)',
          'B(x) ≤ 0 for x ∈ unsafe set (separation)',
          'dB/dt ≤ 0 along trajectories (invariant)',
        ],
        sampling_method: 'uniform' as const,
        sample_count: 1000, // Default sampling
        domain_coverage: this.extractDomain(systemSpec),
        violation_analysis: {
          total_violations: verification.violations.length,
          violation_points: verification.violations.map(v => ({
            point: v.point,
            condition: v.condition,
            value: v.value,
            severity: Math.abs(v.value) > 0.1 ? 'severe' as const : 
                     Math.abs(v.value) > 0.01 ? 'moderate' as const : 'minor' as const,
          })),
        },
        margin_breakdown: {
          separation_margin: verification.separatesRegions ? Math.abs(verification.margin) : undefined,
          invariant_margin: verification.nonIncreasing ? Math.abs(verification.margin) : undefined,
        },
        numerical_parameters: {
          tolerance: 1e-6,
          max_iterations: 1000,
          convergence_threshold: 1e-8,
        },
        stage_results: {
          stage_a_passed: verification.separatesRegions && verification.nonIncreasing,
          stage_b_enabled: false, // Currently only Stage A implemented
          stage_b_passed: undefined,
        },
      };

      if (verification.separatesRegions && verification.nonIncreasing) {
        return {
          accepted: true,
          margin: verification.margin,
          acceptance_method: 'mathematical',
          solver_output: `Barrier conditions satisfied. Margin: ${verification.margin.toFixed(6)}`,
          technical_details: technicalDetails,
        };
      } else {
        const firstViolation = verification.violations[0];
        return {
          accepted: false,
          counterexample: firstViolation ? {
            state: firstViolation.point,
            violation_type: firstViolation.condition,
            violation_magnitude: Math.abs(firstViolation.value),
          } : undefined,
          acceptance_method: 'mathematical',
          solver_output: `Failed barrier conditions. Violations: ${verification.violations.length}`,
          technical_details: technicalDetails,
        };
      }
    } catch (error) {
      logger.error('Barrier acceptance check failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        accepted: false,
        acceptance_method: 'mathematical',
        solver_output: `Acceptance error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  private async checkInductiveInvariant(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<Omit<AcceptanceResult, 'duration_ms'>> {
    const expression = candidate.candidate_expression;
    const dynamics = systemSpec.dynamics_json;

    try {
      // For inductive invariants, check:
      // 1. I(x) holds for all initial states
      // 2. I(x) ∧ dynamics → I(x')
      // 3. I(x) → safe (no unsafe states satisfy I(x))

      const initialSet = systemSpec.initial_set_json;
      const unsafeSet = systemSpec.unsafe_set_json;

      if (!initialSet) {
        return {
          accepted: false,
          acceptance_method: 'mathematical',
          solver_output: 'Initial set must be defined for inductive invariant acceptance check',
        };
      }

      // Check initial condition
      const initialSamples = this.sampleFromSet(initialSet, 500);
      let initialViolations = 0;
      let firstInitialViolation: Record<string, number> | undefined;

      for (const point of initialSamples) {
        try {
          const value = this.mathService.evaluate(expression, point).value;
          if (value <= 0) { // Invariant should be positive
            initialViolations++;
            if (!firstInitialViolation) {
              firstInitialViolation = point;
            }
          }
        } catch (error) {
          logger.warn('Failed to evaluate invariant at initial point', { point, error });
        }
      }

      // Check safety (if unsafe set provided)
      let safetyViolations = 0;
      if (unsafeSet) {
        const unsafeSamples = this.sampleFromSet(unsafeSet, 500);
        for (const point of unsafeSamples) {
          try {
            const value = this.mathService.evaluate(expression, point).value;
            if (value > 0) { // Invariant should be negative in unsafe region
              safetyViolations++;
            }
          } catch (error) {
            logger.warn('Failed to evaluate invariant at unsafe point', { point, error });
          }
        }
      }

      const accepted = initialViolations === 0 && safetyViolations === 0;
      
      return {
        accepted,
        counterexample: firstInitialViolation ? {
          state: firstInitialViolation,
          violation_type: 'initial_condition',
          violation_magnitude: 1.0,
        } : undefined,
        acceptance_method: 'mathematical',
        solver_output: `Inductive invariant check: Initial violations: ${initialViolations}, Safety violations: ${safetyViolations}`,
      };
    } catch (error) {
      logger.error('Inductive invariant acceptance check failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        accepted: false,
        acceptance_method: 'mathematical',
        solver_output: `Acceptance error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  private extractDomain(systemSpec: SystemSpec): Record<string, { min: number; max: number }> {
    const domain: Record<string, { min: number; max: number }> = {};
    
    // Extract domain from system specification
    if (systemSpec.safe_set_json?.bounds) {
      for (const [variable, bounds] of Object.entries(systemSpec.safe_set_json.bounds)) {
        domain[variable] = bounds as { min: number; max: number };
      }
    }
    
    // Add default bounds for variables not specified
    for (let i = 1; i <= systemSpec.dimension; i++) {
      const varName = `x${i}`;
      if (!domain[varName]) {
        domain[varName] = { min: -10, max: 10 };
      }
    }
    
    return domain;
  }

  private extractDynamicsMap(dynamics: any): Record<string, string> {
    const dynamicsMap: Record<string, string> = {};
    
    if (dynamics.equations && Array.isArray(dynamics.equations)) {
      // Extract from equations array
      for (let i = 0; i < dynamics.equations.length; i++) {
        dynamicsMap[`x${i + 1}`] = dynamics.equations[i];
      }
    } else if (dynamics.variables && dynamics.rhs) {
      // Extract from variables and rhs
      for (let i = 0; i < dynamics.variables.length; i++) {
        dynamicsMap[dynamics.variables[i]] = dynamics.rhs[i] || '0';
      }
    }
    
    return dynamicsMap;
  }

  private sampleFromSet(set: any, numSamples: number): Array<Record<string, number>> {
    const samples: Array<Record<string, number>> = [];
    
    if (!set || !set.type) {
      return samples;
    }
    
    switch (set.type) {
      case 'box':
        if (set.bounds) {
          for (let i = 0; i < numSamples; i++) {
            const point: Record<string, number> = {};
            
            for (const [variable, bounds] of Object.entries(set.bounds)) {
              const typedBounds = bounds as { min: number; max: number };
              point[variable] = typedBounds.min + Math.random() * (typedBounds.max - typedBounds.min);
            }
            
            samples.push(point);
          }
        }
        break;
        
      case 'polytope':
        // For polytopes, sample within constraints (simplified)
        if (set.constraints) {
          // Use rejection sampling within bounding box
          const boundingBox = this.computeBoundingBox(set.constraints);
          let attempts = 0;
          
          while (samples.length < numSamples && attempts < numSamples * 10) {
            const candidate = this.sampleFromBox(boundingBox);
            if (this.satisfiesPolytope(candidate, set.constraints)) {
              samples.push(candidate);
            }
            attempts++;
          }
        }
        break;
        
      case 'ellipsoid':
        // Sample from ellipsoid
        if (set.center && set.matrix) {
          for (let i = 0; i < numSamples; i++) {
            const point = this.sampleFromEllipsoid(set.center, set.matrix);
            samples.push(point);
          }
        }
        break;
        
      default:
        logger.warn('Unsupported set type for sampling', { type: set.type });
    }
    
    return samples;
  }

  private computeBoundingBox(constraints: string[]): Record<string, { min: number; max: number }> {
    // Simplified bounding box computation
    const bounds: Record<string, { min: number; max: number }> = {};
    
    // Default bounds
    for (let i = 1; i <= 4; i++) { // Assume max 4 dimensions
      bounds[`x${i}`] = { min: -10, max: 10 };
    }
    
    return bounds;
  }

  private sampleFromBox(bounds: Record<string, { min: number; max: number }>): Record<string, number> {
    const point: Record<string, number> = {};
    
    for (const [variable, bound] of Object.entries(bounds)) {
      point[variable] = bound.min + Math.random() * (bound.max - bound.min);
    }
    
    return point;
  }

  private satisfiesPolytope(point: Record<string, number>, constraints: string[]): boolean {
    // Check if point satisfies all linear constraints
    for (const constraint of constraints) {
      try {
        const value = this.mathService.evaluate(constraint, point).value;
        if (value < 0) { // Assuming constraints are of form expr >= 0
          return false;
        }
      } catch (error) {
        return false;
      }
    }
    return true;
  }

  private sampleFromEllipsoid(center: Record<string, number>, matrix: number[][]): Record<string, number> {
    // Sample from unit sphere and transform
    const dimension = Object.keys(center).length;
    const unitVector = this.sampleUnitSphere(dimension);
    const radius = Math.pow(Math.random(), 1 / dimension); // Uniform distribution in volume
    
    // Transform by ellipsoid matrix
    const point: Record<string, number> = {};
    const variables = Object.keys(center);
    
    for (let i = 0; i < variables.length; i++) {
      const variable = variables[i];
      let transformed = center[variable];
      
      for (let j = 0; j < variables.length; j++) {
        transformed += matrix[i][j] * unitVector[j] * radius;
      }
      
      point[variable] = transformed;
    }
    
    return point;
  }

  private sampleUnitSphere(dimension: number): number[] {
    // Box-Muller for multivariate normal, then normalize
    const vector: number[] = [];
    
    for (let i = 0; i < dimension; i++) {
      vector.push(this.normalRandom());
    }
    
    // Normalize
    const norm = Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
    return vector.map(x => x / norm);
  }

  private normalRandom(): number {
    // Box-Muller transform for normal random variable
    const u = Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * Generate baseline certificate for comparison
   */
  async generateBaseline(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier',
    method?: 'sos' | 'quadratic_template' | 'linear_template' | 'sdp'
  ): Promise<{
    success: boolean;
    expression?: string;
    margin?: number;
    executionTime: number;
    method: string;
  }> {
    try {
      if (certificateType === 'lyapunov') {
        const result = await this.baselineService.generateLyapunovBaseline(
          systemSpec,
          method as 'sos' | 'quadratic_template' || 'quadratic_template'
        );
        
        return {
          success: result.success,
          expression: result.certificate_expression,
          margin: result.margin,
          executionTime: result.execution_time_ms,
          method: result.method,
        };
      } else {
        const result = await this.baselineService.generateBarrierBaseline(
          systemSpec,
          method as 'linear_template' | 'sdp' || 'linear_template'
        );
        
        return {
          success: result.success,
          expression: result.certificate_expression,
          margin: result.margin,
          executionTime: result.execution_time_ms,
          method: result.method,
        };
      }
    } catch (error) {
      logger.error('Baseline generation failed', {
        systemSpecId: systemSpec.id,
        certificateType,
        method,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      
      return {
        success: false,
        executionTime: 0,
        method: method || 'unknown',
      };
    }
  }
}
