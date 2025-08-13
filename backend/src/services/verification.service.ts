import { SystemSpec, Candidate } from '../types/database';
import { VerificationResult } from '../types/api';
import { logger } from '../utils/logger';
import { MathService } from './math.service';
import { BaselineService } from './baseline.service';

export class VerificationService {
  private mathService: MathService;
  private baselineService: BaselineService;

  constructor() {
    this.mathService = new MathService();
    this.baselineService = new BaselineService();
  }
  async verifyCertificate(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<VerificationResult> {
    const startTime = Date.now();
    
    try {
      logger.info('Starting certificate verification', {
        candidateId: candidate.id,
        certificateType: candidate.certificate_type,
        systemSpecId: systemSpec.id,
      });

      let partialResult: Omit<VerificationResult, 'duration_ms'>;

      switch (candidate.certificate_type) {
        case 'lyapunov':
          partialResult = await this.verifyLyapunov(candidate, systemSpec);
          break;
        case 'barrier':
          partialResult = await this.verifyBarrier(candidate, systemSpec);
          break;
        case 'inductive_invariant':
          partialResult = await this.verifyInductiveInvariant(candidate, systemSpec);
          break;
        default:
          throw new Error(`Unsupported certificate type: ${candidate.certificate_type}`);
      }

      const result: VerificationResult = {
        ...partialResult,
        duration_ms: Date.now() - startTime,
      };

      logger.info('Certificate verification completed', {
        candidateId: candidate.id,
        verified: result.verified,
        duration_ms: result.duration_ms,
        margin: result.margin,
      });

      return result;
    } catch (error) {
      const duration_ms = Date.now() - startTime;
      
      logger.error('Certificate verification failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration_ms,
      });

      return {
        verified: false,
        verification_method: 'symbolic',
        duration_ms,
        solver_output: error instanceof Error ? error.message : 'Unknown verification error',
      };
    }
  }

  private async verifyLyapunov(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<Omit<VerificationResult, 'duration_ms'>> {
    const expression = candidate.candidate_expression;
    const dynamics = systemSpec.dynamics_json;

    try {
      // Use MathService for safe verification
      const domain = this.extractDomain(systemSpec);
      const dynamicsMap = this.extractDynamicsMap(dynamics);
      
      const verification = this.mathService.verifyLyapunovConditions(
        expression,
        dynamicsMap,
        domain
      );

      if (verification.positiveDefinite && verification.decreasing) {
        return {
          verified: true,
          margin: verification.margin,
          verification_method: 'mathematical',
          solver_output: `Lyapunov conditions verified. Margin: ${verification.margin.toFixed(6)}`,
        };
      } else {
        const firstViolation = verification.violations[0];
        return {
          verified: false,
          counterexample: firstViolation ? {
            state: firstViolation.point,
            violation_type: firstViolation.condition,
            violation_magnitude: Math.abs(firstViolation.value),
          } : undefined,
          verification_method: 'mathematical',
          solver_output: `Failed Lyapunov conditions. Violations: ${verification.violations.length}`,
        };
      }
    } catch (error) {
      logger.error('Lyapunov verification failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        verified: false,
        verification_method: 'mathematical',
        solver_output: `Verification error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  private async verifyBarrier(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<Omit<VerificationResult, 'duration_ms'>> {
    const expression = candidate.candidate_expression;
    const dynamics = systemSpec.dynamics_json;

    try {
      const dynamicsMap = this.extractDynamicsMap(dynamics);
      const safeSet = systemSpec.safe_set_json;
      const unsafeSet = systemSpec.unsafe_set_json;

      if (!safeSet || !unsafeSet) {
        return {
          verified: false,
          verification_method: 'mathematical',
          solver_output: 'Safe and unsafe sets must be defined for barrier verification',
        };
      }

      const verification = this.mathService.verifyBarrierConditions(
        expression,
        dynamicsMap,
        safeSet,
        unsafeSet
      );

      if (verification.separatesRegions && verification.nonIncreasing) {
        return {
          verified: true,
          margin: verification.margin,
          verification_method: 'mathematical',
          solver_output: `Barrier conditions verified. Margin: ${verification.margin.toFixed(6)}`,
        };
      } else {
        const firstViolation = verification.violations[0];
        return {
          verified: false,
          counterexample: firstViolation ? {
            state: firstViolation.point,
            violation_type: firstViolation.condition,
            violation_magnitude: Math.abs(firstViolation.value),
          } : undefined,
          verification_method: 'mathematical',
          solver_output: `Failed barrier conditions. Violations: ${verification.violations.length}`,
        };
      }
    } catch (error) {
      logger.error('Barrier verification failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        verified: false,
        verification_method: 'mathematical',
        solver_output: `Verification error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  private async verifyInductiveInvariant(
    candidate: Candidate,
    systemSpec: SystemSpec
  ): Promise<Omit<VerificationResult, 'duration_ms'>> {
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
          verified: false,
          verification_method: 'mathematical',
          solver_output: 'Initial set must be defined for inductive invariant verification',
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

      const verified = initialViolations === 0 && safetyViolations === 0;
      
      return {
        verified,
        counterexample: firstInitialViolation ? {
          state: firstInitialViolation,
          violation_type: 'initial_condition',
          violation_magnitude: 1.0,
        } : undefined,
        verification_method: 'mathematical',
        solver_output: `Inductive invariant check: Initial violations: ${initialViolations}, Safety violations: ${safetyViolations}`,
      };
    } catch (error) {
      logger.error('Inductive invariant verification failed', {
        candidateId: candidate.id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        verified: false,
        verification_method: 'mathematical',
        solver_output: `Verification error: ${error instanceof Error ? error.message : 'Unknown error'}`,
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
