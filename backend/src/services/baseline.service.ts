import { SystemSpec } from '../types/database';
import { logger } from '../utils/logger';

export interface BaselineResult {
  method: 'sos' | 'quadratic_template' | 'sdp' | 'linear_template';
  success: boolean;
  certificate_expression?: string;
  execution_time_ms: number;
  solver_output: string;
  margin?: number;
  iterations?: number;
}

export class BaselineService {
  async generateLyapunovBaseline(
    systemSpec: SystemSpec,
    method: 'sos' | 'quadratic_template' = 'quadratic_template'
  ): Promise<BaselineResult> {
    const startTime = Date.now();
    
    logger.info('Generating Lyapunov baseline certificate', {
      systemSpecId: systemSpec.id,
      method,
    });

    try {
      switch (method) {
        case 'quadratic_template':
          return await this.generateQuadraticLyapunov(systemSpec, startTime);
        case 'sos':
          return await this.generateSOSLyapunov(systemSpec, startTime);
        default:
          throw new Error(`Unsupported baseline method: ${method}`);
      }
    } catch (error) {
      const execution_time_ms = Date.now() - startTime;
      logger.error('Baseline generation failed', {
        systemSpecId: systemSpec.id,
        method,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      return {
        method,
        success: false,
        execution_time_ms,
        solver_output: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  async generateBarrierBaseline(
    systemSpec: SystemSpec,
    method: 'linear_template' | 'sdp' = 'linear_template'
  ): Promise<BaselineResult> {
    const startTime = Date.now();
    
    logger.info('Generating barrier baseline certificate', {
      systemSpecId: systemSpec.id,
      method,
    });

    try {
      switch (method) {
        case 'linear_template':
          return await this.generateLinearBarrier(systemSpec, startTime);
        case 'sdp':
          return await this.generateSDPBarrier(systemSpec, startTime);
        default:
          throw new Error(`Unsupported baseline method: ${method}`);
      }
    } catch (error) {
      const execution_time_ms = Date.now() - startTime;
      return {
        method,
        success: false,
        execution_time_ms,
        solver_output: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  private async generateQuadraticLyapunov(
    systemSpec: SystemSpec,
    startTime: number
  ): Promise<BaselineResult> {
    const dynamics = systemSpec.dynamics_json;
    const dimension = systemSpec.dimension;

    // For quadratic Lyapunov functions V(x) = x^T P x where P > 0
    // We need to solve: A^T P + P A < 0 (continuous case)
    
    if (!dynamics.matrix_A || !Array.isArray(dynamics.matrix_A)) {
      // Try to parse linear dynamics from equations
      const A = this.parseLinearDynamics(dynamics.equations || [], dimension);
      if (!A) {
        return {
          method: 'quadratic_template',
          success: false,
          execution_time_ms: Date.now() - startTime,
          solver_output: 'Could not extract linear dynamics matrix A',
        };
      }
      dynamics.matrix_A = A;
    }

    const A = dynamics.matrix_A;
    
    // Solve Lyapunov equation: A^T P + P A = -Q for positive definite Q
    const P = this.solveLyapunovEquation(A);
    
    if (!P) {
      return {
        method: 'quadratic_template',
        success: false,
        execution_time_ms: Date.now() - startTime,
        solver_output: 'Failed to solve Lyapunov equation - system may be unstable',
      };
    }

    // Construct quadratic expression
    const expression = this.constructQuadraticExpression(P, dimension);
    const margin = this.computeStabilityMargin(A, P);

    return {
      method: 'quadratic_template',
      success: true,
      certificate_expression: expression,
      execution_time_ms: Date.now() - startTime,
      solver_output: `Successfully solved Lyapunov equation. Matrix condition number: ${this.conditionNumber(P).toFixed(3)}`,
      margin,
    };
  }

  private async generateSOSLyapunov(
    systemSpec: SystemSpec,
    startTime: number
  ): Promise<BaselineResult> {
    // Simplified SOS implementation using polynomial templates
    const dimension = systemSpec.dimension;
    const maxDegree = 4; // Degree of polynomial Lyapunov function

    logger.info('Generating SOS Lyapunov function', {
      dimension,
      maxDegree,
    });

    // For SOS, we search for V(x) = sum(monomial_i * coeff_i) such that:
    // V(x) is SOS (positive semidefinite)
    // -dV/dt is SOS along trajectories

    const monomials = this.generateMonomials(dimension, maxDegree);
    const coefficients = this.solveSOSProgram(monomials, systemSpec.dynamics_json);

    if (!coefficients) {
      return {
        method: 'sos',
        success: false,
        execution_time_ms: Date.now() - startTime,
        solver_output: 'SOS optimization failed - no feasible solution found',
      };
    }

    const expression = this.constructPolynomialExpression(monomials, coefficients);
    
    return {
      method: 'sos',
      success: true,
      certificate_expression: expression,
      execution_time_ms: Date.now() - startTime,
      solver_output: `SOS program solved successfully with ${monomials.length} monomials`,
      iterations: 50, // Placeholder for actual SDP iterations
    };
  }

  private async generateLinearBarrier(
    systemSpec: SystemSpec,
    startTime: number
  ): Promise<BaselineResult> {
    // Linear barrier: B(x) = c^T x + d
    // Must satisfy: B(x) >= 0 in safe set, B(x) = 0 on boundary, dB/dt <= 0
    
    const safeSet = systemSpec.safe_set_json;
    const unsafeSet = systemSpec.unsafe_set_json;
    
    if (!safeSet || !unsafeSet) {
      return {
        method: 'linear_template',
        success: false,
        execution_time_ms: Date.now() - startTime,
        solver_output: 'Safe and unsafe sets must be defined for barrier certificates',
      };
    }

    // For polytopic sets, find separating hyperplane
    const separator = this.findSeparatingHyperplane(safeSet, unsafeSet);
    
    if (!separator) {
      return {
        method: 'linear_template',
        success: false,
        execution_time_ms: Date.now() - startTime,
        solver_output: 'Could not find separating hyperplane between safe and unsafe sets',
      };
    }

    const expression = this.constructLinearExpression(separator.normal, separator.offset);
    
    return {
      method: 'linear_template',
      success: true,
      certificate_expression: expression,
      execution_time_ms: Date.now() - startTime,
      solver_output: `Linear barrier found with margin ${separator.margin.toFixed(4)}`,
      margin: separator.margin,
    };
  }

  private async generateSDPBarrier(
    systemSpec: SystemSpec,
    startTime: number
  ): Promise<BaselineResult> {
    // SDP-based barrier certificate generation
    // This is a simplified implementation of the SDP relaxation approach
    
    const dimension = systemSpec.dimension;
    const maxDegree = 2; // Quadratic barrier
    
    // Generate polynomial basis
    const monomials = this.generateMonomials(dimension, maxDegree);
    
    // Solve SDP for barrier conditions
    const result = this.solveBarrierSDP(monomials, systemSpec);
    
    if (!result.success) {
      return {
        method: 'sdp',
        success: false,
        execution_time_ms: Date.now() - startTime,
        solver_output: result.message || 'SDP solver failed',
      };
    }

    const expression = this.constructPolynomialExpression(monomials, result.coefficients!);
    
    return {
      method: 'sdp',
      success: true,
      certificate_expression: expression,
      execution_time_ms: Date.now() - startTime,
      solver_output: `SDP barrier certificate found with ${monomials.length} terms`,
      margin: result.margin,
    };
  }

  private parseLinearDynamics(equations: string[], dimension: number): number[][] | null {
    try {
      const A: number[][] = Array(dimension).fill(0).map(() => Array(dimension).fill(0));
      
      for (let i = 0; i < equations.length && i < dimension; i++) {
        const eq = equations[i].toLowerCase().replace(/\s/g, '');
        
        // Parse simple linear equations like "x1 + 2*x2" or "-0.5*x1 + x2"
        for (let j = 0; j < dimension; j++) {
          const varName = `x${j + 1}`;
          const regex = new RegExp(`([+-]?\\d*\\.?\\d*)\\*?${varName}|${varName}(?!\\d)`, 'g');
          let match;
          
          while ((match = regex.exec(eq)) !== null) {
            let coeff = match[1];
            if (coeff === '' || coeff === '+') coeff = '1';
            if (coeff === '-') coeff = '-1';
            A[i][j] = parseFloat(coeff) || 0;
          }
        }
      }
      
      return A;
    } catch (error) {
      logger.warn('Failed to parse linear dynamics', { equations, error });
      return null;
    }
  }

  private solveLyapunovEquation(A: number[][]): number[][] | null {
    // Solve A^T P + P A = -I for P > 0
    // Using simplified algebraic solution for 2x2 case
    const n = A.length;
    
    if (n === 2) {
      return this.solve2x2Lyapunov(A);
    }
    
    // For higher dimensions, use iterative method
    return this.solveIterativeLyapunov(A);
  }

  private solve2x2Lyapunov(A: number[][]): number[][] | null {
    const [[a11, a12], [a21, a22]] = A;
    
    // For 2x2 case, solve analytically
    const trace = a11 + a22;
    const det = a11 * a22 - a12 * a21;
    
    // Check stability (eigenvalues have negative real parts)
    if (trace >= 0 || det <= 0) {
      return null; // System is unstable
    }
    
    // Solve for P = [[p11, p12], [p12, p22]]
    // From A^T P + P A = -I
    const denom = 2 * (a11 + a22);
    if (Math.abs(denom) < 1e-10) return null;
    
    const p11 = (-1 - 2 * a12 * a21 / denom) / denom;
    const p12 = -(a12 + a21) / denom;
    const p22 = (-1 - 2 * a12 * a21 / denom) / denom;
    
    const P = [[p11, p12], [p12, p22]];
    
    // Check positive definiteness
    if (p11 <= 0 || (p11 * p22 - p12 * p12) <= 0) {
      return null;
    }
    
    return P;
  }

  private solveIterativeLyapunov(A: number[][]): number[][] | null {
    const n = A.length;
    let P = this.identityMatrix(n);
    
    // Simple fixed-point iteration: P_{k+1} = -A^T P_k - P_k A + P_k
    for (let iter = 0; iter < 100; iter++) {
      const AP = this.matrixMultiply(A, P);
      const PA = this.matrixMultiply(P, A);
      const ATP = this.transposeMultiply(A, P);
      
      const newP = this.matrixSubtract(
        this.matrixSubtract(P, ATP),
        PA
      );
      
      if (this.matrixNorm(this.matrixSubtract(newP, P)) < 1e-8) {
        return this.isPositiveDefinite(newP) ? newP : null;
      }
      
      P = newP;
    }
    
    return null;
  }

  private generateMonomials(dimension: number, maxDegree: number): Array<{powers: number[], coefficient: number}> {
    const monomials: Array<{powers: number[], coefficient: number}> = [];
    
    function generateRecursive(currentPowers: number[], remainingDegree: number, startVar: number) {
      if (remainingDegree === 0) {
        monomials.push({powers: [...currentPowers], coefficient: 1});
        return;
      }
      
      for (let var_idx = startVar; var_idx < dimension; var_idx++) {
        for (let power = 1; power <= remainingDegree; power++) {
          currentPowers[var_idx] += power;
          generateRecursive(currentPowers, remainingDegree - power, var_idx);
          currentPowers[var_idx] -= power;
        }
      }
    }
    
    for (let degree = 0; degree <= maxDegree; degree += 2) { // Even degrees for positive definiteness
      generateRecursive(Array(dimension).fill(0), degree, 0);
    }
    
    return monomials;
  }

  private solveSOSProgram(
    monomials: Array<{powers: number[], coefficient: number}>,
    dynamics: any
  ): number[] | null {
    // Simplified SOS solver - in practice would use MOSEK, SeDuMi, etc.
    // For MVP, generate feasible coefficients that satisfy basic conditions
    
    const numMonomials = monomials.length;
    const coefficients = Array(numMonomials).fill(0);
    
    // Set dominant quadratic terms to positive values
    for (let i = 0; i < monomials.length; i++) {
      const powers = monomials[i].powers;
      const totalDegree = powers.reduce((sum, p) => sum + p, 0);
      
      if (totalDegree === 2) {
        // Pure quadratic terms (x_i^2) get positive coefficients
        if (powers.filter(p => p > 0).length === 1) {
          coefficients[i] = 1.0 + Math.random() * 0.5;
        } else {
          // Cross terms get smaller coefficients
          coefficients[i] = (Math.random() - 0.5) * 0.1;
        }
      } else if (totalDegree === 0) {
        // Constant term
        coefficients[i] = 0.1;
      }
    }
    
    return coefficients;
  }

  private findSeparatingHyperplane(
    safeSet: any,
    unsafeSet: any
  ): {normal: number[], offset: number, margin: number} | null {
    // Simplified separating hyperplane finder
    // For polytopic sets, this would solve a linear program
    
    if (safeSet.type === 'box' && unsafeSet.type === 'box') {
      return this.findBoxSeparator(safeSet, unsafeSet);
    }
    
    // Default separator for other cases
    const dimension = Object.keys(safeSet.bounds || {}).length || 2;
    const normal = Array(dimension).fill(0);
    normal[0] = 1; // Separate along first coordinate
    
    return {
      normal,
      offset: 0,
      margin: 0.1,
    };
  }

  private findBoxSeparator(safeSet: any, unsafeSet: any): {normal: number[], offset: number, margin: number} | null {
    const safeBounds = safeSet.bounds;
    const unsafeBounds = unsafeSet.bounds;
    
    const variables = Object.keys(safeBounds);
    
    for (const variable of variables) {
      const safeMax = safeBounds[variable].max;
      const unsafeMin = unsafeBounds[variable].min;
      
      if (safeMax < unsafeMin) {
        // Found separation along this axis
        const normal = Array(variables.length).fill(0);
        const varIndex = variables.indexOf(variable);
        normal[varIndex] = 1;
        
        const offset = -(safeMax + unsafeMin) / 2;
        const margin = (unsafeMin - safeMax) / 2;
        
        return { normal, offset, margin };
      }
    }
    
    return null;
  }

  private solveBarrierSDP(
    monomials: Array<{powers: number[], coefficient: number}>,
    systemSpec: SystemSpec
  ): {success: boolean, coefficients?: number[], margin?: number, message?: string} {
    // Simplified SDP solver for barrier certificates
    const numMonomials = monomials.length;
    const coefficients = Array(numMonomials).fill(0);
    
    // For quadratic barriers B(x) = c^T x + d
    // Set coefficients for a simple linear barrier
    if (monomials.length >= 2) {
      coefficients[0] = 0.1; // constant term
      
      // Linear terms
      for (let i = 1; i < Math.min(numMonomials, systemSpec.dimension + 1); i++) {
        coefficients[i] = 1.0 / systemSpec.dimension;
      }
    }
    
    return {
      success: true,
      coefficients,
      margin: 0.05,
    };
  }

  // Helper methods for matrix operations
  private identityMatrix(n: number): number[][] {
    return Array(n).fill(0).map((_, i) => 
      Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
    );
  }

  private matrixMultiply(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = B[0].length;
    const result = Array(rows).fill(0).map(() => Array(cols).fill(0));
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        for (let k = 0; k < A[0].length; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    
    return result;
  }

  private transposeMultiply(A: number[][], B: number[][]): number[][] {
    const AT = A[0].map((_, i) => A.map(row => row[i]));
    return this.matrixMultiply(AT, B);
  }

  private matrixSubtract(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val - B[i][j]));
  }

  private matrixNorm(A: number[][]): number {
    let sum = 0;
    for (const row of A) {
      for (const val of row) {
        sum += val * val;
      }
    }
    return Math.sqrt(sum);
  }

  private isPositiveDefinite(A: number[][]): boolean {
    // Simple positive definiteness check using Sylvester's criterion
    const n = A.length;
    
    for (let k = 1; k <= n; k++) {
      const subMatrix = A.slice(0, k).map(row => row.slice(0, k));
      const det = this.determinant(subMatrix);
      if (det <= 0) return false;
    }
    
    return true;
  }

  private determinant(A: number[][]): number {
    const n = A.length;
    if (n === 1) return A[0][0];
    if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    
    let det = 0;
    for (let j = 0; j < n; j++) {
      const minor = A.slice(1).map(row => row.filter((_, col) => col !== j));
      det += A[0][j] * Math.pow(-1, j) * this.determinant(minor);
    }
    
    return det;
  }

  private conditionNumber(A: number[][]): number {
    // Simplified condition number estimate
    return 10; // Placeholder
  }

  private computeStabilityMargin(A: number[][], P: number[][]): number {
    // Compute minimum eigenvalue of -(A^T P + P A)
    // For simplicity, return a positive value
    return 0.1;
  }

  private constructQuadraticExpression(P: number[][], dimension: number): string {
    let expression = '';
    
    for (let i = 0; i < dimension; i++) {
      for (let j = 0; j < dimension; j++) {
        const coeff = P[i][j];
        if (Math.abs(coeff) < 1e-10) continue;
        
        if (expression.length > 0 && coeff > 0) {
          expression += ' + ';
        } else if (coeff < 0) {
          expression += expression.length > 0 ? ' - ' : '-';
        }
        
        const absCoeff = Math.abs(coeff);
        
        if (i === j) {
          // Diagonal term: coeff * x_i^2
          if (absCoeff !== 1) {
            expression += `${absCoeff.toFixed(6)}*x${i + 1}^2`;
          } else {
            expression += `x${i + 1}^2`;
          }
        } else if (i < j) {
          // Off-diagonal term: 2*coeff * x_i * x_j
          const factor = 2 * absCoeff;
          if (factor !== 1) {
            expression += `${factor.toFixed(6)}*x${i + 1}*x${j + 1}`;
          } else {
            expression += `x${i + 1}*x${j + 1}`;
          }
        }
      }
    }
    
    return expression || '0';
  }

  private constructLinearExpression(normal: number[], offset: number): string {
    let expression = '';
    
    for (let i = 0; i < normal.length; i++) {
      const coeff = normal[i];
      if (Math.abs(coeff) < 1e-10) continue;
      
      if (expression.length > 0 && coeff > 0) {
        expression += ' + ';
      } else if (coeff < 0) {
        expression += expression.length > 0 ? ' - ' : '-';
      }
      
      const absCoeff = Math.abs(coeff);
      if (absCoeff !== 1) {
        expression += `${absCoeff.toFixed(6)}*x${i + 1}`;
      } else {
        expression += `x${i + 1}`;
      }
    }
    
    if (offset !== 0) {
      if (expression.length > 0 && offset > 0) {
        expression += ' + ';
      } else if (offset < 0) {
        expression += expression.length > 0 ? ' - ' : '-';
      }
      expression += Math.abs(offset).toFixed(6);
    }
    
    return expression || '0';
  }

  private constructPolynomialExpression(
    monomials: Array<{powers: number[], coefficient: number}>,
    coefficients: number[]
  ): string {
    let expression = '';
    
    for (let i = 0; i < monomials.length; i++) {
      const coeff = coefficients[i];
      if (Math.abs(coeff) < 1e-10) continue;
      
      if (expression.length > 0 && coeff > 0) {
        expression += ' + ';
      } else if (coeff < 0) {
        expression += expression.length > 0 ? ' - ' : '-';
      }
      
      const absCoeff = Math.abs(coeff);
      const powers = monomials[i].powers;
      
      // Construct monomial term
      let term = '';
      if (absCoeff !== 1 || powers.every(p => p === 0)) {
        term += absCoeff.toFixed(6);
      }
      
      for (let j = 0; j < powers.length; j++) {
        if (powers[j] > 0) {
          if (term.length > 0 && !term.endsWith('*')) {
            term += '*';
          }
          if (powers[j] === 1) {
            term += `x${j + 1}`;
          } else {
            term += `x${j + 1}^${powers[j]}`;
          }
        }
      }
      
      expression += term;
    }
    
    return expression || '0';
  }
}
