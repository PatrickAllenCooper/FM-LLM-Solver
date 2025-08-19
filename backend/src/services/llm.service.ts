import Anthropic from '@anthropic-ai/sdk';
import { LLMConfig, LLMCertificateResponse, LLMCertificateResponseSchema } from '../types/api';
import { SystemSpec } from '../types/database';
import { logger } from '../utils/logger';

export class LLMService {
  private anthropic: Anthropic;

  constructor(apiKey: string) {
    this.anthropic = new Anthropic({
      apiKey: apiKey,
    });
  }

  async generateCertificate(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    config: LLMConfig
  ): Promise<{
    response: LLMCertificateResponse;
    duration_ms: number;
    raw_response: string;
  }> {
    const startTime = Date.now();
    
    try {
      const prompt = this.buildPrompt(systemSpec, certificateType, config.mode);
      
      logger.info('Generating certificate with LLM', {
        systemSpecId: systemSpec.id,
        certificateType,
        mode: config.mode,
        model: config.model,
      });

      const message = await this.anthropic.messages.create({
        model: config.model,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      });

      const duration_ms = Date.now() - startTime;
      
      // Handle Claude 4 refusal stop reason (cast to handle SDK type lag)
      if ((message.stop_reason as string) === 'refusal') {
        logger.warn('Claude 4 model refused to generate content', {
          systemSpecId: systemSpec.id,
          certificateType,
          stop_reason: message.stop_reason,
        });
        throw new Error('Model declined to generate content for safety reasons. Please try adjusting your system specification or certificate type.');
      }
      
      const rawResponse = message.content[0]?.type === 'text' ? message.content[0].text : '';

      // Parse the JSON response
      const parsedResponse = this.parseResponse(rawResponse);
      
      // Validate against schema
      const validatedResponse = LLMCertificateResponseSchema.parse(parsedResponse);

      logger.info('Certificate generated successfully', {
        systemSpecId: systemSpec.id,
        certificateType,
        duration_ms,
        hasExpression: !!validatedResponse.expression,
      });

      return {
        response: validatedResponse,
        duration_ms,
        raw_response: rawResponse,
      };
    } catch (error) {
      const duration_ms = Date.now() - startTime;
      
      logger.error('Failed to generate certificate', {
        systemSpecId: systemSpec.id,
        certificateType,
        duration_ms,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      throw error;
    }
  }

  private buildPrompt(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    mode: LLMConfig['mode']
  ): string {
    const systemDescription = this.formatSystemDescription(systemSpec);
    
    const basePrompt = `You are an expert in formal methods and stability analysis. Your task is to propose a ${certificateType} function for the following dynamical system.

System Description:
${systemDescription}

`;

    const modeSpecificInstructions = this.getModeSpecificInstructions(mode, certificateType);
    const outputFormat = this.getOutputFormat();

    return basePrompt + modeSpecificInstructions + outputFormat;
  }

  private formatSystemDescription(systemSpec: SystemSpec): string {
    const dynamics = systemSpec.dynamics_json;
    let description = `System Type: ${systemSpec.system_type}\n`;
    description += `Dimension: ${systemSpec.dimension}\n`;
    description += `Name: ${systemSpec.name}\n`;
    
    if (systemSpec.description) {
      description += `Description: ${systemSpec.description}\n`;
    }

    if (dynamics.variables) {
      description += `Variables: ${dynamics.variables.join(', ')}\n`;
    }

    if (dynamics.equations) {
      description += `Dynamics:\n`;
      dynamics.equations.forEach((eq: string, idx: number) => {
        description += `  dx${idx + 1}/dt = ${eq}\n`;
      });
    }

    if (dynamics.domain?.constraints) {
      description += `Constraints:\n`;
      dynamics.domain.constraints.forEach((constraint: string) => {
        description += `  ${constraint}\n`;
      });
    }

    if (systemSpec.initial_set_json) {
      description += `Initial Set: ${JSON.stringify(systemSpec.initial_set_json)}\n`;
    }

    if (systemSpec.unsafe_set_json) {
      description += `Unsafe Set: ${JSON.stringify(systemSpec.unsafe_set_json)}\n`;
    }

    return description;
  }

  private getModeSpecificInstructions(
    mode: LLMConfig['mode'],
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant'
  ): string {
    const certificateInstructions = {
      lyapunov: `
A Lyapunov function V(x) must satisfy:
1. V(x) > 0 for all x ≠ 0 (positive definite)
2. V(0) = 0 (zero at equilibrium)
3. dV/dt ≤ 0 along system trajectories (non-increasing)

Note: dV/dt ≤ 0 proves stability, while dV/dt < 0 proves asymptotic stability.
For conservative systems (e.g., undamped oscillators), dV/dt = 0 is valid and proves stability.
`,
      barrier: `
A barrier function B(x) must satisfy:
1. B(x) ≥ 0 for all x in the safe set
2. B(x) = 0 on the boundary between safe and unsafe sets
3. dB/dt ≤ 0 along system trajectories (non-increasing)

This proves that trajectories starting in the safe set remain safe.
`,
      inductive_invariant: `
An inductive invariant I(x) must satisfy:
1. I(x) is true for all initial states
2. If I(x) is true and the system evolves, then I(x') remains true
3. I(x) implies safety (no unsafe states satisfy I(x))

This proves safety by induction.
`,
    };

    const modeInstructions = {
      direct_expression: `
Provide the ${certificateType} function as a direct mathematical expression.
Use standard mathematical notation with the system variables.
`,
      basis_coeffs: `
Express the ${certificateType} function as a linear combination of basis functions.
Provide both the basis functions and their coefficients.
`,
      structure_constraints: `
Provide the ${certificateType} function by specifying its structural form and constraints.
Include any assumptions about the function's structure (e.g., quadratic, polynomial degree).
`,
    };

    return certificateInstructions[certificateType] + modeInstructions[mode];
  }

  private getOutputFormat(): string {
    return `
IMPORTANT: You must respond with valid JSON in exactly this format:

{
  "certificate_type": "lyapunov" | "barrier" | "inductive_invariant",
  "expression": "mathematical expression as string",
  "variables": ["x1", "x2", ...],
  "domain": {
    "bounds": {
      "x1": {"min": -10, "max": 10},
      "x2": {"min": -5, "max": 5}
    },
    "description": "Domain description"
  },
  "properties": {
    "positive_definite": true/false,
    "negative_definite": true/false,
    "decreasing_along_trajectories": true/false,
    "separates_safe_unsafe": true/false
  },
  "reasoning": "Explanation of your reasoning",
  "confidence": 0.85
}

Do not include any text outside of the JSON response. The expression should use standard mathematical notation.
`;
  }

  private parseResponse(rawResponse: string): any {
    try {
      // Extract JSON from response (in case there's extra text)
      const jsonMatch = rawResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in response');
      }

      return JSON.parse(jsonMatch[0]);
    } catch (error) {
      logger.error('Failed to parse LLM response', {
        rawResponse,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw new Error(`Failed to parse LLM response: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      const message = await this.anthropic.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 10,
        messages: [
          {
            role: 'user',
            content: 'Test connection. Respond with "OK".',
          },
        ],
      });

      return message.content[0]?.type === 'text' ? message.content[0].text.trim() === 'OK' : false;
    } catch (error) {
      logger.error('LLM connection test failed', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      return false;
    }
  }
}
