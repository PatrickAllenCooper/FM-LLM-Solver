import { LLMService } from '../../src/services/llm.service';
import { SystemSpec } from '../../src/types/database';
import { LLMConfig } from '../../src/types/api';
import Anthropic from '@anthropic-ai/sdk';

// Mock Anthropic SDK
const mockCreate = jest.fn();
jest.mock('@anthropic-ai/sdk', () => {
  return jest.fn().mockImplementation(() => ({
    messages: {
      create: mockCreate
    }
  }));
});

describe('LLMService', () => {
  let llmService: LLMService;
  let mockAnthropic: jest.Mocked<Anthropic>;
  let mockSystemSpec: SystemSpec;
  let mockConfig: LLMConfig;

  beforeEach(() => {
    jest.clearAllMocks();
    mockCreate.mockClear();
    
    llmService = new LLMService('test-api-key');

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
        domain: {
          constraints: ['x1^2 + x2^2 <= 10'],
        },
      },
      initial_set_json: {
        type: 'box',
        bounds: { x1: { min: -1, max: 1 }, x2: { min: -1, max: 1 } }
      },
      unsafe_set_json: {
        type: 'box',
        bounds: { x1: { min: 5, max: 10 }, x2: { min: 5, max: 10 } }
      },
      hash: 'test-hash',
      description: 'Test system for Lyapunov analysis',
      created_at: new Date(),
      updated_at: new Date(),
    };

    // Create mock LLM configuration
    mockConfig = {
      provider: 'anthropic',
      model: 'claude-sonnet-4-20250514',
      mode: 'direct_expression',
      temperature: 0.0,
      max_tokens: 1000,
      max_attempts: 3,
      timeout_ms: 30000,
    };
  });

  describe('generateCertificate', () => {
    it('should generate Lyapunov certificate successfully', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            domain: {
              bounds: {
                x1: { min: -10, max: 10 },
                x2: { min: -10, max: 10 }
              },
              description: 'State space domain'
            },
            properties: {
              positive_definite: true,
              negative_definite: false,
              decreasing_along_trajectories: true,
              separates_safe_unsafe: false
            },
            reasoning: 'This is a standard quadratic Lyapunov function for a stable linear system.',
            confidence: 0.95
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);

      expect(result.response.certificate_type).toBe('lyapunov');
      expect(result.response.expression).toBe('x1^2 + x2^2');
      expect(result.response.variables).toEqual(['x1', 'x2']);
      expect(result.response.confidence).toBe(0.95);
      expect(result.duration_ms).toBeGreaterThanOrEqual(0);
      expect(result.raw_response).toBeTruthy();

      expect(mockCreate).toHaveBeenCalledWith({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        temperature: 0.0,
        messages: [{
          role: 'user',
          content: expect.stringContaining('Lyapunov function'),
        }],
      });
    });

    it('should generate barrier certificate successfully', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'barrier',
            expression: '10 - x1^2 - x2^2',
            variables: ['x1', 'x2'],
            domain: {
              bounds: {
                x1: { min: -5, max: 5 },
                x2: { min: -5, max: 5 }
              },
              description: 'Safety domain'
            },
            properties: {
              positive_definite: false,
              negative_definite: false,
              decreasing_along_trajectories: false,
              separates_safe_unsafe: true
            },
            reasoning: 'This barrier function separates the safe region from the unsafe region.',
            confidence: 0.88
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.generateCertificate(mockSystemSpec, 'barrier', mockConfig);

      expect(result.response.certificate_type).toBe('barrier');
      expect(result.response.expression).toBe('10 - x1^2 - x2^2');
      expect(result.response.properties?.separates_safe_unsafe).toBe(true);
      expect(result.duration_ms).toBeGreaterThanOrEqual(0);
    });

    it('should generate inductive invariant successfully', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'inductive_invariant',
            expression: 'x1^2 + x2^2 <= 5',
            variables: ['x1', 'x2'],
            domain: {
              bounds: {
                x1: { min: -3, max: 3 },
                x2: { min: -3, max: 3 }
              },
              description: 'Invariant domain'
            },
            properties: {
              positive_definite: false,
              negative_definite: false,
              decreasing_along_trajectories: false,
              separates_safe_unsafe: true
            },
            reasoning: 'This invariant is preserved by the system dynamics.',
            confidence: 0.85
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.generateCertificate(mockSystemSpec, 'inductive_invariant', mockConfig);

      expect(result.response.certificate_type).toBe('inductive_invariant');
      expect(result.response.expression).toBe('x1^2 + x2^2 <= 5');
      expect(result.duration_ms).toBeGreaterThanOrEqual(0);
    });

    it('should handle different LLM modes', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            domain: { bounds: {}, description: '' },
            properties: { positive_definite: true, negative_definite: false, decreasing_along_trajectories: true, separates_safe_unsafe: false },
            reasoning: 'Test',
            confidence: 0.9
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      // Test basis_coeffs mode
      const basisConfig = { ...mockConfig, mode: 'basis_coeffs' as const };
      const result1 = await llmService.generateCertificate(mockSystemSpec, 'lyapunov', basisConfig);
      expect(result1.response).toBeTruthy();

      // Test structure_constraints mode
      const structureConfig = { ...mockConfig, mode: 'structure_constraints' as const };
      const result2 = await llmService.generateCertificate(mockSystemSpec, 'lyapunov', structureConfig);
      expect(result2.response).toBeTruthy();

      // Verify different prompts were generated
      expect(mockCreate).toHaveBeenCalledTimes(2);
    });

    it('should handle API errors gracefully', async () => {
      mockCreate.mockRejectedValue(new Error('API rate limit exceeded'));

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow('API rate limit exceeded');
    });

    it('should handle malformed JSON responses', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: 'This is not valid JSON'
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow('Failed to parse LLM response');
    });

    it('should handle empty responses', async () => {
      const mockResponse = {
        content: []
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow('Failed to parse LLM response');
    });

    it('should validate response schema', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            // Missing required fields
            certificate_type: 'lyapunov',
            // expression is missing
            variables: ['x1', 'x2'],
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow();
    });
  });

  describe('testConnection', () => {
    it('should return true for successful connection', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: 'OK'
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.testConnection();

      expect(result).toBe(true);
      expect(mockCreate).toHaveBeenCalledWith({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 10,
        messages: [{
          role: 'user',
          content: 'Test connection. Respond with "OK".',
        }],
      });
    });

    it('should return false for failed connection', async () => {
      mockCreate.mockRejectedValue(new Error('Network error'));

      const result = await llmService.testConnection();

      expect(result).toBe(false);
    });

    it('should return false for unexpected response', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: 'Unexpected response'
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.testConnection();

      expect(result).toBe(false);
    });
  });

  describe('Prompt generation', () => {
    it('should generate appropriate prompts for different certificate types', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            domain: { bounds: {}, description: '' },
            properties: { positive_definite: true, negative_definite: false, decreasing_along_trajectories: true, separates_safe_unsafe: false },
            reasoning: 'Test',
            confidence: 0.9
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      // Test Lyapunov prompt
      await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);
      const lyapunovCall = mockCreate.mock.calls[0][0];
      expect(lyapunovCall.messages[0].content).toContain('Lyapunov function');
      expect(lyapunovCall.messages[0].content).toContain('positive definite');
      expect(lyapunovCall.messages[0].content).toContain('decreasing');

      // Test Barrier prompt
      await llmService.generateCertificate(mockSystemSpec, 'barrier', mockConfig);
      const barrierCall = mockCreate.mock.calls[1][0];
      expect(barrierCall.messages[0].content).toContain('barrier function');
      expect(barrierCall.messages[0].content).toContain('safe set');
      expect(barrierCall.messages[0].content).toContain('non-increasing');

      // Test Inductive invariant prompt
      await llmService.generateCertificate(mockSystemSpec, 'inductive_invariant', mockConfig);
      const invariantCall = mockCreate.mock.calls[2][0];
      expect(invariantCall.messages[0].content).toContain('inductive invariant');
      expect(invariantCall.messages[0].content).toContain('initial states');
      expect(invariantCall.messages[0].content).toContain('safety');
    });

    it('should include system information in prompts', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            domain: { bounds: {}, description: '' },
            properties: { positive_definite: true, negative_definite: false, decreasing_along_trajectories: true, separates_safe_unsafe: false },
            reasoning: 'Test',
            confidence: 0.9
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);

      const promptContent = mockCreate.mock.calls[0][0].messages[0].content;
      expect(promptContent).toContain('Test Linear System');
      expect(promptContent).toContain('continuous');
      expect(promptContent).toContain('Dimension: 2');
      expect(promptContent).toContain('x1, x2');
      expect(promptContent).toContain('-x1 + 0.1*x2');
      expect(promptContent).toContain('-x2 - 0.1*x1');
    });

    it('should include JSON format requirements', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            domain: { bounds: {}, description: '' },
            properties: { positive_definite: true, negative_definite: false, decreasing_along_trajectories: true, separates_safe_unsafe: false },
            reasoning: 'Test',
            confidence: 0.9
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);

      const promptContent = mockCreate.mock.calls[0][0].messages[0].content;
      expect(promptContent).toContain('valid JSON');
      expect(promptContent).toContain('certificate_type');
      expect(promptContent).toContain('expression');
      expect(promptContent).toContain('variables');
      expect(promptContent).toContain('confidence');
    });
  });

  describe('Error handling and edge cases', () => {
    it('should handle network timeouts', async () => {
      mockCreate.mockImplementation(() => {
        return new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Request timeout')), 100);
        });
      });

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow('Request timeout');
    });

    it('should handle invalid API key', async () => {
      mockCreate.mockRejectedValue(new Error('Invalid API key'));

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow('Invalid API key');
    });

    it('should handle rate limiting', async () => {
      mockCreate.mockRejectedValue(new Error('Rate limit exceeded'));

      await expect(
        llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig)
      ).rejects.toThrow('Rate limit exceeded');
    });

    it('should handle partial JSON in response', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: 'Some text before { "certificate_type": "lyapunov", "expression": "x1^2 + x2^2", "variables": ["x1", "x2"], "domain": {"bounds": {}, "description": ""}, "properties": {"positive_definite": true, "negative_definite": false, "decreasing_along_trajectories": true, "separates_safe_unsafe": false}, "reasoning": "Test", "confidence": 0.9 } some text after'
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);
      expect(result.response.certificate_type).toBe('lyapunov');
      expect(result.response.expression).toBe('x1^2 + x2^2');
    });

    it('should handle very large responses', async () => {
      const largeExpression = 'x1^2 + x2^2 + ' + 'x1*x2 + '.repeat(1000) + '1';
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: largeExpression,
            variables: ['x1', 'x2'],
            domain: { bounds: {}, description: '' },
            properties: { positive_definite: true, negative_definite: false, decreasing_along_trajectories: true, separates_safe_unsafe: false },
            reasoning: 'Very long reasoning text'.repeat(100),
            confidence: 0.9
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const result = await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);
      expect(result.response.expression).toBe(largeExpression);
    });
  });

  describe('Performance', () => {
    it('should complete generation within reasonable time', async () => {
      const mockResponse = {
        content: [{
          type: 'text',
          text: JSON.stringify({
            certificate_type: 'lyapunov',
            expression: 'x1^2 + x2^2',
            variables: ['x1', 'x2'],
            domain: { bounds: {}, description: '' },
            properties: { positive_definite: true, negative_definite: false, decreasing_along_trajectories: true, separates_safe_unsafe: false },
            reasoning: 'Test',
            confidence: 0.9
          })
        }]
      };

      mockCreate.mockResolvedValue(mockResponse as any);

      const startTime = Date.now();
      const result = await llmService.generateCertificate(mockSystemSpec, 'lyapunov', mockConfig);
      const actualDuration = Date.now() - startTime;

      expect(result.duration_ms).toBeGreaterThanOrEqual(0);
      expect(result.duration_ms).toBeDefined();
      expect(actualDuration).toBeLessThan(5000); // Should complete within 5 seconds for mocked response
    });
  });
});
