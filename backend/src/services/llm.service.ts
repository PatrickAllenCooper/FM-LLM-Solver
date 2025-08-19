import Anthropic from '@anthropic-ai/sdk';
import { LLMConfig, LLMCertificateResponse, LLMCertificateResponseSchema, ConversationMessage, ConversationSummary } from '../types/api';
import { SystemSpec } from '../types/database';
import { logger } from '../utils/logger';

export class LLMService {
  private anthropic: Anthropic;

  constructor(apiKey: string) {
    if (!apiKey || apiKey.trim().length === 0) {
      logger.error('Anthropic API key is missing or empty');
      throw new Error('Anthropic API key is required but not provided');
    }
    
    // Log key info for debugging (without exposing the actual key)
    logger.info('Initializing Anthropic client', {
      hasApiKey: !!apiKey,
      keyLength: apiKey.length,
      keyPrefix: apiKey.substring(0, 8) + '...',
    });
    
    this.anthropic = new Anthropic({
      apiKey: apiKey,
      // Ensure we use the correct API version
      defaultHeaders: {
        'anthropic-version': '2023-06-01',
      },
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
      
      // Log API call attempt
      logger.info('Making Anthropic API call', {
        model: config.model,
        hasApiKey: !!process.env.ANTHROPIC_API_KEY,
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
      }).catch((error) => {
        // Capture detailed API error information
        logger.error('Detailed Anthropic API error', {
          systemSpecId: systemSpec.id,
          model: config.model,
          errorMessage: error.message,
          errorName: error.name,
          errorStatus: error.status,
          errorType: error.type,
          errorCode: error.code,
          fullError: JSON.stringify(error, null, 2),
        });
        throw error;
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
      
      // Enhanced error handling for Claude 4 models
      let errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Check if this is a Claude 4 model access issue
      if (errorMessage.includes('Connection error') && config.model.includes('claude-4') || config.model.includes('claude-opus-4') || config.model.includes('claude-sonnet-4')) {
        errorMessage = `Claude 4 model "${config.model}" is not accessible. This may be due to limited API access. Try using Claude 3.5 Sonnet instead.`;
        logger.warn('Claude 4 model access denied', {
          systemSpecId: systemSpec.id,
          model: config.model,
          originalError: error instanceof Error ? error.message : 'Unknown error',
        });
      }
      
      logger.error('Failed to generate certificate', {
        systemSpecId: systemSpec.id,
        certificateType,
        model: config.model,
        duration_ms,
        error: errorMessage,
      });

      throw new Error(errorMessage);
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
        model: 'claude-3-5-sonnet-20241022', // Use known working model for test
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

  // ====== CONVERSATIONAL MODE METHODS ======

  /**
   * Initialize a mathematical conversation about certificate generation
   */
  async initializeConversation(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    config: LLMConfig,
    initialMessage?: string
  ): Promise<ConversationMessage> {
    const startTime = Date.now();
    
    try {
      const systemContext = this.buildConversationSystemPrompt(systemSpec, certificateType);
      const userPrompt = initialMessage || this.getDefaultConversationStarter(certificateType);
      
      logger.info('Initializing mathematical conversation', {
        systemSpecId: systemSpec.id,
        certificateType,
        model: config.model,
        hasInitialMessage: !!initialMessage,
      });

      const message = await this.anthropic.messages.create({
        model: config.model,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        system: systemContext,
        messages: [
          {
            role: 'user',
            content: userPrompt,
          },
        ],
      });

      const assistantResponse = message.content[0]?.type === 'text' ? message.content[0].text : '';
      
      logger.info('Conversation initialized successfully', {
        systemSpecId: systemSpec.id,
        responseLength: assistantResponse.length,
        tokensUsed: message.usage?.output_tokens || 0,
        duration_ms: Date.now() - startTime,
      });

      return {
        role: 'assistant',
        content: assistantResponse,
        timestamp: new Date().toISOString(),
        metadata: {
          token_count: message.usage?.output_tokens || 0,
          model_used: config.model,
          message_type: 'approach',
        },
      };
    } catch (error) {
      logger.error('Failed to initialize conversation', {
        systemSpecId: systemSpec.id,
        certificateType,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration_ms: Date.now() - startTime,
      });
      throw new Error(`Conversation initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Send a message in an ongoing conversation
   */
  async sendConversationMessage(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    conversationHistory: ConversationMessage[],
    newMessage: string,
    config: LLMConfig
  ): Promise<ConversationMessage> {
    const startTime = Date.now();
    
    try {
      const systemContext = this.buildConversationSystemPrompt(systemSpec, certificateType);
      
      // Convert conversation history to Anthropic format
      const messages = conversationHistory.map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
      }));
      
      // Add new user message
      messages.push({
        role: 'user',
        content: newMessage,
      });

      logger.info('Sending conversation message', {
        systemSpecId: systemSpec.id,
        messageCount: messages.length,
        model: config.model,
      });

      const response = await this.anthropic.messages.create({
        model: config.model,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        system: systemContext,
        messages: messages,
      });

      const assistantResponse = response.content[0]?.type === 'text' ? response.content[0].text : '';
      
      logger.info('Conversation message sent successfully', {
        systemSpecId: systemSpec.id,
        responseLength: assistantResponse.length,
        tokensUsed: response.usage?.output_tokens || 0,
        duration_ms: Date.now() - startTime,
      });

      return {
        role: 'assistant',
        content: assistantResponse,
        timestamp: new Date().toISOString(),
        metadata: {
          token_count: response.usage?.output_tokens || 0,
          model_used: config.model,
          message_type: 'refinement',
        },
      };
    } catch (error) {
      logger.error('Failed to send conversation message', {
        systemSpecId: systemSpec.id,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration_ms: Date.now() - startTime,
      });
      throw new Error(`Conversation message failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Summarize a long conversation to preserve context while reducing tokens
   */
  async summarizeConversation(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    conversationHistory: ConversationMessage[],
    config: LLMConfig
  ): Promise<ConversationSummary> {
    const startTime = Date.now();
    
    try {
      const conversationText = conversationHistory
        .map(msg => `${msg.role === 'user' ? 'Researcher' : 'Assistant'}: ${msg.content}`)
        .join('\n\n');

      const summarizationPrompt = `
You are a mathematical research assistant. Please analyze this conversation about ${certificateType} function generation for the given dynamical system and provide a comprehensive summary.

System: ${systemSpec.name} (${certificateType} certificate)

Conversation to summarize:
${conversationText}

Please provide a JSON response with the following structure:
{
  "key_insights": ["list of key mathematical insights discovered"],
  "mathematical_approaches_discussed": ["specific approaches discussed"],
  "final_approach_rationale": "the reasoning for the final approach",
  "conversation_summary": "comprehensive summary preserving mathematical context",
  "total_tokens_used": ${conversationHistory.reduce((sum, msg) => sum + (msg.metadata?.token_count || 0), 0)},
  "summarization_timestamp": "${new Date().toISOString()}"
}
`;

      logger.info('Summarizing conversation', {
        systemSpecId: systemSpec.id,
        messageCount: conversationHistory.length,
        model: config.model,
      });

      const response = await this.anthropic.messages.create({
        model: config.model,
        max_tokens: Math.min(config.max_tokens, 4096), // Ensure enough space for summary
        temperature: 0.1, // Low temperature for consistent summarization
        messages: [
          {
            role: 'user',
            content: summarizationPrompt,
          },
        ],
      });

      const summaryResponse = response.content[0]?.type === 'text' ? response.content[0].text : '';
      const summary = JSON.parse(summaryResponse);
      
      logger.info('Conversation summarized successfully', {
        systemSpecId: systemSpec.id,
        originalMessages: conversationHistory.length,
        summaryLength: summary.conversation_summary?.length || 0,
        duration_ms: Date.now() - startTime,
      });

      return summary;
    } catch (error) {
      logger.error('Failed to summarize conversation', {
        systemSpecId: systemSpec.id,
        messageCount: conversationHistory.length,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration_ms: Date.now() - startTime,
      });
      throw new Error(`Conversation summarization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate final certificate from conversation context
   */
  async generateCertificateFromConversation(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    conversationSummary: ConversationSummary,
    finalInstructions: string | undefined,
    config: LLMConfig
  ): Promise<{
    response: LLMCertificateResponse;
    duration_ms: number;
    raw_response: string;
    conversation_context: string;
  }> {
    const startTime = Date.now();
    
    try {
      const conversationContext = this.buildConversationCertificatePrompt(
        systemSpec,
        certificateType,
        conversationSummary,
        finalInstructions,
        config.mode
      );
      
      logger.info('Generating certificate from conversation', {
        systemSpecId: systemSpec.id,
        certificateType,
        conversationInsights: conversationSummary.key_insights.length,
        model: config.model,
      });

      const message = await this.anthropic.messages.create({
        model: config.model,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        messages: [
          {
            role: 'user',
            content: conversationContext,
          },
        ],
      });

      const duration_ms = Date.now() - startTime;
      const rawResponse = message.content[0]?.type === 'text' ? message.content[0].text : '';
      
      // Handle Claude 4 refusal stop reason
      if ((message.stop_reason as string) === 'refusal') {
        logger.warn('Claude 4 model refused to generate certificate from conversation', {
          systemSpecId: systemSpec.id,
          certificateType,
          stop_reason: message.stop_reason,
        });
        throw new Error('Model declined to generate certificate for safety reasons. Please review conversation content.');
      }
      
      const response = this.parseResponse(rawResponse);
      
      logger.info('Certificate generated from conversation successfully', {
        systemSpecId: systemSpec.id,
        certificateType,
        duration_ms,
        tokensUsed: message.usage?.output_tokens || 0,
      });

      return {
        response,
        duration_ms,
        raw_response: rawResponse,
        conversation_context: conversationSummary.conversation_summary,
      };
    } catch (error) {
      logger.error('Failed to generate certificate from conversation', {
        systemSpecId: systemSpec.id,
        certificateType,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration_ms: Date.now() - startTime,
      });
      throw new Error(`Certificate generation from conversation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // ====== CONVERSATIONAL MODE HELPER METHODS ======

  private buildConversationSystemPrompt(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant'
  ): string {
    const systemDescription = this.formatSystemDescription(systemSpec);
    
    return `You are a mathematical research assistant specializing in formal methods and stability analysis. You are helping a researcher explore approaches for generating ${certificateType} functions for dynamical systems.

SYSTEM SPECIFICATION:
${systemDescription}

YOUR ROLE:
- Engage in mathematical dialogue about ${certificateType} function theory and approaches
- Ask clarifying questions about mathematical requirements and preferences
- Propose and discuss different mathematical strategies
- Provide insights about stability theory and formal verification
- Guide the conversation toward a well-reasoned mathematical approach
- Be thorough in your mathematical reasoning and explanations

CONVERSATION GUIDELINES:
- Focus on mathematical rigor and theoretical soundness
- Discuss trade-offs between different approaches
- Ask about specific mathematical properties the researcher is interested in
- Explain your reasoning for proposed approaches
- Be collaborative - this is a dialogue, not a lecture
- When ready, you can propose specific mathematical expressions for discussion

Do not generate a final certificate unless explicitly asked to "publish" or "finalize" the result. Instead, engage in mathematical conversation and refinement.`;
  }

  private getDefaultConversationStarter(certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant'): string {
    const starters = {
      lyapunov: `I'd like to explore approaches for generating a Lyapunov function for this system. Can you help me think through different mathematical strategies? What are some key considerations for this particular system?`,
      barrier: `I'm interested in generating a barrier certificate for this system. What approaches might work well given the system dynamics and safety requirements? Let's discuss the mathematical considerations.`,
      inductive_invariant: `I want to develop an inductive invariant for this system. What mathematical approaches should we consider? Can you help me think through the theoretical requirements?`,
    };
    
    return starters[certificateType];
  }

  private buildConversationCertificatePrompt(
    systemSpec: SystemSpec,
    certificateType: 'lyapunov' | 'barrier' | 'inductive_invariant',
    conversationSummary: ConversationSummary,
    finalInstructions: string | undefined,
    mode: LLMConfig['mode']
  ): string {
    const systemDescription = this.formatSystemDescription(systemSpec);
    const outputFormat = this.getOutputFormat();

    return `You are a mathematical research assistant. Based on our detailed conversation, please generate a ${certificateType} function for the following system.

SYSTEM SPECIFICATION:
${systemDescription}

CONVERSATION SUMMARY:
Key Insights: ${conversationSummary.key_insights.join(', ')}
Approaches Discussed: ${conversationSummary.mathematical_approaches_discussed.join(', ')}
Final Approach Rationale: ${conversationSummary.final_approach_rationale}

FULL CONVERSATION CONTEXT:
${conversationSummary.conversation_summary}

${finalInstructions ? `ADDITIONAL INSTRUCTIONS: ${finalInstructions}` : ''}

Based on our conversation and the insights we've developed together, please generate a ${certificateType} function using the ${mode} approach we discussed.

${this.getModeSpecificInstructions(mode, certificateType)}

${outputFormat}`;
  }
}
