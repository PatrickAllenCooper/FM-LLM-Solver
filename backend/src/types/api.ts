import { z } from 'zod';

// Request/Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface AuthRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  token: string;
  user: {
    id: string;
    email: string;
    role: string;
  };
  expires_at: string;
}

// LLM Configuration
export const LLMConfigSchema = z.object({
  provider: z.literal('anthropic'),
  model: z.string().default('claude-3-5-sonnet-20241022'),
  temperature: z.number().min(0).max(1).default(0.0),
  max_tokens: z.number().min(1).max(4096).default(2048),
  max_attempts: z.number().min(1).max(10).default(3),
  mode: z.enum(['direct_expression', 'basis_coeffs', 'structure_constraints']),
  timeout_ms: z.number().default(30000),
});

export type LLMConfig = z.infer<typeof LLMConfigSchema>;

// System specification schemas
export const DynamicsSchema = z.object({
  type: z.enum(['polynomial', 'nonlinear', 'linear', 'piecewise']),
  variables: z.array(z.string()),
  equations: z.array(z.string()), // Mathematical expressions
  domain: z.object({
    bounds: z.record(z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    })).optional(),
    constraints: z.array(z.string()).optional(), // Inequality constraints
  }).optional(),
});

export const SystemSpecRequestSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
  system_type: z.enum(['continuous', 'discrete', 'hybrid']),
  dimension: z.number().min(1).max(20),
  dynamics: DynamicsSchema,
  constraints: z.record(z.any()).optional(),
  initial_set: z.record(z.any()).optional(),
  unsafe_set: z.record(z.any()).optional(),
});

export type SystemSpecRequest = z.infer<typeof SystemSpecRequestSchema>;

// Certificate generation request  
export const CertificateGenerationRequestSchema = z.object({
  system_spec_id: z.string().min(1), // Accept Firestore IDs, not just UUIDs
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  generation_method: z.enum(['llm', 'sos', 'sdp', 'quadratic_template', 'conversational']),
  llm_config: LLMConfigSchema.optional(),
  baseline_comparison: z.boolean().default(false),
  // Conversational mode specific fields
  conversation_id: z.string().optional(), // For publishing from existing conversation
});

export type CertificateGenerationRequest = z.infer<typeof CertificateGenerationRequestSchema>;

// Re-run acceptance with custom parameters for experimental analysis
export const AcceptanceParametersSchema = z.object({
  sampling_method: z.enum(['uniform', 'sobol', 'lhs', 'adaptive']).default('uniform'),
  sample_count: z.number().min(100).max(10000).default(1000),
  tolerance: z.number().min(1e-12).max(1e-3).default(1e-6),
  max_iterations: z.number().min(100).max(10000).default(1000),
  convergence_threshold: z.number().min(1e-12).max(1e-3).default(1e-8),
  enable_stage_b: z.boolean().default(false),
  custom_margins: z.object({
    positivity_threshold: z.number().optional(),
    decreasing_threshold: z.number().optional(),
    separation_threshold: z.number().optional(),
  }).optional(),
});

export type AcceptanceParameters = z.infer<typeof AcceptanceParametersSchema>;

// Conversational mode types for mathematical dialogue
export const ConversationMessageSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string().min(1),
  timestamp: z.string().datetime(),
  metadata: z.object({
    token_count: z.number().optional(),
    model_used: z.string().optional(),
    message_type: z.enum(['question', 'insight', 'approach', 'refinement', 'final']).optional(),
  }).optional(),
});

export const ConversationSummarySchema = z.object({
  key_insights: z.array(z.string()),
  mathematical_approaches_discussed: z.array(z.string()),
  final_approach_rationale: z.string(),
  conversation_summary: z.string(),
  total_tokens_used: z.number(),
  summarization_timestamp: z.string().datetime(),
});

export const ConversationSchema = z.object({
  id: z.string(),
  system_spec_id: z.string(),
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  status: z.enum(['active', 'summarized', 'published', 'abandoned']),
  messages: z.array(ConversationMessageSchema),
  summary: ConversationSummarySchema.optional(),
  final_certificate_id: z.string().optional(),
  created_by: z.string(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  published_at: z.string().datetime().optional(),
  token_count: z.number().default(0),
  message_count: z.number().default(0),
});

// Request schemas for conversation management
export const StartConversationSchema = z.object({
  system_spec_id: z.string().min(1),
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  initial_message: z.string().min(1).optional(),
});

export const SendMessageSchema = z.object({
  message: z.string().min(1),
  request_insights: z.boolean().default(false), // Request mathematical insights
  force_summarize: z.boolean().default(false), // Force conversation summarization
});

export const PublishCertificateFromConversationSchema = z.object({
  conversation_id: z.string(),
  final_instructions: z.string().optional(), // Additional instructions for final generation
});

export type ConversationMessage = z.infer<typeof ConversationMessageSchema>;
export type ConversationSummary = z.infer<typeof ConversationSummarySchema>;
export type Conversation = z.infer<typeof ConversationSchema>;
export type StartConversationRequest = z.infer<typeof StartConversationSchema>;
export type SendMessageRequest = z.infer<typeof SendMessageSchema>;
export type PublishCertificateFromConversationRequest = z.infer<typeof PublishCertificateFromConversationSchema>;

// Additional conversational schemas for backend validation
export const StartConversationRequestSchema = z.object({
  system_spec_id: z.string().min(1),
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  initial_message: z.string().min(1).optional(),
});

export const SendMessageRequestSchema = z.object({
  message: z.string().min(1),
  request_insights: z.boolean().default(false),
  force_summarize: z.boolean().default(false),
});

export const PublishFromConversationRequestSchema = z.object({
  conversation_id: z.string(),
  final_instructions: z.string().optional(),
});

export type StartConversationRequestType = z.infer<typeof StartConversationRequestSchema>;
export type SendMessageRequestType = z.infer<typeof SendMessageRequestSchema>;
export type PublishFromConversationRequestType = z.infer<typeof PublishFromConversationRequestSchema>;

// Certificate response from LLM
export const LLMCertificateResponseSchema = z.object({
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  expression: z.string(), // Mathematical expression as string
  variables: z.array(z.string()),
  domain: z.object({
    bounds: z.record(z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    })).optional(),
    description: z.string().optional(),
  }).optional(),
  properties: z.object({
    positive_definite: z.boolean().optional(),
    negative_definite: z.boolean().optional(),
    decreasing_along_trajectories: z.boolean().optional(),
    separates_safe_unsafe: z.boolean().optional(),
  }).optional(),
  reasoning: z.string().optional(),
  confidence: z.number().min(0).max(1).optional(),
});

export type LLMCertificateResponse = z.infer<typeof LLMCertificateResponseSchema>;

// Detailed acceptance result for experimental analysis
export interface AcceptanceResult {
  accepted: boolean;
  margin?: number;
  counterexample?: {
    state: Record<string, number>;
    violation_type: string;
    violation_magnitude: number;
  };
  acceptance_method: 'symbolic' | 'numerical' | 'smt' | 'mathematical';
  duration_ms: number;
  solver_output?: string;
  // Enhanced technical details for experimental work
  technical_details?: {
    conditions_checked: string[];
    sampling_method: 'uniform' | 'sobol' | 'lhs' | 'adaptive';
    sample_count: number;
    domain_coverage: Record<string, { min: number; max: number }>;
    violation_analysis: {
      total_violations: number;
      violation_points: Array<{
        point: Record<string, number>;
        condition: string;
        value: number;
        severity: 'minor' | 'moderate' | 'severe';
      }>;
    };
    margin_breakdown: {
      positivity_margin?: number;
      decreasing_margin?: number;
      separation_margin?: number;
      invariant_margin?: number;
    };
    numerical_parameters: {
      tolerance: number;
      max_iterations: number;
      convergence_threshold: number;
    };
    stage_results: {
      stage_a_passed: boolean;
      stage_b_enabled: boolean;
      stage_b_passed?: boolean;
    };
  };
}

// Experiment configuration
export const ExperimentConfigSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
  system_specs: z.array(z.string().uuid()),
  llm_configs: z.array(LLMConfigSchema),
  baseline_methods: z.array(z.enum(['sos', 'sdp', 'quadratic_template'])),
  budget_per_spec: z.object({
    max_total_attempts: z.number().min(1).default(10),
    max_time_minutes: z.number().min(1).default(60),
  }),
  acceptance_config: z.object({
    timeout_seconds: z.number().min(1).default(30),
    numerical_precision: z.number().default(1e-6),
  }),
});

export type ExperimentConfig = z.infer<typeof ExperimentConfigSchema>;

// Pagination
export interface PaginationParams {
  page: number;
  limit: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    total_pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

// Error types
export interface ValidationError {
  field: string;
  message: string;
  value?: any;
}

export interface ApiError {
  code: string;
  message: string;
  details?: ValidationError[];
}
